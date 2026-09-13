"""Layer 3: re-used or lightly modified figures, via perceptual hashing.

Every extracted figure (raster image or rendered vector cluster) gets:
  * sha256 of its source bytes          -> identical embedded object  (`exact`)
  * pHash (DCT) and dHash, 64 bit each  -> rescaled / recompressed / recoloured (`same`)
  * a 128 px grayscale thumbnail        -> multi-scale normalised cross-correlation
                                           template match, which finds one figure
                                           cropped out of the other (`cropped`)
                                           and gives a global NCC confidence number

Verdicts, strongest first: exact, same, cropped, similar (pHash within the
looser threshold).  All exact/same/cropped partners of an A figure are
reported; a `similar` partner only when it is the best available.
No model, no network.

`score_pair` is deliberately a standalone function: a vision-embedding scorer
(DINOv2/CLIP) for re-drawn figures could be added beside it later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import imagehash
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

from .extract import Document, Figure

THUMB = 64
CROP_SIDE = 128
CROP_SCALES = np.geomspace(0.25, 1.0, 9)


@dataclass
class ImageParams:
    phash_threshold: int = 10
    similar_threshold: int = 16
    dhash_threshold: int = 16
    crop_ncc: float = 0.85           # template-match NCC needed for `cropped` (unrelated figures reach ~0.80)
    min_std: float = 4.0             # near-blank renders are ignored
    max_side: int = 768              # hashing works on a downscaled copy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FigureHashes:
    figure: int
    phash: str
    dhash: str
    blank: bool
    thumb: np.ndarray | None     # 64x64, zero-mean, unit-norm
    gray: np.ndarray | None      # <=128 px grayscale float32


@dataclass
class FigureMatch:
    a: int
    b: int
    verdict: str        # exact | same | cropped | similar
    phash_distance: int
    dhash_distance: int
    ncc: float          # whole-figure correlation
    crop_ncc: float     # best template-match correlation, either direction
    a_page: int
    b_page: int
    a_caption: str
    b_caption: str
    a_path: str
    b_path: str
    a_bbox: tuple[float, float, float, float]
    b_bbox: tuple[float, float, float, float]


@dataclass
class ImageResult:
    matches: list[FigureMatch]
    n_a: int
    n_b: int
    blank_a: int
    blank_b: int
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- hashing -----------------------------------------------------------------

def _load(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def hash_figure(fig: Figure, params: ImageParams | None = None) -> FigureHashes:
    params = params or ImageParams()
    img = _load(fig.path, params.max_side)
    g = img.convert("L")
    gray_full = np.asarray(g, dtype=np.float32)
    if float(gray_full.std()) < params.min_std:
        return FigureHashes(fig.id, "", "", True, None, None)
    thumb = np.asarray(g.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    thumb = thumb / norm if norm > 0 else thumb
    small = g.copy()
    small.thumbnail((CROP_SIDE, CROP_SIDE))
    gray = np.asarray(small, dtype=np.float32)
    return FigureHashes(fig.id, str(imagehash.phash(img)), str(imagehash.dhash(img)), False, thumb, gray)


def _ncc(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip((a * b).sum(), -1.0, 1.0))


def _ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation of `template` at every valid
    position of `image` (the classic match_template formulation, via FFT)."""
    th, tw = template.shape
    ih, iw = image.shape
    if th > ih or tw > iw or th < 4 or tw < 4:
        return np.zeros((0, 0), dtype=np.float32)
    t = template - template.mean()
    tnorm = float(np.sqrt((t * t).sum()))
    if tnorm < 1e-6:
        return np.zeros((0, 0), dtype=np.float32)
    ones = np.ones_like(template)
    n = float(th * tw)
    s1 = fftconvolve(image, ones, mode="valid")
    s2 = fftconvolve(image * image, ones, mode="valid")
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    xcorr = fftconvolve(image, t[::-1, ::-1], mode="valid")
    denom = np.sqrt(var) * tnorm
    out = np.where(denom > 1e-3 * tnorm, xcorr / np.maximum(denom, 1e-6), 0.0)
    return np.clip(out, -1.0, 1.0)


def _ncc_at_width(big: np.ndarray, small: np.ndarray, tw: int) -> float:
    bh, bw = big.shape
    sh, sw = small.shape
    th = min(int(round(tw * sh / sw)), bh)   # clamp: a 1 px aspect rounding must not reject the scale
    if tw < 12 or th < 12 or tw > bw:
        return -1.0
    t = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((tw, th), Image.BILINEAR),
                   dtype=np.float32)
    m = _ncc_map(big, t)
    return float(m.max()) if m.size else -1.0


def crop_score(big: np.ndarray, small: np.ndarray) -> float:
    """Best NCC of `small`, rescaled over a range of widths, slid over `big`.
    High when `small` is a (possibly rescaled) crop of `big`.  Widths are
    fractions of the largest template that fits inside `big`; a coarse sweep
    is followed by a fine sweep around the best coarse scale, because NCC
    falls off quickly when the template is a few percent off scale."""
    bh, bw = big.shape
    sh, sw = small.shape
    max_tw = min(bw, int(bh * sw / sh))
    if max_tw < 12:
        return -1.0
    scores: dict[int, float] = {}
    for frac in CROP_SCALES:
        tw = int(round(max_tw * frac))
        if tw not in scores:
            scores[tw] = _ncc_at_width(big, small, tw)
    best_tw = max(scores, key=scores.get)
    for f in np.linspace(0.9, 1.1, 9):
        tw = min(int(round(best_tw * f)), max_tw)
        if tw not in scores:
            scores[tw] = _ncc_at_width(big, small, tw)
    return max(scores.values())


def score_pair(fa: Figure, fb: Figure, ha: FigureHashes, hb: FigureHashes,
               params: ImageParams) -> FigureMatch | None:
    if ha.blank or hb.blank:
        return None
    pd = int(imagehash.hex_to_hash(ha.phash) - imagehash.hex_to_hash(hb.phash))
    dd = int(imagehash.hex_to_hash(ha.dhash) - imagehash.hex_to_hash(hb.dhash))
    ncc = round(_ncc(ha.thumb, hb.thumb), 3)
    if fa.sha256 == fb.sha256:
        verdict, cscore = "exact", 1.0
    elif pd <= params.phash_threshold and dd <= params.dhash_threshold:
        verdict, cscore = "same", ncc
    else:
        cscore = max(crop_score(ha.gray, hb.gray), crop_score(hb.gray, ha.gray))
        if cscore >= params.crop_ncc:
            verdict = "cropped"
        elif pd <= params.similar_threshold:
            verdict = "similar"
        else:
            return None
    return FigureMatch(fa.id, fb.id, verdict, pd, dd, ncc, round(float(cscore), 3),
                       fa.page, fb.page, fa.caption, fb.caption, fa.path, fb.path, fa.bbox, fb.bbox)


_RANK = {"exact": 0, "same": 1, "cropped": 2, "similar": 3}


def compare(doc_a: Document, doc_b: Document, params: ImageParams | None = None) -> ImageResult:
    params = params or ImageParams()
    ha = {f.id: hash_figure(f, params) for f in doc_a.figures}
    hb = {f.id: hash_figure(f, params) for f in doc_b.figures}
    matches: list[FigureMatch] = []
    for fa in doc_a.figures:
        candidates = [m for fb in doc_b.figures
                      if (m := score_pair(fa, fb, ha[fa.id], hb[fb.id], params)) is not None]
        if not candidates:
            continue
        candidates.sort(key=lambda m: (_RANK[m.verdict], m.phash_distance, -m.crop_ncc))
        matches.append(candidates[0])
        matches.extend(m for m in candidates[1:] if m.verdict != "similar")
    return ImageResult(matches, len(doc_a.figures), len(doc_b.figures),
                       sum(h.blank for h in ha.values()), sum(h.blank for h in hb.values()),
                       params.to_dict())
