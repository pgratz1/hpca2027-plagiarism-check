"""PDF -> Document: text units (paragraphs and sentences with page + bbox) and
figures (raster images and rendered vector-drawing clusters).

Uses PyMuPDF only; nothing here touches the network.

Per page:
  1. page.get_text("dict") in natural content order (LaTeX PDFs emit columns
     in reading order; sorting by y would interleave two-column text).
  2. Drop running headers/footers (same digit-blind text in the top/bottom
     margin on >= 3 pages), bare page numbers, and rotated text (arXiv stamps).
  3. Classify blocks: heading / caption / abstract / body / reference.  The
     reference list is dropped unless `include_references` (bibliographies
     legitimately overlap between related papers).  Blocks without a real
     word (glyphs from drawing fonts) and short labels inside vector figures
     are dropped too.
  4. Figures: raster images via get_images/get_image_rects, vector figures via
     cluster_drawings rendered at `figure_dpi`.  Each figure is written as a
     PNG and tagged with the nearest caption.
  5. OCR fallback through the `tesseract` binary when a page yields fewer than
     `ocr_min_chars` characters *and* carries a page-sized image (a scan).
     `ocr="on"` forces OCR on every page; `ocr="off"` disables it.

Results are cached per (pdf sha256, schema, options); see pdfsim.cache.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from . import cache as _cache
from .textnorm import normalize, split_sentences, tokenize

BBox = tuple[float, float, float, float]

_CAPTION_RE = re.compile(r"^(fig\.?|figure|table|tab\.?|listing|algorithm)\s*\d+", re.I)
_REFERENCES_RE = re.compile(r"^(\d+\.?\s*|[ivx]+\.\s*)?(references|bibliography)\s*$", re.I)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.?|[A-Z]\.)\s+[A-Z]")
_ABSTRACT_HEADING_RE = re.compile(r"^abstract\s*$", re.I)
_ABSTRACT_INLINE_RE = re.compile(r"^abstract\b", re.I)
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_DIGITS = re.compile(r"\d+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_ALPHA = re.compile(r"[^\W\d_]{2,}")

SENTENCE_KINDS = ("body", "abstract", "caption", "reference")
FIGURE_TEXT_MAX_TOKENS = 8   # shorter text inside a vector figure is an axis label, not prose


# --- data model --------------------------------------------------------------

@dataclass
class Line:
    start: int
    end: int
    page: int
    bbox: BBox


@dataclass
class TextUnit:
    id: int
    kind: str
    text: str
    page: int
    para: int                      # parent paragraph id (== id for paragraphs)
    span: tuple[int, int]          # char span inside the parent paragraph text
    lines: list[Line] = field(default_factory=list)              # paragraphs only
    rects: list[tuple[int, BBox]] = field(default_factory=list)  # (page, bbox) covering the unit

    @property
    def n_tokens(self) -> int:
        return len(tokenize(self.text))

    def rects_for(self, start: int, end: int) -> list[tuple[int, BBox]]:
        """(page, bbox) rectangles covering chars [start, end) of a paragraph.
        Partial lines are clipped proportionally to character position."""
        out: list[tuple[int, BBox]] = []
        for ln in self.lines:
            if ln.end <= start or ln.start >= end:
                continue
            x0, y0, x1, y1 = ln.bbox
            n = max(ln.end - ln.start, 1)
            w = x1 - x0
            a = max(start, ln.start) - ln.start
            b = min(end, ln.end) - ln.start
            out.append((ln.page, (x0 + w * a / n, y0, x0 + w * b / n, y1)))
        return out


@dataclass
class Figure:
    id: int
    page: int
    bbox: BBox
    path: str
    kind: str          # raster | vector
    sha256: str        # raw image bytes (raster) or rendered PNG bytes (vector)
    width: int
    height: int
    caption: str = ""


@dataclass
class Document:
    path: str
    sha256: str
    n_pages: int
    title: str
    body_font_size: float
    paragraphs: list[TextUnit]
    sentences: list[TextUnit]
    figures: list[Figure]
    ocr_pages: list[int]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Document":
        def unit(u: dict[str, Any]) -> TextUnit:
            return TextUnit(
                id=u["id"], kind=u["kind"], text=u["text"], page=u["page"], para=u["para"],
                span=tuple(u["span"]),
                lines=[Line(l["start"], l["end"], l["page"], tuple(l["bbox"])) for l in u.get("lines", [])],
                rects=[(p, tuple(b)) for p, b in u.get("rects", [])],
            )
        return Document(
            path=d["path"], sha256=d["sha256"], n_pages=d["n_pages"], title=d.get("title", ""),
            body_font_size=d.get("body_font_size", 0.0),
            paragraphs=[unit(u) for u in d["paragraphs"]],
            sentences=[unit(u) for u in d["sentences"]],
            figures=[Figure(**dict(f, bbox=tuple(f["bbox"]))) for f in d["figures"]],
            ocr_pages=list(d.get("ocr_pages", [])),
        )


@dataclass
class ExtractOptions:
    include_references: bool = False
    ocr: str = "auto"             # auto | on | off
    ocr_min_chars: int = 200
    figure_dpi: int = 150
    min_figure_pt: float = 40.0   # smallest displayed width/height of a figure, in points
    min_image_px: int = 48        # smallest raster image dimension, in pixels
    max_page_fraction: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- raw page structures -----------------------------------------------------

@dataclass
class _RawLine:
    text: str
    bbox: BBox
    sizes: list[tuple[float, int]]     # (font size, char count)
    bold_chars: int
    chars: int


@dataclass
class _RawBlock:
    page: int
    bbox: BBox
    lines: list[_RawLine]

    def text_len(self) -> int:
        return sum(len(l.text) for l in self.lines)


def _page_raw_blocks(page: pymupdf.Page, page_no: int) -> list[_RawBlock]:
    out: list[_RawBlock] = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines: list[_RawLine] = []
        for ln in b.get("lines", []):
            dx, dy = ln.get("dir", (1, 0))
            if abs(dx - 1) > 0.05 or abs(dy) > 0.05:
                continue  # rotated text (arXiv stamp, watermark)
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"])
            sizes = [(round(float(s["size"]), 1), len(s["text"])) for s in spans]
            bold = sum(len(s["text"]) for s in spans
                       if (s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower())
            chars = sum(len(s["text"]) for s in spans)
            lines.append(_RawLine(text, tuple(ln["bbox"]), sizes, bold, chars))
        if lines:
            out.append(_RawBlock(page_no, tuple(b["bbox"]), lines))
    return out


def _join_lines(lines: list[_RawLine], page: int) -> tuple[str, list[Line]]:
    """Concatenate normalised line texts, joining hyphenated line breaks, and
    record each line's char range so any span can be mapped back to a bbox."""
    text = ""
    lmap: list[Line] = []
    for ln in lines:
        piece = normalize(_CONTROL.sub("", ln.text))
        if not piece:
            continue
        if text:
            if text.endswith("-") and piece[:1].islower():
                text = text[:-1]
                lmap[-1].end -= 1
            else:
                text += " "
        start = len(text)
        text += piece
        lmap.append(Line(start, len(text), page, ln.bbox))
    return text, lmap


# --- OCR fallback -----------------------------------------------------------

def _ocr_page(page: pymupdf.Page, page_no: int, dpi: int = 300) -> list[_RawBlock]:
    """Render the page and run tesseract in TSV mode, rebuilding paragraphs
    (block, par) and lines with bboxes converted back to PDF points."""
    if shutil.which("tesseract") is None:
        raise RuntimeError("OCR requested but the `tesseract` binary is not installed")
    pix = page.get_pixmap(dpi=dpi)
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "page.png"
        pix.save(str(img))
        res = subprocess.run(["tesseract", str(img), "-", "--psm", "3", "tsv"],
                             capture_output=True, text=True, check=True)
    scale = 72.0 / dpi
    paras: dict[tuple[int, int], dict[int, list[tuple[str, BBox]]]] = defaultdict(lambda: defaultdict(list))
    for row in res.stdout.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        word = cols[11].strip()
        if not word:
            continue
        left, top, w, h = (float(cols[i]) for i in (6, 7, 8, 9))
        bbox = (left * scale, top * scale, (left + w) * scale, (top + h) * scale)
        paras[(int(cols[2]), int(cols[3]))][int(cols[4])].append((word, bbox))
    blocks: list[_RawBlock] = []
    for key in sorted(paras):
        lines: list[_RawLine] = []
        for ln_no in sorted(paras[key]):
            words = paras[key][ln_no]
            text = " ".join(w for w, _ in words)
            bbox = _union([b for _, b in words])
            size = round(bbox[3] - bbox[1], 1)
            lines.append(_RawLine(text, bbox, [(size, len(text))], 0, len(text)))
        if lines:
            blocks.append(_RawBlock(page_no, _union([l.bbox for l in lines]), lines))
    return blocks


def _has_full_page_image(page: pymupdf.Page, fraction: float = 0.5) -> bool:
    """True when one embedded image covers most of the page: the signature of
    a scanned page, which is the only case auto-OCR should handle."""
    page_area = page.rect.width * page.rect.height
    for info in page.get_images(full=True):
        try:
            rects = page.get_image_rects(info[0])
        except Exception:
            continue
        if any(r.width * r.height >= fraction * page_area for r in rects):
            return True
    return False


def _union(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# --- text classification ----------------------------------------------------

def _body_font_size(blocks: list[_RawBlock]) -> float:
    c: Counter[float] = Counter()
    for b in blocks:
        for ln in b.lines:
            for size, n in ln.sizes:
                c[size] += n
    return c.most_common(1)[0][0] if c else 10.0


def _block_size(b: _RawBlock) -> float:
    tot = sum(n for ln in b.lines for _, n in ln.sizes)
    if not tot:
        return 0.0
    return sum(s * n for ln in b.lines for s, n in ln.sizes) / tot


def _bold_fraction(b: _RawBlock) -> float:
    chars = sum(ln.chars for ln in b.lines)
    return (sum(ln.bold_chars for ln in b.lines) / chars) if chars else 0.0


def _header_footer_keys(pages: list[list[_RawBlock]], page_heights: list[float]) -> set[str]:
    """Digit-blind texts that recur in the top/bottom margins of >= 3 pages
    (>= 2 for very short documents)."""
    n_pages = len(pages)
    if n_pages < 2:
        return set()
    threshold = 3 if n_pages >= 3 else 2
    seen: Counter[str] = Counter()
    for blocks, h in zip(pages, page_heights):
        keys = set()
        for b in blocks:
            if b.bbox[3] < 0.08 * h or b.bbox[1] > 0.92 * h:
                keys.add(_margin_key(b))
        seen.update(keys)
    return {k for k, n in seen.items() if n >= threshold}


def _margin_key(b: _RawBlock) -> str:
    return _DIGITS.sub("#", normalize(" ".join(l.text for l in b.lines)).lower())


def _is_margin_noise(b: _RawBlock, page_h: float, hf_keys: set[str]) -> bool:
    in_margin = b.bbox[3] < 0.08 * page_h or b.bbox[1] > 0.92 * page_h
    if not in_margin:
        return False
    text = normalize(" ".join(l.text for l in b.lines))
    return bool(_PAGE_NUMBER_RE.match(text)) or _margin_key(b) in hf_keys


# --- figures -----------------------------------------------------------------

def _overlap_fraction(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return inter / area_a


def _to_rgb(pix: pymupdf.Pixmap) -> pymupdf.Pixmap:
    if pix.colorspace is None or pix.n - pix.alpha not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return pix


def _page_figures(doc: pymupdf.Document, page: pymupdf.Page, page_no: int, fig_dir: Path,
                  opts: ExtractOptions, next_id: int) -> list[Figure]:
    figs: list[Figure] = []
    page_area = page.rect.width * page.rect.height
    raster_rects: list[BBox] = []

    def ok_rect(r: pymupdf.Rect) -> bool:
        if r.width < opts.min_figure_pt or r.height < opts.min_figure_pt:
            return False
        return r.width * r.height <= opts.max_page_fraction * page_area

    for info in page.get_images(full=True):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            if not ok_rect(r):
                continue
            try:
                pix = _to_rgb(pymupdf.Pixmap(doc, xref))
            except Exception:
                continue
            if pix.width < opts.min_image_px or pix.height < opts.min_image_px:
                continue
            try:
                raw = doc.extract_image(xref)["image"]
            except Exception:
                raw = pix.tobytes("png")
            path = fig_dir / f"fig_p{page_no + 1:02d}_{next_id + len(figs):03d}_raster.png"
            pix.save(str(path))
            bbox = (r.x0, r.y0, r.x1, r.y1)
            raster_rects.append(bbox)
            figs.append(Figure(next_id + len(figs), page_no, bbox, str(path), "raster",
                               hashlib.sha256(raw).hexdigest(), pix.width, pix.height))

    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for r in clusters:
        if not ok_rect(r):
            continue
        bbox = (r.x0, r.y0, r.x1, r.y1)
        if any(_overlap_fraction(bbox, rr) > 0.8 or _overlap_fraction(rr, bbox) > 0.8 for rr in raster_rects):
            continue  # frame around an embedded image, or the image itself
        try:
            pix = _to_rgb(page.get_pixmap(clip=r, dpi=opts.figure_dpi))
        except Exception:
            continue
        png = pix.tobytes("png")
        path = fig_dir / f"fig_p{page_no + 1:02d}_{next_id + len(figs):03d}_vector.png"
        path.write_bytes(png)
        figs.append(Figure(next_id + len(figs), page_no, bbox, str(path), "vector",
                           hashlib.sha256(png).hexdigest(), pix.width, pix.height))
    return figs


def _attach_captions(figures: list[Figure], captions: list[TextUnit]) -> None:
    for fig in figures:
        best, best_d = "", 1e9
        for cap in captions:
            if cap.page != fig.page or not cap.rects:
                continue
            _, cb = cap.rects[0]
            horiz = min(cb[2], fig.bbox[2]) - max(cb[0], fig.bbox[0])
            if horiz <= 0:
                continue
            below = cb[1] - fig.bbox[3]      # caption under the figure
            above = fig.bbox[1] - cb[3]      # caption over a table
            d = below if below >= -5 else (above + 20 if above >= -5 else 1e9)
            if 0 <= d < min(best_d, 80):
                best, best_d = cap.text, d
        fig.caption = best


# --- main entry point -------------------------------------------------------

def extract(pdf_path: str | Path, options: ExtractOptions | None = None,
            cache_dir: str | Path = "cache") -> Document:
    """Extract (or load from cache) the Document for one PDF."""
    opts = options or ExtractOptions()
    pdf_path = Path(pdf_path)
    sha = _cache.pdf_sha256(pdf_path)
    key = _cache.cache_key(sha, opts.to_dict())
    entry = _cache.entry_dir(cache_dir, key)
    data = _cache.load(cache_dir, key)
    if data is not None:
        doc = Document.from_dict(data)
        if all(Path(f.path).is_file() for f in doc.figures):
            doc.path = str(pdf_path)
            return doc
    fig_dir = entry / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    doc = _extract(pdf_path, sha, opts, fig_dir)
    _cache.store(cache_dir, key, doc.to_dict())
    return doc


def _extract(pdf_path: Path, sha: str, opts: ExtractOptions, fig_dir: Path) -> Document:
    pdf = pymupdf.open(str(pdf_path))
    n_pages = pdf.page_count
    pages_raw: list[list[_RawBlock]] = []
    heights: list[float] = []
    ocr_pages: list[int] = []
    for page_no, page in enumerate(pdf):
        blocks = _page_raw_blocks(page, page_no)
        chars = sum(b.text_len() for b in blocks)
        looks_scanned = chars < opts.ocr_min_chars and _has_full_page_image(page)
        if opts.ocr == "on" or (opts.ocr == "auto" and looks_scanned):
            try:
                blocks = _ocr_page(page, page_no)
                ocr_pages.append(page_no)
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                if opts.ocr == "on":
                    raise
                print(f"warning: OCR skipped for page {page_no + 1} of {pdf_path.name}: {exc}", file=sys.stderr)
        pages_raw.append(blocks)
        heights.append(page.rect.height)

    all_blocks = [b for blocks in pages_raw for b in blocks]
    body_size = _body_font_size(all_blocks)
    hf_keys = _header_footer_keys(pages_raw, heights)

    figures: list[Figure] = []
    for page_no, page in enumerate(pdf):
        figures.extend(_page_figures(pdf, page, page_no, fig_dir, opts, len(figures)))
    vector_rects: dict[int, list[BBox]] = {}
    for f in figures:
        if f.kind == "vector":
            vector_rects.setdefault(f.page, []).append(f.bbox)

    paragraphs: list[TextUnit] = []
    title, title_size = "", 0.0
    in_refs = False
    next_is_abstract = False
    for page_no, blocks in enumerate(pages_raw):
        for b in blocks:
            if _is_margin_noise(b, heights[page_no], hf_keys):
                continue
            text, lines = _join_lines(b.lines, page_no)
            if not text or not _ALPHA.search(text):
                continue  # empty, or glyph soup from a drawing font (no real word)
            ntok = len(tokenize(text))
            if ntok <= FIGURE_TEXT_MAX_TOKENS and any(
                    _overlap_fraction(b.bbox, r) > 0.8 for r in vector_rects.get(page_no, [])):
                continue  # axis label / box text inside a vector figure
            size = _block_size(b)
            is_heading = ntok <= 15 and (size >= body_size * 1.15 or _bold_fraction(b) > 0.8
                                         or bool(_NUMBERED_HEADING_RE.match(text)))
            if page_no == 0 and is_heading and ntok >= 3 and size > title_size:
                title, title_size = text, size
            if _CAPTION_RE.match(text):
                kind = "caption"
            elif is_heading:
                if _REFERENCES_RE.match(text):
                    in_refs = True
                    continue
                in_refs = False
                next_is_abstract = bool(_ABSTRACT_HEADING_RE.match(text))
                kind = "heading"
            elif in_refs:
                kind = "reference"
            elif next_is_abstract or (_ABSTRACT_INLINE_RE.match(text) and ntok >= 20):
                kind, next_is_abstract = "abstract", False
            else:
                kind = "body"
            if kind == "reference" and not opts.include_references:
                continue
            prev = paragraphs[-1] if paragraphs else None
            if (kind == "body" and prev is not None and prev.kind == "body"
                    and prev.text[-1:] not in ".!?:" and text[:1].islower()):
                # continuation of a paragraph across a column/page break
                offset = len(prev.text) + 1
                prev.text += " " + text
                prev.lines.extend(Line(l.start + offset, l.end + offset, l.page, l.bbox) for l in lines)
                prev.span = (0, len(prev.text))
                prev.rects = [(l.page, l.bbox) for l in prev.lines]
                continue
            uid = len(paragraphs)
            paragraphs.append(TextUnit(uid, kind, text, page_no, uid, (0, len(text)), lines,
                                       [(l.page, l.bbox) for l in lines]))

    sentences: list[TextUnit] = []
    for p in paragraphs:
        if p.kind not in SENTENCE_KINDS:
            continue
        pos = 0
        for s in split_sentences(p.text):
            start = p.text.find(s, pos)
            if start < 0:
                start = pos
            end = start + len(s)
            pos = end
            sentences.append(TextUnit(len(sentences), p.kind, s, p.page, p.id, (start, end),
                                      [], p.rects_for(start, end)))

    _attach_captions(figures, [p for p in paragraphs if p.kind == "caption"])
    pdf.close()

    return Document(str(pdf_path), sha, n_pages, title, body_size, paragraphs, sentences,
                    figures, ocr_pages)
