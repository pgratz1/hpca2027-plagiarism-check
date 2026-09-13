"""Layer 2: paraphrase / semantic similarity (weaker evidence; reported apart).

Sentences and paragraphs of both documents are embedded with a local
sentence-transformers model and compared by cosine similarity.  A pair is
reported only when it is above threshold, is a mutual nearest neighbour, and
neither side is already inside a verbatim run or near-verbatim sentence pair,
so this layer shows only what the verbatim layer did *not* find.

Offline by construction: the Hugging Face offline switches are set below,
before the model library is imported, so a model missing from the local cache
raises ModelNotAvailable instead of starting a download.  Fetch it once with
scripts/prefetch_models.py.
"""

from __future__ import annotations

import os

for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
               ("HF_HUB_DISABLE_TELEMETRY", "1"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_k, _v)

from dataclasses import asdict, dataclass  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from .extract import BBox, Document, TextUnit  # noqa: E402
from .verbatim import TEXT_KINDS, Loc, VerbatimResult  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LEVELS = ("sentence", "paragraph")


class ModelNotAvailable(RuntimeError):
    """The embedding model is not in the local cache and downloads are off."""


@dataclass
class SemanticParams:
    model: str = DEFAULT_MODEL
    sentence_threshold: float = 0.80
    paragraph_threshold: float = 0.75
    min_sentence_tokens: int = 8
    min_paragraph_tokens: int = 25
    mutual: bool = True                 # require mutual nearest neighbours
    max_verbatim_fraction: float = 0.5  # units covered this much by runs are skipped
    include_references: bool = False
    device: str | None = None           # None = cuda if available else cpu
    batch_size: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticPair:
    level: str
    a_id: int
    b_id: int
    a_text: str
    b_text: str
    score: float
    a_page: int
    b_page: int
    a_rects: list[tuple[int, BBox]]
    b_rects: list[tuple[int, BBox]]


@dataclass
class SemanticResult:
    pairs: list[SemanticPair]
    n_a: dict[str, int]
    n_b: dict[str, int]
    mean_max_sim: dict[str, float]
    fraction_matched_a: dict[str, float]
    fraction_matched_b: dict[str, float]
    histogram: dict[str, list[int]]     # row-max cosine, 10 bins on [0, 1]
    model: str
    device: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- model -------------------------------------------------------------------

_MODELS: dict[tuple[str, str | None], Any] = {}


def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    key = (name, device)
    if key in _MODELS:
        return _MODELS[key]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ModelNotAvailable("sentence-transformers is not installed in this environment") from exc
    try:
        model = SentenceTransformer(name, device=device)
    except Exception as exc:
        raise ModelNotAvailable(
            f"embedding model {name!r} is not in the local Hugging Face cache and downloads are "
            f"disabled. Run `make models MODEL={name}` once (network) and retry.") from exc
    _MODELS[key] = model
    return model


# --- unit selection ----------------------------------------------------------

def _intervals(locs_per_run: list[list[Loc]]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for locs in locs_per_run:
        for loc in locs:
            out.setdefault(loc.para, []).append((loc.start, loc.end))
    return out


def covered_fraction(unit: TextUnit, intervals: dict[int, list[tuple[int, int]]]) -> float:
    cs, ce = unit.span
    length = ce - cs
    if length <= 0:
        return 0.0
    covered = sum(max(0, min(e, ce) - max(s, cs)) for s, e in intervals.get(unit.para, []))
    return covered / length


def _units(doc: Document, level: str, params: SemanticParams,
           intervals: dict[int, list[tuple[int, int]]], exclude_ids: set[int]) -> list[TextUnit]:
    kinds = set(TEXT_KINDS) if params.include_references else set(TEXT_KINDS) - {"reference"}
    src = doc.sentences if level == "sentence" else doc.paragraphs
    min_tok = params.min_sentence_tokens if level == "sentence" else params.min_paragraph_tokens
    return [u for u in src
            if u.kind in kinds and u.id not in exclude_ids and u.n_tokens >= min_tok
            and covered_fraction(u, intervals) < params.max_verbatim_fraction]


# --- public API --------------------------------------------------------------

def compare(doc_a: Document, doc_b: Document, params: SemanticParams | None = None,
            verbatim_result: VerbatimResult | None = None) -> SemanticResult:
    params = params or SemanticParams()
    model = load_model(params.model, params.device)
    ia = _intervals([r.a_locs for r in verbatim_result.runs]) if verbatim_result else {}
    ib = _intervals([r.b_locs for r in verbatim_result.runs]) if verbatim_result else {}
    near_a = {p.a_sentence for p in verbatim_result.sentence_pairs} if verbatim_result else set()
    near_b = {p.b_sentence for p in verbatim_result.sentence_pairs} if verbatim_result else set()
    # near-verbatim sentences count as covered too, so their paragraph is not
    # re-reported here as a paraphrase
    for sid in near_a:
        u = doc_a.sentences[sid]
        ia.setdefault(u.para, []).append(u.span)
    for sid in near_b:
        u = doc_b.sentences[sid]
        ib.setdefault(u.para, []).append(u.span)

    pairs: list[SemanticPair] = []
    n_a: dict[str, int] = {}
    n_b: dict[str, int] = {}
    mean_max: dict[str, float] = {}
    frac_a: dict[str, float] = {}
    frac_b: dict[str, float] = {}
    hist: dict[str, list[int]] = {}
    for level in LEVELS:
        thr = params.sentence_threshold if level == "sentence" else params.paragraph_threshold
        ua = _units(doc_a, level, params, ia, near_a if level == "sentence" else set())
        ub = _units(doc_b, level, params, ib, near_b if level == "sentence" else set())
        n_a[level], n_b[level] = len(ua), len(ub)
        if not ua or not ub:
            mean_max[level], frac_a[level], frac_b[level], hist[level] = 0.0, 0.0, 0.0, [0] * 10
            continue
        ea = model.encode([u.text for u in ua], batch_size=params.batch_size,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        eb = model.encode([u.text for u in ub], batch_size=params.batch_size,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sim = np.asarray(ea) @ np.asarray(eb).T
        row_max, col_max = sim.max(axis=1), sim.max(axis=0)
        mean_max[level] = round(float(row_max.mean()), 4)
        frac_a[level] = round(float((row_max >= thr).mean()), 4)
        frac_b[level] = round(float((col_max >= thr).mean()), 4)
        hist[level] = np.histogram(np.clip(row_max, 0, 1), bins=10, range=(0.0, 1.0))[0].tolist()
        best_b = sim.argmax(axis=1)
        best_a = sim.argmax(axis=0)
        for i, j in enumerate(best_b):
            score = float(sim[i, j])
            if score < thr or (params.mutual and best_a[j] != i):
                continue
            a, b = ua[i], ub[j]
            pairs.append(SemanticPair(level, a.id, b.id, a.text, b.text, round(score, 3),
                                      a.page, b.page, a.rects, b.rects))
    return SemanticResult(pairs, n_a, n_b, mean_max, frac_a, frac_b, hist,
                          params.model, str(model.device), params.to_dict())
