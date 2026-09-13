"""Layer 1: verbatim and near-verbatim text overlap (the strongest evidence).

Two complementary detectors over the normalised text of two Documents:

* Winnowed k-gram fingerprints (Schleimer, Wilkerson & Aiken 2003; the MOSS
  algorithm).  Every shared fingerprint is verified token-by-token and grown
  into a maximal run of matching tokens; runs separated by small edits are
  merged.  This yields copied passages and a document-level coverage figure.
* Sentence alignment with rapidfuzz for sentences that were edited enough to
  fragment the k-gram matches (word swaps, insertions).

No model, no network.  Runs and sentence pairs are scored with
rapidfuzz.fuzz.ratio on their token strings and labelled `exact` or `near`.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from .extract import BBox, Document, TextUnit
from .textnorm import tokenize_with_offsets

TEXT_KINDS = ("body", "abstract", "caption", "reference")

# Phrases so common in papers that sharing them is not evidence of anything.
DEFAULT_BOILERPLATE = [
    "the rest of the paper is organized as follows",
    "the remainder of this paper is organized as follows",
    "this paper is organized as follows",
    "the rest of this paper is organized as follows",
    "to the best of our knowledge",
    "the contributions of this paper are as follows",
    "we make the following contributions",
    "this work was supported in part by",
    "the authors would like to thank",
    "section 2 describes",
    "the remainder of this paper",
]


@dataclass
class VerbatimParams:
    kgram: int = 6
    window: int = 4
    max_gap: int = 3            # tokens of edit tolerated between merged runs
    min_run: int = 12           # tokens; shorter matched runs are ignored
    near_threshold: float = 85  # fuzz.ratio, 0..100
    exact_threshold: float = 100  # identical token sequence; anything edited is `near`
    min_sentence_tokens: int = 6
    max_fingerprint_positions: int = 50  # skip k-grams repeated this often (tables, boilerplate)
    include_references: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Loc:
    para: int
    start: int    # char offsets inside the paragraph text
    end: int
    page: int
    rects: list[tuple[int, BBox]]


@dataclass
class Run:
    id: int
    a_start: int          # token indices in the A stream, [start, end)
    a_end: int
    b_start: int
    b_end: int
    a_text: str
    b_text: str
    score: float
    verdict: str          # exact | near
    a_locs: list[Loc]
    b_locs: list[Loc]

    @property
    def length(self) -> int:
        return self.a_end - self.a_start


@dataclass
class SentencePair:
    a_sentence: int
    b_sentence: int
    a_text: str
    b_text: str
    score: float
    verdict: str          # exact | near
    a_page: int
    b_page: int
    a_rects: list[tuple[int, BBox]]
    b_rects: list[tuple[int, BBox]]


@dataclass
class VerbatimResult:
    runs: list[Run]
    sentence_pairs: list[SentencePair]
    tokens_a: int
    tokens_b: int
    covered_a: int
    covered_b: int
    longest_run: int
    params: dict[str, Any]

    @property
    def coverage_a(self) -> float:
        return self.covered_a / self.tokens_a if self.tokens_a else 0.0

    @property
    def coverage_b(self) -> float:
        return self.covered_b / self.tokens_b if self.tokens_b else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coverage_a"] = self.coverage_a
        d["coverage_b"] = self.coverage_b
        return d


# --- token stream ------------------------------------------------------------

@dataclass
class _Stream:
    tokens: list[str] = field(default_factory=list)
    para: list[int] = field(default_factory=list)     # paragraph id per token
    start: list[int] = field(default_factory=list)    # char offsets inside that paragraph
    end: list[int] = field(default_factory=list)
    paragraphs: dict[int, TextUnit] = field(default_factory=dict)


def _stream(doc: Document, params: VerbatimParams) -> _Stream:
    kinds = set(TEXT_KINDS) if params.include_references else set(TEXT_KINDS) - {"reference"}
    s = _Stream()
    for p in doc.paragraphs:
        if p.kind not in kinds:
            continue
        s.paragraphs[p.id] = p
        for t in tokenize_with_offsets(p.text):
            s.tokens.append(t.text)
            s.para.append(p.id)
            s.start.append(t.start)
            s.end.append(t.end)
    return s


def _locs(s: _Stream, start: int, end: int) -> list[Loc]:
    """Paragraph char spans covered by tokens [start, end) of a stream."""
    out: list[Loc] = []
    i = start
    while i < end:
        pid = s.para[i]
        j = i
        while j + 1 < end and s.para[j + 1] == pid:
            j += 1
        p = s.paragraphs[pid]
        cs, ce = s.start[i], s.end[j]
        out.append(Loc(pid, cs, ce, p.page, p.rects_for(cs, ce)))
        i = j + 1
    return out


def _text(s: _Stream, start: int, end: int) -> str:
    return " ".join(p.text[l.start:l.end] for l in _locs(s, start, end)
                    for p in (s.paragraphs[l.para],))


# --- winnowing ---------------------------------------------------------------

def _fingerprints(tokens: list[str], k: int, w: int) -> dict[int, list[int]]:
    """Winnowed fingerprints: {hash: [positions]}.  Guarantees that any shared
    token run of length >= k + w - 1 shares at least one fingerprint."""
    n = len(tokens)
    if n < k:
        return {}
    hashes = [zlib.crc32(" ".join(tokens[i:i + k]).encode()) for i in range(n - k + 1)]
    fps: dict[int, list[int]] = {}
    if len(hashes) <= w:
        h = min(range(len(hashes)), key=lambda i: (hashes[i], -i))
        fps.setdefault(hashes[h], []).append(h)
        return fps
    last = -1
    for i in range(len(hashes) - w + 1):
        # rightmost minimum in window [i, i + w)
        m = i
        for j in range(i, i + w):
            if hashes[j] <= hashes[m]:
                m = j
        if m != last:
            fps.setdefault(hashes[m], []).append(m)
            last = m
    return fps


def _extend(a: list[str], b: list[str], i: int, j: int, k: int) -> tuple[int, int, int] | None:
    """Verify a seed (i, j) and grow it to a maximal exact match.  Returns
    (a_start, b_start, length) or None on a hash collision."""
    if a[i:i + k] != b[j:j + k]:
        return None
    s, t = i, j
    while s > 0 and t > 0 and a[s - 1] == b[t - 1]:
        s -= 1
        t -= 1
    e, f = i + k, j + k
    while e < len(a) and f < len(b) and a[e] == b[f]:
        e += 1
        f += 1
    return s, t, e - s


def _exact_runs(a: list[str], b: list[str], params: VerbatimParams) -> list[tuple[int, int, int, int]]:
    """Maximal exact runs as (a_start, a_end, b_start, b_end)."""
    fa = _fingerprints(a, params.kgram, params.window)
    fb = _fingerprints(b, params.kgram, params.window)
    seen: set[tuple[int, int]] = set()
    runs: list[tuple[int, int, int, int]] = []
    for h, pa in fa.items():
        pb = fb.get(h)
        if not pb:
            continue
        if len(pa) > params.max_fingerprint_positions or len(pb) > params.max_fingerprint_positions:
            continue
        for i in pa:
            for j in pb:
                r = _extend(a, b, i, j, params.kgram)
                if r is None:
                    continue
                s, t, n = r
                key = (s, t)
                if key in seen:
                    continue
                seen.add(key)
                runs.append((s, s + n, t, t + n))
    runs.sort()
    return runs


def _merge_runs(runs: list[tuple[int, int, int, int]], max_gap: int) -> list[tuple[int, int, int, int]]:
    """Merge exact runs separated by a small edit on both sides (same or
    nearly the same diagonal), and drop runs overlapping an earlier one."""
    merged: list[list[int]] = []
    for r in runs:
        a0, a1, b0, b1 = r
        if merged:
            m = merged[-1]
            ga, gb = a0 - m[1], b0 - m[3]
            if -1 <= ga <= max_gap and -1 <= gb <= max_gap and abs(ga - gb) <= max_gap:
                m[1], m[3] = max(m[1], a1), max(m[3], b1)
                continue
            if a0 < m[1] and b0 < m[3]:
                continue  # nested inside the previous run
        merged.append([a0, a1, b0, b1])
    return [tuple(m) for m in merged]  # type: ignore[misc]


def _token_string(text: str) -> str:
    return " ".join(t.text for t in tokenize_with_offsets(text))


def load_boilerplate(path: str | Path | None) -> set[str]:
    phrases = set(_token_string(p) for p in DEFAULT_BOILERPLATE)
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                phrases.add(_token_string(line))
    return phrases


def _is_boilerplate(token_string: str, phrases: set[str]) -> bool:
    return token_string in phrases or any(token_string in p for p in phrases)


# --- public API --------------------------------------------------------------

def compare(doc_a: Document, doc_b: Document, params: VerbatimParams | None = None,
            boilerplate: set[str] | None = None) -> VerbatimResult:
    params = params or VerbatimParams()
    phrases = boilerplate if boilerplate is not None else load_boilerplate(None)
    sa, sb = _stream(doc_a, params), _stream(doc_b, params)

    runs: list[Run] = []
    for a0, a1, b0, b1 in _merge_runs(_exact_runs(sa.tokens, sb.tokens, params), params.max_gap):
        if a1 - a0 < params.min_run:
            continue
        ta, tb = " ".join(sa.tokens[a0:a1]), " ".join(sb.tokens[b0:b1])
        if _is_boilerplate(ta, phrases):
            continue
        score = fuzz.ratio(ta, tb)
        if score < params.near_threshold:
            continue
        verdict = "exact" if score >= params.exact_threshold else "near"
        runs.append(Run(len(runs), a0, a1, b0, b1, _text(sa, a0, a1), _text(sb, b0, b1),
                        round(score, 1), verdict, _locs(sa, a0, a1), _locs(sb, b0, b1)))

    covered_a = [False] * len(sa.tokens)
    covered_b = [False] * len(sb.tokens)
    for r in runs:
        for i in range(r.a_start, r.a_end):
            covered_a[i] = True
        for j in range(r.b_start, r.b_end):
            covered_b[j] = True

    pairs = _sentence_pairs(doc_a, doc_b, sa, sb, covered_a, covered_b, params, phrases)

    return VerbatimResult(
        runs=runs, sentence_pairs=pairs,
        tokens_a=len(sa.tokens), tokens_b=len(sb.tokens),
        covered_a=sum(covered_a), covered_b=sum(covered_b),
        longest_run=max((r.length for r in runs), default=0),
        params=params.to_dict(),
    )


def _sentence_coverage(doc: Document, s: _Stream, covered: list[bool]) -> dict[int, float]:
    """Fraction of each sentence's tokens that lie inside a run."""
    by_para: dict[int, list[int]] = {}
    for idx, pid in enumerate(s.para):
        by_para.setdefault(pid, []).append(idx)
    out: dict[int, float] = {}
    for sent in doc.sentences:
        idxs = by_para.get(sent.para)
        if not idxs:
            continue
        cs, ce = sent.span
        inside = [i for i in idxs if s.start[i] >= cs and s.end[i] <= ce]
        if inside:
            out[sent.id] = sum(covered[i] for i in inside) / len(inside)
    return out


def _sentence_pairs(doc_a: Document, doc_b: Document, sa: _Stream, sb: _Stream,
                    covered_a: list[bool], covered_b: list[bool], params: VerbatimParams,
                    phrases: set[str]) -> list[SentencePair]:
    kinds = set(sa.paragraphs[p].kind for p in sa.paragraphs)
    cov_a = _sentence_coverage(doc_a, sa, covered_a)
    cov_b = _sentence_coverage(doc_b, sb, covered_b)

    def eligible(doc: Document, cov: dict[int, float]) -> list[TextUnit]:
        out = []
        for s in doc.sentences:
            if s.kind not in kinds or s.n_tokens < params.min_sentence_tokens:
                continue
            if cov.get(s.id, 0.0) >= 0.8:
                continue  # already reported inside a run
            if _is_boilerplate(_token_string(s.text), phrases):
                continue
            out.append(s)
        return out

    ea, eb = eligible(doc_a, cov_a), eligible(doc_b, cov_b)
    if not ea or not eb:
        return []
    ta = [_token_string(s.text) for s in ea]
    tb = [_token_string(s.text) for s in eb]
    matrix = process.cdist(ta, tb, scorer=fuzz.ratio, score_cutoff=params.near_threshold, workers=-1)
    pairs: list[SentencePair] = []
    for i, row in enumerate(matrix):
        j = int(row.argmax())
        score = float(row[j])
        if score < params.near_threshold:
            continue
        verdict = "exact" if score >= params.exact_threshold else "near"
        a, b = ea[i], eb[j]
        pairs.append(SentencePair(a.id, b.id, a.text, b.text, round(score, 1), verdict,
                                  a.page, b.page, a.rects, b.rects))
    return pairs
