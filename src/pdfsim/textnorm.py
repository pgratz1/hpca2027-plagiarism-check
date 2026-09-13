"""Text normalization and segmentation shared by every layer.

Pure string processing: no model, no network.  Everything downstream
(fingerprints, fuzzy matching, embeddings) consumes the output of
`normalize`, `split_sentences` and `tokenize`, so a change here changes
cached extraction results: bump `pdfsim.cache.SCHEMA_VERSION` when editing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Glyphs that PDF text extraction commonly emits, folded to plain ASCII so
# the same sentence typeset by two toolchains compares equal.
_CHAR_MAP = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "­": "",
})

# "archi-\ntecture" -> "architecture".  Only joins when the continuation is
# lowercase, which is how word-internal hyphenation looks; "Intel-\nBased"
# keeps its hyphen.
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=[a-z])")
_NEWLINES = re.compile(r"\s*\n\s*")
_SPACES = re.compile(r"[ \t\r\f\v]+")


def normalize(text: str) -> str:
    """Fold ligatures/quotes/dashes, NFKC-normalise, join hyphenated line
    breaks, and collapse all whitespace to single spaces."""
    text = text.translate(_CHAR_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub("", text)
    text = _NEWLINES.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


# --- sentences ---------------------------------------------------------------

_ABBREVIATIONS = {
    "fig", "figs", "eq", "eqs", "eqn", "sec", "secs", "ref", "refs", "tab",
    "e.g", "i.e", "et", "al", "vs", "cf", "no", "approx", "etc", "resp",
    "dr", "prof", "mr", "mrs", "ms", "vol", "pp", "ch", "dept", "univ",
    "inc", "ltd", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "st", "nd", "rd", "th",
}

# Candidate boundary: terminal punctuation (optionally closing quote/paren),
# whitespace, then something that can start a sentence.
_SENT_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def _ends_with_abbreviation(chunk: str) -> bool:
    words = chunk.split()
    if not words:
        return False
    last = words[-1].rstrip(".!?\"')]").lower()
    if not last:
        return False
    if last in _ABBREVIATIONS:
        return True
    # single-letter initials ("J. Smith") and enumerations ("(a) ... b.")
    return len(last) == 1 and last.isalpha()


def split_sentences(text: str, min_tokens: int = 1) -> list[str]:
    """Split normalised text into sentences with a regex plus an abbreviation
    list.  Deliberately dependency-free (no NLTK/spaCy downloads)."""
    sentences: list[str] = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        chunk = text[start:m.start() + len(m.group().rstrip())]
        if _ends_with_abbreviation(text[start:m.start()]):
            continue
        sentences.append(chunk.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    if min_tokens > 1:
        sentences = [s for s in sentences if len(tokenize(s)) >= min_tokens]
    return sentences


# --- tokens ------------------------------------------------------------------

# Runs of Unicode letters/digits; punctuation, underscores and whitespace are
# separators.  Matching is done on the original string so offsets are exact,
# and each token is lower-cased afterwards.
_TOKEN = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _TOKEN.finditer(text)]


def tokenize_with_offsets(text: str) -> list[Token]:
    return [Token(m.group().lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
