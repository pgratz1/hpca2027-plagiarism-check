# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

`pdfsim`: a fully local pairwise PDF similarity checker used by the HPCA 2027
PC chair to compare two paper submissions (text and figures). Built
2026-09-13. **Read `README.md` first**; it documents usage, outputs, options
and the offline guarantee. This file only adds what a coding session needs.

## Hard invariants

- **Nothing leaves the machine.** `scripts/prefetch_models.py` is the only
  code allowed to open a network connection, and only for model weights.
  `scripts/compare_pdfs.py` and `src/pdfsim/semantic.py` set the Hugging Face
  offline variables *before* importing torch/transformers; keep it that way.
  `tests/conftest.py` monkeypatches sockets to fail, so any new network use
  breaks `make test`.
- **The three layers are never merged into one score.** Verbatim (layer 1)
  is the headline; paraphrase (layer 2) is labelled "weaker evidence" and
  reports only what layer 1 did not cover; figures (layer 3) are separate.
  The user asked for this explicitly.
- Do not commit PDFs, `cache/` or `outputs/` (gitignored).

## Running things

- Python env: `~/envs/hpca-plagiarism/bin/python` (3.13, torch+CUDA,
  PyMuPDF, rapidfuzz, sentence-transformers, imagehash). `make` targets use
  it; `requirements.txt` is the installer.
- `make test` (about 8 s), `make compare A=... B=...`, `make models` (network,
  once), `make clean-cache`.
- Import as `pymupdf`, not the deprecated `fitz`.

## Code map and conventions

- `src/pdfsim/extract.py` turns a PDF into `Document` (paragraph and sentence
  `TextUnit`s with per-line bboxes, `Figure`s as PNGs). Results are cached by
  `cache.py` under `cache/<sha256[:16]>-v<SCHEMA_VERSION>-<options>/`.
  **Bump `SCHEMA_VERSION` in `cache.py` whenever `extract.py` or
  `textnorm.py` changes what they emit.**
- `verbatim.py`: winnowing over word tokens, run growth and merging, then
  rapidfuzz sentence alignment for what fragments. `Run.a_locs` /
  `SentencePair.a_rects` carry page + bbox for the annotated PDFs.
- `semantic.py`: sentence-transformers embeddings; units covered by layer-1
  runs or near-sentences are excluded before matching.
- `images.py`: pHash/dHash + multi-scale NCC template match for crops.
  `score_pair` is the place to plug in a vision-embedding scorer later.
- `report.py`: `ComparisonResult` -> JSON / HTML (jinja2 template inline) /
  annotated PDFs. Colours live in `COLORS` and `CSS_COLORS`.
- Thresholds are dataclass defaults (`VerbatimParams`, `SemanticParams`,
  `ImageParams`) mirrored by CLI flags in `scripts/compare_pdfs.py`; change
  both. Calibration so far: synthetic papers (`tests/pdfgen.py`) and
  pdflatex two-column papers; unrelated figures reach crop NCC 0.80, real
  crops score at least 0.95, hence the 0.85 default.
- Tests build their PDFs at run time from `tests/corpus.py` text; add new
  cases there rather than checking in PDFs.
