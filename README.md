# pdfsim: local pairwise PDF similarity (text + figures)

Compare two PDFs, typically two paper submissions or a submission and a prior
publication, and see what they share. Everything runs on this machine: no PDF,
no extracted text and no image ever leaves it, and no hosted AI service is
involved.

Three kinds of evidence are detected and **always reported separately**,
because they are not equally strong:

| layer | what it finds | how | colour in annotated PDFs |
|---|---|---|---|
| 1. verbatim | copied or lightly edited passages and sentences | winnowed word k-gram fingerprints (the MOSS algorithm) + rapidfuzz sentence alignment | red = exact, orange = near |
| 2. paraphrase | reworded passages that say the same thing | local sentence-embedding model, cosine similarity, mutual nearest neighbours; **only** what layer 1 did not already flag | blue (labelled "weaker evidence") |
| 3. figures | re-used figures: rescaled, recompressed, recoloured, cropped | pHash/dHash perceptual hashes + normalised cross-correlation template matching | green box |

Re-drawn figures (same plot, different styling) are out of scope; see
*Limitations*.

## Setup (once)

```sh
make env      # creates ~/envs/hpca-plagiarism (Python 3.13) and installs requirements.txt
make models   # downloads the embedding model into ~/.cache/huggingface  <-- the ONLY network step
```

`tesseract` (OCR fallback for scanned PDFs) is optional and already installed
on this machine. CUDA is used automatically when available; CPU works too.

## Usage

```sh
make compare A=paper1.pdf B=paper2.pdf                 # writes outputs/paper1_vs_paper2/
make compare A=a.pdf B=b.pdf OUT=/tmp/x ARGS='--no-semantic'
~/envs/hpca-plagiarism/bin/python scripts/compare_pdfs.py a.pdf b.pdf -o out/   # same thing
```

Console summary:

```
paper1.pdf vs paper2.pdf
  verbatim : 23.5% of A / 27.7% of B inside 2 runs (2 exact, 0 near; longest 64 words); 1 near-verbatim sentences
  semantic : 2 paraphrase-only pairs (1 sentences, 1 paragraphs); mean max-sim sent 0.68 / para 0.62   [weaker evidence]
  figures  : 0 exact, 2 same, 0 cropped, 0 similar   (3 vs 3 figures)
  json     : outputs/paper1_vs_paper2/result.json
  report   : outputs/paper1_vs_paper2/report.html
  pdf A    : outputs/paper1_vs_paper2/paper1.annotated.pdf
  pdf B    : outputs/paper1_vs_paper2/paper2.annotated.pdf
```

Output files:

- `report.html`: self-contained. Summary cards per layer, then every match
  side by side (differing words in edited passages are highlighted), then
  matched figures as thumbnails with their captions and distances.
- `<A>.annotated.pdf`, `<B>.annotated.pdf`: copies of the inputs with
  highlight annotations; hover a highlight for the partner page and score.
- `result.json`: everything, with page numbers, character offsets, bounding
  boxes and scores, for scripting.

Two 12-page papers take about 5 s (3 s of which is loading the embedding
model). Extraction is cached per PDF under `cache/`.

## Reading the numbers

- **Verbatim coverage** is the share of a document's words that lie inside a
  copied run of at least 12 words. Identical papers give 100 %. Unrelated
  papers give 0 to 2 % (shared boilerplate; see `data/boilerplate.txt`).
  A single 100-word copied paragraph in a 6,000-word paper is 1.7 %, so read
  the run list, not just the percentage.
- `exact` means the word sequence is identical; `near` means edited, with the
  rapidfuzz ratio shown (default cut-off 85).
- **Paraphrase pairs** are model judgements. The report prints the
  distribution of every unit's best cosine so thresholds can be tuned per
  corpus: `--sem-threshold` (sentences, default 0.80) and
  `--sem-para-threshold` (paragraphs, default 0.75). Raise them if unrelated
  papers produce pairs; lower them if a known rewrite is missed.
- **Figure verdicts**: `exact` = identical embedded bytes; `same` = pHash
  distance at most 10 (survives rescaling, JPEG, recolouring); `cropped` =
  one figure is a sub-region of the other (template-match NCC at least 0.85);
  `similar` = pHash within 16, worth a look. Vector figures are rendered at
  150 dpi and hashed like images.

## Options

`scripts/compare_pdfs.py --help` lists everything. The ones that matter:

| option | default | effect |
|---|---|---|
| `--include-references` | off | compare bibliographies too (they legitimately overlap) |
| `--ocr auto\|on\|off` | auto | OCR only pages that look scanned / every page / never |
| `--kgram`, `--window`, `--min-run`, `--max-gap` | 6, 4, 12, 3 | fingerprint length, winnowing window, shortest reported run, edit gap merged inside a run |
| `--near-threshold` | 85 | rapidfuzz ratio for `near` |
| `--model`, `--device` | all-MiniLM-L6-v2, auto | embedding model (must be prefetched), torch device |
| `--sem-threshold`, `--sem-para-threshold`, `--no-mutual` | 0.80, 0.75 | paraphrase cut-offs and nearest-neighbour rule |
| `--phash-threshold`, `--similar-threshold`, `--crop-ncc` | 10, 16, 0.85 | figure cut-offs |
| `--no-semantic`, `--no-images`, `--no-annotate`, `--json-only` | | skip layers or outputs |
| `--boilerplate FILE` | data/boilerplate.txt | phrases never reported |

## Privacy / offline guarantee

- `scripts/prefetch_models.py` (`make models`) is the only code that opens a
  network connection, and it fetches model weights only.
- `scripts/compare_pdfs.py` sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` before any ML library is imported. A missing model
  is an error message, never a download.
- The test suite runs with outbound TCP connections monkeypatched to raise,
  so the full pipeline is exercised with no network at all (`make test`).
- Inputs are only read; outputs go to the output directory and `cache/`,
  both gitignored.

## Limitations

- Re-drawn figures (same data, new plot style) are not matched. `images.py`
  keeps its scoring in one function so a local vision embedding (DINOv2 or
  CLIP) can be added later.
- A panel copied out of a multi-panel vector figure is found only if
  PyMuPDF's drawing clustering splits the panels; otherwise the `cropped`
  check has to catch it.
- Text inside vector figures (axis labels) is excluded from text matching;
  longer table cells are kept.
- Scanned PDFs depend on tesseract quality; OCR'd pages are listed in the
  report header.
- Thresholds were calibrated on synthetic and LaTeX-generated test papers,
  not on real submissions. Check a known pair before trusting the defaults.

## Layout

```
src/pdfsim/    extract.py (PDF -> text units + figures, cached)   textnorm.py   cache.py
               verbatim.py (layer 1)   semantic.py (layer 2)   images.py (layer 3)   report.py
scripts/       compare_pdfs.py (CLI)   prefetch_models.py (network, once)
tests/         pdfgen.py + corpus.py build synthetic two-column papers; no PDF fixtures checked in
data/          boilerplate.txt
cache/ outputs/   generated, gitignored
```

`make test` runs the suite (about 8 s with the model cached).
