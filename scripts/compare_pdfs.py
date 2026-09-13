#!/usr/bin/env python
"""Compare two PDFs for shared text and figures, fully offline.

    python scripts/compare_pdfs.py A.pdf B.pdf [-o outputs/A_vs_B] [--no-semantic] [--no-images]

Writes into the output directory:
    result.json               every match with pages, offsets and scores
    report.html               self-contained side-by-side report
    <A>.annotated.pdf         copies of the inputs with highlights:
    <B>.annotated.pdf         red = exact, orange = near-verbatim,
                              blue = paraphrase, green = matched figure
and prints a summary.  The three evidence layers are never merged into one
score: verbatim copying is direct evidence, paraphrase pairs are model
judgements, figure matches are perceptual-hash results.

Network: none.  Hugging Face offline mode is forced below, before any ML
library is imported; the embedding model must have been fetched once with
scripts/prefetch_models.py (`make models`).  Without it the semantic layer is
skipped with a warning and the other two layers still run.
"""

from __future__ import annotations

import os

for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
               ("HF_HUB_DISABLE_TELEMETRY", "1"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pdfsim.extract import ExtractOptions, extract  # noqa: E402
from pdfsim.images import ImageParams  # noqa: E402
from pdfsim.images import compare as image_compare  # noqa: E402
from pdfsim.report import ComparisonResult, annotate_pdfs, summary_text, write_html, write_json  # noqa: E402
from pdfsim.verbatim import VerbatimParams, load_boilerplate  # noqa: E402
from pdfsim.verbatim import compare as verbatim_compare  # noqa: E402

DEFAULT_BOILERPLATE = ROOT / "data" / "boilerplate.txt"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="first PDF")
    ap.add_argument("b", type=Path, help="second PDF")
    ap.add_argument("-o", "--out", type=Path, help="output directory (default: outputs/<A>_vs_<B>)")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache", help="extraction cache (default: cache/)")
    ap.add_argument("--no-semantic", action="store_true", help="skip the paraphrase layer (no model load)")
    ap.add_argument("--no-images", action="store_true", help="skip figure matching")
    ap.add_argument("--no-annotate", action="store_true", help="do not write annotated PDFs")
    ap.add_argument("--json-only", action="store_true", help="write result.json only")
    ap.add_argument("--quiet", action="store_true", help="no summary on stdout")

    ex = ap.add_argument_group("extraction")
    ex.add_argument("--include-references", action="store_true", help="keep the bibliography in the comparison")
    ex.add_argument("--ocr", choices=("auto", "on", "off"), default="auto",
                    help="tesseract OCR: auto = scanned pages only (default), on = every page, off = never")
    ex.add_argument("--figure-dpi", type=int, default=150, help="render DPI for vector figures (default 150)")

    vb = ap.add_argument_group("verbatim layer")
    vb.add_argument("--kgram", type=int, default=6, help="fingerprint k-gram length in words (default 6)")
    vb.add_argument("--window", type=int, default=4, help="winnowing window (default 4)")
    vb.add_argument("--min-run", type=int, default=12, help="shortest reported run, in words (default 12)")
    vb.add_argument("--max-gap", type=int, default=3, help="edit gap tolerated inside a run, in words (default 3)")
    vb.add_argument("--near-threshold", type=float, default=85, help="rapidfuzz ratio for `near` (default 85)")
    vb.add_argument("--boilerplate", type=Path, default=DEFAULT_BOILERPLATE,
                    help="file of phrases to ignore (default data/boilerplate.txt)")

    se = ap.add_argument_group("semantic layer")
    se.add_argument("--model", default=None, help="sentence-transformers model id (default all-MiniLM-L6-v2)")
    se.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu (default: auto)")
    se.add_argument("--sem-threshold", type=float, default=0.80, help="sentence cosine threshold (default 0.80)")
    se.add_argument("--sem-para-threshold", type=float, default=0.75, help="paragraph cosine threshold (default 0.75)")
    se.add_argument("--no-mutual", action="store_true", help="do not require mutual nearest neighbours")

    im = ap.add_argument_group("figure layer")
    im.add_argument("--phash-threshold", type=int, default=10, help="pHash Hamming distance for `same` (default 10)")
    im.add_argument("--similar-threshold", type=int, default=16, help="pHash distance for `similar` (default 16)")
    im.add_argument("--crop-ncc", type=float, default=0.85, help="template-match NCC for `cropped` (default 0.85)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for p in (args.a, args.b):
        if not p.is_file():
            print(f"error: {p} is not a file", file=sys.stderr)
            return 2
    out_dir = args.out or (ROOT / "outputs" / f"{args.a.stem}_vs_{args.b.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t = time.time()
    opts = ExtractOptions(include_references=args.include_references, ocr=args.ocr, figure_dpi=args.figure_dpi)
    doc_a = extract(args.a, opts, cache_dir=args.cache_dir)
    doc_b = extract(args.b, opts, cache_dir=args.cache_dir)
    timings["extract"] = time.time() - t

    t = time.time()
    vparams = VerbatimParams(kgram=args.kgram, window=args.window, min_run=args.min_run, max_gap=args.max_gap,
                             near_threshold=args.near_threshold, include_references=args.include_references)
    verbatim = verbatim_compare(doc_a, doc_b, vparams, load_boilerplate(args.boilerplate))
    timings["verbatim"] = time.time() - t

    images = None
    images_note = "disabled with --no-images"
    if not args.no_images:
        t = time.time()
        images = image_compare(doc_a, doc_b, ImageParams(phash_threshold=args.phash_threshold,
                                                         similar_threshold=args.similar_threshold,
                                                         crop_ncc=args.crop_ncc))
        images_note = ""
        timings["figures"] = time.time() - t

    semantic = None
    semantic_note = "disabled with --no-semantic"
    if not args.no_semantic:
        from pdfsim.semantic import DEFAULT_MODEL, ModelNotAvailable, SemanticParams
        from pdfsim.semantic import compare as semantic_compare
        t = time.time()
        sparams = SemanticParams(model=args.model or DEFAULT_MODEL, device=args.device,
                                 sentence_threshold=args.sem_threshold,
                                 paragraph_threshold=args.sem_para_threshold, mutual=not args.no_mutual,
                                 include_references=args.include_references)
        try:
            semantic = semantic_compare(doc_a, doc_b, sparams, verbatim_result=verbatim)
            semantic_note = ""
        except ModelNotAvailable as exc:
            semantic_note = str(exc)
            print(f"warning: semantic layer skipped: {exc}", file=sys.stderr)
        timings["semantic"] = time.time() - t

    result = ComparisonResult(doc_a, doc_b, verbatim, semantic, images, semantic_note, images_note, timings)
    outputs: dict[str, Path] = {"json": write_json(result, out_dir / "result.json")}
    if not args.json_only:
        outputs["report"] = write_html(result, out_dir / "report.html")
        if not args.no_annotate:
            for side, path in annotate_pdfs(result, out_dir).items():
                outputs[f"pdf {side.upper()}"] = path
    if not args.quiet:
        print(summary_text(result, outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
