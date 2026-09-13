import json
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_pdfs import main  # noqa: E402
from pdfsim.report import diff_marks  # noqa: E402


def test_diff_marks():
    a, b = diff_marks("the quick brown fox", "the slow brown fox jumps")
    assert "<mark>quick</mark>" in str(a) and "<mark>slow</mark>" in str(b) and "<mark>jumps</mark>" in str(b)
    assert "<mark>the</mark>" not in str(a)


def test_cli_end_to_end(papers, tmp_path):
    out = tmp_path / "out"
    rc = main([str(papers["a"]), str(papers["b"]), "-o", str(out), "--cache-dir", str(tmp_path / "cache"), "--quiet"])
    assert rc == 0
    data = json.loads((out / "result.json").read_text())
    s = data["summary"]
    assert s["verbatim"]["exact_runs"] == 1 and s["verbatim"]["near_runs"] == 1 and s["verbatim"]["near_sentences"] == 1
    assert s["figures"] == {"n_a": 3, "n_b": 4, "exact": 0, "same": 2, "cropped": 1, "similar": 0}
    if s["semantic"] is not None:
        assert s["semantic"]["paragraph_pairs"] == 1
    else:
        pytest.skip(f"semantic layer unavailable: {data['semantic_note']}")
    html = (out / "report.html").read_text()
    assert "Verbatim / near-verbatim" in html and "<mark>" in html and "data:image/png;base64," in html
    assert "weaker evidence" in html
    for name in ("paper_a", "paper_b"):
        pdf = pymupdf.open(out / f"{name}.annotated.pdf")
        kinds = [a.type[1] for page in pdf for a in page.annots()]
        assert kinds.count("Highlight") >= 5 and kinds.count("Square") == 3
        assert all(a.info["title"] == "pdfsim" for page in pdf for a in page.annots())


def test_cli_layers_can_be_disabled(papers, tmp_path):
    out = tmp_path / "out"
    rc = main([str(papers["a"]), str(papers["c"]), "-o", str(out), "--cache-dir", str(tmp_path / "cache"),
               "--no-semantic", "--no-images", "--json-only", "--quiet"])
    assert rc == 0
    data = json.loads((out / "result.json").read_text())
    assert data["semantic"] is None and data["images"] is None
    assert data["summary"]["verbatim"]["runs"] == 0
    assert not (out / "report.html").exists() and not list(out.glob("*.annotated.pdf"))


def test_cli_rejects_missing_file(tmp_path):
    assert main([str(tmp_path / "nope.pdf"), str(tmp_path / "nope2.pdf"), "-o", str(tmp_path)]) == 2
