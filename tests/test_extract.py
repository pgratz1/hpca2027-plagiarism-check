import shutil
from pathlib import Path

import pytest

from pdfsim import cache as cache_mod
from pdfsim.extract import ExtractOptions, extract


@pytest.fixture(scope="module")
def doc_a(papers, cache_dir):
    return extract(papers["a"], ExtractOptions(), cache_dir=cache_dir)


def test_structure(doc_a):
    assert doc_a.n_pages == 2
    assert doc_a.title.startswith("A Bandwidth-Aware Stride Prefetcher")
    assert {"abstract", "heading", "body", "caption"} <= {p.kind for p in doc_a.paragraphs}
    texts = [p.text for p in doc_a.paragraphs]
    assert not any("Proceedings of the Test Conference" in t for t in texts), "running header not dropped"
    assert not any(t.strip() in ("1", "2") for t in texts), "page number not dropped"
    assert not any(p.kind == "reference" for p in doc_a.paragraphs)
    assert not any("Computing Surveys" in t for t in texts), "bibliography leaked into the text"
    assert doc_a.ocr_pages == []
    assert any(p.kind == "body" and p.text.startswith("We observe that prefetch accuracy") for p in doc_a.paragraphs)
    abstract = [p for p in doc_a.paragraphs if p.kind == "abstract"]
    assert len(abstract) == 1 and "Hardware prefetchers" in abstract[0].text


def test_references_flag(papers, cache_dir):
    doc = extract(papers["a"], ExtractOptions(include_references=True), cache_dir=cache_dir)
    refs = [p for p in doc.paragraphs if p.kind == "reference"]
    assert len(refs) == 3 and refs[0].text.startswith("[1]")


def test_sentences_map_back_to_paragraphs_and_pages(doc_a):
    assert doc_a.sentences
    for s in doc_a.sentences:
        assert s.rects, s.text
        for page, (x0, y0, x1, y1) in s.rects:
            assert 0 <= page < doc_a.n_pages
            assert 0 <= x0 < x1 <= 612 and 0 <= y0 < y1 <= 792
        parent = doc_a.paragraphs[s.para]
        assert parent.text[s.span[0]:s.span[1]] == s.text


def test_figures(doc_a):
    assert sorted(f.kind for f in doc_a.figures) == ["raster", "raster", "vector"]
    assert {f.caption.split(":")[0] for f in doc_a.figures} == {"Fig. 1", "Fig. 2", "Fig. 3"}
    for f in doc_a.figures:
        assert Path(f.path).is_file()
        assert f.width >= 48 and f.height >= 48
        assert len(f.sha256) == 64


def test_cache_roundtrip(papers, cache_dir):
    d1 = extract(papers["b"], ExtractOptions(), cache_dir=cache_dir)
    entry = cache_mod.entry_dir(cache_dir, cache_mod.cache_key(d1.sha256, ExtractOptions().to_dict())) / "document.json"
    mtime = entry.stat().st_mtime_ns
    d2 = extract(papers["b"], ExtractOptions(), cache_dir=cache_dir)
    assert entry.stat().st_mtime_ns == mtime, "cache hit should not rewrite the entry"
    assert d1.to_dict() == d2.to_dict()


def test_cache_rebuilds_when_figures_missing(papers, tmp_path):
    d1 = extract(papers["c"], ExtractOptions(), cache_dir=tmp_path)
    Path(d1.figures[0].path).unlink()
    d2 = extract(papers["c"], ExtractOptions(), cache_dir=tmp_path)
    assert d2.figures and all(Path(f.path).is_file() for f in d2.figures)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_forced(papers, tmp_path):
    doc = extract(papers["c"], ExtractOptions(ocr="on"), cache_dir=tmp_path)
    assert doc.ocr_pages == [0]
    text = " ".join(p.text for p in doc.paragraphs).lower()
    assert "voltage" in text and "occupancy" in text
    assert all(s.rects for s in doc.sentences)
