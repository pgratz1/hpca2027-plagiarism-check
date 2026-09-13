import random

import pytest

from corpus import A_COPIED
from pdfsim import verbatim
from pdfsim.extract import ExtractOptions, extract
from pdfsim.textnorm import tokenize
from pdfsim.verbatim import VerbatimParams, _fingerprints, _is_boilerplate, _merge_runs, load_boilerplate


@pytest.fixture(scope="module")
def docs(papers, cache_dir):
    return {k: extract(p, ExtractOptions(), cache_dir=cache_dir) for k, p in papers.items()}


def test_identical_documents(docs):
    r = verbatim.compare(docs["a"], docs["a"])
    assert r.coverage_a == 1.0 and r.coverage_b == 1.0
    assert len(r.runs) == 1 and r.runs[0].verdict == "exact" and r.sentence_pairs == []


def test_copied_and_edited_paragraphs(docs):
    r = verbatim.compare(docs["a"], docs["b"])
    exact = [x for x in r.runs if x.verdict == "exact"]
    assert len(exact) == 1
    assert exact[0].a_text == exact[0].b_text == A_COPIED.rstrip(".")  # runs span tokens, not punctuation
    assert exact[0].length == len(tokenize(A_COPIED))
    near = [x for x in r.runs if x.verdict == "near"]
    assert len(near) == 1 and near[0].score < 100 and "confirms a pattern after" in near[0].a_text
    assert 0.2 < r.coverage_a < 0.45 and 0.2 < r.coverage_b < 0.45
    assert r.longest_run == exact[0].length
    assert any(p.verdict == "near" and p.a_text.startswith("The stride detector") for p in r.sentence_pairs)
    for run in r.runs:
        assert all(loc.rects for loc in run.a_locs + run.b_locs)
        assert run.a_locs[0].page == 0


def test_unrelated_documents(docs):
    r = verbatim.compare(docs["a"], docs["c"])
    assert r.runs == [] and r.sentence_pairs == []
    assert r.coverage_a == 0.0 and r.longest_run == 0


def test_references_only_compared_on_request(papers, tmp_path):
    opts = ExtractOptions(include_references=True)
    a, b = extract(papers["a"], opts, cache_dir=tmp_path), extract(papers["b"], opts, cache_dir=tmp_path)
    r0 = verbatim.compare(a, b)
    r1 = verbatim.compare(a, b, VerbatimParams(include_references=True))
    hits = lambda r: [x.a_text for x in r.runs] + [p.a_text for p in r.sentence_pairs]  # noqa: E731
    assert not any("Computing Surveys" in t for t in hits(r0))
    assert any("Computing Surveys" in t for t in hits(r1))


def test_winnowing_guarantee():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(2000)]
    k, w = 6, 4
    for _ in range(30):
        shared = [rng.choice(vocab) for _ in range(k + w - 1)]
        a = [rng.choice(vocab) for _ in range(rng.randint(0, 60))] + shared + [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        b = [rng.choice(vocab) for _ in range(rng.randint(0, 60))] + shared + [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        assert set(_fingerprints(a, k, w)) & set(_fingerprints(b, k, w))


def test_merge_runs():
    runs = [(0, 10, 100, 110), (12, 20, 112, 120), (50, 60, 300, 310)]
    assert _merge_runs(runs, max_gap=3) == [(0, 20, 100, 120), (50, 60, 300, 310)]
    assert _merge_runs(runs, max_gap=1) == runs


def test_boilerplate(tmp_path):
    f = tmp_path / "bp.txt"
    f.write_text("# comment\nthis text is ignored\n")
    phrases = load_boilerplate(f)
    assert _is_boilerplate("this text is ignored", phrases)
    assert _is_boilerplate("the rest of the paper is organized as follows", phrases)
    assert not _is_boilerplate("our prefetcher improves performance by 14 percent", phrases)
