import pytest

pytest.importorskip("sentence_transformers")

from pdfsim import semantic, verbatim  # noqa: E402
from pdfsim.extract import ExtractOptions, extract  # noqa: E402


@pytest.fixture(scope="module")
def model():
    try:
        return semantic.load_model()
    except semantic.ModelNotAvailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def docs(papers, cache_dir):
    return {k: extract(p, ExtractOptions(), cache_dir=cache_dir) for k, p in papers.items()}


def test_paraphrase_found_and_verbatim_excluded(docs, model):
    v = verbatim.compare(docs["a"], docs["b"])
    r = semantic.compare(docs["a"], docs["b"], verbatim_result=v)
    assert any(p.level == "paragraph" and p.a_text.startswith("Our evaluation uses")
               and p.b_text.startswith("We evaluate on") and p.score > 0.8 for p in r.pairs)
    assert not any(p.a_text.startswith("We observe that prefetch") for p in r.pairs), "verbatim run leaked"
    assert not any(p.level == "sentence" and p.a_text.startswith("The stride detector") for p in r.pairs)
    assert r.device and sum(r.histogram["sentence"]) == r.n_a["sentence"]
    assert all(p.a_rects and p.b_rects for p in r.pairs)


def test_unrelated_documents(docs, model):
    r = semantic.compare(docs["a"], docs["c"])
    assert r.pairs == []
    assert r.mean_max_sim["sentence"] < 0.6 and r.fraction_matched_a["sentence"] == 0.0


def test_identical_documents_have_nothing_paraphrase_only(docs, model):
    v = verbatim.compare(docs["a"], docs["a"])
    r = semantic.compare(docs["a"], docs["a"], verbatim_result=v)
    assert r.pairs == [] and r.n_a == {"sentence": 0, "paragraph": 0}


def test_missing_model_fails_offline_without_download():
    with pytest.raises(semantic.ModelNotAvailable):
        semantic.load_model("sentence-transformers/this-model-does-not-exist-pdfsim")
