import io

import numpy as np
import pytest
from PIL import Image

from pdfgen import crop, degrade, make_image
from pdfsim import images
from pdfsim.extract import ExtractOptions, Figure, extract


@pytest.fixture(scope="module")
def docs(papers, cache_dir):
    return {k: extract(p, ExtractOptions(), cache_dir=cache_dir) for k, p in papers.items()}


def _verdicts(r):
    return {(m.a_caption.split(":")[0], m.b_caption.split(":")[0]): m.verdict for m in r.matches}


def test_reused_figures(docs):
    r = images.compare(docs["a"], docs["b"])
    assert _verdicts(r) == {("Fig. 1", "Fig. 2"): "same", ("Fig. 1", "Fig. 4"): "cropped", ("Fig. 2", "Fig. 3"): "same"}
    same = [m for m in r.matches if m.verdict == "same"]
    assert all(m.phash_distance <= 2 and m.ncc > 0.95 for m in same)


def test_identical_documents(docs):
    r = images.compare(docs["a"], docs["a"])
    assert len(r.matches) == 3 and all(m.verdict == "exact" for m in r.matches)


def test_unrelated_documents(docs):
    assert images.compare(docs["a"], docs["c"]).matches == []
    assert images.compare(docs["b"], docs["c"]).matches == []


def _gray(png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png)).convert("L")
    img.thumbnail((images.CROP_SIDE, images.CROP_SIDE))
    return np.asarray(img, dtype=np.float32)


def test_crop_score():
    full = make_image(11)
    assert images.crop_score(_gray(full), _gray(crop(full, 0.3))) > 0.9
    assert images.crop_score(_gray(full), _gray(degrade(crop(full, 0.3)))) > 0.85
    assert images.crop_score(_gray(full), _gray(make_image(12))) < 0.8


def test_blank_figures_ignored(tmp_path):
    p = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), (255, 255, 255)).save(p)
    fig = Figure(0, 0, (0, 0, 100, 100), str(p), "vector", "abc", 200, 200)
    h = images.hash_figure(fig)
    assert h.blank
    assert images.score_pair(fig, fig, h, h, images.ImageParams()) is None
