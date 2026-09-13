"""Synthetic two-column 'paper' PDFs for tests, built with PyMuPDF so no
binary fixtures are checked in.  Every page carries a running header and a
page-number footer so header/footer stripping is exercised."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

PAGE_W, PAGE_H = 612.0, 792.0
MARGIN = 54.0
COL_GAP = 18.0
COL_W = (PAGE_W - 2 * MARGIN - COL_GAP) / 2
TOP = 90.0
BOTTOM = PAGE_H - 60.0
BODY_PT = 9.0
GAP = 7.0


def make_image(seed: int, size: tuple[int, int] = (320, 240)) -> bytes:
    """Deterministic pseudo-figure (coloured boxes and lines) as PNG bytes."""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = rng.randint(0, size[0] - 40), rng.randint(0, size[1] - 40)
        x1, y1 = x0 + rng.randint(20, 150), y0 + rng.randint(20, 120)
        d.rectangle([x0, y0, min(x1, size[0] - 1), min(y1, size[1] - 1)],
                    fill=tuple(rng.randint(0, 255) for _ in range(3)))
    for _ in range(6):
        d.line([rng.randint(0, size[0]), rng.randint(0, size[1]),
                rng.randint(0, size[0]), rng.randint(0, size[1])],
               fill=tuple(rng.randint(0, 120) for _ in range(3)), width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def degrade(png: bytes, scale: float = 0.5, quality: int = 60) -> bytes:
    """Rescale + JPEG-recompress an image, as a re-used figure would be."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.resize((max(8, int(img.width * scale)), max(8, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def crop(png: bytes, frac: float = 0.25) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * frac), 0, w, h))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class PaperBuilder:
    def __init__(self, title: str, header: str = "Proceedings of the Test Conference 2027"):
        self.doc = pymupdf.open()
        self.header = header
        self.title = title
        self.page: pymupdf.Page | None = None
        self.col = 0
        self.y = TOP
        self.n_fig = 0
        self._new_page()
        self._title()

    # --- layout ---------------------------------------------------------
    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        n = self.doc.page_count
        self.page.insert_text((MARGIN, 30), self.header, fontsize=8, fontname="helv")
        self.page.insert_text((PAGE_W / 2 - 10, PAGE_H - 30), f"{n}", fontsize=8, fontname="helv")
        self.col, self.y = 0, TOP

    def _x(self) -> float:
        return MARGIN + self.col * (COL_W + COL_GAP)

    def _advance_column(self) -> None:
        if self.col == 0:
            self.col, self.y = 1, TOP
        else:
            self._new_page()

    def _title(self) -> None:
        rect = pymupdf.Rect(MARGIN, 48, PAGE_W - MARGIN, 80)
        self.page.insert_textbox(rect, self.title, fontsize=16, fontname="hebo", align=1)

    def _place_text(self, text: str, fontsize: float, fontname: str, needed: float | None = None) -> None:
        for _ in range(3):
            rect = pymupdf.Rect(self._x(), self.y, self._x() + COL_W, BOTTOM)
            if rect.height < max(needed or 0.0, 2.5 * fontsize):
                self._advance_column()
                continue
            unused = self.page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
            if unused >= 0:
                self.y = BOTTOM - unused + GAP
                return
            self._advance_column()
        raise RuntimeError("paragraph does not fit in a column")

    # --- content --------------------------------------------------------
    def heading(self, text: str) -> "PaperBuilder":
        self._place_text(text, 11, "hebo", needed=40)
        return self

    def paragraph(self, text: str) -> "PaperBuilder":
        self._place_text(text, BODY_PT, "helv")
        return self

    def image(self, data: bytes, caption: str, height: float = 120.0) -> "PaperBuilder":
        if self.y + height + 30 > BOTTOM:
            self._advance_column()
        rect = pymupdf.Rect(self._x(), self.y, self._x() + COL_W, self.y + height)
        self.page.insert_image(rect, stream=data, keep_proportion=True)
        self.y = rect.y1 + 4
        self.n_fig += 1
        self._place_text(f"Fig. {self.n_fig}: {caption}", 8, "helv")
        return self

    def vector(self, seed: int, caption: str, height: float = 110.0) -> "PaperBuilder":
        if self.y + height + 30 > BOTTOM:
            self._advance_column()
        rng = random.Random(seed)
        x0, y0 = self._x(), self.y
        shape = self.page.new_shape()
        shape.draw_rect(pymupdf.Rect(x0, y0, x0 + COL_W, y0 + height))
        for _ in range(8):
            r = pymupdf.Rect(x0 + rng.uniform(0, COL_W - 40), y0 + rng.uniform(0, height - 30),
                             0, 0)
            r.x1, r.y1 = r.x0 + rng.uniform(15, 40), r.y0 + rng.uniform(10, 30)
            shape.draw_rect(r)
            shape.finish(color=(0, 0, 0), fill=(rng.random(), rng.random(), rng.random()), width=0.8)
        for _ in range(5):
            shape.draw_line((x0 + rng.uniform(0, COL_W), y0 + rng.uniform(0, height)),
                            (x0 + rng.uniform(0, COL_W), y0 + rng.uniform(0, height)))
        shape.finish(color=(0.2, 0.2, 0.6), width=1.2)
        shape.commit()
        self.y = y0 + height + 4
        self.n_fig += 1
        self._place_text(f"Fig. {self.n_fig}: {caption}", 8, "helv")
        return self

    def references(self, entries: list[str]) -> "PaperBuilder":
        self.heading("References")
        for e in entries:
            self.paragraph(e)
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.doc.save(str(path))
        self.doc.close()
        return path


def build_paper_a(path: Path) -> Path:
    from corpus import (A_ABSTRACT, A_COPIED, A_CONCLUSION, A_EDITED, A_INTRO, A_PARAPHRASE,
                        A_REFERENCES, A_RESULTS)
    b = PaperBuilder("A Bandwidth-Aware Stride Prefetcher for Many-Core Processors")
    b.paragraph(A_ABSTRACT)
    b.heading("1 Introduction").paragraph(A_INTRO).paragraph(A_COPIED)
    b.image(make_image(1), "Prefetcher block diagram.")
    b.heading("2 Design").paragraph(A_EDITED)
    b.vector(7, "Timeliness histogram for the stride detector.")
    b.heading("3 Methodology").paragraph(A_PARAPHRASE)
    b.image(make_image(2), "Simulated system configuration.")
    b.heading("4 Results").paragraph(A_RESULTS)
    b.heading("5 Conclusion").paragraph(A_CONCLUSION)
    b.references(A_REFERENCES)
    return b.save(path)


def build_paper_b(path: Path) -> Path:
    from corpus import (B_ABSTRACT, B_COPIED, B_CONCLUSION, B_EDITED, B_INTRO, B_PARAPHRASE,
                        B_REFERENCES, B_RESULTS)
    b = PaperBuilder("Congestion-Aware Adaptive Routing for Networks on Chip")
    b.paragraph(B_ABSTRACT)
    b.heading("1 Introduction").paragraph(B_INTRO)
    b.image(make_image(3), "Mesh topology with adaptive routers.")
    b.heading("2 Background").paragraph(B_COPIED).paragraph(B_EDITED)
    b.image(degrade(make_image(1)), "Reference prefetcher block diagram.")
    b.heading("3 Methodology").paragraph(B_PARAPHRASE)
    b.vector(7, "Timeliness histogram, reproduced.")
    b.heading("4 Results").paragraph(B_RESULTS)
    b.image(crop(make_image(1)), "Cropped block diagram.")
    b.heading("5 Conclusion").paragraph(B_CONCLUSION)
    b.references(B_REFERENCES)
    return b.save(path)


def build_paper_c(path: Path) -> Path:
    from corpus import C_PARAGRAPHS
    b = PaperBuilder("Occupancy-Guided Near-Threshold Operation for GPUs")
    b.paragraph(C_PARAGRAPHS[0])
    b.heading("1 Introduction").paragraph(C_PARAGRAPHS[1])
    b.image(make_image(4), "GPU voltage domains.")
    b.heading("2 Controller").paragraph(C_PARAGRAPHS[2])
    b.vector(9, "Voltage versus occupancy.")
    b.heading("3 Results").paragraph(C_PARAGRAPHS[3])
    b.references(["[1] Someone, Something about GPUs, 2018."])
    return b.save(path)
