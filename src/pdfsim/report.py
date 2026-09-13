"""Report writers: result.json, a self-contained report.html, and
highlight-annotated copies of both PDFs.

The three evidence layers are never merged into one number.  Colours are
fixed so a reader learns them once:

  red    = exact verbatim run           orange = near-verbatim run / sentence
  blue   = paraphrase (semantic layer)  green  = matched figure
"""

from __future__ import annotations

import base64
import datetime as _dt
import difflib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from jinja2 import Environment, BaseLoader, select_autoescape
from markupsafe import Markup, escape
from PIL import Image

from . import __version__
from .extract import Document
from .images import ImageResult
from .verbatim import VerbatimResult

COLORS = {
    "exact": (0.95, 0.35, 0.35),
    "near": (1.0, 0.65, 0.2),
    "semantic": (0.35, 0.55, 1.0),
    "figure": (0.2, 0.75, 0.35),
}
CSS_COLORS = {"exact": "#e05555", "near": "#f0a030", "semantic": "#4d8cff", "figure": "#2fbf5f"}
FIGURE_VERDICTS = ("exact", "same", "cropped", "similar")


@dataclass
class ComparisonResult:
    a: Document
    b: Document
    verbatim: VerbatimResult
    semantic: Any | None = None          # pdfsim.semantic.SemanticResult
    images: ImageResult | None = None
    semantic_note: str = ""              # why the semantic layer is absent
    images_note: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        v = self.verbatim
        out: dict[str, Any] = {
            "verbatim": {
                "coverage_a": round(v.coverage_a, 4), "coverage_b": round(v.coverage_b, 4),
                "tokens_a": v.tokens_a, "tokens_b": v.tokens_b,
                "runs": len(v.runs),
                "exact_runs": sum(r.verdict == "exact" for r in v.runs),
                "near_runs": sum(r.verdict == "near" for r in v.runs),
                "longest_run": v.longest_run,
                "near_sentences": len(v.sentence_pairs),
            },
            "semantic": None,
            "figures": None,
        }
        if self.semantic is not None:
            s = self.semantic
            out["semantic"] = {
                "pairs": len(s.pairs),
                "sentence_pairs": sum(p.level == "sentence" for p in s.pairs),
                "paragraph_pairs": sum(p.level == "paragraph" for p in s.pairs),
                "mean_max_sim": s.mean_max_sim,
                "fraction_matched_a": s.fraction_matched_a,
                "fraction_matched_b": s.fraction_matched_b,
                "model": s.model, "device": s.device,
            }
        if self.images is not None:
            im = self.images
            out["figures"] = {"n_a": im.n_a, "n_b": im.n_b,
                              **{k: sum(m.verdict == k for m in im.matches) for k in FIGURE_VERDICTS}}
        return out


def _doc_meta(doc: Document) -> dict[str, Any]:
    return {"path": doc.path, "name": doc.name, "sha256": doc.sha256, "n_pages": doc.n_pages,
            "title": doc.title, "n_paragraphs": len(doc.paragraphs),
            "n_sentences": len(doc.sentences), "n_figures": len(doc.figures),
            "ocr_pages": [p + 1 for p in doc.ocr_pages]}


# --- JSON --------------------------------------------------------------------

def write_json(result: ComparisonResult, path: str | Path) -> Path:
    data = {
        "pdfsim_version": __version__,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "a": _doc_meta(result.a),
        "b": _doc_meta(result.b),
        "summary": result.summary(),
        "timings_s": {k: round(t, 2) for k, t in result.timings.items()},
        "verbatim": result.verbatim.to_dict(),
        "semantic": result.semantic.to_dict() if result.semantic is not None else None,
        "semantic_note": result.semantic_note,
        "images": result.images.to_dict() if result.images is not None else None,
        "images_note": result.images_note,
    }
    path = Path(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    return path


# --- stdout summary ----------------------------------------------------------

def summary_text(result: ComparisonResult, outputs: dict[str, Path] | None = None) -> str:
    s = result.summary()
    v = s["verbatim"]
    lines = [f"{result.a.name} vs {result.b.name}"]
    lines.append(
        f"  verbatim : {v['coverage_a']:.1%} of A / {v['coverage_b']:.1%} of B inside "
        f"{v['runs']} runs ({v['exact_runs']} exact, {v['near_runs']} near; longest {v['longest_run']} words); "
        f"{v['near_sentences']} near-verbatim sentences")
    if s["semantic"] is not None:
        m = s["semantic"]
        lines.append(
            f"  semantic : {m['pairs']} paraphrase-only pairs ({m['sentence_pairs']} sentences, "
            f"{m['paragraph_pairs']} paragraphs); mean max-sim sent {m['mean_max_sim'].get('sentence', 0):.2f} / "
            f"para {m['mean_max_sim'].get('paragraph', 0):.2f}   [weaker evidence]")
    else:
        lines.append(f"  semantic : skipped ({result.semantic_note or 'disabled'})")
    if s["figures"] is not None:
        f = s["figures"]
        lines.append(
            f"  figures  : {f['exact']} exact, {f['same']} same, {f['cropped']} cropped, {f['similar']} similar"
            f"   ({f['n_a']} vs {f['n_b']} figures)")
    else:
        lines.append(f"  figures  : skipped ({result.images_note or 'disabled'})")
    for label, p in (outputs or {}).items():
        lines.append(f"  {label:9}: {p}")
    return "\n".join(lines)


# --- HTML --------------------------------------------------------------------

def diff_marks(a_text: str, b_text: str) -> tuple[Markup, Markup]:
    """Both texts with the words that differ wrapped in <mark>."""
    ta, tb = a_text.split(), b_text.split()
    key = lambda w: w.lower().strip(".,;:()[]\"'")  # noqa: E731
    sm = difflib.SequenceMatcher(a=[key(w) for w in ta], b=[key(w) for w in tb], autojunk=False)
    out_a: list[Markup] = []
    out_b: list[Markup] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a, seg_b = " ".join(ta[i1:i2]), " ".join(tb[j1:j2])
        if tag == "equal":
            out_a.append(escape(seg_a))
            out_b.append(escape(seg_b))
        else:
            if seg_a:
                out_a.append(Markup("<mark>") + escape(seg_a) + Markup("</mark>"))
            if seg_b:
                out_b.append(Markup("<mark>") + escape(seg_b) + Markup("</mark>"))
    return Markup(" ").join(out_a), Markup(" ").join(out_b)


def _thumb_b64(path: str, max_w: int = 360) -> str:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    if img.width > max_w:
        img = img.resize((max_w, max(1, int(img.height * max_w / img.width))))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>pdfsim: {{ a.name }} vs {{ b.name }}</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 32px;color:#222;background:#fafafa;max-width:1400px}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 8px;padding-bottom:4px;border-bottom:2px solid #ddd}
 .meta{color:#555;font-size:13px} .cards{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}
 .card{flex:1 1 280px;background:#fff;border:1px solid #ddd;border-left:6px solid #999;border-radius:6px;padding:12px 16px}
 .card h3{margin:0 0 6px;font-size:15px} .card .big{font-size:26px;font-weight:600}
 .card.verbatim{border-left-color:{{ css.exact }}} .card.semantic{border-left-color:{{ css.semantic }}} .card.figures{border-left-color:{{ css.figure }}}
 .weak{display:inline-block;background:#eef;color:#335;font-size:11px;padding:1px 6px;border-radius:3px;margin-left:6px}
 table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 16px} th,td{border:1px solid #ddd;padding:6px 8px;vertical-align:top;text-align:left}
 th{background:#f0f0f0;font-size:12px} td.txt{width:42%;font-size:13px} td.num{white-space:nowrap;font-size:12px;color:#444}
 .badge{display:inline-block;padding:1px 7px;border-radius:3px;color:#fff;font-size:11px;font-weight:600;text-transform:uppercase}
 .badge.exact{background:{{ css.exact }}} .badge.near{background:{{ css.near }}} .badge.semantic{background:{{ css.semantic }}}
 .badge.same,.badge.cropped,.badge.similar{background:{{ css.figure }}}
 mark{background:#ffe08a;padding:0 1px} img.fig{max-width:360px;border:1px solid #ccc;background:#fff}
 .hist{font-family:monospace;font-size:12px;color:#555} .note{color:#666;font-size:13px} .params{font-family:monospace;font-size:11px;color:#666;white-space:pre-wrap}
 .empty{color:#777;font-style:italic}
</style></head><body>
<h1>{{ a.name }} <span class="meta">vs</span> {{ b.name }}</h1>
<div class="meta">A: <b>{{ a.title or '(no title found)' }}</b> &middot; {{ a.n_pages }} pages &middot; {{ a.paragraphs|length }} paragraphs &middot; {{ a.figures|length }} figures{% if a.ocr_pages %} &middot; OCR on pages {{ a.ocr_pages|map('int')|map('string')|join(', ') }}{% endif %}<br>
B: <b>{{ b.title or '(no title found)' }}</b> &middot; {{ b.n_pages }} pages &middot; {{ b.paragraphs|length }} paragraphs &middot; {{ b.figures|length }} figures<br>
generated {{ generated }} by pdfsim {{ version }}, fully offline{% if timings %} &middot; {% for k, t in timings.items() %}{{ k }} {{ '%.1f'|format(t) }}s{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}</div>

<div class="cards">
 <div class="card verbatim"><h3>1. Verbatim / near-verbatim text</h3>
  <div class="big">{{ '%.1f'|format(s.verbatim.coverage_a*100) }}% of A &middot; {{ '%.1f'|format(s.verbatim.coverage_b*100) }}% of B</div>
  <div>{{ s.verbatim.runs }} copied runs ({{ s.verbatim.exact_runs }} exact, {{ s.verbatim.near_runs }} near), longest {{ s.verbatim.longest_run }} words<br>{{ s.verbatim.near_sentences }} near-verbatim sentence pairs</div></div>
 <div class="card semantic"><h3>2. Paraphrase-only text <span class="weak">weaker evidence</span></h3>
  {% if s.semantic %}<div class="big">{{ s.semantic.pairs }} pairs</div>
  <div>{{ s.semantic.sentence_pairs }} sentences, {{ s.semantic.paragraph_pairs }} paragraphs above threshold<br>
  mean best-match cosine: sentences {{ '%.2f'|format(s.semantic.mean_max_sim.sentence) }}, paragraphs {{ '%.2f'|format(s.semantic.mean_max_sim.paragraph) }}<br>
  <span class="note">{{ s.semantic.model }} on {{ s.semantic.device }}</span></div>
  {% else %}<div class="empty">skipped: {{ semantic_note or 'disabled' }}</div>{% endif %}</div>
 <div class="card figures"><h3>3. Figures</h3>
  {% if s.figures %}<div class="big">{{ s.figures.exact + s.figures.same + s.figures.cropped }} re-used</div>
  <div>{{ s.figures.exact }} identical, {{ s.figures.same }} same, {{ s.figures.cropped }} cropped, {{ s.figures.similar }} similar<br>of {{ s.figures.n_a }} figures in A and {{ s.figures.n_b }} in B</div>
  {% else %}<div class="empty">skipped: {{ images_note or 'disabled' }}</div>{% endif %}</div>
</div>
<p class="note">Layers are reported separately on purpose: verbatim runs are direct evidence of copying; paraphrase pairs are model judgements that need a human read; figure matches are perceptual-hash results. In the annotated PDFs: <span class="badge exact">exact</span> <span class="badge near">near</span> <span class="badge semantic">paraphrase</span> <span class="badge same">figure</span>.</p>

<h2>1. Verbatim / near-verbatim runs</h2>
{% if runs %}
<table><tr><th>#</th><th>verdict</th><th>words</th><th>A (page)</th><th>B (page)</th></tr>
{% for r in runs %}<tr><td class="num">{{ r.id }}</td><td class="num"><span class="badge {{ r.verdict }}">{{ r.verdict }}</span><br>{{ r.score }}</td><td class="num">{{ r.length }}</td>
<td class="txt">p{{ r.a_page }}: {{ r.a_html }}</td><td class="txt">p{{ r.b_page }}: {{ r.b_html }}</td></tr>
{% endfor %}</table>
{% else %}<p class="empty">No shared runs of {{ vparams.min_run }}+ words.</p>{% endif %}

<h3>Near-verbatim sentences (outside the runs above)</h3>
{% if sents %}
<table><tr><th>verdict</th><th>A (page)</th><th>B (page)</th></tr>
{% for p in sents %}<tr><td class="num"><span class="badge {{ p.verdict }}">{{ p.verdict }}</span><br>{{ p.score }}</td>
<td class="txt">p{{ p.a_page }}: {{ p.a_html }}</td><td class="txt">p{{ p.b_page }}: {{ p.b_html }}</td></tr>
{% endfor %}</table>
{% else %}<p class="empty">None.</p>{% endif %}

<h2>2. Paraphrase-only matches <span class="weak">weaker evidence</span></h2>
{% if semantic %}
<p class="note">Mutual nearest neighbours by cosine similarity of {{ semantic.model }} embeddings (sentences &ge; {{ semantic.params.sentence_threshold }}, paragraphs &ge; {{ semantic.params.paragraph_threshold }}); units already inside a verbatim run are excluded.
Distribution of each A unit's best cosine in B (bins 0.0&ndash;1.0): <span class="hist">sentences {{ semantic.histogram.sentence }} &middot; paragraphs {{ semantic.histogram.paragraph }}</span></p>
{% if semantic.pairs %}
<table><tr><th>level</th><th>cosine</th><th>A (page)</th><th>B (page)</th></tr>
{% for p in sem_pairs %}<tr><td class="num">{{ p.level }}</td><td class="num">{{ '%.3f'|format(p.score) }}</td>
<td class="txt">p{{ p.a_page }}: {{ p.a_html }}</td><td class="txt">p{{ p.b_page }}: {{ p.b_html }}</td></tr>
{% endfor %}</table>
{% else %}<p class="empty">No paraphrase-only pairs above threshold.</p>{% endif %}
{% else %}<p class="empty">Skipped: {{ semantic_note or 'disabled' }}</p>{% endif %}

<h2>3. Figures</h2>
{% if images %}
{% if figs %}
<table><tr><th>verdict</th><th>distances</th><th>A (page)</th><th>B (page)</th></tr>
{% for m in figs %}<tr><td class="num"><span class="badge {{ m.verdict }}">{{ m.verdict }}</span></td>
<td class="num">pHash {{ m.phash_distance }}<br>dHash {{ m.dhash_distance }}<br>NCC {{ m.ncc }}<br>crop NCC {{ m.crop_ncc }}</td>
<td>p{{ m.a_page }}: {{ m.a_caption or '(no caption)' }}<br><img class="fig" src="data:image/png;base64,{{ m.a_b64 }}"></td>
<td>p{{ m.b_page }}: {{ m.b_caption or '(no caption)' }}<br><img class="fig" src="data:image/png;base64,{{ m.b_b64 }}"></td></tr>
{% endfor %}</table>
{% else %}<p class="empty">No re-used figures among {{ images.n_a }} in A and {{ images.n_b }} in B.</p>{% endif %}
{% else %}<p class="empty">Skipped: {{ images_note or 'disabled' }}</p>{% endif %}

<h2>Parameters</h2>
<div class="params">verbatim: {{ vparams }}
{% if semantic %}semantic: {{ semantic.params }}{% endif %}
{% if images %}images:   {{ images.params }}{% endif %}</div>
</body></html>
"""


def write_html(result: ComparisonResult, path: str | Path) -> Path:
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=True))
    tpl = env.from_string(_TEMPLATE)
    v = result.verbatim
    runs = []
    for r in v.runs:
        a_html, b_html = diff_marks(r.a_text, r.b_text) if r.verdict != "exact" else (escape(r.a_text), escape(r.b_text))
        runs.append(dict(id=r.id, verdict=r.verdict, score=r.score, length=r.length,
                         a_page=r.a_locs[0].page + 1 if r.a_locs else "?",
                         b_page=r.b_locs[0].page + 1 if r.b_locs else "?", a_html=a_html, b_html=b_html))
    sents = []
    for p in v.sentence_pairs:
        a_html, b_html = diff_marks(p.a_text, p.b_text)
        sents.append(dict(verdict=p.verdict, score=p.score, a_page=p.a_page + 1, b_page=p.b_page + 1,
                          a_html=a_html, b_html=b_html))
    sem_pairs = []
    if result.semantic is not None:
        for p in sorted(result.semantic.pairs, key=lambda p: (p.level != "paragraph", -p.score)):
            sem_pairs.append(dict(level=p.level, score=p.score, a_page=p.a_page + 1, b_page=p.b_page + 1,
                                  a_html=escape(p.a_text), b_html=escape(p.b_text)))
    figs = []
    if result.images is not None:
        for m in result.images.matches:
            figs.append(dict(verdict=m.verdict, phash_distance=m.phash_distance, dhash_distance=m.dhash_distance,
                             ncc=m.ncc, crop_ncc=m.crop_ncc, a_page=m.a_page + 1, b_page=m.b_page + 1,
                             a_caption=m.a_caption, b_caption=m.b_caption,
                             a_b64=_thumb_b64(m.a_path), b_b64=_thumb_b64(m.b_path)))
    html = tpl.render(
        a=result.a, b=result.b, s=result.summary(), css=CSS_COLORS, version=__version__,
        generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), timings=result.timings,
        runs=runs, sents=sents, vparams=v.params, semantic=result.semantic, sem_pairs=sem_pairs,
        images=result.images, figs=figs, semantic_note=result.semantic_note, images_note=result.images_note,
    )
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path


# --- annotated PDFs ----------------------------------------------------------

def _marks(result: ComparisonResult, side: str) -> list[tuple[int, tuple, str, str]]:
    """(page, bbox, layer, note) for every highlight on one side."""
    other = "B" if side == "a" else "A"
    marks: list[tuple[int, tuple, str, str]] = []
    for r in result.verbatim.runs:
        locs, olocs = (r.a_locs, r.b_locs) if side == "a" else (r.b_locs, r.a_locs)
        opage = olocs[0].page + 1 if olocs else "?"
        note = f"{r.verdict} run #{r.id}: {r.length} words, score {r.score}; matches {other} p{opage}"
        for loc in locs:
            for page, bbox in loc.rects:
                marks.append((page, bbox, r.verdict, note))
    for p in result.verbatim.sentence_pairs:
        rects = p.a_rects if side == "a" else p.b_rects
        opage = (p.b_page if side == "a" else p.a_page) + 1
        note = f"{p.verdict} sentence, score {p.score}; matches {other} p{opage}"
        for page, bbox in rects:
            marks.append((page, bbox, p.verdict, note))
    if result.semantic is not None:
        for p in result.semantic.pairs:
            rects = p.a_rects if side == "a" else p.b_rects
            opage = (p.b_page if side == "a" else p.a_page) + 1
            note = f"paraphrase ({p.level}), cosine {p.score}; matches {other} p{opage} [weaker evidence]"
            for page, bbox in rects:
                marks.append((page, bbox, "semantic", note))
    if result.images is not None:
        for m in result.images.matches:
            page, bbox = (m.a_page, m.a_bbox) if side == "a" else (m.b_page, m.b_bbox)
            opage = (m.b_page if side == "a" else m.a_page) + 1
            note = f"figure {m.verdict}: pHash {m.phash_distance}, NCC {m.ncc}; matches {other} p{opage}"
            marks.append((page, bbox, "figure", note))
    return marks


def annotate_pdfs(result: ComparisonResult, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    outputs: dict[str, Path] = {}
    for side, doc in (("a", result.a), ("b", result.b)):
        pdf = pymupdf.open(doc.path)
        if pdf.is_encrypted and not pdf.authenticate(""):
            pdf.close()
            continue
        for page_no, bbox, layer, note in _marks(result, side):
            if page_no >= pdf.page_count:
                continue
            rect = pymupdf.Rect(bbox)
            if rect.width < 1 or rect.height < 1:
                continue
            page = pdf[page_no]
            try:
                if layer == "figure":
                    annot = page.add_rect_annot(rect)
                    annot.set_border(width=2)
                    annot.set_colors(stroke=COLORS["figure"])
                else:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=COLORS[layer])
                annot.set_info(title="pdfsim", content=note)
                annot.update()
            except Exception:
                continue
        out = out_dir / f"{Path(doc.path).stem}.annotated.pdf"
        pdf.save(str(out), garbage=1, deflate=True)
        pdf.close()
        outputs[side] = out
    return outputs
