"""Build the 24 + 6 slide deck from results/tables, results/figures and presentation/assets.

One message per slide: the title states the finding, the body shows the evidence.
Every number on a slide and in the speaker notes is read from results/tables at build
time; a missing table value renders as an em dash. Speaker notes are bound to slides
by the IDs in presentation/speaker_notes.md and may use {fact} placeholders.
"""

from __future__ import annotations

import json
import math
import re
import string
import sys
from pathlib import Path

import pandas as pd
import yaml
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visualization import style as S  # noqa: E402

ASSETS = ROOT / "presentation" / "assets"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
NOTES_MD = ROOT / "presentation" / "speaker_notes.md"
OUT = ROOT / "presentation" / "deep-clustering-evolution-selection-guide.pptx"
REPO_URL = "https://github.com/KQYaili/DeepClustering-Survey"

W, H = 13.333, 7.5
MX = 0.65                      # side margin
CW = W - 2 * MX                # content width
FONT = "Microsoft YaHei"
DASH = "—"

SLIDE_IDS = [
    "S01_cover", "S02_question", "S03_statistics_roots", "S04_pixel_distance", "S05_core_idea",
    "S06_paradigms", "S07_dec", "S08_idec", "S09_vade", "S10_contrastive", "S11_pretrained",
    "S12_setup", "S13_benchmark", "S14_geometry", "S15_representation_vs_clusterer",
    "S16_pixels_vs_dinov2", "S17_noise", "S18_sample_size", "S19_components", "S20_cost",
    "S21_stability", "S22_verdict", "S23_selection_guide", "S24_limits_and_close",
    "A01_benchmark_table", "A02_ablations", "A03_sensitivity", "A04_encoder_clusterer_table",
    "A05_stability_table", "A06_references",
]


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#"))


# =============================================================================== facts
def _f(value, digits=3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DASH
    return DASH if math.isnan(v) else f"{v:.{digits}f}"


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _pct(value):
    return DASH if value is None else f"{value:.0%}"


def _read_csv(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_tables(tables_dir: Path) -> dict:
    names = ["e01_summary", "e02_summary", "e02_fixed_epochs", "e03_noise", "e03_sensitivity_summary",
             "e04_summary", "benchmark_cost_frontiers", "coverage", "all_runs"]
    tables = {n: _read_csv(tables_dir, n) for n in names}
    sel_path = tables_dir / "selection.json"
    try:
        tables["selection"] = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    except json.JSONDecodeError:
        tables["selection"] = {}
    return tables


def _lookup(df: pd.DataFrame, column: str, **conds):
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in conds.items():
        if key not in df:
            return None
        if isinstance(value, float):
            mask &= pd.to_numeric(df[key], errors="coerce").sub(value).abs() < 1e-9
        else:
            mask &= df[key].astype(str).eq(str(value))
    rows = df[mask]
    return _num(rows.iloc[0][column]) if len(rows) and column in rows else None


def compute_facts(t: dict) -> dict:
    """Every number quoted in a title, label or speaker note."""
    facts: dict = {}
    e01 = t["e01_summary"]
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        for m in ("kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"):
            facts[f"ari_{ds}_{m}"] = _lookup(e01, "ari_mean", dataset=ds, method=m)
            facts[f"sd_{ds}_{m}"] = _lookup(e01, "ari_std", dataset=ds, method=m)
        rows = e01[e01.dataset.eq(ds)] if not e01.empty else e01
        if len(rows):
            best = rows.loc[rows.ari_mean.idxmax()]
            facts[f"best_{ds}"] = S.LABEL.get(best.method, best.method)
            facts[f"best_{ds}_ari"] = _num(best.ari_mean)
            facts[f"max_scratch_{ds}"] = _num(rows.ari_mean.max())
            facts[f"min_scratch_{ds}"] = _num(rows.ari_mean.min())
    # encoder x clusterer
    e04 = t["e04_summary"]
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        for enc in ("random_cnn", "ae", "simclr", "dinov2", "clip"):
            for cl in ("kmeans", "gmm", "dec_head"):
                facts[f"e04_{ds}_{enc}_{cl}"] = _lookup(e04, "ari_mean", dataset=ds, encoder=enc, clusterer=cl)
        km = [facts[f"e04_{ds}_{e}_kmeans"] for e in ("random_cnn", "ae", "simclr", "dinov2", "clip")]
        km = [v for v in km if v is not None]
        facts[f"encoder_range_{ds}"] = (max(km) - min(km)) if len(km) >= 2 else None
        spreads = []
        for enc in ("random_cnn", "ae", "simclr", "dinov2", "clip"):
            vals = [facts[f"e04_{ds}_{enc}_{c}"] for c in ("kmeans", "gmm", "dec_head")]
            vals = [v for v in vals if v is not None]
            if len(vals) == 3:
                spreads.append(max(vals) - min(vals))
        facts[f"clusterer_range_{ds}"] = (sum(spreads) / len(spreads)) if spreads else None
        facts[f"clusterer_range_max_{ds}"] = max(spreads) if spreads else None
        heads = [facts[f"e04_{ds}_{e}_{c}"] for e in ("dinov2", "clip") for c in ("gmm", "dec_head")]
        kms = [facts[f"e04_{ds}_{e}_kmeans"] for e in ("dinov2", "clip")]
        facts[f"pretrained_kmeans_best_{ds}"] = (
            all(h is not None for h in heads) and all(k is not None for k in kms)
            and max(heads[:2]) <= kms[0] + 1e-9 and max(heads[2:]) <= kms[1] + 1e-9)
        best_pre = [v for v in kms if v is not None]
        facts[f"pretrained_best_{ds}"] = max(best_pre) if best_pre else None
        enc_km = {e: facts[f"e04_{ds}_{e}_kmeans"] for e in ("random_cnn", "ae", "simclr", "dinov2", "clip")
                  if facts[f"e04_{ds}_{e}_kmeans"] is not None}
        if enc_km:
            top = max(enc_km, key=enc_km.get)
            facts[f"best_enc_{ds}"] = S.ENCODER_LABEL[top]
            facts[f"best_enc_{ds}_ari"] = enc_km[top]
    # Second-stage clusterers on frozen pretrained features, relative to K-means on the same features
    gmm_d, head_d = [], []
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        for enc in ("dinov2", "clip"):
            km = facts.get(f"e04_{ds}_{enc}_kmeans")
            g, h = facts.get(f"e04_{ds}_{enc}_gmm"), facts.get(f"e04_{ds}_{enc}_dec_head")
            if km is not None and g is not None:
                gmm_d.append(g - km)
            if km is not None and h is not None:
                head_d.append(h - km)
    facts["pre_gmm_delta_mean"] = sum(gmm_d) / len(gmm_d) if gmm_d else None
    facts["pre_gmm_delta_max"] = max(gmm_d) if gmm_d else None
    facts["pre_gmm_delta_absmax"] = max(abs(v) for v in gmm_d) if gmm_d else None
    facts["pre_head_delta_mean"] = sum(head_d) / len(head_d) if head_d else None
    facts["pre_head_delta_max"] = max(head_d) if head_d else None
    facts["pre_pairs"] = len(gmm_d)
    facts["pre_gmm_drop"] = -facts["pre_gmm_delta_mean"] if gmm_d else None
    facts["pre_head_drop"] = -facts["pre_head_delta_mean"] if head_d else None
    ranges = [facts[f"encoder_range_{d}"] for d in ("mnist", "fashion_mnist", "cifar10")]
    facts["encoder_range_max"] = max([r for r in ranges if r is not None], default=None)
    cranges = [facts[f"clusterer_range_{d}"] for d in ("mnist", "fashion_mnist", "cifar10")]
    facts["clusterer_range_mean_max"] = max([r for r in cranges if r is not None], default=None)
    # noise
    noise = t["e03_noise"]
    if not noise.empty:
        fm = noise[noise.dataset.eq("fashion_mnist")]
        top = float(fm.noise_std.max())
        facts["noise_max"] = top
        for m in ("kmeans", "pca_kmeans", "ae_kmeans", "dec", "idec", "vade", "gmm", "simclr_kmeans", "cc"):
            c0 = _lookup(fm, "ari_mean", method=m, noise_std=0.0)
            c1 = _lookup(fm, "ari_mean", method=m, noise_std=top)
            facts[f"noise_clean_{m}"], facts[f"noise_hi_{m}"] = c0, c1
            facts[f"keep_{m}"] = (c1 / c0) if c0 and c1 is not None else None
        facts["gmm_clusters_at_01"] = _lookup(fm, "n_nonempty_mean", method="gmm", noise_std=0.1)
    # sample size and other sensitivity
    sens = t["e03_sensitivity_summary"]
    for m in ("kmeans", "ae_kmeans", "dec", "idec", "cc"):
        for frac in (0.01, 0.05, 0.1, 0.25, 1.0):
            facts[f"sf_{m}_{frac}"] = _lookup(sens, "ari_mean", factor="sample_frac", value=frac, method=m)
        for k in (5, 8, 10, 12, 15, 20):
            facts[f"k_{m}_{k}"] = _lookup(sens, "ari_mean", factor="k", value=float(k), method=m)
    facts["n_train_1pct"] = _lookup(sens, "n_train_mean", factor="sample_frac", value=0.01, method="kmeans")
    # components and lambda
    e02 = t["e02_summary"]
    for key, factor, value in (("full", "freeze_encoder", "False"), ("frozen", "freeze_encoder", "True"),
                               ("random_init", "center_init", "random"), ("no_pretrain", "pretrain", "False")):
        facts[f"comp_{key}"] = _lookup(e02, "ari_mean", factor=factor, value=value)
        facts[f"comp_{key}_sd"] = _lookup(e02, "ari_std", factor=factor, value=value)
    for lam in (0.0, 0.01, 0.1, 1.0, 10.0):
        facts[f"lam_es_{lam}"] = _lookup(e02, "ari_mean", factor="lambda_recon", value=str(lam))
        facts[f"lam_es_ep_{lam}"] = _lookup(e02, "epochs_ran_mean", factor="lambda_recon", value=str(lam))
        facts[f"lam_fx_{lam}"] = _lookup(t["e02_fixed_epochs"], "ari_mean", lambda_recon=lam)
        facts[f"lam_fx_sd_{lam}"] = _lookup(t["e02_fixed_epochs"], "ari_std", lambda_recon=lam)
    runs = t["all_runs"]
    if not runs.empty:
        fx = runs[runs.experiment.eq("E02_fixed")].pivot_table(index="seed", columns="lambda_recon", values="ari")
        if 0.0 in fx and len(fx):
            for lam in (1.0, 10.0):
                if lam in fx:
                    diff = (fx[lam] - fx[0.0]).dropna()
                    facts[f"lam_fx_gap_{lam:g}"] = float(-diff.mean()) if len(diff) else None
                    facts[f"lam_fx_all_lower_{lam:g}"] = bool(len(diff) and (diff < 0).all())
                    facts[f"lam_fx_seeds"] = int(len(diff))
    for d in (2, 10, 32, 64, 128):
        facts[f"latent_{d}"] = _lookup(e02, "ari_mean", factor="latent_dim", value=str(d))
    # cost
    cost = t["benchmark_cost_frontiers"]
    if not cost.empty:
        fm = cost[cost.dataset.eq("fashion_mnist")]
        for m in ("pca_kmeans", "kmeans", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc", "gmm"):
            facts[f"time_{m}"] = _lookup(fm, "time_train_e2e_mean_sec", method=m)
            facts[f"cost_ari_{m}"] = _lookup(fm, "ari_mean", method=m)
        if facts.get("time_dec") and facts.get("time_pca_kmeans"):
            facts["dec_time_ratio"] = facts["time_dec"] / facts["time_pca_kmeans"]
            facts["dec_ari_gain"] = facts["cost_ari_dec"] - facts["cost_ari_pca_kmeans"]
        front = fm[fm.front_ari_time_e2e.astype(str).str.lower().eq("true")] if "front_ari_time_e2e" in fm else fm.iloc[0:0]
        facts["front_methods"] = list(front.sort_values("time_train_e2e_mean_sec").method)
    # stability
    sel = t.get("selection") or {}
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        entry = sel.get(ds) or {}
        choice = entry.get("choice")
        facts[f"stable_{ds}"] = S.LABEL.get(choice, choice) if choice else None
        ev = (entry.get("evidence") or {}).get(choice or "", {})
        facts[f"stable_{ds}_jac"] = _num(ev.get("macro_cluster_jaccard"))
        facts[f"stable_{ds}_ari"] = facts.get(f"ari_{ds}_{choice}") if choice else None
        facts[f"gmm_noisy_clusters_{ds}"] = _num(((entry.get("evidence") or {}).get("gmm") or {}).get("noisy_nonempty"))
    # runs
    cov = t["coverage"]
    if not cov.empty:
        ok = cov[cov.status.eq("ok")]
        facts["n_records"] = int(len(ok))
        facts["n_fits"] = int((~ok.experiment.eq("E03_noise")).sum())
        facts["n_noise"] = int(ok.experiment.eq("E03_noise").sum())
    pix = ASSETS / "pixel_distance.yaml"
    if pix.exists():
        info = yaml.safe_load(pix.read_text(encoding="utf-8"))
        facts.update({f"pix_{k}": v for k, v in info.items()})
    return facts


class Facts(dict):
    """Formats facts for text: f('ari_mnist_dec') -> '0.669', missing -> em dash."""

    def f(self, key, digits=3):
        return _f(self.get(key), digits)

    def p(self, key):
        return _pct(self.get(key))


class NotesFormatter(string.Formatter):
    def __init__(self, facts: Facts):
        super().__init__()
        self.facts = facts

    def get_field(self, field_name, args, kwargs):
        # Fact names may contain dots (e.g. sf_dec_0.01); never treat them as attribute access.
        return self.facts.get(field_name), field_name

    def format_field(self, value, spec):
        if value is None:
            return DASH
        if isinstance(value, float) and math.isnan(value):
            return DASH
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return str(value)


# =============================================================================== drawing
class Deck:
    def __init__(self, facts: Facts, figures_dir: Path = FIGS):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.facts = facts
        self.figs = figures_dir
        self.ids: list[str] = []
        self.page = 0

    # ---- slide chrome
    def slide(self, sid: str, kicker: str | None = None, title: str | None = None, lede: str | None = None,
              dark: bool = False, number: bool = True, source: str | None = None):
        slide = self.prs.slides.add_slide(self.blank)
        self.ids.append(sid)
        self.page += 1
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = rgb(S.NIGHT if dark else S.SURFACE)
        if kicker:
            self.text(slide, MX, 0.40, CW, 0.30, kicker, size=12, color=S.SECONDARY, spacing=1.5)
        if title:
            self.text(slide, MX, 0.66, CW, 0.62, title, size=27, bold=True, color=S.INK, line=1.0)
        if lede:
            self.text(slide, MX, 1.30, CW, 0.45, lede, size=15, color=S.SECONDARY, line=1.2)
        if number and not dark:
            label = str(self.page) if sid.startswith("S") else sid[:3]
            self.text(slide, W - MX - 1.0, 7.02, 1.0, 0.28, label, size=10.5, color=S.MUTED, align="r")
        if source:
            self.text(slide, MX, 7.02, CW - 1.2, 0.28, source, size=10.5, color=S.MUTED)
        return slide

    # ---- primitives
    def text(self, slide, x, y, w, h, content, size=16, bold=False, color=S.BODY, align="l", anchor="t",
             line=1.15, spacing=None, italic=False, font=FONT, after=0):
        """Add a text box. `content` is a string (\\n separates paragraphs) or a list of paragraphs,
        each a string or a list of (text, {overrides}) runs. **bold** marks bold runs."""
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
        paragraphs = content.split("\n") if isinstance(content, str) else content
        for i, para in enumerate(paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
            p.line_spacing = line
            if after:
                p.space_after = Pt(after)
            runs = self._runs(para)
            for text, over in runs:
                run = p.add_run()
                run.text = text
                self._font(run, size=over.get("size", size), bold=over.get("bold", bold),
                           color=over.get("color", color), italic=over.get("italic", italic),
                           spacing=over.get("spacing", spacing), font=over.get("font", font),
                           baseline=-25000 if over.get("sub") else 30000 if over.get("sup") else None)
        return box

    @staticmethod
    def _runs(para):
        if isinstance(para, list):
            return [(r, {}) if isinstance(r, str) else r for r in para]
        parts = re.split(r"(\*\*[^*]+\*\*)", para)
        return [(p[2:-2], {"bold": True}) if p.startswith("**") else (p, {}) for p in parts if p]

    @staticmethod
    def _font(run, size, bold, color, italic=False, spacing=None, font=FONT, baseline=None):
        f = run.font
        f.size = Pt(size)
        f.bold = bold
        f.italic = italic
        f.color.rgb = rgb(color)
        rpr = run._r.get_or_add_rPr()
        for tag in ("a:latin", "a:ea", "a:cs"):
            el = rpr.find(qn(tag))
            if el is None:
                el = etree.SubElement(rpr, qn(tag))
            el.set("typeface", font)
        if spacing:
            rpr.set("spc", str(int(spacing * 100)))
        if baseline:
            rpr.set("baseline", str(baseline))

    @staticmethod
    def _plain(shape):
        style = shape._element.find(qn("p:style"))
        if style is not None:
            shape._element.remove(style)

    def rect(self, slide, x, y, w, h, fill=S.PANEL, line=None, radius=0.0, line_w=1.0):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        self._plain(shape)
        if radius:
            shape.adjustments[0] = radius
        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(fill)
        else:
            shape.fill.background()
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(line_w)
        else:
            shape.line.fill.background()
        return shape

    def oval(self, slide, x, y, d, fill, line=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        self._plain(shape)
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    def box(self, slide, x, y, w, h, label, fill=S.PANEL, color=S.INK, size=15, bold=True, line=None,
            radius=0.08, sub=None, sub_size=11.5, sub_color=None):
        shape = self.rect(slide, x, y, w, h, fill=fill, line=line, radius=radius)
        runs = [(label, {})] if isinstance(label, str) else [(r, {}) if isinstance(r, str) else r for r in label]
        paras = [[(text, {"bold": bold, "size": size, "color": color, **over}) for text, over in runs]]
        if sub:
            paras.append([(sub, {"bold": False, "size": sub_size, "color": sub_color or S.SECONDARY})])
        self.text(slide, x + 0.08, y, w - 0.16, h, paras, align="c", anchor="m", line=1.1)
        return shape

    def line(self, slide, x1, y1, x2, y2, color=S.MUTED, width=1.5, arrow=False, dash=False):
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        self._plain(conn)
        conn.line.color.rgb = rgb(color)
        conn.line.width = Pt(width)
        ln = conn.line._get_or_add_ln()
        if dash:
            prst = etree.SubElement(ln, qn("a:prstDash"))
            prst.set("val", "sysDash")
        if arrow:
            tail = etree.SubElement(ln, qn("a:tailEnd"))
            tail.set("type", "triangle")
            tail.set("w", "med")
            tail.set("len", "med")
        return conn

    def arrow(self, slide, x1, y1, x2, y2, color=S.SECONDARY, width=1.75):
        return self.line(slide, x1, y1, x2, y2, color=color, width=width, arrow=True)

    def path(self, slide, points, color=S.SECONDARY, width=1.75, arrow=True):
        for i in range(len(points) - 1):
            last = i == len(points) - 2
            self.line(slide, *points[i], *points[i + 1], color=color, width=width, arrow=arrow and last)

    def image(self, slide, path: Path, x, y, w=None, h=None):
        if not Path(path).exists():
            ph = self.rect(slide, x, y, w or 4, h or 3, fill=S.PANEL)
            self.text(slide, x, y, w or 4, h or 3, f"缺少 {Path(path).name}", size=12, color=S.MUTED,
                      align="c", anchor="m")
            return ph
        kwargs = {}
        if w is not None:
            kwargs["width"] = Inches(w)
        if h is not None:
            kwargs["height"] = Inches(h)
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kwargs)

    def equation(self, slide, name, x, y, scale=1.0, align="l", max_w=None):
        path = ASSETS / f"eq_{name}.png"
        from PIL import Image

        with Image.open(path) as im:
            w_in, h_in = im.size[0] / 600 * scale, im.size[1] / 600 * scale
        if max_w and w_in > max_w:
            h_in *= max_w / w_in
            w_in = max_w
        if align == "c":
            x = x - w_in / 2
        elif align == "r":
            x = x - w_in
        self.image(slide, path, x, y, w=w_in, h=h_in)
        return w_in, h_in

    def figure(self, slide, name, x, y, w):
        return self.image(slide, self.figs / f"{name}.png", x, y, w=w)

    def dot(self, slide, x, y, color, d=0.16):
        return self.oval(slide, x, y, d, color)

    def stat(self, slide, x, y, w, value, label, color=S.INK, value_size=40, label_size=14):
        self.text(slide, x, y, w, 0.75, value, size=value_size, bold=True, color=color, line=1.0)
        self.text(slide, x, y + value_size / 72 * 1.18, w, 0.9, label, size=label_size, color=S.SECONDARY,
                  line=1.25)

    def rule(self, slide, x, y, w, color=S.INK, width=2.0):
        return self.line(slide, x, y, x + w, y, color=color, width=width)

    def table(self, slide, x, y, w, rows, col_w=None, size=11.5, header_size=11.5, row_h=0.36,
              align=None, bold_cells=None, header_fill=None):
        n_rows, n_cols = len(rows), len(rows[0])
        shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(row_h * n_rows))
        table = shape.table
        tbl_pr = shape._element.graphic.graphicData.tbl.tblPr
        for attr in ("firstRow", "bandRow"):
            tbl_pr.set(attr, "0")
        style = tbl_pr.find(qn("a:tableStyleId"))
        if style is not None:
            tbl_pr.remove(style)
        col_w = col_w or [w / n_cols] * n_cols
        for j, cw in enumerate(col_w):
            table.columns[j].width = Inches(cw)
        bold_cells = bold_cells or set()
        for i, row in enumerate(rows):
            table.rows[i].height = Inches(row_h)
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.02)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(header_fill if (i == 0 and header_fill) else S.SURFACE)
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[
                    (align[j] if align else ("l" if j == 0 else "c"))]
                run = p.add_run()
                run.text = str(value)
                head = i == 0
                self._font(run, size=header_size if head else size, bold=head or (i, j) in bold_cells,
                           color=S.SECONDARY if head else S.INK)
                self._cell_border(cell, bottom=S.HAIRLINE if i < n_rows - 1 else S.MUTED,
                                  top=S.MUTED if i == 0 else None, bottom_w=0.75 if i else 1.0)
        return shape

    @staticmethod
    def _cell_border(cell, bottom=None, top=None, bottom_w=0.75):
        tc_pr = cell._tc.get_or_add_tcPr()
        for tag, color, width in (("a:lnL", None, 0), ("a:lnR", None, 0), ("a:lnT", top, 1.0),
                                  ("a:lnB", bottom, bottom_w)):
            el = tc_pr.find(qn(tag))
            if el is not None:
                tc_pr.remove(el)
            ln = etree.SubElement(tc_pr, qn(tag))
            if color:
                ln.set("w", str(int(Pt(width))))
                solid = etree.SubElement(ln, qn("a:solidFill"))
                clr = etree.SubElement(solid, qn("a:srgbClr"))
                clr.set("val", color.lstrip("#"))
            else:
                ln.set("w", "0")
                etree.SubElement(ln, qn("a:noFill"))
        # schema order: borders must precede the fill
        fill = tc_pr.find(qn("a:solidFill"))
        if fill is not None:
            tc_pr.remove(fill)
            tc_pr.append(fill)


# =============================================================================== slides
SECTION = {"intro": "引言", "stat": "统计基础", "method": "方法谱系", "exp": "实验证据", "end": "结论与选型",
           "app": "附录"}


def s01_cover(d: Deck):
    s = d.slide("S01_cover", dark=True, number=False)
    mosaic = ASSETS / "cover_mosaic.png"
    if mosaic.exists():
        from PIL import Image

        with Image.open(mosaic) as im:
            ratio = im.size[0] / im.size[1]
        mw = H * ratio
        d.image(s, mosaic, W - mw, 0, w=mw, h=H)
    d.text(s, MX + 0.05, 1.55, 6.2, 0.4, "第九组 · 课程综述汇报", size=13, color="#A1A1AA", spacing=2)
    d.text(s, MX, 2.05, 6.4, 1.0, "深度聚类", size=54, bold=True, color="#FFFFFF", line=1.0)
    d.text(s, MX, 3.12, 6.4, 0.6, "范式演进、对照实验与方法选择", size=26, color="#E4E4E7", line=1.0)
    d.rule(s, MX + 0.03, 4.02, 0.6, color=S.PRETRAINED, width=2.5)
    d.text(s, MX, 4.25, 5.4, 1.0, "面对一批没有标签的图像，\n先改表示，还是先换聚类器？", size=18, color="#D4D4D8",
           line=1.35)
    d.text(s, MX, 6.62, 5.6, 0.5, "MNIST · Fashion-MNIST · CIFAR-10　|　11 种方法 · 3 个随机种子", size=11.5,
           color="#8E8E93")
    d.text(s, W - 5.6, 7.08, 5.35, 0.3, "右图：CIFAR-10 测试图按 DINOv2 表示的 t-SNE 位置排列", size=9.5,
           color="#8E8E93", align="r")


def s02_question(d: Deck):
    s = d.slide("S02_question", SECTION["intro"], "先改表示，还是先换聚类器？",
                lede="同一批图像，可以换一个更好的“坐标系”，也可以换一个更复杂的分组规则。下面三个命题都能用实验检验。")
    items = [
        ("表示比聚类器更重要", "固定聚类器换编码器，与固定编码器换聚类器，比较哪一边带来的 ARI 变化更大。",
         "5 种编码器 × 3 种聚类器 × 3 个数据集"),
        ("联合优化有条件地有效", "边学表示边聚类（DEC、IDEC）在干净、充足的数据上占优，但在噪声和小样本下未必。",
         "测试噪声、训练样本量、组件消融"),
        ("有了预训练表示，简单聚类器就够了", "在冻结的 DINOv2、CLIP 特征上，K-means 不弱于更复杂的聚类头。",
         "K-means vs GMM vs DEC 头"),
    ]
    col_w, gap, y0 = 3.75, 0.39, 2.2
    for i, (claim, how, design) in enumerate(items):
        x = MX + i * (col_w + gap)
        d.rule(s, x, y0, col_w, color=S.INK, width=2)
        d.text(s, x, y0 + 0.2, 1.0, 0.8, str(i + 1), size=40, bold=True, color="#D4D4D8", line=1.0)
        d.text(s, x, y0 + 0.98, col_w, 0.85, claim, size=20, bold=True, color=S.INK, line=1.15, anchor="b")
        d.text(s, x, y0 + 2.0, col_w, 1.3, how, size=15, color=S.BODY, line=1.4)
        d.text(s, x, y0 + 3.3, col_w, 0.6, [[("检验：", {"color": S.SECONDARY}), (design, {"bold": True})]],
               size=13.5, color=S.INK, line=1.3)
    steps = ["统计基础", "方法谱系", "实验证据", "结论与选型"]
    y = 6.45
    for i, step in enumerate(steps):
        x = MX + i * 3.0
        d.dot(s, x, y + 0.07, S.INK if i == 0 else "#D4D4D8", d=0.13)
        d.text(s, x + 0.25, y, 2.5, 0.3, step, size=12.5, color=S.SECONDARY)
        if i < len(steps) - 1:
            d.line(s, x + 1.35, y + 0.135, x + 2.85, y + 0.135, color=S.HAIRLINE, width=1.25)


def s03_statistics(d: Deck):
    s = d.slide("S03_statistics_roots", SECTION["stat"], "深度聚类建立在多元统计的几块基石上",
                source="王学民《应用多元统计分析》第六版，第 2、3、6、7、8 章")
    cols = [
        ("第 2 章", "距离与协方差", "隐空间中的距离", "相似性在学到的坐标里度量"),
        ("第 3 章", "多元正态分布", "高斯混合先验", "VaDE 让每个簇是一团正态云"),
        ("第 6 章", "聚类分析", "K-means 与簇中心", "DEC 用簇中心定义软分配"),
        ("第 7 章", "主成分分析", "自编码器", "非线性的降维与重构"),
        ("第 8 章", "因子分析", "潜变量生成模型", "VAE / VaDE：潜变量生成观测"),
    ]
    cw, gap, x0 = 2.2, 0.258, MX
    top, mid = 1.85, 4.2
    for i, (chap, concept, deep, note) in enumerate(cols):
        x = x0 + i * (cw + gap)
        d.rect(s, x, top, cw, 1.3, fill=S.PANEL, radius=0.06)
        d.text(s, x + 0.2, top + 0.2, cw - 0.4, 0.3, chap, size=12, color=S.SECONDARY)
        d.text(s, x + 0.2, top + 0.55, cw - 0.4, 0.6, concept, size=19, bold=True, color=S.INK)
        d.arrow(s, x + cw / 2, top + 1.42, x + cw / 2, mid - 0.12, color=S.MUTED, width=1.5)
        d.rule(s, x, mid, cw, color=S.DEEP, width=2.5)
        d.text(s, x, mid + 0.2, cw, 0.5, deep, size=18, bold=True, color=S.INK)
        d.text(s, x, mid + 0.67, cw, 0.8, note, size=13.5, color=S.SECONDARY, line=1.3)
    d.text(s, MX, 6.2, CW, 0.5, [[("教材里的坐标是给定的或线性的；", {}),
                                    ("深度聚类把坐标本身变成了可学习的。", {"bold": True, "color": S.INK})]],
           size=17, color=S.BODY)


def s04_pixel_distance(d: Deck):
    f = d.facts
    s = d.slide("S04_pixel_distance", SECTION["stat"], "K-means 在像素空间里量错了距离",
                source=f"Fashion-MNIST 训练集，像素归一化到 [0, 1]；随机抽取 {f.get('pix_n_shirts', 1000):,} 张 T 恤统计")
    d.text(s, MX, 1.75, 5.0, 0.3, "K-means 的目标", size=13, color=S.SECONDARY)
    d.equation(s, "kmeans", MX, 2.12, scale=0.92)
    d.text(s, MX, 3.15, 4.9, 1.6,
           "把每个点分给最近的中心。它默认：**欧氏距离近，就是同一类。**\n"
           "图像被拉成 784 维像素向量后，这个距离主要反映位置、亮度和背景，而不是“这是什么”。",
           size=16, color=S.BODY, line=1.45, after=8)
    # image triplet
    x0, y0, side = 6.15, 1.85, 1.75
    names = [("pixel_anchor", "T 恤 A"), ("pixel_moved", f"A 右移 {f.get('pix_shift', 4)} 像素"),
             ("pixel_other", f"另一类：{f.get('pix_other_class', '衬衫')} B")]
    xs = [x0, x0 + 2.25, x0 + 4.5]
    for (name, label), x in zip(names, xs):
        d.image(s, ASSETS / f"{name}.png", x, y0, w=side, h=side)
        d.text(s, x - 0.2, y0 + side + 0.1, side + 0.4, 0.35, label, size=13.5, color=S.BODY, align="c")
    # dimension lines: A to A' and A to B
    def dim(xa, xb, y, label, value):
        d.line(s, xa, y, xb, y, color=S.BODY, width=1.25)
        d.line(s, xa, y - 0.08, xa, y + 0.08, color=S.BODY, width=1.25)
        d.line(s, xb, y - 0.08, xb, y + 0.08, color=S.BODY, width=1.25)
        d.text(s, (xa + xb) / 2 - 1.2, y - 0.4, 2.4, 0.34, [[(label, {"color": S.SECONDARY}), (value, {"bold": True})]],
               size=15, color=S.INK, align="c", anchor="b")
    yb = y0 + side + 0.95
    dim(xs[0] + side / 2, xs[1] + side / 2, yb, "像素距离 ", _f(f.get("pix_d_moved"), 1))
    dim(xs[0] + side / 2, xs[2] + side / 2, yb + 0.68, "像素距离 ", _f(f.get("pix_d_other"), 1))
    frac = f.get("pix_fraction_farther")
    d.stat(s, x0, 5.55, 6.5, DASH if frac is None else f"{frac:.1%}",
           f"随机抽取的 T 恤右移 {f.get('pix_shift', 4)} 像素后，离原图比离最近的“非 T 恤”图像还远",
           value_size=34, label_size=14)


def s05_core_idea(d: Deck):
    s = d.slide("S05_core_idea", SECTION["stat"], "深度聚类：先学一个让距离有意义的表示")
    y, h = 2.05, 1.25
    xs = [MX, 3.25, 5.85, 8.45, 10.85]
    labels = [("图像 x", "784 / 3072 维像素"), (["编码器 f", ("θ", {"sub": True})], "神经网络"), ("表示 z", "10 维左右"),
              ("聚类", "K-means、软分配、混合模型"), ("簇 ĉ", "每张图的类别编号")]
    fills = [S.PANEL, "#E3EEFB", S.PANEL, S.PANEL, S.PANEL]
    widths = [1.95, 1.95, 1.95, 1.95, 1.83]
    for (label, sub), x, fill, w in zip(labels, xs, fills, widths):
        d.box(s, x, y, w, h, label, fill=fill, size=17, sub=sub, sub_size=12)
    for i in range(4):
        d.arrow(s, xs[i] + widths[i] + 0.05, y + h / 2, xs[i + 1] - 0.06, y + h / 2, color=S.SECONDARY)
    d.equation(s, "pipeline", W / 2, 3.7, scale=0.9, align="c")
    qy = 4.65
    qs = [("没有标签，编码器靠什么信号训练？", "重构输入 · 用自己的预测自训练 · 同一张图的两个增强视图 · 外部数据预训练"),
          ("编码器和聚类器怎么配合？", "先学后分 · 边学边分 · 交替迭代 · 用潜变量解释生成过程")]
    for i, (q, a) in enumerate(qs):
        x = MX + i * 6.1
        d.text(s, x, qy, 0.5, 0.5, "①②"[i], size=24, bold=True, color=S.DEEP)
        d.text(s, x + 0.55, qy + 0.02, 5.3, 0.5, q, size=18, bold=True, color=S.INK)
        d.text(s, x + 0.55, qy + 0.55, 5.3, 1.0, a, size=14.5, color=S.SECONDARY, line=1.4)
    d.text(s, MX, 6.35, CW, 0.4, "后面所有方法，都是这两个问题的不同答案。", size=15, color=S.BODY)


def _mini_node(d, s, x, y, w, h, label, fill, color=S.INK, size=12.5):
    d.box(s, x, y, w, h, label, fill=fill, color=color, size=size, radius=0.18)


def s06_paradigms(d: Deck):
    s = d.slide("S06_paradigms", SECTION["method"], "表示与聚类的四种配合方式长期并存",
                lede="按 Zhou 等（ACM Computing Surveys, 2025）的交互方式分类：年份不决定方法属于哪一类。",
                source="实心圆点：本项目实验中运行的方法；空心圆点：仅文献综述。")
    cols = [
        ("多阶段", "先学特征，再固定特征分组", [("AE + K-means", S.DEEP, True), ("SimCLR + K-means", S.CONTRASTIVE, True)]),
        ("联合优化", "同一个目标同时更新表示和簇", [("DEC、IDEC", S.DEEP, True), ("CC", S.CONTRASTIVE, True)]),
        ("生成式", "用潜在簇解释图像如何生成", [("VaDE", S.GENERATIVE, True)]),
        ("迭代式", "本轮簇编号监督下一轮特征", [("DeepCluster", S.MUTED, False), ("SwAV", S.MUTED, False)]),
    ]
    cw, gap, y0 = 2.85, 0.21, 1.92
    rep, clu = "#E3EEFB", "#ECECEE"
    for i, (name, desc, methods) in enumerate(cols):
        x = MX + i * (cw + gap)
        d.rect(s, x, y0, cw, 2.0, fill=S.PANEL, radius=0.05)
        bx, by, bw, bh = x + 0.35, y0 + 0.62, 0.95, 0.5
        if i == 0:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + bh / 2, bx + 1.21, by + bh / 2)
        elif i == 1:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + bh / 2 - 0.1, bx + 1.21, by + bh / 2 - 0.1)
            d.arrow(s, bx + 1.21, by + bh / 2 + 0.1, bx + bw + 0.04, by + bh / 2 + 0.1)
            d.text(s, x, by + bh + 0.12, cw, 0.3, "共享一个损失", size=11, color=S.SECONDARY, align="c")
        elif i == 2:
            for j, lab in enumerate(["c", "z", "x"]):
                d.oval(s, bx + 0.05 + j * 0.8, by, 0.5, rep if lab == "z" else clu)
                d.text(s, bx + 0.05 + j * 0.8, by, 0.5, 0.5, lab, size=14, bold=True, color=S.INK, align="c",
                       anchor="m", italic=True, font="Cambria Math")
                if j < 2:
                    d.arrow(s, bx + 0.58 + j * 0.8, by + 0.25, bx + 0.82 + j * 0.8, by + 0.25)
            d.text(s, x, by + bh + 0.12, cw, 0.3, "簇 → 隐变量 → 图像", size=11, color=S.SECONDARY, align="c")
        else:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + 0.14, bx + 1.21, by + 0.14)
            d.path(s, [(bx + 1.72, by + bh + 0.02), (bx + 1.72, by + bh + 0.25), (bx + 0.48, by + bh + 0.25),
                       (bx + 0.48, by + bh + 0.04)])
            d.text(s, x, by + bh + 0.3, cw, 0.3, "伪标签", size=11, color=S.SECONDARY, align="c")
        d.text(s, x, y0 + 2.2, cw, 0.45, name, size=20, bold=True, color=S.INK)
        d.text(s, x, y0 + 2.68, cw, 0.6, desc, size=14, color=S.BODY, line=1.3)
        for j, (mname, color, ran) in enumerate(methods):
            yy = y0 + 3.3 + j * 0.38
            if ran:
                d.dot(s, x, yy + 0.08, color, d=0.15)
            else:
                d.oval(s, x, yy + 0.08, 0.15, S.SURFACE, line=S.MUTED)
            d.text(s, x + 0.27, yy, cw - 0.3, 0.35, mname, size=14, color=S.INK if ran else S.SECONDARY)
    by = 6.12
    d.rect(s, MX, by, CW, 0.66, fill="#FDF0EA", radius=0.12)
    d.dot(s, MX + 0.3, by + 0.255, S.PRETRAINED, d=0.15)
    d.text(s, MX + 0.6, by + 0.01, CW - 0.8, 0.64,
           [[("外部预训练：", {"bold": True, "color": S.INK}),
             ("表示来自课程数据之外（DINOv2、CLIP），再接简单聚类器；TAC、VLM4Cluster 进一步引入文本语义。", {})]],
           size=15, color=S.BODY, anchor="m")


def _flow_row(d, s, items, y, h=0.78, x0=None, gap=0.42, size=14):
    """items: list of (label, sub, fill, width)."""
    x = x0 if x0 is not None else MX
    centers = []
    for i, (label, sub, fill, w) in enumerate(items):
        d.box(s, x, y, w, h, label, fill=fill, size=size, sub=sub, sub_size=11)
        centers.append((x, x + w))
        if i < len(items) - 1:
            d.arrow(s, x + w + 0.04, y + h / 2, x + w + gap - 0.05, y + h / 2)
        x += w + gap
    return centers


def s07_dec(d: Deck):
    s = d.slide("S07_dec", SECTION["method"], "DEC：用自己的高置信预测训练自己",
                source="Xie, Girshick & Farhadi, Unsupervised Deep Embedding for Clustering Analysis, ICML 2016")
    steps = [("① 软分配", "Student-t 核：离簇中心越近，归属概率越高", "dec_q"),
             ("② 目标分布", "平方后按簇大小归一：放大高置信样本，防止大簇吞并小簇", "dec_p"),
             ("③ 自训练", "最小化 KL，同时更新编码器参数和簇中心", "dec_loss")]
    y = 1.72
    for title, desc, eq in steps:
        d.text(s, MX, y, 1.5, 0.4, title, size=17, bold=True, color=S.INK)
        d.text(s, MX + 1.55, y + 0.03, 5.6, 0.4, desc, size=14, color=S.SECONDARY)
        _, eh = d.equation(s, eq, MX + 1.55, y + 0.48, scale=0.78)
        y += 0.48 + eh + 0.42
    # diagram on the right
    x0, y0 = 8.25, 1.95
    d.box(s, x0, y0, 1.25, 0.62, "图像 x", fill=S.PANEL, size=14)
    d.box(s, x0 + 1.75, y0, 1.45, 0.62, "编码器", fill="#E3EEFB", size=14)
    d.box(s, x0 + 1.95, y0 + 1.25, 1.05, 0.62, "z", fill=S.PANEL, size=15)
    d.box(s, x0 + 1.55, y0 + 2.5, 1.85, 0.62, "软分配 Q", fill=S.PANEL, size=14)
    d.box(s, x0 + 1.55, y0 + 3.75, 1.85, 0.62, "目标分布 P", fill=S.PANEL, size=14)
    d.arrow(s, x0 + 1.29, y0 + 0.31, x0 + 1.71, y0 + 0.31)
    d.arrow(s, x0 + 2.475, y0 + 0.66, x0 + 2.475, y0 + 1.21)
    d.arrow(s, x0 + 2.475, y0 + 1.91, x0 + 2.475, y0 + 2.46)
    d.arrow(s, x0 + 2.475, y0 + 3.16, x0 + 2.475, y0 + 3.71)
    d.text(s, x0 + 2.6, y0 + 3.25, 1.6, 0.35, "平方、归一", size=11, color=S.SECONDARY)
    # feedback
    d.path(s, [(x0 + 3.44, y0 + 4.06), (x0 + 4.25, y0 + 4.06), (x0 + 4.25, y0 + 0.31), (x0 + 3.24, y0 + 0.31)],
           color=S.DEEP, width=2)
    d.text(s, x0 + 3.05, y0 + 1.25, 1.1, 0.8, "KL(P‖Q)\n反向更新", size=12.5, bold=True, color=S.DEEP, align="r")
    d.text(s, x0 - 0.05, y0 + 4.6, 4.4, 0.6, "训练全程不读标签；簇分配变化 < 0.1% 时早停。", size=12.5,
           color=S.SECONDARY)


def s08_idec(d: Deck):
    s = d.slide("S08_idec", SECTION["method"], "IDEC：保留解码器，防止表示被聚类目标拉坏",
                source="Guo et al., Improved Deep Embedded Clustering with Local Structure Preservation, IJCAI 2017")
    d.equation(s, "idec_loss", MX, 1.85, scale=0.95)
    d.text(s, MX, 2.75, 5.9, 2.6,
           "只用 KL 时，早期分错的样本会被不断强化，隐空间的局部结构可能被扭曲。\n"
           "**重构项要求 z 仍能还原输入**，λ 控制“压紧簇”与“保留信息”之间的权衡。",
           size=16, color=S.BODY, line=1.5, after=10)
    d.text(s, MX, 4.65, 5.9, 1.2,
           "实现说明：本项目的 λ 乘在每张图的平方重构误差（对像素求和）上；原文把 γ 乘在聚类项上、重构误差对像素取平均，"
           "两者的换算是 λ = 1/(γD)，D 为像素数。",
           size=12.5, color=S.SECONDARY, line=1.4)
    x0, y0 = 6.55, 1.95
    d.box(s, x0, y0 + 1.05, 1.15, 0.62, "图像 x", fill=S.PANEL, size=14)
    d.box(s, x0 + 1.55, y0 + 1.05, 1.25, 0.62, "编码器", fill="#E3EEFB", size=14)
    d.box(s, x0 + 3.2, y0 + 1.05, 0.85, 0.62, "z", fill=S.PANEL, size=15)
    d.box(s, x0 + 4.45, y0, 1.6, 0.62, "软分配 Q", fill=S.PANEL, size=13.5)
    d.box(s, x0 + 4.45, y0 + 2.1, 1.6, 0.62, ["解码器 g", ("φ", {"sub": True})], fill="#E3EEFB", size=13.5)
    d.arrow(s, x0 + 1.19, y0 + 1.36, x0 + 1.51, y0 + 1.36)
    d.arrow(s, x0 + 2.84, y0 + 1.36, x0 + 3.16, y0 + 1.36)
    d.path(s, [(x0 + 4.09, y0 + 1.2), (x0 + 4.25, y0 + 1.2), (x0 + 4.25, y0 + 0.31), (x0 + 4.41, y0 + 0.31)])
    d.path(s, [(x0 + 4.09, y0 + 1.52), (x0 + 4.25, y0 + 1.52), (x0 + 4.25, y0 + 2.41), (x0 + 4.41, y0 + 2.41)])
    d.text(s, x0 + 4.45, y0 + 0.7, 1.9, 0.35, "聚类损失 KL", size=12, color=S.DEEP, bold=True)
    d.text(s, x0 + 4.45, y0 + 2.8, 1.9, 0.35, "重构损失 × λ", size=12, color=S.DEEP, bold=True)
    d.text(s, x0, y0 + 3.6, 5.7, 0.9, "DEC 是 λ = 0 的特例：解码器只用于预训练，之后被丢弃。",
           size=14.5, color=S.BODY, line=1.35)


def s09_vade(d: Deck):
    s = d.slide("S09_vade", SECTION["method"], "VaDE：把高斯混合模型搬进隐空间",
                source="Jiang et al., Variational Deep Embedding, IJCAI 2017")
    d.text(s, MX, 1.75, 6, 0.3, "生成过程", size=13, color=S.SECONDARY)
    d.equation(s, "vade_gen", MX, 2.1, scale=0.85)
    d.text(s, MX, 2.75, 6, 0.3, "训练目标（证据下界 ELBO）", size=13, color=S.SECONDARY)
    d.equation(s, "vade_elbo", MX, 3.1, scale=0.85)
    # diagram
    y = 4.15
    nodes = [("c", "簇类别", S.PANEL), ("z", "隐变量", "#F3E6F1"), ("x", "图像", S.PANEL)]
    for i, (sym, lab, fill) in enumerate(nodes):
        x = MX + 0.2 + i * 2.1
        d.oval(s, x, y, 0.85, fill)
        d.text(s, x, y, 0.85, 0.85, sym, size=22, bold=True, color=S.INK, align="c", anchor="m", italic=True,
               font="Cambria Math")
        d.text(s, x - 0.3, y + 0.95, 1.45, 0.3, lab, size=12.5, color=S.SECONDARY, align="c")
        if i < 2:
            d.arrow(s, x + 0.9, y + 0.42, x + 2.05, y + 0.42, color=S.GENERATIVE, width=2)
    d.text(s, MX + 0.95, y - 0.45, 3.8, 0.3, [["生成：π → N(μ", ("c", {"sub": True}), ", σ", ("c", {"sub": True}),
                                               "²) → 解码器"]], size=11.5, color=S.GENERATIVE)
    d.path(s, [(MX + 4.62, y + 1.35), (MX + 4.62, y + 1.6), (MX + 2.62, y + 1.6), (MX + 2.62, y + 1.33)],
           color=S.SECONDARY)
    d.text(s, MX + 2.75, y + 1.62, 2.8, 0.3, "推断：编码器 q(z, c | x)", size=11.5, color=S.SECONDARY)
    # comparison on the right
    x0 = 7.55
    rows = [("与 K-means 相比", "每个簇有自己的方差和先验比例 π，是“概率云”而不是一个点。"),
            ("与像素空间的 GMM 相比", "先降到 10 维再建模，避开 784 维协方差难以估计的问题。"),
            ("与教材的联系", "第 3 章多元正态 + 第 8 章潜变量模型，用神经网络把两者连起来。")]
    for i, (head, body) in enumerate(rows):
        y0 = 1.85 + i * 1.5
        d.rule(s, x0, y0, 5.1, color=S.GENERATIVE if i == 0 else S.HAIRLINE, width=2 if i == 0 else 1)
        d.text(s, x0, y0 + 0.15, 5.1, 0.4, head, size=16, bold=True, color=S.INK)
        d.text(s, x0, y0 + 0.58, 5.1, 0.85, body, size=14.5, color=S.BODY, line=1.4)


def s10_contrastive(d: Deck):
    s = d.slide("S10_contrastive", SECTION["method"], "对比学习：同一张图的两个增强视图应当靠近",
                source="Chen et al., SimCLR, ICML 2020 · Li et al., Contrastive Clustering, AAAI 2021")
    x0, y0 = MX, 2.0
    d.box(s, x0, y0 + 0.95, 1.2, 0.7, "图像 x", fill=S.PANEL, size=14)
    d.box(s, x0 + 1.75, y0, 1.6, 0.62, "增强视图 1", fill=S.PANEL, size=13.5, sub="裁剪、翻转、亮度", sub_size=10.5)
    d.box(s, x0 + 1.75, y0 + 1.98, 1.6, 0.62, "增强视图 2", fill=S.PANEL, size=13.5, sub="裁剪、翻转、亮度", sub_size=10.5)
    d.path(s, [(x0 + 1.24, y0 + 1.15), (x0 + 1.5, y0 + 1.15), (x0 + 1.5, y0 + 0.31), (x0 + 1.71, y0 + 0.31)])
    d.path(s, [(x0 + 1.24, y0 + 1.45), (x0 + 1.5, y0 + 1.45), (x0 + 1.5, y0 + 2.29), (x0 + 1.71, y0 + 2.29)])
    d.box(s, x0 + 3.85, y0 + 0.85, 1.55, 0.9, "共享编码器", fill="#E2F3EE", size=14)
    d.arrow(s, x0 + 3.39, y0 + 0.31, x0 + 3.81, y0 + 1.0)
    d.arrow(s, x0 + 3.39, y0 + 2.29, x0 + 3.81, y0 + 1.6)
    d.box(s, x0 + 5.9, y0, 2.35, 0.75, "实例头：行对比", fill=S.PANEL, size=13.5, sub="同一张图的两视图互为正样本", sub_size=10.5)
    d.box(s, x0 + 5.9, y0 + 1.85, 2.35, 0.75, "簇头：列对比（CC）", fill=S.PANEL, size=13.5, sub="同一个簇在两视图中一致", sub_size=10.5)
    d.path(s, [(x0 + 5.44, y0 + 1.1), (x0 + 5.66, y0 + 1.1), (x0 + 5.66, y0 + 0.375), (x0 + 5.86, y0 + 0.375)])
    d.path(s, [(x0 + 5.44, y0 + 1.5), (x0 + 5.66, y0 + 1.5), (x0 + 5.66, y0 + 2.225), (x0 + 5.86, y0 + 2.225)])
    d.equation(s, "nt_xent", x0 + 0.05, 5.0, scale=0.82)
    d.text(s, x0 + 4.2, 5.15, 4.3, 0.6, "NT-Xent：把正样本对拉近，把其余样本推远", size=13, color=S.SECONDARY)
    xr = 9.55
    rows = [("SimCLR + K-means", S.CONTRASTIVE, "多阶段：只训练实例头学特征，再离线做 K-means。"),
            ("CC", S.CONTRASTIVE, "联合优化：实例级与簇级对比同时训练，簇头直接给出分配。")]
    for i, (name, color, body) in enumerate(rows):
        y = 2.0 + i * 1.55
        d.dot(s, xr, y + 0.1, color, d=0.16)
        d.text(s, xr + 0.3, y, 3.0, 0.4, name, size=16, bold=True, color=S.INK)
        d.text(s, xr + 0.3, y + 0.45, 3.1, 1.0, body, size=13.5, color=S.BODY, line=1.35)
    d.rect(s, xr, 5.05, 3.13, 1.45, fill=S.PANEL, radius=0.06)
    d.text(s, xr + 0.2, 5.15, 2.75, 1.25, "本项目受课程算力限制：两者都用 128 维小型 CNN、训练 100 轮；CC 未加原文的簇熵正则。",
           size=12, color=S.SECONDARY, line=1.4, anchor="m")


def s11_pretrained(d: Deck):
    s = d.slide("S11_pretrained", SECTION["method"], "预训练模型把“学表示”交给了海量外部数据",
                source="Oquab et al., TMLR 2024 · Radford et al., ICML 2021 · Li et al., ICML 2024 · Hu et al., 2026（预印本）")
    y = 2.0
    items = [("CIFAR-10 图像", None, S.PANEL, 1.85), ("冻结编码器", "DINOv2 ViT-S/14 · CLIP ViT-B/32", "#FDF0EA", 2.6),
             ("特征向量", "384 / 512 维，L2 归一化", S.PANEL, 2.15), ("聚类器", "K-means · GMM · DEC 头", S.PANEL, 2.3)]
    _flow_row(d, s, items, y, h=0.95, gap=0.48, size=15)
    d.text(s, MX + 2.33, y + 1.05, 2.6, 0.3, "参数不更新", size=11.5, color=S.PRETRAINED, align="c")
    cards = [("DINOv2", "约 1.42 亿张无标签图像自监督预训练；只用图像。"),
             ("CLIP", "4 亿对图文配对，用对比学习对齐图像和文本。"),
             ("TAC", "用 WordNet 名词给图像聚类提供外部语义引导。"),
             ("VLM4Cluster", "在 20 个数据集上比较 17 种方法的效果、鲁棒性与成本（2026 预印本）。")]
    cw, gap = 2.85, 0.21
    for i, (name, body) in enumerate(cards):
        x = MX + i * (cw + gap)
        d.rule(s, x, 3.75, cw, color=S.PRETRAINED if i < 2 else S.HAIRLINE, width=2 if i < 2 else 1)
        d.text(s, x, 3.92, cw, 0.4, name, size=17, bold=True, color=S.INK)
        d.text(s, x, 4.38, cw, 1.2, body, size=13.5, color=S.BODY, line=1.4)
    d.text(s, MX, 3.42, 4.0, 0.3, "本项目实验中使用", size=11.5, color=S.PRETRAINED)
    d.text(s, MX + 2 * (cw + gap), 3.42, 4.0, 0.3, "文献前沿", size=11.5, color=S.SECONDARY)
    d.rect(s, MX, 6.0, CW, 0.7, fill=S.PANEL, radius=0.12)
    d.text(s, MX + 0.3, 6.0, CW - 0.6, 0.7,
           "这些模型在课程开始前就见过大量外部数据，训练成本不可与从零训练直接比较，实验中单独报告。",
           size=14.5, color=S.BODY, anchor="m")


def s12_setup(d: Deck):
    f = d.facts
    s = d.slide("S12_setup", SECTION["exp"], "实验设计：3 个数据集、11 种方法，标签只在评分时使用")
    rows = [("mnist", "MNIST", "手写数字 · 28×28 灰度", "训练 2 万 / 测试 1 万"),
            ("fashion_mnist", "Fashion-MNIST", "服饰 · 28×28 灰度", "训练 2 万 / 测试 1 万"),
            ("cifar10", "CIFAR-10", "自然图像 · 32×32 彩色", "训练 1 万 / 测试 2 千")]
    y = 1.8
    for key, name, desc, size in rows:
        d.text(s, MX, y, 2.3, 0.35, name, size=16, bold=True, color=S.INK)
        d.text(s, MX, y + 0.38, 2.3, 0.3, desc, size=11.5, color=S.SECONDARY)
        d.text(s, MX, y + 0.66, 2.3, 0.3, size, size=11.5, color=S.SECONDARY)
        d.image(s, ASSETS / f"strip_{key}.png", MX + 2.35, y + 0.05, w=4.9)
        y += 1.22
    x0 = 8.25
    d.text(s, x0, 1.8, 4.4, 0.35, "方法（按家族着色，全篇一致）", size=13, color=S.SECONDARY)
    fams = [(S.CLASSICAL, "经典", "K-means · PCA + K-means · GMM"),
            (S.DEEP, "自编码器系", "AE + K-means · DEC · IDEC"),
            (S.GENERATIVE, "生成式", "VaDE"),
            (S.CONTRASTIVE, "对比学习", "SimCLR + K-means · CC"),
            (S.PRETRAINED, "外部预训练", "DINOv2 · CLIP（冻结）")]
    for i, (color, fam, names) in enumerate(fams):
        yy = 2.25 + i * 0.66
        d.dot(s, x0, yy + 0.09, color, d=0.17)
        d.text(s, x0 + 0.3, yy, 1.4, 0.35, fam, size=14, bold=True, color=S.INK)
        d.text(s, x0 + 1.62, yy + 0.01, 2.81, 0.6, names, size=12.5, color=S.BODY, line=1.2)
    py = 5.6
    d.rule(s, MX, py, CW, color=S.HAIRLINE, width=1)
    rules = [("训练不读标签", "fit 接口只接收图像；测试标签只在最后计算 ACC、NMI、ARI 时读取"),
             ("预先固定的协议", "超参数、早停（簇分配变化 < 0.1%）在看到测试结果前写入配置"),
             ("3 个随机种子", "主要对比报告 3 个种子的均值 ± 标准差；稳定性诊断用种子 0"
              + (f"。共 {f['n_fits']:,} 次训练" if isinstance(f.get("n_fits"), int) else ""))]
    for i, (head, body) in enumerate(rules):
        x = MX + i * 4.05
        d.text(s, x, py + 0.2, 3.85, 0.35, head, size=15, bold=True, color=S.INK)
        d.text(s, x, py + 0.6, 3.85, 0.8, body, size=12.5, color=S.SECONDARY, line=1.35)


def s13_benchmark(d: Deck):
    s = d.slide("S13_benchmark", SECTION["exp"], "从零训练：DEC 与 VaDE 领先，CIFAR-10 上全部失效",
                source="ARI（调整兰德指数）：0 相当于随机分组，1 与真实类别完全一致。完整 ACC / NMI / ARI 见附录 A01。")
    d.figure(s, "benchmark_ari", MX, 1.75, CW)


def s14_geometry(d: Deck):
    s = d.slide("S14_geometry", SECTION["exp"], "非线性表示先把类拉开，DEC 再把每个簇压紧",
                source="Fashion-MNIST 测试集 2,000 张图（种子 0）在各方法表示空间中的 t-SNE；颜色是该方法自己的预测簇，不使用真实标签。")
    d.figure(s, "embeddings_tsne", MX, 1.75, CW)
    d.text(s, MX, 6.05, CW, 0.7,
           "AE + K-means、DEC、IDEC 共用同一个预训练自编码器，差别只来自第二阶段：DEC 的自训练把簇压成紧凑的“岛”，IDEC 的重构项让它们保持更松散的形状。",
           size=14, color=S.BODY, line=1.4)


def s15_repr_vs_clusterer(d: Deck):
    f = d.facts
    er, cr = f.get("encoder_range_max"), f.get("clusterer_range_mean_max")
    title = (f"换表示，ARI 最多变化 {er:.2f}；换聚类器，平均变化不到 {math.ceil(cr * 100) / 100:.2f}"
             if er is not None and cr is not None else "换表示的影响远大于换聚类器")
    lede = None
    if all(f.get(f"best_enc_{ds}") for ds in ("mnist", "fashion_mnist", "cifar10")):
        c, fa, m = f["best_enc_cifar10"], f["best_enc_fashion_mnist"], f["best_enc_mnist"]
        head = f"CIFAR-10 与 Fashion-MNIST 上是 {c}" if c == fa else f"CIFAR-10 上是 {c}，Fashion-MNIST 上是 {fa}"
        tail = f"MNIST 上是在本数据上训练的{m}" if m == "自编码器" else f"MNIST 上是 {m}"
        lede = f"但最好的表示因数据而异：{head}，{tail}。"
    s = d.slide("S15_representation_vs_clusterer", SECTION["exp"], title, lede=lede,
                source="每行是一种编码器，三个标记是在同一表示上换用的三种聚类器；自编码器、SimCLR 与 K-means 的组合即基准实验中的同一批模型。")
    d.figure(s, "representation_vs_clusterer", MX, 1.72, CW)
    d.text(s, MX, 6.42, CW, 0.4,
           [[("读图：", {"color": S.SECONDARY}), ("行与行之间的距离是换表示的影响，", {"bold": True, "color": S.INK}),
             ("同一行三个标记之间的距离是换聚类器的影响。", {"bold": True, "color": S.INK})]],
           size=14.5, color=S.BODY)


def s16_pixels_vs_dinov2(d: Deck):
    f = d.facts
    s = d.slide("S16_pixels_vs_dinov2", SECTION["exp"], "同一批 CIFAR-10 图像：像素空间按颜色排，DINOv2 空间按物体排",
                source="300 张测试图（种子 0）按各自表示的 t-SNE 坐标放进 20 × 15 网格；排列只用表示，不用标签。")
    pw = 5.85
    for i, (name, head, ari) in enumerate([("mosaic_pixels", "像素空间", f.f("ari_cifar10_kmeans", 2)),
                                           ("mosaic_dinov2", "DINOv2 表示空间", f.f("e04_cifar10_dinov2_kmeans", 2))]):
        x = MX + i * (pw + CW - 2 * pw)
        d.image(s, ASSETS / f"{name}.png", x, 1.75, w=pw)
        d.text(s, x, 6.2, pw - 2.2, 0.4, head, size=17, bold=True, color=S.INK)
        d.text(s, x + pw - 2.6, 6.2, 2.6, 0.4, [[("K-means ARI ", {"color": S.SECONDARY}), (ari, {"bold": True})]],
               size=17, color=S.INK, align="r")


def s17_noise(d: Deck):
    f = d.facts
    sigma = f.get("noise_max", 0.3)
    title = f"加噪后排名反转：K-means 保留 {f.p('keep_kmeans')}，DEC 只剩 {f.p('keep_dec')}"
    s = d.slide("S17_noise", SECTION["exp"], title,
                source="噪声只加在测试图上（截断到 [0, 1]），模型不重新训练；Fashion-MNIST，3 个随机种子均值。")
    d.figure(s, "noise_robustness", MX - 0.1, 1.62, 7.55)
    x0 = 8.35
    d.text(s, x0, 1.8, 4.3, 0.3, "同一张测试图在不同噪声强度下", size=12.5, color=S.SECONDARY)
    for i, std in enumerate([0.0, 0.1, 0.2, 0.3]):
        x = x0 + i * 1.07
        d.image(s, ASSETS / f"noise_{str(std).replace('.', 'p')}.png", x, 2.2, w=0.92, h=0.92)
        d.text(s, x, 3.17, 0.92, 0.3, f"σ = {std:g}", size=11.5, color=S.SECONDARY, align="c")
    d.text(s, x0, 3.85, 4.35, 1.3,
           [[("干净图上越强的深度方法，加噪后掉得越多：", {"bold": True, "color": S.INK}),
             ("学到的特征对像素扰动敏感；而各向同性噪声让像素到每个中心的距离增加大致相同的量，K-means 的最近中心基本不变。", {})]],
           size=14, color=S.BODY, line=1.45)
    gmm_k = f.get("gmm_clusters_at_01")
    d.text(s, x0, 5.35, 4.35, 1.3,
           [[("像素 GMM 一碰噪声就塌缩：", {"bold": True, "color": S.INK}),
             (f"边缘背景像素在训练集上几乎不变，标准化后微小噪声被放大，σ = 0.1 时全部测试图落进 {gmm_k:.0f} 个簇。"
              if gmm_k else "边缘背景像素在训练集上几乎不变，标准化后微小噪声被放大，全部测试图落进同一个簇。", {})]],
           size=14, color=S.BODY, line=1.45)


def s18_sample_size(d: Deck):
    f = d.facts
    km, dec = f.get("sf_kmeans_0.01"), f.get("sf_dec_0.01")
    n = f.get("n_train_1pct")
    if km is not None and dec is not None and km > dec:
        title = f"训练图只有 {n:,.0f} 张时，联合优化反而不如 K-means" if n else "样本很少时，联合优化反而不如 K-means"
    else:
        title = "训练样本量对联合优化的影响"
    s = d.slide("S18_sample_size", SECTION["exp"], title,
                source="从 2 万张训练图中无放回抽样；训练轮数上限不变；测试集固定为 1 万张。3 个随机种子均值 ± 标准差。")
    d.figure(s, "sample_size", MX - 0.1, 1.62, 7.55)
    x0 = 8.4
    d.stat(s, x0, 1.85, 4.2, f"{f.f('sf_kmeans_0.01')} vs {f.f('sf_dec_0.01')}",
           f"{n:,.0f} 张训练图时，K-means 与 DEC 的 ARI" if n else "1% 训练图时，K-means 与 DEC 的 ARI",
           value_size=30, label_size=13.5)
    d.stat(s, x0, 3.35, 4.2, f"{f.f('sf_dec_1.0')} vs {f.f('sf_kmeans_1.0')}", "全部 2 万张训练图时，DEC 与 K-means 的 ARI",
           value_size=30, label_size=13.5)
    d.text(s, x0, 4.95, 4.25, 1.5,
           "DEC 依赖自编码器先学出好的表示；样本太少时预训练学不充分，自训练又会放大初始错误。",
           size=14, color=S.BODY, line=1.45)


def s19_components(d: Deck):
    f = d.facts
    full, frozen = f.get("comp_full"), f.get("comp_frozen")
    nopre, rnd = f.get("comp_no_pretrain"), f.get("comp_random_init")
    init_matters = None not in (full, frozen, nopre, rnd) and (full - frozen) < min(full - nopre, full - rnd)
    recon_hurts = bool(f.get("lam_fx_all_lower_1") and f.get("lam_fx_all_lower_10"))
    if init_matters and recon_hurts:
        title = "IDEC 的收益主要来自初始化；加重重构项反而降低 ARI"
    elif init_matters:
        title = "IDEC 的收益主要来自预训练初始化，而不是联合微调"
    else:
        title = "IDEC 组件消融：预训练、初始化与联合微调"
    s = d.slide("S19_components", SECTION["exp"], title,
                source="Fashion-MNIST，IDEC（λ = 1，隐空间 10 维），3 个随机种子；右图误差线为同一种子上差值的标准差，“固定 50 轮”关闭早停与追加轮次后重跑。")
    d.figure(s, "components_lambda", MX, 1.6, CW)
    lines = []
    if None not in (full, frozen, nopre, rnd):
        lines.append([("左：去掉 AE 预训练 −", {}), (f"{full - nopre:.2f}", {"bold": True, "color": S.INK}),
                      ("，簇中心随机初始化 −", {}), (f"{full - rnd:.2f}", {"bold": True, "color": S.INK}),
                      ("，冻结编码器只学簇中心 −", {}), (f"{full - frozen:.2f}", {"bold": True, "color": S.INK}),
                      ("。", {})])
    if f.get("lam_fx_gap_1") is not None and f.get("lam_fx_gap_10") is not None:
        lines.append([("右：固定 50 轮时，λ ≤ 0.1 与 DEC 几乎相同；λ = 1、10 平均低 ", {}),
                      (f"{f['lam_fx_gap_1']:.2f}", {"bold": True, "color": S.INK}), ("、", {}),
                      (f"{f['lam_fx_gap_10']:.2f}", {"bold": True, "color": S.INK}),
                      (f"，{f.get('lam_fx_seeds', 3)} 个种子上都更低。" if recon_hurts else "。", {})])
    if lines:
        d.text(s, MX, 6.12, CW, 0.8, lines, size=14, color=S.BODY, line=1.35)


def s20_cost(d: Deck):
    f = d.facts
    ratio, gain = f.get("dec_time_ratio"), f.get("dec_ari_gain")
    title = (f"DEC 多花约 {ratio:.0f} 倍训练时间，换来 ARI +{gain:.2f}" if ratio and gain is not None
             else "精度与训练成本")
    s = d.slide("S20_cost", SECTION["exp"], title,
                source="Fashion-MNIST，3 个随机种子均值；时间为同一台笔记本 GPU 上的端到端训练时间，DEC/IDEC 计入自编码器预训练。")
    d.figure(s, "cost_frontier", MX - 0.1, 1.62, 7.55)
    x0 = 8.4
    d.text(s, x0, 1.85, 4.25, 1.2,
           [[("黑线是帕累托前沿：", {"bold": True, "color": S.INK}),
             ("在不多花时间的前提下，没有别的方法能得到更高的 ARI。", {})]], size=14, color=S.BODY, line=1.45)
    y = 3.15
    for m in f.get("front_methods") or []:
        d.dot(s, x0, y + 0.1, S.FAMILY.get(m, S.MUTED), d=0.15)
        d.text(s, x0 + 0.28, y, 2.3, 0.35, S.LABEL.get(m, m), size=14, bold=True, color=S.INK)
        d.text(s, x0 + 2.5, y, 1.75, 0.35, f"{f.f('time_' + m, 1)} 秒 · {f.f('cost_ari_' + m)}", size=13,
               color=S.SECONDARY, align="r")
        y += 0.46
    d.text(s, x0, y + 0.3, 4.25, 1.3, "对比学习路线训练最久、分数最低：小网络、100 轮不足以学出语义特征。",
           size=14, color=S.BODY, line=1.45)


def s21_stability(d: Deck):
    f = d.facts
    sel_ok = all(f.get(f"stable_{ds}") for ds in ("mnist", "fashion_mnist", "cifar10"))
    s = d.slide("S21_stability", SECTION["exp"], "没有标签时，“稳定”选不出“正确”",
                source="稳定性：训练图加 σ = 0.1 噪声前后，簇成员的平均 Jaccard 一致度（种子 0，不读任何标签）。ARI 用测试标签事后核对。")
    if sel_ok:
        d.figure(s, "stability_vs_ari", MX, 1.55, CW)
        d.text(s, MX, 6.2, CW, 0.6,
               [[("三个数据集上，最稳定的都是经典方法，最准确的都是深度方法。", {"bold": True, "color": S.INK}),
                 ("稳定性能排除塌缩的方法，但不能代替少量标注样本的检验。", {})]], size=15, color=S.BODY)
    else:
        d.rect(s, MX, 2.2, CW, 3.0, fill=S.PANEL, radius=0.04)
        d.text(s, MX, 2.2, CW, 3.0, "稳定性诊断证据不足：缺少完整的无标签稳定性记录。", size=18, color=S.SECONDARY,
               align="c", anchor="m")


def _pretrained_heads_text(f: Facts) -> str:
    g, gmax, h, hmax, n = (f.get(k) for k in ("pre_gmm_delta_mean", "pre_gmm_delta_max", "pre_head_delta_mean",
                                              "pre_head_delta_max", "pre_pairs"))
    if None in (g, gmax, h, hmax, n):
        return DASH
    gmm = f"GMM 平均比 K-means {'低' if g < 0 else '高'} {abs(g):.2f}（单组最多高 {max(gmax, 0):.2f}）"
    head = (f"DEC 聚类头 {n} 组全部更低，平均低 {abs(h):.2f}" if hmax < 0
            else f"DEC 聚类头平均{'低' if h < 0 else '高'} {abs(h):.2f}")
    return f"冻结的 DINOv2 / CLIP 特征上（3 个数据集 × 2 个模型），{gmm}；{head}。"


def s22_verdict(d: Deck):
    f = d.facts
    s = d.slide("S22_verdict", SECTION["end"], "三个命题的回答")
    rows = [
        ("表示比聚类器更重要", "成立", S.INK,
         f"换表示让 ARI 最多变化 {f.f('encoder_range_max', 2)}，同一表示上换聚类器平均只变化约 "
         f"{f.f('clusterer_range_mean_max', 2)}。最好的表示取决于领域：DINOv2 在 CIFAR-10 上 "
         f"{f.f('e04_cifar10_dinov2_kmeans')}，在 MNIST 上只有 {f.f('e04_mnist_dinov2_kmeans')}"
         f"（自编码器 {f.f('ari_mnist_ae_kmeans')}）。"),
        ("联合优化有条件地有效", "有条件", S.SECONDARY,
         f"干净、充足的数据上 DEC 最好（MNIST {f.f('ari_mnist_dec')}）；但测试噪声 σ = {f.get('noise_max', 0.3):g} 时只保留 "
         f"{f.p('keep_dec')}（K-means {f.p('keep_kmeans')}），只有 1% 训练图时不如 K-means。"),
        ("有了预训练表示，简单聚类器就够了", "成立", S.INK,
         _pretrained_heads_text(f)),
    ]
    y = 1.75
    for i, (claim, verdict, vcolor, evidence) in enumerate(rows):
        d.rule(s, MX, y, CW, color=S.INK if i == 0 else S.HAIRLINE, width=2 if i == 0 else 1)
        d.text(s, MX, y + 0.28, 0.6, 0.6, str(i + 1), size=30, bold=True, color="#D4D4D8", line=1.0)
        d.text(s, MX + 0.65, y + 0.32, 3.6, 0.9, claim, size=19, bold=True, color=S.INK, line=1.2)
        pill = d.rect(s, MX + 4.45, y + 0.36, 1.05, 0.42, fill=S.INK if verdict == "成立" else S.SURFACE,
                      line=None if verdict == "成立" else S.INK, radius=0.5)
        d.text(s, MX + 4.45, y + 0.36, 1.05, 0.42, verdict, size=13, bold=True,
               color="#FFFFFF" if verdict == "成立" else S.INK, align="c", anchor="m")
        d.text(s, MX + 5.8, y + 0.3, CW - 5.8, 1.2, evidence, size=14.5, color=S.BODY, line=1.45)
        y += 1.55


def s23_selection_guide(d: Deck):
    f = d.facts
    s = d.slide("S23_selection_guide", SECTION["end"], "方法选择：先问表示从哪里来，再问数据和预算",
                source="建议来自本项目三个数据集上的带标签基准；在新数据上使用前，需要少量人工标注样本核对。")
    qx, qw, qh = MX, 4.5, 0.82
    ax, aw = 7.9, 4.78
    qs = [(1.7, "有与数据领域匹配的预训练模型吗？", "冻结特征 + K-means", S.PRETRAINED,
           f"CIFAR-10：DINOv2 + K-means {f.f('e04_cifar10_dinov2_kmeans')}，从零训练 ≤ {f.f('max_scratch_cifar10')}；"
           f"但在手写数字上只有 {f.f('e04_mnist_dinov2_kmeans')}"),
          (3.05, "样本很少、噪声大，或预算很紧？", "PCA + K-means / 像素 K-means", S.CLASSICAL,
           f"加噪保留 {f.p('keep_pca_kmeans')}；训练 {f.f('time_pca_kmeans', 1)} 秒"),
          (4.4, "需要概率解释或生成样本？", "VaDE", S.GENERATIVE,
           f"MNIST {f.f('ari_mnist_vade')}，与 DEC 相当，并给出每簇的方差和先验"),
          (5.75, "以上都不是", "DEC（自编码器预训练 + 自训练）", S.DEEP,
           f"干净数据上最高：MNIST {f.f('ari_mnist_dec')} · Fashion {f.f('ari_fashion_mnist_dec')}")]
    for i, (y, q, a, color, ev) in enumerate(qs):
        last = i == len(qs) - 1
        d.box(s, qx, y, qw, qh, q, fill=S.PANEL if not last else S.SURFACE, size=15.5, bold=not last,
              color=S.INK if not last else S.SECONDARY, line=S.HAIRLINE if last else None)
        d.arrow(s, qx + qw + 0.08, y + qh / 2, ax - 0.1, y + qh / 2, color=S.SECONDARY)
        if not last:
            d.text(s, qx + qw + 0.25, y + qh / 2 - 0.33, 1.0, 0.3, "是", size=12, color=S.SECONDARY)
            d.arrow(s, qx + 0.45, y + qh + 0.03, qx + 0.45, y + 1.35 - 0.04, color=S.MUTED)
            d.text(s, qx + 0.6, y + qh + 0.08, 0.6, 0.3, "否", size=12, color=S.SECONDARY)
        d.rect(s, ax, y, 0.09, qh, fill=color)
        d.text(s, ax + 0.28, y + 0.04, aw - 0.3, 0.4, a, size=16, bold=True, color=S.INK)
        d.text(s, ax + 0.28, y + 0.45, aw - 0.3, 0.35, ev, size=12, color=S.SECONDARY)


def s24_limits(d: Deck):
    s = d.slide("S24_limits_and_close", SECTION["end"], "局限与展望")
    lim = ["簇数 K 已知（= 10）；K 未知时需要另行估计，估错会明显拉低 ARI（附录 A03）。",
           "课程算力：SimCLR、CC 用 128 维小 CNN、训练 100 轮，CC 未加簇熵正则；对比学习路线被低估。",
           "只有 3 个小规模图像数据集；结论未覆盖文本、表格和高分辨率图像。",
           "预训练模型含外部数据，成本不可与从零训练直接比较。"]
    fut = ["未知 K 的自适应聚类：让模型自己决定分几类。",
           "无标签模型选择：超越扰动稳定性，找到与语义一致的选择准则。",
           "文本引导的聚类：沿 TAC、VLM4Cluster 方向，检验语言先验的鲁棒性。"]
    for col, (head, items, color) in enumerate([("局限", lim, S.SECONDARY), ("展望", fut, S.DEEP)]):
        x = MX + col * 6.25
        w = 5.8
        d.rule(s, x, 1.75, w, color=S.INK, width=2)
        d.text(s, x, 1.92, w, 0.45, head, size=19, bold=True, color=S.INK)
        y = 2.55
        for item in items:
            d.dot(s, x, y + 0.12, color, d=0.12)
            d.text(s, x + 0.3, y, w - 0.3, 0.9, item, size=14.5, color=S.BODY, line=1.4)
            y += 0.82
    d.rect(s, 0, 6.05, W, 1.45, fill=S.NIGHT)
    d.text(s, MX, 6.05, 7.5, 1.45, [[("谢谢，欢迎提问", {"bold": True, "size": 24, "color": "#FFFFFF"})],
                                   [(REPO_URL.replace("https://", ""), {"size": 12.5, "color": "#A1A1AA"})]],
           anchor="m", line=1.4)
    qr = ASSETS / "qr_repo.png"
    if qr.exists():
        d.image(s, qr, W - MX - 1.15, 6.2, w=1.15, h=1.15)
        d.text(s, W - MX - 4.2, 6.05, 2.9, 1.45, "代码、协议、原始记录\n与本幻灯片的生成脚本", size=11.5, color="#A1A1AA",
               align="r", anchor="m", line=1.35)


# ------------------------------------------------------------------------------- appendix
def a01_table(d: Deck, t: dict):
    s = d.slide("A01_benchmark_table", SECTION["app"], "A01　从零训练基准：完整指标",
                source="均值 ± 标准差（3 个随机种子）。加粗为每列最高。")
    e01 = t["e01_summary"]
    methods = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
    head = ["方法"] + [f"{S.DATASET_LABEL[ds]}\n{m}" for ds in ("mnist", "fashion_mnist", "cifar10")
                       for m in ("ACC", "NMI", "ARI")]
    rows, bold = [head], set()
    best = {}
    for j, (ds, metric) in enumerate([(ds, m) for ds in ("mnist", "fashion_mnist", "cifar10")
                                       for m in ("acc", "nmi", "ari")]):
        sub = e01[e01.dataset.eq(ds)] if not e01.empty else e01
        if len(sub):
            best[(ds, metric)] = sub.loc[sub[f"{metric}_mean"].idxmax(), "method"]
    for i, m in enumerate(methods, start=1):
        row = [S.LABEL[m]]
        for j, (ds, metric) in enumerate([(ds, mm) for ds in ("mnist", "fashion_mnist", "cifar10")
                                           for mm in ("acc", "nmi", "ari")], start=1):
            mean = _lookup(e01, f"{metric}_mean", dataset=ds, method=m)
            sd = _lookup(e01, f"{metric}_std", dataset=ds, method=m)
            row.append(DASH if mean is None else f"{mean:.3f}±{sd:.3f}" if sd is not None else f"{mean:.3f}")
            if best.get((ds, metric)) == m:
                bold.add((i, j))
        rows.append(row)
    d.table(s, MX, 1.7, CW, rows, col_w=[1.83] + [1.133] * 9, size=11, header_size=10.5, row_h=0.43,
            bold_cells=bold)


def a02_ablations(d: Deck, t: dict):
    f = d.facts
    s = d.slide("A02_ablations", SECTION["app"], "A02　IDEC 消融：隐空间维度与重构权重 λ",
                source="Fashion-MNIST，3 个随机种子。早停版与基准协议相同；固定版关闭早停与追加轮次，每个 λ 训练 50 轮。")
    d.figure(s, "latent_dim", MX, 1.75, 5.6)
    rows = [["λ", "早停：ARI", "早停：实际轮数", "固定 50 轮：ARI"]]
    for lam in (0.0, 0.01, 0.1, 1.0, 10.0):
        ep = f.get(f"lam_es_ep_{lam}")
        fx, sd = f.get(f"lam_fx_{lam}"), f.get(f"lam_fx_sd_{lam}")
        rows.append([f"{lam:g}", f.f(f"lam_es_{lam}"), DASH if ep is None else f"{ep:.0f}",
                     DASH if fx is None else (f"{fx:.3f}±{sd:.3f}" if sd is not None else f"{fx:.3f}")])
    d.table(s, 6.75, 2.0, 5.9, rows, col_w=[0.9, 1.5, 1.6, 1.9], size=13, header_size=12, row_h=0.5)
    d.text(s, 6.75, 5.25, 5.9, 1.3,
           "早停只看训练集簇分配是否稳定，λ 越大，簇分配越晚稳定、训练轮数越多；固定轮数后，λ 的作用不再与训练时长混在一起。",
           size=13, color=S.SECONDARY, line=1.45)


def a03_sensitivity(d: Deck, t: dict):
    s = d.slide("A03_sensitivity", SECTION["app"], "A03　簇数设错、训练集混入噪声图、类别不平衡",
                source="Fashion-MNIST；K-means、AE + K-means、DEC、IDEC 为 3 个随机种子均值 ± 标准差，CC 为单种子。不平衡设置按训练集标签构造比例，评价在均衡的测试集上。")
    d.figure(s, "sensitivity_k_outliers_imbalance", MX, 1.75, CW)


def a04_encoder_table(d: Deck, t: dict):
    s = d.slide("A04_encoder_clusterer_table", SECTION["app"], "A04　编码器 × 聚类器：完整 ARI",
                source="均值 ± 标准差（3 个随机种子）；自编码器 / SimCLR + K-means 与基准实验为同一批模型。")
    e04 = t["e04_summary"]
    head = ["编码器"] + [f"{S.DATASET_LABEL[ds]}\n{S.CLUSTERER_LABEL[c]}" for ds in ("mnist", "fashion_mnist", "cifar10")
                         for c in ("kmeans", "gmm", "dec_head")]
    rows, bold = [head], set()
    for i, enc in enumerate(("random_cnn", "ae", "simclr", "dinov2", "clip"), start=1):
        row = [S.ENCODER_LABEL[enc]]
        for j, (ds, c) in enumerate([(ds, c) for ds in ("mnist", "fashion_mnist", "cifar10")
                                      for c in ("kmeans", "gmm", "dec_head")], start=1):
            mean = _lookup(e04, "ari_mean", dataset=ds, encoder=enc, clusterer=c)
            sd = _lookup(e04, "ari_std", dataset=ds, encoder=enc, clusterer=c)
            row.append(DASH if mean is None else f"{mean:.3f}±{sd:.3f}" if sd is not None else f"{mean:.3f}")
        rows.append(row)
    d.table(s, MX, 1.8, CW, rows, col_w=[1.5] + [1.17] * 9, size=11, header_size=10.5, row_h=0.55)


def a05_stability_table(d: Deck, t: dict):
    f = d.facts
    s = d.slide("A05_stability_table", SECTION["app"], "A05　无标签稳定性诊断：完整记录",
                source="训练集中抽 2,000 张图，比较加 σ = 0.1 噪声前后的簇分配（种子 0）；要求两次预测都包含全部 K 个簇。")
    sel = t.get("selection") or {}
    methods = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
    head = ["方法"] + [f"{S.DATASET_LABEL[ds]}\n{lab}" for ds in ("mnist", "fashion_mnist", "cifar10")
                       for lab in ("稳定性", "ARI")]
    rows, bold = [head], set()
    for i, m in enumerate(methods, start=1):
        row = [S.LABEL[m]]
        for j, ds in enumerate(("mnist", "fashion_mnist", "cifar10")):
            ev = ((sel.get(ds) or {}).get("evidence") or {}).get(m) or {}
            jac = _num(ev.get("macro_cluster_jaccard"))
            row += [DASH if jac is None else f"{jac:.3f}", f.f(f"ari_{ds}_{m}")]
            if (sel.get(ds) or {}).get("choice") == m:
                bold.add((i, 1 + 2 * j))
        rows.append(row)
    d.table(s, MX, 1.8, 8.3, rows, col_w=[1.7] + [1.1] * 6, size=11.5, header_size=10.5, row_h=0.43,
            bold_cells=bold)
    d.equation(s, "jaccard", 9.45, 2.0, scale=0.75)
    d.text(s, 9.3, 3.0, 3.4, 3.0,
           [["A", ("k", {"sub": True}), "、B", ("k", {"sub": True}),
             "：同一个模型在加噪前、后分到第 k 个簇的训练图集合。"],
            "加粗：每个数据集上稳定性最高的方法，它们都不是 ARI 最高的方法。"], size=12.5, color=S.SECONDARY,
           line=1.45, after=6)


def a06_references(d: Deck, t: dict):
    s = d.slide("A06_references", SECTION["app"], "A06　参考文献")
    refs = [
        "Zhou, S. et al. A Comprehensive Survey on Deep Clustering: Taxonomy, Challenges, and Future Directions. ACM Computing Surveys, 2025.",
        "Xie, J., Girshick, R., Farhadi, A. Unsupervised Deep Embedding for Clustering Analysis. ICML, 2016.",
        "Guo, X. et al. Improved Deep Embedded Clustering with Local Structure Preservation. IJCAI, 2017.",
        "Jiang, Z. et al. Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering. IJCAI, 2017.",
        "Caron, M. et al. Deep Clustering for Unsupervised Learning of Visual Features. ECCV, 2018.",
        "Chen, T. et al. A Simple Framework for Contrastive Learning of Visual Representations. ICML, 2020.",
        "Caron, M. et al. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. NeurIPS, 2020.",
        "Li, Y. et al. Contrastive Clustering. AAAI, 2021.",
        "Radford, A. et al. Learning Transferable Visual Models from Natural Language Supervision. ICML, 2021.",
        "Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR, 2024.",
        "Li, Y. et al. Image Clustering with External Guidance. ICML, 2024.",
        "Hu, Y. et al. VLM4Cluster: Benchmarking Deep Clustering in the Era of Vision-Language Pre-training. arXiv:2609.36648, 2026（预印本）.",
        "王学民.《应用多元统计分析》（第六版）. 上海财经大学出版社.",
    ]
    d.text(s, MX, 1.6, CW, 5.3, refs, size=12.5, color=S.BODY, line=1.25, after=7)


SLIDES = [s01_cover, s02_question, s03_statistics, s04_pixel_distance, s05_core_idea, s06_paradigms, s07_dec,
          s08_idec, s09_vade, s10_contrastive, s11_pretrained, s12_setup, s13_benchmark, s14_geometry,
          s15_repr_vs_clusterer, s16_pixels_vs_dinov2, s17_noise, s18_sample_size, s19_components, s20_cost,
          s21_stability, s22_verdict, s23_selection_guide, s24_limits]
APPENDIX = [a01_table, a02_ablations, a03_sensitivity, a04_encoder_table, a05_stability_table, a06_references]


# =============================================================================== notes
def parse_speaker_notes(raw: str) -> list[tuple[str, str]]:
    blocks = re.split(r"^## \[([A-Za-z0-9_]+)\][^\n]*\n", raw, flags=re.MULTILINE)
    return [(blocks[i], blocks[i + 1].strip()) for i in range(1, len(blocks) - 1, 2)]


def inject_speaker_notes(prs, slide_ids: list[str], facts: Facts, notes_path: Path = NOTES_MD) -> None:
    notes = parse_speaker_notes(notes_path.read_text(encoding="utf-8"))
    ids = [sid for sid, _ in notes]
    if ids != slide_ids:
        missing = [s for s in slide_ids if s not in ids]
        extra = [s for s in ids if s not in slide_ids]
        raise ValueError(f"Speaker notes do not match the slides. Missing: {missing}; unexpected: {extra}")
    fmt = NotesFormatter(facts)
    for slide, (sid, body) in zip(prs.slides, notes):
        text = fmt.format(body)
        slide.notes_slide.notes_text_frame.text = f"[{sid}]\n{text}"


def make_qr(path: Path) -> None:
    import qrcode

    qr = qrcode.QRCode(border=1, box_size=12, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(REPO_URL)
    qr.make(fit=True)
    qr.make_image(fill_color="#17171A", back_color="#FFFFFF").save(path)


def build(tables_dir: Path = TABLES, out_path: Path = OUT, notes_path: Path = NOTES_MD,
          figures_dir: Path = FIGS) -> Path:
    tables = load_tables(Path(tables_dir))
    facts = Facts(compute_facts(tables))
    qr = ASSETS / "qr_repo.png"
    if not qr.exists():
        try:
            make_qr(qr)
        except ImportError:
            pass
    deck = Deck(facts, Path(figures_dir))
    for make in SLIDES:
        make(deck)
    for make in APPENDIX:
        make(deck, tables)
    if deck.ids != SLIDE_IDS:
        raise ValueError(f"Slide order changed: {deck.ids}")
    inject_speaker_notes(deck.prs, deck.ids, facts, Path(notes_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deck.prs.save(out_path)
    print(f"Built {len(deck.ids)} slides -> {out_path}")
    return out_path


if __name__ == "__main__":
    build()
