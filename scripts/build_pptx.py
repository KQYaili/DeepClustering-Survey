"""Build the 25 + 6 slide deck from results/tables, results/figures and presentation/assets.

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
    "S11b_architectures",
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
            facts[f"params_{ds}_{m}"] = _lookup(e01, "n_parameters_mean", dataset=ds, method=m)
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

    def k(self, key):
        v = _num(self.get(key))
        if v is None:
            return DASH
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        if v >= 100_000:
            return f"{v / 1_000:.0f}K"
        if v >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:.0f}"


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

    def rect(self, slide, x, y, w, h, fill=S.PANEL, line=None, radius=0.0, line_w=1.0, dash=False):
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
            if dash:
                ln = shape.line._get_or_add_ln()
                prst = etree.SubElement(ln, qn("a:prstDash"))
                prst.set("val", "sysDash")
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
            radius=0.08, sub=None, sub_size=11.5, sub_color=None, line_w=1.0, dash=False):
        shape = self.rect(slide, x, y, w, h, fill=fill, line=line, radius=radius, line_w=line_w, dash=dash)
        runs = [(label, {})] if isinstance(label, str) else [(r, {}) if isinstance(r, str) else r for r in label]
        paras = [[(text, {"bold": bold, "size": size, "color": color, **over}) for text, over in runs]]
        if sub:
            for part in (sub.split("\n") if isinstance(sub, str) else [sub]):
                paras.append([(part, {"bold": False, "size": sub_size, "color": sub_color or S.SECONDARY})])
        self.text(slide, x + 0.06, y, w - 0.12, h, paras, align="c", anchor="m", line=1.12)
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

    def family_legend(self, slide, x, y, include_pretrained: bool = False):
        items = [(S.CLASSICAL, "经典方法"), (S.DEEP, "自编码器系"), (S.GENERATIVE, "生成式"),
                 (S.CONTRASTIVE, "对比学习")]
        if include_pretrained:
            items.append((S.PRETRAINED, "外部预训练"))
        cx = x
        for color, label in items:
            self.dot(slide, cx, y + 0.055, color, d=0.13)
            w = 0.22 + len(label) * 0.165 + 0.28
            self.text(slide, cx + 0.20, y, w - 0.20, 0.26, label, size=11, color=S.SECONDARY)
            cx += w

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
    d.text(s, MX + 0.05, 1.50, 6.2, 0.4, "第九组 · 课程综述汇报", size=13, color="#A1A1AA", spacing=2)
    d.text(s, MX, 1.98, 6.4, 1.0, "深度聚类", size=54, bold=True, color="#FFFFFF", line=1.0)
    d.text(s, MX, 3.05, 6.4, 0.6, "范式演进、对照实验与方法选择", size=26, color="#E4E4E7", line=1.0)
    d.rule(s, MX + 0.03, 3.92, 0.6, color=S.PRETRAINED, width=2.5)
    d.text(s, MX, 4.16, 5.7, 1.35,
           "面对一批没有标签的图像，\n先改“看图的方式（表示）”，\n还是先换“分组规则（聚类器）”？",
           size=18, color="#E4E4E7", line=1.38)
    d.text(s, MX, 6.62, 5.8, 0.45, "MNIST · Fashion-MNIST · CIFAR-10　|　11 种方法 · 3 个随机种子", size=11.5,
           color="#A1A1AA")
    # Dark scrim pill behind the right caption so text stays crisp over the bright CIFAR-10 mosaic.
    d.rect(s, W - 6.38, 6.96, 6.10, 0.36, fill=S.NIGHT, line="#2C2F38", radius=0.45, line_w=0.75)
    d.text(s, W - 6.25, 6.96, 5.84, 0.36, "右图：没给任何标签，预训练模型已把鸟、车、马、飞机自动排成了片",
           size=10, color="#D4D4D8", align="c", anchor="m")


def s02_question(d: Deck):
    s = d.slide("S02_question", SECTION["intro"], "先改表示，还是先换聚类器？",
                lede="给无标签图像分组，两条路可选：换更好的特征坐标（表示），或换更复杂的分组算法（聚类器）。三个命题都用实验检验：")
    items = [
        ("表示比聚类器更重要",
         "固定聚类器换特征，与固定特征换聚类器，比哪边带来的分组准确度（ARI）差距更大。",
         "5 种编码器 × 3 种聚类器 × 3 数据集"),
        ("联合优化有条件地有效",
         "边学特征边分组（DEC、IDEC）在干净大样本下占优，但有噪声或小样本时未必。",
         "测试加噪 · 缩减样本量 · 组件拆解"),
        ("有了预训练表示，简单聚类就够了",
         "用现成视觉大模型（DINOv2、CLIP）提取特征后，最基础的 K-means 就够用。",
         "K-means vs 高斯混合(GMM) vs DEC 聚类头"),
    ]
    col_w, gap, y0 = 3.75, 0.39, 2.15
    for i, (claim, how, design) in enumerate(items):
        x = MX + i * (col_w + gap)
        d.rule(s, x, y0, col_w, color=S.INK, width=2)
        d.text(s, x, y0 + 0.18, 1.0, 0.75, str(i + 1), size=38, bold=True, color="#D4D4D8", line=1.0)
        d.text(s, x, y0 + 0.92, col_w, 0.85, claim, size=19.5, bold=True, color=S.INK, line=1.15, anchor="b")
        d.text(s, x, y0 + 1.95, col_w, 1.25, how, size=14.5, color=S.BODY, line=1.45)
        d.text(s, x, y0 + 3.25, col_w, 0.65, [[("怎么验：", {"color": S.SECONDARY}), (design, {"bold": True})]],
               size=13, color=S.INK, line=1.3)
    steps = ["1 统计基础", "2 方法谱系", "3 实验证据", "4 结论与选型"]
    y = 6.42
    for i, step in enumerate(steps):
        x = MX + i * 3.05
        d.dot(s, x, y + 0.08, S.INK if i == 0 else "#D4D4D8", d=0.13)
        d.text(s, x + 0.24, y, 2.5, 0.3, step, size=12.5, bold=(i == 0), color=S.INK if i == 0 else S.SECONDARY)
        if i < len(steps) - 1:
            d.line(s, x + 1.55, y + 0.145, x + 2.92, y + 0.145, color=S.HAIRLINE, width=1.25)


def s03_statistics(d: Deck):
    s = d.slide("S03_statistics_roots", SECTION["stat"], "深度聚类建立在多元统计的几块基石上",
                source="王学民《应用多元统计分析》第六版，第 2、3、6、7、8 章")
    cols = [
        ("第 2 章", "距离与协方差", "隐空间里的距离", "在学到的坐标里算远近"),
        ("第 3 章", "多元正态分布", "高斯混合先验", "VaDE 让每类成一团正态云"),
        ("第 6 章", "聚类分析", "K-means 与中心", "DEC 用簇中心算归属概率"),
        ("第 7 章", "主成分分析", "自编码器", "神经网络版的非线性降维"),
        ("第 8 章", "因子分析", "潜变量生成模型", "用隐变量反向还原原图"),
    ]
    cw, gap, x0 = 2.2, 0.258, MX
    d.text(s, MX, 1.48, CW, 0.28, "多元统计教材（人工给定或线性坐标）", size=12, color=S.SECONDARY)
    d.text(s, MX, 3.86, CW, 0.28, "深度聚类（让神经网络自动学新坐标）", size=12, bold=True, color=S.DEEP)
    top, mid = 1.82, 4.22
    for i, (chap, concept, deep, note) in enumerate(cols):
        x = x0 + i * (cw + gap)
        d.rect(s, x, top, cw, 1.28, fill=S.PANEL, radius=0.06)
        d.text(s, x + 0.18, top + 0.18, cw - 0.36, 0.28, chap, size=12, color=S.SECONDARY)
        d.text(s, x + 0.18, top + 0.52, cw - 0.36, 0.6, concept, size=18.5, bold=True, color=S.INK)
        d.arrow(s, x + cw / 2, top + 1.38, x + cw / 2, 3.74, color=S.MUTED, width=1.5)
        d.rule(s, x, mid, cw, color=S.DEEP, width=2.5)
        d.text(s, x, mid + 0.18, cw, 0.48, deep, size=17.5, bold=True, color=S.INK)
        d.text(s, x, mid + 0.66, cw, 0.8, note, size=13.5, color=S.SECONDARY, line=1.35)
    d.rect(s, MX, 6.08, CW, 0.64, fill=S.PANEL, radius=0.12)
    d.text(s, MX + 0.28, 6.08, CW - 0.56, 0.64,
           [[("核心跨越：", {"bold": True, "color": S.INK}),
             ("教材里的坐标是固定的或线性的；", {}),
             ("深度聚类把坐标本身变成了可学习的。", {"bold": True, "color": S.DEEP})]],
           size=16, color=S.BODY, anchor="m")


def s04_pixel_distance(d: Deck):
    f = d.facts
    s = d.slide("S04_pixel_distance", SECTION["stat"], "K-means 在像素空间里量错了距离",
                source=f"Fashion-MNIST 训练集，像素归一化到 [0, 1]；随机抽取 {f.get('pix_n_shirts', 1000):,} 张 T 恤统计")
    d.text(s, MX, 1.72, 5.0, 0.3, "K-means 的目标函数", size=13, color=S.SECONDARY)
    d.equation(s, "kmeans", MX, 2.06, scale=0.88)
    d.text(s, MX, 3.04, 5.1, 0.35,
           [[("x", {"italic": True, "font": "Cambria Math"}), ("i", {"sub": True, "italic": True}),
             ("：第 i 张图（784 个像素值）　", {}),
             ("μ", {"italic": True, "font": "Cambria Math"}), ("k", {"sub": True, "italic": True}),
             ("：第 k 组中心", {})]],
           size=12.5, color=S.SECONDARY)
    d.text(s, MX, 3.56, 4.95, 1.9,
           "把每张图分给最近的中心。它默认：**欧氏距离近 = 同一类。**\n"
           "但图像拉成 784 维像素向量后，这个距离主要量的是物体位置、亮度和背景，而不是“这是什么”。",
           size=15.5, color=S.BODY, line=1.48, after=10)
    # image triplet
    x0, y0, side = 6.15, 1.78, 1.72
    names = [("pixel_anchor", "原图：T 恤 A"), ("pixel_moved", f"同件右移 {f.get('pix_shift', 4)} 像素"),
             ("pixel_other", f"另一类：{f.get('pix_other_class', '衬衫')} B")]
    xs = [x0, x0 + 2.25, x0 + 4.5]
    for (name, label), x in zip(names, xs):
        d.image(s, ASSETS / f"{name}.png", x, y0, w=side, h=side)
        d.text(s, x - 0.25, y0 + side + 0.08, side + 0.5, 0.35, label, size=13, color=S.BODY, align="c")
    # color-contrasted dimension lines so 11.3 > 4.8 hits immediately
    def dim(xa, xb, y, label, value, suffix, accent):
        d.line(s, xa, y, xb, y, color=accent, width=1.8)
        d.line(s, xa, y - 0.08, xa, y + 0.08, color=accent, width=1.8)
        d.line(s, xb, y - 0.08, xb, y + 0.08, color=accent, width=1.8)
        d.text(s, (xa + xb) / 2 - 2.0, y - 0.45, 4.0, 0.34,
               [[(label, {"color": S.SECONDARY}), (value, {"bold": True, "color": accent}),
                 (suffix, {"bold": True, "color": accent})]],
               size=14, color=S.INK, align="c", anchor="b")
    yb = y0 + side + 1.00
    dim(xs[0] + side / 2, xs[1] + side / 2, yb, "同类错位距离 ", _f(f.get("pix_d_moved"), 1), "（反而更远！）",
        S.PRETRAINED)
    dim(xs[0] + side / 2, xs[2] + side / 2, yb + 0.68, "不同类距离 ", _f(f.get("pix_d_other"), 1), "（反而更近！）",
        S.DEEP)
    frac = f.get("pix_fraction_farther")
    d.stat(s, x0, 5.50, 6.5, DASH if frac is None else f"{frac:.1%}",
           f"随机抽取的 T 恤右移 {f.get('pix_shift', 4)} 像素后，离原图比离最近的“非 T 恤”图像还远",
           value_size=34, label_size=14)


def s05_core_idea(d: Deck):
    s = d.slide("S05_core_idea", SECTION["stat"], "深度聚类：先学一个让距离有意义的表示")
    y, h = 1.80, 1.22
    xs = [MX, 3.25, 5.85, 8.45, 10.85]
    labels = [("图像 x", "784 / 3072 维像素"), (["编码器 f", ("θ", {"sub": True})], "先“看懂”图的神经网络"),
              ("表示 z", "浓缩特征（约 10 维）"), ("聚类器", "按新坐标划组"), ("簇 ĉ", "输出的类别编号")]
    fills = [S.PANEL, "#E3EEFB", S.PANEL, S.PANEL, S.PANEL]
    widths = [1.95, 1.95, 1.95, 1.95, 1.83]
    for (label, sub), x, fill, w in zip(labels, xs, fills, widths):
        d.box(s, x, y, w, h, label, fill=fill, size=17, sub=sub, sub_size=11.5)
    for i in range(4):
        d.arrow(s, xs[i] + widths[i] + 0.05, y + h / 2, xs[i + 1] - 0.06, y + h / 2, color=S.SECONDARY)
    d.equation(s, "pipeline", W / 2, 3.28, scale=0.88, align="c")
    d.rect(s, MX, 3.98, CW, 0.50, fill=S.PANEL, radius=0.14)
    d.text(s, MX + 0.25, 3.98, CW - 0.5, 0.50,
           [[("直觉：", {"bold": True, "color": S.DEEP}),
             ("像素里两件错位 T 恤相距很远；经过编码器压缩成特征 z 后，同类图像自然聚在一起。", {})]],
           size=14, color=S.BODY, align="c", anchor="m")
    qy = 4.78
    qs = [("没有标签，编码器靠什么信号训练？", "还原原图（重构） · 用高把握预测教自己 · 同图两张变形照对比 · 直接借用大模型"),
          ("编码器和聚类器怎么配合？", "先学后分 · 边学边分 · 交替迭代 · 用潜变量解释图像怎么生成")]
    for i, (q, a) in enumerate(qs):
        x = MX + i * 6.1
        d.text(s, x, qy, 0.5, 0.5, "①②"[i], size=24, bold=True, color=S.DEEP)
        d.text(s, x + 0.55, qy + 0.02, 5.3, 0.45, q, size=17.5, bold=True, color=S.INK)
        d.text(s, x + 0.55, qy + 0.54, 5.3, 0.9, a, size=14, color=S.SECONDARY, line=1.4)
    d.text(s, MX, 6.40, CW, 0.4, "后面所有深度聚类方法，本质上都是这两个问题的不同答案。", size=15, color=S.BODY)


def _mini_node(d, s, x, y, w, h, label, fill, color=S.INK, size=12.5):
    d.box(s, x, y, w, h, label, fill=fill, color=color, size=size, radius=0.18)


def s06_paradigms(d: Deck):
    s = d.slide("S06_paradigms", SECTION["method"], "表示与聚类的四种配合方式长期并存",
                lede="按 Zhou 等（ACM Computing Surveys, 2025）的交互方式分类：不是新淘汰旧，而是四条路线并存。",
                source="实心圆点：本项目实验中运行的方法；空心圆点：仅文献综述。")
    cols = [
        ("多阶段", "先画地图，再划区", "先训好特征并锁死，再独立分组",
         [("AE + K-means", S.DEEP, True), ("SimCLR + K-means", S.CONTRASTIVE, True)]),
        ("联合优化", "边画地图，边划区", "同一个损失同时更新特征与簇中心",
         [("DEC、IDEC", S.DEEP, True), ("CC", S.CONTRASTIVE, True)]),
        ("生成式", "给每类画概率云", "假设每类是一团正态分布再画成图",
         [("VaDE", S.GENERATIVE, True)]),
        ("迭代式", "边猜边教自己", "用本轮分组当伪标签训下一轮特征",
         [("DeepCluster", S.MUTED, False), ("SwAV", S.MUTED, False)]),
    ]
    cw, gap, y0 = 2.85, 0.21, 1.88
    rep, clu = "#E3EEFB", "#ECECEE"
    for i, (name, metaphor, desc, methods) in enumerate(cols):
        x = MX + i * (cw + gap)
        d.rect(s, x, y0, cw, 1.84, fill=S.PANEL, radius=0.05)
        bx, by, bw, bh = x + 0.35, y0 + 0.50, 0.95, 0.48
        if i == 0:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + bh / 2, bx + 1.21, by + bh / 2)
            d.text(s, x, by + bh + 0.14, cw, 0.3, "单向传递，互不干扰", size=11, color=S.SECONDARY, align="c")
        elif i == 1:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + bh / 2 - 0.1, bx + 1.21, by + bh / 2 - 0.1)
            d.arrow(s, bx + 1.21, by + bh / 2 + 0.1, bx + bw + 0.04, by + bh / 2 + 0.1)
            d.text(s, x, by + bh + 0.14, cw, 0.3, "共享一个总损失", size=11, color=S.SECONDARY, align="c")
        elif i == 2:
            for j, lab in enumerate(["c", "z", "x"]):
                d.oval(s, bx + 0.05 + j * 0.8, by, 0.48, rep if lab == "z" else clu)
                d.text(s, bx + 0.05 + j * 0.8, by, 0.48, 0.48, lab, size=14, bold=True, color=S.INK, align="c",
                       anchor="m", italic=True, font="Cambria Math")
                if j < 2:
                    d.arrow(s, bx + 0.56 + j * 0.8, by + 0.24, bx + 0.82 + j * 0.8, by + 0.24)
            d.text(s, x, by + bh + 0.14, cw, 0.3, "类别 c → 特征 z → 图像 x", size=11, color=S.SECONDARY, align="c")
        else:
            _mini_node(d, s, bx, by, bw, bh, "表示", rep)
            _mini_node(d, s, bx + 1.25, by, bw, bh, "聚类", clu)
            d.arrow(s, bx + bw + 0.04, by + 0.14, bx + 1.21, by + 0.14)
            d.path(s, [(bx + 1.72, by + bh + 0.02), (bx + 1.72, by + bh + 0.24), (bx + 0.48, by + bh + 0.24),
                       (bx + 0.48, by + bh + 0.04)])
            d.text(s, x, by + bh + 0.30, cw, 0.3, "猜的伪标签回传", size=11, color=S.SECONDARY, align="c")
        d.text(s, x, y0 + 1.98, cw, 0.40, name, size=19.5, bold=True, color=S.INK)
        d.text(s, x, y0 + 2.40, cw, 0.30, metaphor, size=12.5, bold=True, color=S.DEEP)
        d.text(s, x, y0 + 2.74, cw, 0.58, desc, size=13.5, color=S.BODY, line=1.3)
        for j, (mname, color, ran) in enumerate(methods):
            yy = y0 + 3.36 + j * 0.36
            if ran:
                d.dot(s, x, yy + 0.08, color, d=0.14)
            else:
                d.oval(s, x, yy + 0.08, 0.14, S.SURFACE, line=S.MUTED)
            d.text(s, x + 0.25, yy, cw - 0.28, 0.34, mname, size=13.5, color=S.INK if ran else S.SECONDARY)
    by = 6.12
    d.rect(s, MX, by, CW, 0.66, fill="#FDF0EA", radius=0.12)
    d.dot(s, MX + 0.28, by + 0.255, S.PRETRAINED, d=0.15)
    d.text(s, MX + 0.56, by + 0.01, CW - 0.76, 0.64,
           [[("外部预训练（直接借用现成“识图大脑”）：", {"bold": True, "color": S.INK}),
             ("用海量外部图预训练好特征（DINOv2、CLIP），冻结参数再接简单聚类器；TAC、VLM4Cluster 进一步引入文字语义。", {})]],
           size=14, color=S.BODY, anchor="m")


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
    s = d.slide("S07_dec", SECTION["method"], "DEC：用自己最有把握的预测训练自己",
                source="Xie, Girshick & Farhadi, Unsupervised Deep Embedding for Clustering Analysis, ICML 2016")
    steps = [("① 算归属概率", "Student-t 核：图特征 z 离簇中心 μ 越近，归属概率 q 越高", "dec_q"),
             ("② 造锐化目标", "把 q 平方放大把握；除簇大小 f 防止大簇吞掉小簇", "dec_p"),
             ("③ 自己教自己", "最小化 KL(P‖Q)，逼当前预测 Q 向更笃定的 P 靠拢", "dec_loss")]
    y = 1.62
    for title, desc, eq in steps:
        d.text(s, MX, y, 1.75, 0.36, title, size=16, bold=True, color=S.INK)
        d.text(s, MX + 1.80, y + 0.02, 5.25, 0.36, desc, size=13.5, color=S.SECONDARY)
        _, eh = d.equation(s, eq, MX + 1.80, y + 0.42, scale=0.70)
        y += 0.42 + eh + 0.24
    # Concrete intuitive numeric illustration of squaring/sharpening
    d.rect(s, MX, 5.42, 7.05, 1.32, fill=S.PANEL, radius=0.08)
    d.text(s, MX + 0.22, 5.54, 6.65, 0.32, "直觉小例子：为什么“平方”有效？（数学示意，设两簇等大）",
           size=13, bold=True, color=S.DEEP)
    d.text(s, MX + 0.22, 5.90, 6.65, 0.78,
           "原归属概率 q = [0.70, 0.30]  →  平方再归一得目标 p = [0.84, 0.16]\n"
           "本来较像第 1 类，就被推得更笃定——像“强者愈强”，把松散的簇往中心压紧。",
           size=12.5, color=S.BODY, line=1.42)
    # Right: real two-stage DEC neural network architecture
    d.text(s, 7.92, 1.48, 4.76, 0.28, "DEC 网络架构（阶段① AE 预训练 → 阶段② 弃解码器自训练）",
           size=11.5, bold=True, color=S.DEEP)
    cx = 10.62
    d.box(s, cx - 1.25, 1.82, 2.50, 0.50, "输入图像 x", fill=S.PANEL, size=12.5,
          sub="784 维灰度 / CIFAR 3×32²", sub_size=9.5)
    d.arrow(s, cx, 2.34, cx, 2.54)
    # Encoder container with exact layers
    d.rect(s, cx - 1.52, 2.56, 3.04, 1.12, fill="#EEF5FD", line=S.DEEP, radius=0.08, line_w=1.25)
    d.text(s, cx - 1.44, 2.61, 2.88, 0.26, [[("编码器 f", {}), ("θ", {"sub": True}), ("（联合微调）", {})]],
           size=11, bold=True, color=S.DEEP, align="c")
    d.box(s, cx - 1.38, 2.91, 2.76, 0.31, "MLP: 784 → 500 → 500 → 2000 → 10", fill=S.SURFACE,
          color=S.INK, size=9.5, bold=True, line=S.HAIRLINE, radius=0.14)
    d.box(s, cx - 1.38, 3.28, 2.76, 0.31, "CIFAR: Conv 32→64→128 → FC(10)", fill=S.SURFACE,
          color=S.SECONDARY, size=9.2, bold=False, line=S.HAIRLINE, radius=0.14)
    d.arrow(s, cx, 3.70, cx, 3.92)
    # Bottleneck z + dashed discarded decoder on left
    d.box(s, cx - 0.86, 3.94, 1.72, 0.44, "隐特征 z ∈ ℝ¹⁰", fill="#DCEBFA", color=S.INK,
          size=12, bold=True, line=S.DEEP, line_w=1.5)
    d.line(s, cx - 0.88, 4.16, 9.48, 4.16, color=S.MUTED, width=1.25, arrow=True, dash=True)
    d.box(s, 7.92, 3.80, 1.52, 0.72, "对称解码器", fill=S.SURFACE, color=S.SECONDARY, size=10.5,
          sub="仅预训练用\n阶段②丢弃", sub_size=8.8, line=S.MUTED, dash=True)
    # Clustering layer: learnable centers + Q + P
    d.arrow(s, cx, 4.40, cx, 4.66)
    d.box(s, 7.92, 4.68, 1.56, 0.56, ["簇中心 {μ", ("k", {"sub": True}), "}"], fill="#EEF5FD", color=S.DEEP, size=10.5,
          sub="K×10 · K-means 初始化", sub_size=8.5, line=S.DEEP)
    d.arrow(s, 9.50, 4.96, cx - 0.88, 4.96, color=S.DEEP, width=1.5)
    d.box(s, cx - 0.86, 4.68, 1.72, 0.56, "软分配 Q", fill=S.PANEL, size=11.5,
          sub="Student-t 核", sub_size=9.5, line=S.HAIRLINE)
    d.arrow(s, cx, 5.26, cx, 5.56)
    d.text(s, cx + 0.08, 5.28, 1.35, 0.26, "平方、按簇归一", size=9.2, color=S.SECONDARY)
    d.box(s, cx - 0.86, 5.58, 1.72, 0.50, "锐化目标 P", fill=S.PANEL, size=11.5,
          sub="高置信目标分布", sub_size=9.5, line=S.DEEP)
    # KL feedback loop to encoder & centers
    d.path(s, [(cx + 0.88, 5.83), (12.56, 5.83), (12.56, 3.12), (cx + 1.54, 3.12)], color=S.DEEP, width=1.8)
    d.line(s, cx + 0.88, 4.96, 12.56, 4.96, color=S.DEEP, width=1.2, dash=True)
    d.box(s, 11.56, 4.02, 1.12, 0.56, "KL(P‖Q)", fill=S.DEEP, color="#FFFFFF", size=10.5,
          sub="反向更新", sub_size=8.8, sub_color="#E3EEFB", radius=0.14)
    d.text(s, 7.92, 6.26, 4.76, 0.48, "训练全程不读标签；簇分配变化 < 0.1% 时早停。", size=12,
           color=S.SECONDARY, line=1.35)


def s08_idec(d: Deck):
    s = d.slide("S08_idec", SECTION["method"], "IDEC：保留解码器，防止表示被聚类目标拉坏",
                source="Guo et al., Improved Deep Embedded Clustering with Local Structure Preservation, IJCAI 2017")
    d.equation(s, "idec_loss", MX, 1.75, scale=0.88)
    d.text(s, MX, 2.58, 5.55, 2.1,
           "只用 KL 自训练时，早期分错的样本会被越拉越偏，隐空间容易被扭曲。\n"
           "**可以想成一场拔河：**聚类项 KL 拼命把同类往中心压紧；**重构项拉住编码器，要求 z 仍能画回原图**，系数 λ 控制两者的比重。",
           size=15, color=S.BODY, line=1.5, after=10)
    d.rect(s, MX, 4.78, 5.55, 1.32, fill=S.PANEL, radius=0.08)
    d.text(s, MX + 0.20, 4.86, 5.15, 1.16,
           "实现口径说明：本项目的 λ 乘在每张图的平方重构误差（对像素求和）上；原论文把 γ 乘在聚类项上、重构误差按像素取平均，两者换算为 λ = 1/(γD)，D 为像素数。",
           size=11.8, color=S.SECONDARY, line=1.42, anchor="m")
    # Right: real hourglass dual-branch architecture
    x0 = 6.42
    d.text(s, x0, 1.52, 6.26, 0.28, "IDEC 对称自编码器 + 双分支联合优化架构", size=12, bold=True, color=S.DEEP)
    d.box(s, x0, 3.04, 0.90, 0.80, "图像 x", fill=S.PANEL, size=12.5, sub="784 /\n3×32²", sub_size=9.2)
    d.arrow(s, x0 + 0.92, 3.44, x0 + 1.12, 3.44)
    # Encoder block
    d.rect(s, x0 + 1.14, 2.66, 1.54, 1.56, fill="#EEF5FD", line=S.DEEP, radius=0.08, line_w=1.25)
    d.text(s, x0 + 1.18, 2.72, 1.46, 0.26, [[("编码器 f", {}), ("θ", {"sub": True})]],
           size=11.5, bold=True, color=S.DEEP, align="c")
    d.box(s, x0 + 1.24, 3.02, 1.34, 0.34, "784→500→500", fill=S.SURFACE, color=S.INK, size=8.8, bold=True,
          line=S.HAIRLINE, radius=0.14)
    d.box(s, x0 + 1.24, 3.41, 1.34, 0.32, "→2000 (ReLU)", fill=S.SURFACE, color=S.INK, size=8.8, bold=False,
          line=S.HAIRLINE, radius=0.14)
    d.box(s, x0 + 1.24, 3.79, 1.34, 0.32, "CIFAR: Conv×3", fill=S.SURFACE, color=S.SECONDARY, size=8.5,
          bold=False, line=S.HAIRLINE, radius=0.14)
    d.arrow(s, x0 + 2.70, 3.44, x0 + 2.90, 3.44)
    # Bottleneck z
    d.box(s, x0 + 2.92, 3.04, 0.78, 0.80, "特征 z", fill="#DCEBFA", color=S.INK, size=12, bold=True,
          sub="10 维", sub_size=9.5, line=S.DEEP, line_w=1.6)
    # Split to two branches
    d.path(s, [(x0 + 3.72, 3.26), (x0 + 3.90, 3.26), (x0 + 3.90, 2.34), (x0 + 4.06, 2.34)], color=S.DEEP, width=1.6)
    d.path(s, [(x0 + 3.72, 3.62), (x0 + 3.90, 3.62), (x0 + 3.90, 4.44), (x0 + 4.06, 4.44)], color=S.DEEP, width=1.6)
    # Upper branch: Clustering (KL)
    d.box(s, x0 + 4.08, 1.94, 1.18, 0.80, "软分配 Q", fill=S.PANEL, size=11.5,
          sub="Student-t\n+ 簇中心 {μₖ}", sub_size=8.8, line=S.DEEP)
    d.arrow(s, x0 + 5.28, 2.34, x0 + 5.42, 2.34, color=S.DEEP, width=1.5)
    d.box(s, x0 + 5.44, 1.98, 0.82, 0.72, "KL(P‖Q)", fill=S.DEEP, color="#FFFFFF", size=10,
          sub="压紧簇", sub_size=8.8, sub_color="#E3EEFB")
    # Lower branch: Reconstruction (retained in IDEC, dropped in DEC)
    d.rect(s, x0 + 3.98, 3.76, 2.30, 1.66, fill="#FAFBFD", line=S.DEEP, radius=0.08, line_w=1.2, dash=True)
    d.box(s, x0 + 4.08, 3.90, 1.18, 1.06, ["解码器 g", ("φ", {"sub": True})], fill="#EEF5FD", color=S.INK, size=11.5,
          sub="2000→500→\n500→784\nSigmoid", sub_size=8.4, line=S.DEEP)
    d.arrow(s, x0 + 5.28, 4.43, x0 + 5.42, 4.43, color=S.DEEP, width=1.5)
    d.box(s, x0 + 5.44, 3.98, 0.76, 0.90, "还原 x̂", fill=S.PANEL, color=S.INK, size=10.5,
          sub="λ·SSE\n保细节", sub_size=8.8, sub_color=S.DEEP, line=S.DEEP)
    d.text(s, x0 + 4.02, 5.06, 2.22, 0.30, "IDEC 保留重构；DEC(λ=0) 丢弃", size=9.2, bold=True,
           color=S.DEEP, align="c")
    d.text(s, x0, 5.78, 6.26, 0.55,
           "DEC 是 λ = 0 的特例：解码器只在预训练时用一次，微调时直接丢弃。",
           size=13.5, color=S.BODY, line=1.35)


def s09_vade(d: Deck):
    s = d.slide("S09_vade", SECTION["method"], "VaDE：把高斯混合模型搬进隐空间",
                source="Jiang et al., Variational Deep Embedding, IJCAI 2017 · 本项目实现中重构项采用每图平方误差（SSE），非 Bernoulli 交叉熵。")
    d.text(s, MX, 1.52, 6.5, 0.28, "生成过程（模型怎么“想象”出一张图）", size=12.5, color=S.SECONDARY)
    d.equation(s, "vade_gen", MX, 1.82, scale=0.76)
    d.text(s, MX, 2.34, 6.6, 0.30, "① 按比例 π 抽类别 c  →  ② 在该类正态云抽特征 z  →  ③ 解码画成图 x",
           size=11.8, bold=True, color=S.GENERATIVE)
    d.text(s, MX, 2.74, 6.5, 0.28, "训练目标（证据下界 ELBO）", size=12.5, color=S.SECONDARY)
    d.equation(s, "vade_elbo", MX, 3.04, scale=0.76)
    d.text(s, MX, 3.58, 6.6, 0.30, "直觉：既要“还原得像原图”（前项 SSE），又要“特征贴合高斯混合云”（后项 KL）。",
           size=11.8, color=S.BODY)
    # Real VaDE variational architecture diagram
    d.text(s, MX, 4.08, 6.55, 0.26, "VaDE 变分编码器 + 隐空间 GMM 先验网络架构",
           size=11.5, bold=True, color=S.GENERATIVE)
    # GMM Prior sitting cleanly above z
    d.rect(s, MX + 2.88, 4.40, 2.08, 0.56, fill="#F3E6F1", line=S.GENERATIVE, radius=0.08, line_w=1.4)
    d.text(s, MX + 2.94, 4.40, 1.96, 0.56,
           [[("可学习 GMM 先验", {"bold": True, "size": 11, "color": S.GENERATIVE})],
            [("π", {}), ("c", {"sub": True}), (", μ", {}), ("c", {"sub": True}), (", σ", {}),
             ("c", {"sub": True}), ("²（K=10 团高斯云）", {})]],
           size=8.8, color=S.SECONDARY, align="c", anchor="m", line=1.12)
    d.line(s, MX + 3.92, 4.98, MX + 3.92, 5.28, color=S.GENERATIVE, width=1.6, arrow=True)
    d.text(s, MX + 4.02, 5.00, 1.65, 0.26, [[("KL 正则 & 后验 γ", {}), ("c", {"sub": True})]],
           size=9.2, bold=True, color=S.GENERATIVE)
    # Main horizontal encoder -> (mu, logvar) -> reparam z -> decoder -> x_hat
    ym = 5.66
    d.box(s, MX, ym - 0.34, 0.68, 0.68, "图 x", fill=S.PANEL, size=11.5, sub="784", sub_size=9)
    d.arrow(s, MX + 0.70, ym, MX + 0.84, ym)
    d.box(s, MX + 0.86, ym - 0.44, 1.18, 0.88, "编码器", fill="#F7ECF5", color=S.INK, size=11.5,
          sub="784→500→\n500→2000\n(CIFAR Conv)", sub_size=8.2, line=S.GENERATIVE)
    # Fork to mu & logvar heads
    d.path(s, [(MX + 2.06, ym - 0.14), (MX + 2.18, ym - 0.14), (MX + 2.18, ym - 0.28), (MX + 2.28, ym - 0.28)],
           color=S.GENERATIVE, width=1.4)
    d.path(s, [(MX + 2.06, ym + 0.14), (MX + 2.18, ym + 0.14), (MX + 2.18, ym + 0.28), (MX + 2.28, ym + 0.28)],
           color=S.GENERATIVE, width=1.4)
    d.box(s, MX + 2.30, ym - 0.48, 0.76, 0.40, "均值 μ", fill=S.SURFACE, color=S.INK, size=10,
          sub="10d", sub_size=8.2, line=S.GENERATIVE)
    d.box(s, MX + 2.30, ym + 0.08, 0.76, 0.40, "log σ²", fill=S.SURFACE, color=S.INK, size=10,
          sub="10d", sub_size=8.2, line=S.GENERATIVE)
    # Merge into reparameterized z
    d.path(s, [(MX + 3.08, ym - 0.28), (MX + 3.22, ym - 0.28), (MX + 3.22, ym - 0.10), (MX + 3.34, ym - 0.10)],
           color=S.GENERATIVE, width=1.4)
    d.path(s, [(MX + 3.08, ym + 0.28), (MX + 3.22, ym + 0.28), (MX + 3.22, ym + 0.10), (MX + 3.34, ym + 0.10)],
           color=S.GENERATIVE, width=1.4)
    d.box(s, MX + 3.36, ym - 0.36, 1.12, 0.72, "z = μ+σ⊙ε", fill="#F3E6F1", color=S.INK, size=10.5, bold=True,
          sub="重参数采样 10d", sub_size=8.5, line=S.GENERATIVE, line_w=1.6)
    d.arrow(s, MX + 4.50, ym, MX + 4.66, ym, color=S.GENERATIVE, width=1.5)
    d.box(s, MX + 4.68, ym - 0.44, 1.14, 0.88, "对称解码器", fill="#F7ECF5", color=S.INK, size=11,
          sub="10→2000→\n500→784\nSigmoid", sub_size=8.2, line=S.GENERATIVE)
    d.arrow(s, MX + 5.84, ym, MX + 5.96, ym, color=S.GENERATIVE, width=1.5)
    d.box(s, MX + 5.98, ym - 0.34, 0.64, 0.68, "x̂", fill=S.PANEL, size=12, sub="SSE", sub_size=9, line=S.GENERATIVE)
    d.text(s, MX, 6.28, 6.62, 0.45,
           [[("推断：由编码器得 μ 及每簇后验 γ", {}), ("c", {"sub": True}), (" 判类别；生成：从 GMM 抽 z 经解码器画图。", {})]],
           size=11.5, color=S.SECONDARY, line=1.35)
    # comparison on the right
    x0 = 7.62
    rows = [("与 K-means 相比", "每个簇有自己的方差和先验比例 π，是一团“概率云”，而不只是一个孤立中心点。"),
            ("与像素空间 GMM 相比", "先降到 10 维再建模，避开 784 维像素协方差难以估计的难题。"),
            ("与统计教材的联系", "第 3 章多元正态 + 第 8 章潜变量模型，用神经网络把两者连起来。")]
    for i, (head, body) in enumerate(rows):
        y0 = 1.82 + i * 1.52
        d.rule(s, x0, y0, 5.06, color=S.GENERATIVE if i == 0 else S.HAIRLINE, width=2 if i == 0 else 1)
        d.text(s, x0, y0 + 0.15, 5.06, 0.38, head, size=16, bold=True, color=S.INK)
        d.text(s, x0, y0 + 0.58, 5.06, 0.88, body, size=14.5, color=S.BODY, line=1.42)


def s10_contrastive(d: Deck):
    s = d.slide("S10_contrastive", SECTION["method"], "对比学习：同一张图的两张变形照应当靠近",
                source="Chen et al., SimCLR, ICML 2020 · Li et al., Contrastive Clustering, AAAI 2021")
    x0 = MX
    d.text(s, x0, 1.48, 8.65, 0.28, "SimCLR & CC 孪生网络架构（共享 SmallCNN 骨干 + 双对比头）",
           size=11.8, bold=True, color=S.CONTRASTIVE)
    # Input + Two augmented views
    d.box(s, x0, 2.66, 1.04, 0.78, "图像 x", fill=S.PANEL, size=13, sub="例：猫图", sub_size=9.5)
    d.box(s, x0 + 1.32, 1.82, 1.44, 0.68, "变形视图 x̃₁", fill=S.PANEL, size=12, sub="随机裁剪 + 变亮", sub_size=9.5)
    d.box(s, x0 + 1.32, 3.60, 1.44, 0.68, "变形视图 x̃₂", fill=S.PANEL, size=12, sub="水平翻转 + 调色", sub_size=9.5)
    d.path(s, [(x0 + 1.06, 2.90), (x0 + 1.18, 2.90), (x0 + 1.18, 2.16), (x0 + 1.30, 2.16)])
    d.path(s, [(x0 + 1.06, 3.20), (x0 + 1.18, 3.20), (x0 + 1.18, 3.94), (x0 + 1.30, 3.94)])
    # Real SmallCNN Shared Backbone container
    bx, bw = x0 + 3.08, 2.38
    d.arrow(s, x0 + 2.78, 2.16, bx - 0.04, 2.68, color=S.CONTRASTIVE, width=1.6)
    d.arrow(s, x0 + 2.78, 3.94, bx - 0.04, 3.42, color=S.CONTRASTIVE, width=1.6)
    d.rect(s, bx, 1.80, bw, 2.52, fill="#E8F6F1", line=S.CONTRASTIVE, radius=0.07, line_w=1.4)
    d.text(s, bx + 0.08, 1.86, bw - 0.16, 0.26, "共享骨干 SmallCNN", size=11.5, bold=True,
           color=S.CONTRASTIVE, align="c")
    layers = [
        ("Conv 3×3 (32) · ReLU · Pool", S.SURFACE, S.INK, False),
        ("Conv 3×3 (64) · ReLU · Pool", S.SURFACE, S.INK, False),
        ("Conv 3×3 (128) · ReLU · GAP", S.SURFACE, S.INK, False),
        ("FC → 表征 h₁, h₂ ∈ ℝ¹²⁸", "#D4EFE6", S.INK, True),
    ]
    for idx, (txt, fill, col, is_b) in enumerate(layers):
        d.box(s, bx + 0.12, 2.18 + idx * 0.44, bw - 0.24, 0.36, txt, fill=fill, color=col,
              size=9.5, bold=is_b, line=S.HAIRLINE if not is_b else S.CONTRASTIVE, radius=0.14)
    d.text(s, bx + 0.08, 3.98, bw - 0.16, 0.26, "两视图共享同一套卷积参数", size=9.2,
           color=S.CONTRASTIVE, align="c")
    # Two real heads: Instance MLP + CC Cluster Linear+Softmax
    hx, hw = x0 + 5.76, 2.90
    d.path(s, [(bx + bw + 0.02, 2.72), (hx - 0.16, 2.72), (hx - 0.16, 2.34), (hx - 0.02, 2.34)],
           color=S.CONTRASTIVE, width=1.6)
    d.path(s, [(bx + bw + 0.02, 3.38), (hx - 0.16, 3.38), (hx - 0.16, 3.74), (hx - 0.02, 3.74)],
           color=S.CONTRASTIVE, width=1.6)
    d.box(s, hx, 1.80, hw, 1.10, "实例头（行对比 · SimCLR & CC）", fill=S.PANEL, color=S.INK, size=11,
          sub="MLP: 128 → 128 → ReLU → 128\nNT-Xent (τ=0.5)：同图拉近、异图推远", sub_size=9.2,
          line=S.CONTRASTIVE)
    d.box(s, hx, 3.20, hw, 1.12, "簇头（列对比 · 仅 CC）", fill=S.PANEL, color=S.INK, size=11,
          sub="Linear: 128 → K(10) + Softmax\n列对比 (τ=1.0)：同簇两视图一致", sub_size=9.2,
          line=S.CONTRASTIVE, dash=True)
    # NT-Xent formula + plain intuition below
    d.equation(s, "nt_xent", x0 + 0.05, 4.98, scale=0.80)
    d.text(s, x0 + 4.10, 5.02, 4.50, 1.05,
           [[("NT-Xent 直觉：", {"bold": True, "color": S.INK}),
             ("分子为同图两视图相似度（拉大），分母为与其他所有图的相似度（压小）。", {})]],
           size=13, color=S.SECONDARY, line=1.42)
    xr = 9.55
    rows = [("SimCLR + K-means", S.CONTRASTIVE, "多阶段：只用实例头学特征 h，最后离线做 K-means。"),
            ("CC（对比聚类）", S.CONTRASTIVE, "联合优化：实例头与簇头联合训练，直接输出簇分配。")]
    for i, (name, color, body) in enumerate(rows):
        y = 1.85 + i * 1.54
        d.dot(s, xr, y + 0.10, color, d=0.16)
        d.text(s, xr + 0.28, y, 2.9, 0.38, name, size=15.5, bold=True, color=S.INK)
        d.text(s, xr + 0.28, y + 0.44, 2.9, 0.96, body, size=13.5, color=S.BODY, line=1.38)
    d.rect(s, xr, 5.00, 3.13, 1.50, fill=S.PANEL, radius=0.06)
    d.text(s, xr + 0.18, 5.08, 2.78, 1.34,
           "实验算力限制：两者均用 128 维小型 CNN、训练 100 轮（非原论文 ResNet）；CC 未加簇熵正则，因此对比路线被低估。",
           size=11.8, color=S.SECONDARY, line=1.4, anchor="m")


def s11_pretrained(d: Deck):
    s = d.slide("S11_pretrained", SECTION["method"], "预训练模型把“学表示”交给了海量外部数据",
                source="Oquab et al., TMLR 2024 · Radford et al., ICML 2021 · Li et al., ICML 2024 · Hu et al., 2026（预印本）")
    # Real Pretrained ViT + L2 Norm + 3 parallel downstream clusterers architecture
    d.box(s, MX, 1.84, 2.02, 1.24, "输入图像", fill=S.PANEL, size=13.5,
          sub="MNIST / Fashion / CIFAR-10\n转 RGB · Resize 224²", sub_size=9.5)
    d.arrow(s, MX + 2.04, 2.46, MX + 2.28, 2.46)
    # Frozen ViT backbone container
    bx = MX + 2.30
    d.rect(s, bx, 1.56, 3.44, 1.80, fill="#FDF0EA", line=S.PRETRAINED, radius=0.07, line_w=1.4, dash=True)
    d.text(s, bx + 0.10, 1.63, 3.24, 0.26, "冻结预训练骨干（参数锁死不重训）", size=11, bold=True,
           color=S.PRETRAINED, align="c")
    d.box(s, bx + 0.14, 1.98, 3.16, 0.58, "DINOv2 ViT-S/14  →  384 维", fill=S.SURFACE, color=S.INK, size=11.5,
          sub="LVD-142M 无标签自然图自监督", sub_size=9.2, line=S.PRETRAINED)
    d.box(s, bx + 0.14, 2.66, 3.16, 0.58, "CLIP ViT-B/32   →  512 维", fill=S.SURFACE, color=S.INK, size=11.5,
          sub="OpenAI 4 亿图文对对比对齐", sub_size=9.2, line=S.PRETRAINED)
    d.arrow(s, bx + 3.46, 2.46, bx + 3.70, 2.46, color=S.PRETRAINED, width=1.6)
    # L2 normalization
    nx = bx + 3.72
    d.box(s, nx, 1.94, 1.48, 1.04, "L2 归一化", fill=S.PANEL, color=S.INK, size=12.5,
          sub="单位球面特征\n384 / 512 维", sub_size=9.5, line=S.PRETRAINED)
    # Fork to 3 parallel downstream clusterers (matching E04)
    cx = nx + 1.82
    cw_c = W - MX - cx
    ys_c = [1.56, 2.21, 2.86]
    d.line(s, nx + 1.50, 2.46, nx + 1.66, 2.46, color=S.SECONDARY, width=1.4)
    d.line(s, nx + 1.66, ys_c[0] + 0.25, nx + 1.66, ys_c[2] + 0.25, color=S.SECONDARY, width=1.4)
    heads = [
        ("● K-means", "直接划组（无额外参数）", "#FDF0EA", S.PRETRAINED),
        ("■ Diag-GMM", "标准化 + 对角协方差混合", S.PANEL, S.HAIRLINE),
        ("▲ DEC 聚类头", "Linear D→64→ReLU→10 + KL", S.PANEL, S.HAIRLINE),
    ]
    for yc, (hname, hsub, hfill, hline) in zip(ys_c, heads):
        d.arrow(s, nx + 1.66, yc + 0.25, cx - 0.04, yc + 0.25, color=S.SECONDARY, width=1.4)
        d.rect(s, cx, yc, cw_c, 0.50, fill=hfill, line=hline, radius=0.14)
        d.text(s, cx + 0.14, yc, 1.38, 0.50, hname, size=11, bold=True, color=S.INK, anchor="m")
        d.text(s, cx + 1.48, yc, cw_c - 1.56, 0.50, hsub, size=9.8, color=S.SECONDARY, anchor="m")
    # 4 literature & method cards below
    cards = [("DINOv2", "纯视觉自监督", "1.42 亿无标签图自监督训练，纯视觉学物体语义。"),
             ("CLIP", "图文对比对齐", "4 亿对图文配对训练，让视觉特征与自然语言对齐。"),
             ("TAC", "外部名词引导", "ICML 2024：引入 WordNet 名词语义，辅助图像聚类。"),
             ("VLM4Cluster", "2026 预印本基准", "在 20 个数据集上系统比较 17 种方法的效果、鲁棒性与成本。")]
    cw, gap = 2.85, 0.21
    d.text(s, MX, 3.60, 4.0, 0.26, "本项目实验中使用（冻结提特征）", size=11, bold=True, color=S.PRETRAINED)
    d.text(s, MX + 2 * (cw + gap), 3.60, 4.0, 0.26, "文献前沿方向", size=11, color=S.SECONDARY)
    for i, (name, role, body) in enumerate(cards):
        x = MX + i * (cw + gap)
        d.rule(s, x, 3.90, cw, color=S.PRETRAINED if i < 2 else S.HAIRLINE, width=2 if i < 2 else 1)
        d.text(s, x, 4.04, cw, 0.36, name, size=16.5, bold=True, color=S.INK)
        d.text(s, x, 4.40, cw, 0.26, role, size=11, bold=True, color=S.PRETRAINED if i < 2 else S.SECONDARY)
        d.text(s, x, 4.72, cw, 1.12, body, size=13, color=S.BODY, line=1.36)
    d.rect(s, MX, 6.02, CW, 0.68, fill=S.PANEL, radius=0.12)
    d.text(s, MX + 0.30, 6.02, CW - 0.60, 0.68,
           [[("公平口径：", {"bold": True, "color": S.INK}),
             ("预训练模型在实验前已见过海量外部图像，训练成本不可与从零训练直接比较，因此单独分组报告。", {})]],
           size=13.8, color=S.BODY, anchor="m")


def s11b_architectures(d: Deck):
    f = d.facts
    s = d.slide("S11b_architectures", SECTION["method"], "各模型网络架构总览：从像素统计到预训练骨干",
                source="参数量为 Fashion-MNIST（MLP 784 维）与 CIFAR-10（CNN 3×32²）实测均值，读取自 results/tables/e01_summary.csv。")
    # Column headers
    cols = [
        (MX, 1.92, "家族与方法"),
        (2.68, 1.48, "输入处理"),
        (4.36, 3.24, "核心表示网络（编码器 / 骨干）"),
        (7.80, 2.48, "特征与输出头"),
        (10.48, 2.20, "聚类目标 & 实测参数"),
    ]
    for cx, cw, title in cols:
        d.text(s, cx, 1.38, cw, 0.26, title, size=11.5, bold=True, color=S.SECONDARY)
    d.rule(s, MX, 1.66, CW, color=S.HAIRLINE, width=1.2)
    lanes = [
        (S.CLASSICAL, "#F4F4F5", "经典方法", "K-means · PCA · GMM",
         "拉平像素\n784 / 3072 维",
         "无神经网络", "原像素 / PCA(10) / 标准化",
         "784 或 10 维向量", "直接度量欧氏/高斯距离",
         "最近中心 / 对角 GMM",
         f"参数 {f.k('params_fashion_mnist_kmeans')}–{f.k('params_fashion_mnist_gmm')}"),
        (S.DEEP, "#EEF5FD", "自编码器系", "AE+KM · DEC · IDEC",
         "原始图像\n1×28² / 3×32²",
         "MLP: 784→500→500→2000→10", "CIFAR-10: Conv 32→64→128→10",
         "z ∈ ℝ¹⁰ + 对称解码器", "DEC 微调丢弃 / IDEC 保留",
         "K-means / KL [+ λ·SSE]",
         f"MLP {f.k('params_fashion_mnist_dec')} · CNN {f.k('params_cifar10_dec')}"),
        (S.GENERATIVE, "#F7ECF5", "生成式", "VaDE",
         "原始图像\n1×28² / 3×32²",
         "同上编码器 → (μ, log σ²) → 重参数", "每簇对应隐空间高斯云 N(μₖ, σₖ²I)",
         "z ∈ ℝ¹⁰ + 解码器 + GMM 先验", "同时学习各簇先验 πₖ, μₖ, σₖ²",
         "ELBO = 每图 SSE + KL",
         f"MLP {f.k('params_fashion_mnist_vade')} · CNN {f.k('params_cifar10_vade')}"),
        (S.CONTRASTIVE, "#E8F6F1", "对比学习", "SimCLR+KM · CC",
         "两份随机增强\n裁剪 / 翻转 / 亮度",
         "共享 SmallCNN: Conv 32→64→128", "GAP → FC(128)（非 ResNet）",
         "h ∈ ℝ¹²⁸ + 实例 MLP(128)", "CC 加簇头 Linear 128→10",
         "NT-Xent [+ 列对比]",
         f"仅 ~{f.k('params_fashion_mnist_simclr_kmeans')}（小 CNN）"),
        (S.PRETRAINED, "#FDF0EA", "外部预训练", "DINOv2 · CLIP（冻结）",
         "转 RGB · 224²\nImageNet 归一",
         "冻结 ViT-S/14 (DINOv2) / ViT-B/32", "参数锁死不重训，仅前向提特征",
         "384 / 512 维 + L2 归一", "外接 K-means / GMM / DEC 头",
         "冻结提特征 + 独立聚类",
         "外部预训练（下游极轻）"),
    ]
    y0, lh, gap = 1.76, 0.90, 0.11
    for i, (color, tint, fam, mlist, inp, enc_top, enc_sub, head_top, head_sub, loss_top, param_sub) in enumerate(lanes):
        y = y0 + i * (lh + gap)
        # Subtle row container + left family stripe
        d.rect(s, MX, y, CW, lh, fill=S.SURFACE, line=S.HAIRLINE, radius=0.05, line_w=0.8)
        d.rect(s, MX, y, 0.09, lh, fill=color)
        # Col 1: Family & models
        d.text(s, MX + 0.18, y + 0.14, 1.72, 0.30, fam, size=13, bold=True, color=color)
        d.text(s, MX + 0.18, y + 0.46, 1.72, 0.34, mlist, size=9.8, color=S.BODY, line=1.18)
        # Col 2: Input box
        d.box(s, 2.68, y + 0.12, 1.48, lh - 0.24, inp.split("\n")[0], fill=S.PANEL, color=S.INK, size=10.5,
              sub=inp.split("\n")[1], sub_size=8.8)
        d.arrow(s, 4.18, y + lh / 2, 4.34, y + lh / 2, color=color, width=1.4)
        # Col 3: Backbone / encoder box
        d.box(s, 4.36, y + 0.10, 3.24, lh - 0.20, enc_top, fill=tint, color=S.INK, size=10.5,
              sub=enc_sub, sub_size=8.8, line=color, line_w=1.1, dash=(i == 4))
        d.arrow(s, 7.62, y + lh / 2, 7.78, y + lh / 2, color=color, width=1.4)
        # Col 4: Latent & heads box
        d.box(s, 7.80, y + 0.10, 2.48, lh - 0.20, head_top, fill=S.PANEL, color=S.INK, size=10.2,
              sub=head_sub, sub_size=8.6, line=S.HAIRLINE)
        d.arrow(s, 10.30, y + lh / 2, 10.46, y + lh / 2, color=color, width=1.4)
        # Col 5: Objective & measured params
        d.box(s, 10.48, y + 0.10, 2.12, lh - 0.20, loss_top, fill=S.SURFACE, color=S.INK, size=10.0,
              sub=param_sub, sub_size=8.8, sub_color=color, line=color, line_w=0.9)


def s12_setup(d: Deck):
    f = d.facts
    s = d.slide("S12_setup", SECTION["exp"], "实验设计：3 个数据集、11 种方法，标签只在评分时使用")
    rows = [("mnist", "MNIST", "手写数字 · 28×28 灰度", "训练 2 万 / 测试 1 万"),
            ("fashion_mnist", "Fashion-MNIST", "服饰 · 28×28 灰度", "训练 2 万 / 测试 1 万"),
            ("cifar10", "CIFAR-10", "自然图像 · 32×32 彩色", "训练 1 万 / 测试 2 千")]
    y = 1.78
    for key, name, desc, size in rows:
        d.text(s, MX, y, 2.3, 0.34, name, size=16, bold=True, color=S.INK)
        d.text(s, MX, y + 0.36, 2.3, 0.28, desc, size=11.5, color=S.SECONDARY)
        d.text(s, MX, y + 0.64, 2.3, 0.28, size, size=11.5, color=S.SECONDARY)
        d.image(s, ASSETS / f"strip_{key}.png", MX + 2.35, y + 0.04, w=4.9)
        y += 1.20
    x0 = 8.25
    d.text(s, x0, 1.78, 4.4, 0.32, "方法（按家族着色，全篇一致）", size=13, color=S.SECONDARY)
    fams = [(S.CLASSICAL, "经典方法", "K-means · PCA + K-means · GMM"),
            (S.DEEP, "自编码器系", "AE + K-means · DEC · IDEC"),
            (S.GENERATIVE, "生成式", "VaDE"),
            (S.CONTRASTIVE, "对比学习", "SimCLR + K-means · CC"),
            (S.PRETRAINED, "外部预训练", "DINOv2 · CLIP（冻结）")]
    for i, (color, fam, names) in enumerate(fams):
        yy = 2.22 + i * 0.64
        d.dot(s, x0, yy + 0.08, color, d=0.16)
        d.text(s, x0 + 0.28, yy, 1.45, 0.34, fam, size=14, bold=True, color=S.INK)
        d.text(s, x0 + 1.66, yy + 0.01, 2.78, 0.56, names, size=12.5, color=S.BODY, line=1.2)
    py = 5.50
    d.rule(s, MX, py, CW, color=S.HAIRLINE, width=1)
    rules = [
        ("训练绝不读标签",
         "fit 接口只输入图像；真实标签仅在最后评分（ACC / NMI / ARI）时读取。"),
        ("怎么看核心指标 ARI",
         "0 = 随机乱分，0.5 = 明显分出大类，1.0 = 与真实分类完全一致（后续图表主看 ARI）。"),
        ("先锁协议 · 3 种子均值",
         "超参数与早停规则事先写入配置；主实验跑 3 个随机种子报均值±标准差"
         + (f"（共 {f['n_fits']:,} 次训练）。" if isinstance(f.get("n_fits"), int) else "。")),
    ]
    for i, (head, body) in enumerate(rules):
        x = MX + i * 4.05
        d.text(s, x, py + 0.18, 3.85, 0.34, head, size=14.5, bold=True, color=S.DEEP if i == 1 else S.INK)
        d.text(s, x, py + 0.56, 3.85, 0.88, body, size=12.5, color=S.SECONDARY, line=1.38)


def s13_benchmark(d: Deck):
    f = d.facts
    s = d.slide("S13_benchmark", SECTION["exp"], "从零训练：DEC 与 VaDE 领先，CIFAR-10 上全部失效",
                source="ARI（调整兰德指数）：0 ≈ 随机乱分，1 = 与真实类别完全一致。完整 ACC / NMI / ARI 见附录 A01。")
    d.family_legend(s, MX, 1.26)
    d.figure(s, "benchmark_ari", MX + 0.16, 1.56, 11.70)
    d.text(s, MX, 6.36, CW, 0.50,
           [[("读图总结：", {"bold": True, "color": S.INK}),
             ("左→右图像变复杂。灰度图上自编码器/生成式明显领先；彩色自然图 CIFAR-10 上从零训练全部 ≤ ", {}),
             (f.f("max_scratch_cifar10"), {"bold": True, "color": S.INK}),
             ("（近乎随机）。", {})]],
           size=14, color=S.BODY)


def s14_geometry(d: Deck):
    s = d.slide("S14_geometry", SECTION["exp"], "非线性表示先把类拉开，DEC 再把每个簇压紧",
                lede="用 t-SNE 把高维特征压到二维平面观察：每个点是一张服饰图，颜色是模型自己分出的组（不读真实标签）。",
                source="Fashion-MNIST 测试集 2,000 张图（种子 0）；AE + K-means、DEC、IDEC 共用同一个预训练自编码器。")
    d.figure(s, "embeddings_tsne", MX + 0.12, 1.82, 11.80)
    d.text(s, MX, 6.02, CW, 0.72,
           [[("从左到右：", {"bold": True, "color": S.INK}),
             ("① PCA 线性降维各簇粘连 → ② 自编码器拉开大类 → ③ DEC 压成紧凑“孤岛” → ④ IDEC 为保留原图，簇更松散。", {})]],
           size=14, color=S.BODY, line=1.42)


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
    d.text(s, MX, 6.42, CW, 0.44,
           [[("读图诀窍：", {"bold": True, "color": S.DEEP}),
             ("上下跨行比（换“看图方式”）跨度极大，", {"bold": True, "color": S.INK}),
             ("左右同行比（换“分组算法”）挤在一起——选对特征远比换聚类器重要。", {})]],
           size=14, color=S.BODY)


def s16_pixels_vs_dinov2(d: Deck):
    f = d.facts
    s = d.slide("S16_pixels_vs_dinov2", SECTION["exp"], "同一批 CIFAR-10 图像：像素空间按颜色排，DINOv2 空间按物体排",
                source="300 张测试图（种子 0）按各自表示的 t-SNE 坐标放进 20 × 15 网格；排列只用特征，不用任何标签。")
    pw = 5.85
    panels = [
        ("mosaic_pixels", "像素空间", "按颜色与明暗排：蓝天飞机和蓝海轮船混成一团", f.f("ari_cifar10_kmeans", 2)),
        ("mosaic_dinov2", "DINOv2 表示空间", "按物体语义排：猫狗、飞机船、车、马鹿各自成片",
         f.f("e04_cifar10_dinov2_kmeans", 2)),
    ]
    for i, (name, head, hint, ari) in enumerate(panels):
        x = MX + i * (pw + CW - 2 * pw)
        d.image(s, ASSETS / f"{name}.png", x, 1.66, w=pw)
        d.text(s, x, 6.12, pw - 2.4, 0.36, head, size=17, bold=True, color=S.INK)
        d.text(s, x + pw - 2.6, 6.12, 2.6, 0.36, [[("K-means ARI ", {"color": S.SECONDARY}), (ari, {"bold": True})]],
               size=17, color=S.INK, align="r")
        d.text(s, x, 6.50, pw, 0.34, hint, size=12.5, color=S.SECONDARY)


def s17_noise(d: Deck):
    f = d.facts
    title = f"加噪后排名反转：K-means 保留 {f.p('keep_kmeans')}，DEC 只剩 {f.p('keep_dec')}"
    s = d.slide("S17_noise", SECTION["exp"], title,
                source="噪声只加在测试图上（截断到 [0, 1]），模型不重新训练；Fashion-MNIST，3 个随机种子均值。")
    d.figure(s, "noise_robustness", MX - 0.1, 1.62, 7.55)
    x0 = 8.35
    d.text(s, x0, 1.75, 4.3, 0.3, "同一张测试图在不同雪花噪声强度下", size=12.5, color=S.SECONDARY)
    for i, std in enumerate([0.0, 0.1, 0.2, 0.3]):
        x = x0 + i * 1.07
        d.image(s, ASSETS / f"noise_{str(std).replace('.', 'p')}.png", x, 2.14, w=0.92, h=0.92)
        d.text(s, x, 3.10, 0.92, 0.28, f"σ = {std:g}", size=11.5, color=S.SECONDARY, align="c")
    d.text(s, x0, 3.68, 4.35, 1.48,
           [[("深度特征更怕噪点：", {"bold": True, "color": S.INK}),
             ("干净图上越强的深度方法，加噪后掉得越惨；而均匀随机噪点对所有像素一视同仁，到各簇中心的距离同增，所以 K-means 挑出的“最近中心”基本不变。", {})]],
           size=13.5, color=S.BODY, line=1.44)
    gmm_k = f.get("gmm_clusters_at_01")
    d.text(s, x0, 5.26, 4.35, 1.45,
           [[("像素 GMM 一碰噪就全挤进 1 簇：", {"bold": True, "color": S.INK}),
             (f"边缘黑背景在训练集里几乎不变（方差近 0），标准化后微小噪点被急剧放大，σ = 0.1 时全部测试图落进 {gmm_k:.0f} 个簇。"
              if gmm_k else "边缘黑背景在训练集里几乎不变（方差近 0），标准化后微小噪点被急剧放大，全部测试图落进同一个簇。", {})]],
           size=13.5, color=S.BODY, line=1.44)


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
    x0, cw = 8.35, 4.33
    # Card 1: small sample (K-means leads)
    d.rect(s, x0, 1.78, cw, 1.38, fill=S.PANEL, radius=0.08)
    d.text(s, x0 + 0.22, 1.90, cw - 0.44, 0.30,
           f"{n:,.0f} 张训练图（1% 样本）· K-means 占优" if n else "1% 样本 · K-means 占优",
           size=12.5, bold=True, color=S.CLASSICAL_DARK)
    d.text(s, x0 + 0.22, 2.28, cw - 0.44, 0.72,
           [[("K-means ", {"size": 14, "color": S.CLASSICAL_DARK}),
             (f.f("sf_kmeans_0.01"), {"size": 27, "bold": True, "color": S.CLASSICAL_DARK}),
             ("   vs   ", {"size": 13, "color": S.MUTED}),
             ("DEC ", {"size": 14, "color": S.DEEP}),
             (f.f("sf_dec_0.01"), {"size": 27, "bold": True, "color": S.DEEP})]],
           line=1.0, anchor="m")
    # Card 2: full sample (DEC leads)
    d.rect(s, x0, 3.36, cw, 1.38, fill="#EEF5FD", radius=0.08)
    d.text(s, x0 + 0.22, 3.48, cw - 0.44, 0.30, "全部 2 万张训练图（100% 样本）· DEC 反超",
           size=12.5, bold=True, color=S.DEEP)
    d.text(s, x0 + 0.22, 3.86, cw - 0.44, 0.72,
           [[("DEC ", {"size": 14, "color": S.DEEP}),
             (f.f("sf_dec_1.0"), {"size": 27, "bold": True, "color": S.DEEP}),
             ("   vs   ", {"size": 13, "color": S.MUTED}),
             ("K-means ", {"size": 14, "color": S.CLASSICAL_DARK}),
             (f.f("sf_kmeans_1.0"), {"size": 27, "bold": True, "color": S.CLASSICAL_DARK})]],
           line=1.0, anchor="m")
    d.text(s, x0, 5.02, cw, 1.50,
           [[("原因直觉：", {"bold": True, "color": S.INK}),
             ("DEC 依赖自编码器先把“看图方式”学好；样本太少时预训练没学透，后续自训练反而会把初始错误越放越大。", {})]],
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
    d.figure(s, "components_lambda", MX, 1.56, CW)
    lines = []
    if None not in (full, frozen, nopre, rnd):
        lines.append([("左（谁功劳最大）：", {"bold": True, "color": S.INK}),
                      ("去掉 AE 预训练 −", {}), (f"{full - nopre:.2f}", {"bold": True, "color": S.INK}),
                      ("，随机初始化 −", {}), (f"{full - rnd:.2f}", {"bold": True, "color": S.INK}),
                      ("，锁死编码器仅 −", {}), (f"{full - frozen:.2f}", {"bold": True, "color": S.INK}),
                      (" → 好起点比联合微调更关键。", {"bold": True, "color": S.DEEP})])
    if f.get("lam_fx_gap_1") is not None and f.get("lam_fx_gap_10") is not None:
        lines.append([("右（重构项有帮倒忙吗）：", {"bold": True, "color": S.INK}),
                      ("固定 50 轮后，λ ≤ 0.1 与 DEC 持平；λ = 1、10 平均低 ", {}),
                      (f"{f['lam_fx_gap_1']:.2f}", {"bold": True, "color": S.INK}), ("、", {}),
                      (f"{f['lam_fx_gap_10']:.2f}", {"bold": True, "color": S.INK}),
                      (f"（{f.get('lam_fx_seeds', 3)} 种子全降）→ 过强重构阻碍聚拢。" if recon_hurts else "。", {})])
    if lines:
        d.text(s, MX, 6.10, CW, 0.85, lines, size=13.5, color=S.BODY, line=1.42)


def s20_cost(d: Deck):
    f = d.facts
    ratio, gain = f.get("dec_time_ratio"), f.get("dec_ari_gain")
    title = (f"DEC 多花约 {ratio:.0f} 倍训练时间，换来 ARI +{gain:.2f}" if ratio and gain is not None
             else "精度与训练成本")
    s = d.slide("S20_cost", SECTION["exp"], title,
                source="Fashion-MNIST，3 个随机种子均值；时间为同一台笔记本 GPU 上的端到端训练时间，DEC/IDEC 计入自编码器预训练。")
    d.figure(s, "cost_frontier", MX - 0.1, 1.62, 7.55)
    x0 = 8.4
    d.text(s, x0, 1.78, 4.25, 1.08,
           [[("黑线 = 性价比最优线（帕累托前沿）：", {"bold": True, "color": S.INK}),
             ("想比线上方法更准，就必须多花训练时间。", {})]], size=14, color=S.BODY, line=1.45)
    d.text(s, x0, 2.86, 4.25, 0.28, "最优线上的选择（耗时 · ARI）", size=12, color=S.SECONDARY)
    y = 3.22
    for m in f.get("front_methods") or []:
        d.dot(s, x0, y + 0.10, S.FAMILY.get(m, S.MUTED), d=0.15)
        d.text(s, x0 + 0.28, y, 2.25, 0.34, S.LABEL.get(m, m), size=14, bold=True, color=S.INK)
        d.text(s, x0 + 2.45, y, 1.80, 0.34, f"{f.f('time_' + m, 1)} 秒 · {f.f('cost_ari_' + m)}", size=13,
               color=S.SECONDARY, align="r")
        y += 0.45
    d.rect(s, x0, y + 0.22, 4.25, 1.08, fill=S.PANEL, radius=0.08)
    d.text(s, x0 + 0.20, y + 0.30, 3.85, 0.92,
           "对比学习路线耗时最长、分数最低：课程算力下用小 CNN 训 100 轮，不足以学出语义特征。",
           size=12.5, color=S.SECONDARY, line=1.4, anchor="m")


def s21_stability(d: Deck):
    f = d.facts
    sel_ok = all(f.get(f"stable_{ds}") for ds in ("mnist", "fashion_mnist", "cifar10"))
    s = d.slide("S21_stability", SECTION["exp"], "没有标签时，“稳定”选不出“正确”",
                source="稳定性：训练图加 σ = 0.1 噪声前后，簇成员的平均 Jaccard 一致度（种子 0，不读任何标签）。ARI 用测试标签事后核对。")
    if sel_ok:
        d.family_legend(s, MX, 1.24)
        d.figure(s, "stability_vs_ari", MX + 0.08, 1.54, 11.88)
        d.text(s, MX, 6.08, CW, 0.78,
               [[("反例结论：", {"bold": True, "color": S.DEEP}),
                 ("三个数据集上，最“稳”（实线圈）的都是经典方法，最“准”（虚线圈）的都是深度方法。", {"bold": True, "color": S.INK}),
                 ("“每次分得一样” ≠ “分得对”：稳定性只能筛掉崩塌模型，不能代替少量人工抽检。", {})]],
               size=14, color=S.BODY, line=1.42)
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
    return f"在冻结 DINOv2 / CLIP 特征上（共 {n} 组），{gmm}；{head}。"


def s22_verdict(d: Deck):
    f = d.facts
    s = d.slide("S22_verdict", SECTION["end"], "三个命题的回答")
    rows = [
        ("表示比聚类器更重要", "成立",
         [("特征影响远超聚类器：", {"bold": True, "color": S.INK}),
          (f"换表示让 ARI 最多差 {f.f('encoder_range_max', 2)}，换聚类器平均仅差约 {f.f('clusterer_range_mean_max', 2)}。"
           f"但要选对领域：DINOv2 在自然图 CIFAR-10 达 {f.f('e04_cifar10_dinov2_kmeans')}，"
           f"在手写数字 MNIST 仅 {f.f('e04_mnist_dinov2_kmeans')}（低于本域自编码器 {f.f('ari_mnist_ae_kmeans')}）。", {})]),
        ("联合优化有条件地有效", "有条件",
         [("仅在干净大样本下占优：", {"bold": True, "color": S.INK}),
          (f"干净充足数据上 DEC 最好（MNIST {f.f('ari_mnist_dec')}）；"
           f"但加噪 σ = {f.get('noise_max', 0.3):g} 时仅保留 {f.p('keep_dec')}（K-means 留 {f.p('keep_kmeans')}），"
           f"仅 1% 训练图时反被 K-means 反超。", {})]),
        ("有了预训练表示，简单聚类就够了", "成立",
         [("无需堆复杂聚类头：", {"bold": True, "color": S.INK}),
          (_pretrained_heads_text(f), {})]),
    ]
    y = 1.75
    for i, (claim, verdict, runs) in enumerate(rows):
        d.rule(s, MX, y, CW, color=S.INK if i == 0 else S.HAIRLINE, width=2 if i == 0 else 1)
        d.text(s, MX, y + 0.28, 0.6, 0.6, str(i + 1), size=30, bold=True, color="#D4D4D8", line=1.0)
        d.text(s, MX + 0.65, y + 0.32, 3.6, 0.9, claim, size=19, bold=True, color=S.INK, line=1.2)
        d.rect(s, MX + 4.45, y + 0.36, 1.05, 0.42, fill=S.INK if verdict == "成立" else S.SURFACE,
               line=None if verdict == "成立" else S.INK, radius=0.5)
        d.text(s, MX + 4.45, y + 0.36, 1.05, 0.42, verdict, size=13, bold=True,
               color="#FFFFFF" if verdict == "成立" else S.INK, align="c", anchor="m")
        d.text(s, MX + 5.75, y + 0.28, CW - 5.75, 1.25, [runs], size=14.2, color=S.BODY, line=1.46)
        y += 1.56


def s23_selection_guide(d: Deck):
    f = d.facts
    s = d.slide("S23_selection_guide", SECTION["end"], "方法选择：先问表示从哪里来，再问数据和预算",
                source="建议来自本项目三个数据集上的带标签基准；在新数据上使用前，需要少量人工标注样本核对。")
    qx, qw, qh = MX, 4.70, 0.84
    ax, aw = 7.82, 4.86
    qs = [(1.68, "① 有与数据领域匹配的预训练模型吗？", "冻结预训练特征 + K-means", S.PRETRAINED,
           f"自然图 CIFAR-10 达 {f.f('e04_cifar10_dinov2_kmeans')}（从零训练 ≤ {f.f('max_scratch_cifar10')}）；手写数字仅 {f.f('e04_mnist_dinov2_kmeans')}，需看领域是否匹配"),
          (3.04, "② 样本很少、噪声大，或算力预算紧？", "PCA + K-means / 像素 K-means", S.CLASSICAL,
           f"抗噪最强（加噪保留 {f.p('keep_pca_kmeans')}），训练仅 {f.f('time_pca_kmeans', 1)} 秒，小样本下最稳妥"),
          (4.40, "③ 需要概率解释，或想生成新样本？", "VaDE（隐空间高斯混合）", S.GENERATIVE,
           f"MNIST 达 {f.f('ari_mnist_vade')}（逼近 DEC），可输出每簇概率分布与先验"),
          (5.76, "④ 以上都不是（干净、样本充足）", "DEC（自编码器预训练 + 自训练）", S.DEEP,
           f"从零训练干净数据最高：MNIST {f.f('ari_mnist_dec')} · Fashion {f.f('ari_fashion_mnist_dec')}")]
    for i, (y, q, a, color, ev) in enumerate(qs):
        last = i == len(qs) - 1
        d.box(s, qx, y, qw, qh, q, fill=S.PANEL if not last else S.SURFACE, size=14.8, bold=not last,
              color=S.INK if not last else S.SECONDARY, line=S.HAIRLINE if last else None)
        d.arrow(s, qx + qw + 0.08, y + qh / 2, ax - 0.12, y + qh / 2, color=S.SECONDARY)
        if not last:
            d.text(s, qx + qw + 0.24, y + qh / 2 - 0.33, 1.0, 0.3, "是", size=12, color=S.SECONDARY)
            d.arrow(s, qx + 0.45, y + qh + 0.03, qx + 0.45, y + 1.36 - 0.04, color=S.MUTED)
            d.text(s, qx + 0.60, y + qh + 0.08, 0.6, 0.3, "否", size=12, color=S.SECONDARY)
        d.rect(s, ax, y, 0.09, qh, fill=color)
        d.text(s, ax + 0.26, y + 0.02, aw - 0.28, 0.36, a, size=15.5, bold=True, color=S.INK)
        d.text(s, ax + 0.26, y + 0.39, aw - 0.28, 0.46, ev, size=11.8, color=S.SECONDARY, line=1.28)


def s24_limits(d: Deck):
    s = d.slide("S24_limits_and_close", SECTION["end"], "局限与展望")
    lim = [
        ("假设已知簇数 K：", "实验固定 K = 10；未知 K 时需另行估计，估错会明显拉低 ARI（见附录 A03）。"),
        ("算力限制对比学习：", "SimCLR、CC 仅用 128 维小 CNN 训 100 轮，CC 未加簇熵正则，对比路线被低估。"),
        ("数据规模与模态有限：", "只测试了 3 个小尺寸图像集，结论尚未涵盖文本、表格与高分辨率图像。"),
        ("预训练成本不可直比：", "DINOv2 / CLIP 依赖海量外部数据，训练成本不可与从零训练直接对比。"),
    ]
    fut = [
        ("未知 K 的自适应聚类：", "让模型自己发现数据该分几类，而非人工事先指定簇数。"),
        ("更可靠的无标签选型：", "超越加噪稳定性，寻找能反映真实语义的无标签评估准则。"),
        ("文字引导的多模态聚类：", "沿 TAC、VLM4Cluster 方向，用语言先验提升语义分组并检验鲁棒性。"),
    ]
    for col, (head, items, color) in enumerate([("局限", lim, S.SECONDARY), ("展望", fut, S.DEEP)]):
        x = MX + col * 6.25
        w = 5.8
        d.rule(s, x, 1.72, w, color=S.INK, width=2)
        d.text(s, x, 1.88, w, 0.42, head, size=19, bold=True, color=S.INK)
        y = 2.50
        for lead, rest in items:
            d.dot(s, x, y + 0.11, color, d=0.12)
            d.text(s, x + 0.28, y, w - 0.28, 0.85, [[(lead, {"bold": True, "color": S.INK}), (rest, {})]],
                   size=14, color=S.BODY, line=1.42)
            y += 0.84
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
    d.table(s, 6.75, 1.92, 5.9, rows, col_w=[0.9, 1.5, 1.6, 1.9], size=13, header_size=12, row_h=0.48)
    d.text(s, 6.75, 5.05, 5.9, 1.55,
           "左图：2 维压丢信息、128 维距离被稀释，10 维附近效果最好。\n"
           "右表：早停时 λ 越大训练越久；固定 50 轮后，λ 的作用不再与训练时长混在一起。",
           size=13, color=S.SECONDARY, line=1.45, after=6)


def a03_sensitivity(d: Deck, t: dict):
    s = d.slide("A03_sensitivity", SECTION["app"], "A03　簇数设错、训练集混入噪声图、类别不平衡",
                source="Fashion-MNIST；K-means / AE+KM / DEC / IDEC 为 3 种子均值±标准差，CC 为单种子。不平衡按训练集构造，均衡测试集评分。")
    d.figure(s, "sensitivity_k_outliers_imbalance", MX, 1.75, CW)


def a04_encoder_table(d: Deck, t: dict):
    s = d.slide("A04_encoder_clusterer_table", SECTION["app"], "A04　编码器 × 聚类器：完整 ARI",
                source="均值 ± 标准差（3 个随机种子）；加粗为各数据集最高。自编码器 / SimCLR + K-means 与基准实验同源。")
    e04 = t["e04_summary"]
    head = ["编码器"] + [f"{S.DATASET_LABEL[ds]}\n{S.CLUSTERER_LABEL[c]}" for ds in ("mnist", "fashion_mnist", "cifar10")
                         for c in ("kmeans", "gmm", "dec_head")]
    rows, bold = [head], set()
    best_by_ds = {}
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        sub = e04[e04.dataset.eq(ds)] if not e04.empty else e04
        if len(sub):
            top = sub.loc[sub.ari_mean.idxmax()]
            best_by_ds[ds] = (str(top.encoder), str(top.clusterer))
    for i, enc in enumerate(("random_cnn", "ae", "simclr", "dinov2", "clip"), start=1):
        row = [S.ENCODER_LABEL[enc]]
        for j, (ds, c) in enumerate([(ds, c) for ds in ("mnist", "fashion_mnist", "cifar10")
                                      for c in ("kmeans", "gmm", "dec_head")], start=1):
            mean = _lookup(e04, "ari_mean", dataset=ds, encoder=enc, clusterer=c)
            sd = _lookup(e04, "ari_std", dataset=ds, encoder=enc, clusterer=c)
            row.append(DASH if mean is None else f"{mean:.3f}±{sd:.3f}" if sd is not None else f"{mean:.3f}")
            if best_by_ds.get(ds) == (enc, c):
                bold.add((i, j))
        rows.append(row)
    d.table(s, MX, 1.8, CW, rows, col_w=[1.5] + [1.17] * 9, size=11, header_size=10.5, row_h=0.55,
            bold_cells=bold)


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
    d.equation(s, "jaccard", 9.45, 1.95, scale=0.75)
    d.text(s, 9.3, 2.92, 3.4, 3.3,
           [["直觉：交集 ÷ 并集；越接近 1，说明加噪前后每组里的人几乎没变。"],
            ["A", ("k", {"sub": True}), "、B", ("k", {"sub": True}),
             "：同一模型在加噪前、后分到第 k 个簇的训练图集合。"],
            "加粗：各数据集上稳定性最高的方法，它们都不是 ARI 最高的方法。"], size=12.5, color=S.SECONDARY,
           line=1.45, after=8)


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
          s08_idec, s09_vade, s10_contrastive, s11_pretrained, s11b_architectures, s12_setup, s13_benchmark,
          s14_geometry, s15_repr_vs_clusterer, s16_pixels_vs_dinov2, s17_noise, s18_sample_size, s19_components,
          s20_cost, s21_stability, s22_verdict, s23_selection_guide, s24_limits]
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
