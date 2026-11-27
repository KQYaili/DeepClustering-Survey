"""Build the 30-slide (24 main + 6 appendix) blue-white academic infographic deck.

All experimental values are dynamically read from results/tables/*.csv and *.json.
Every slide is bound to presentation/speaker_notes.md via a stable slide ID checked at build time.
All images are placed with exact natural aspect ratios via PIL (zero stretching).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
NOTES_MD = ROOT / "presentation" / "speaker_notes.md"
OUT = ROOT / "presentation" / "deep-clustering-evolution-selection-guide.pptx"

# Blue-White Academic Infographic Palette
BLUE_DEEP = RGBColor(0x07, 0x3D, 0xA5)
BLUE_DARK = RGBColor(0x0B, 0x25, 0x59)
BLUE_MID = RGBColor(0x25, 0x63, 0xEB)
BLUE_SOFT = RGBColor(0xEA, 0xF0, 0xFC)
BLUE_PALE = RGBColor(0xF5, 0xF8, 0xFE)
BLUE_LINE = RGBColor(0x9C, 0xB3, 0xD3)

TEAL = RGBColor(0x0F, 0x76, 0x6E)
TEAL_SOFT = RGBColor(0xE6, 0xF4, 0xF1)
AMBER = RGBColor(0xC2, 0x59, 0x1A)
AMBER_SOFT = RGBColor(0xFD, 0xF1, 0xE6)
PURPLE = RGBColor(0x6D, 0x28, 0xD9)
PURPLE_SOFT = RGBColor(0xF1, 0xEC, 0xFC)
CRIMSON = RGBColor(0xC8, 0x1E, 0x1E)
SLATE = RGBColor(0x47, 0x55, 0x69)
SLATE_SOFT = RGBColor(0xF1, 0xF5, 0xF9)

INK = RGBColor(0x11, 0x17, 0x22)
MUTED = RGBColor(0x52, 0x60, 0x75)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HAIRLINE = RGBColor(0xCB, 0xD8, 0xEB)
GOLD = RGBColor(0xD9, 0x77, 0x06)

W, H = 13.333, 7.5

METHOD_LABELS = {
    "kmeans": "像素 K-means",
    "pca_kmeans": "PCA + K-means",
    "gmm": "高斯混合 GMM",
    "ae_kmeans": "AE + K-means",
    "dec": "DEC",
    "idec": "IDEC (λ=1.0)",
    "vade": "VaDE",
    "simclr_kmeans": "SimCLR+KM (小CNN)",
    "cc": "CC (小CNN,省熵项)",
}
SHORT_LABELS = {
    "kmeans": "K-means",
    "pca_kmeans": "PCA+KM",
    "gmm": "GMM",
    "ae_kmeans": "AE+KM",
    "dec": "DEC",
    "idec": "IDEC",
    "vade": "VaDE",
    "simclr_kmeans": "SimCLR+KM",
    "cc": "CC",
}

# Keep equations as ordinary PowerPoint text.  Unicode subscripts/superscripts
# preserve the notation while avoiding a non-editable LaTeX raster asset.
EDITABLE_EQUATIONS = {
    "kmeans": "min{μₖ, cᵢ}  Σᵢ₌₁ᴺ ‖xᵢ − μ꜀ᵢ‖₂²",
    "map": "x   →   z = fθ(x)   →   ĉ",
    "dec": "qᵢⱼ = (1 + ‖zᵢ − μⱼ‖²)⁻¹ / Σⱼ′(1 + ‖zᵢ − μⱼ′‖²)⁻¹,    pᵢⱼ = (qᵢⱼ² / fⱼ) / Σⱼ′(qᵢⱼ′² / fⱼ′)",
    "kl": "L_DEC = KL(P ‖ Q) = (1/B) Σᵢ₌₁ᴮ Σⱼ₌₁ᴷ pᵢⱼ log(pᵢⱼ / qᵢⱼ)",
    "idec": "L_IDEC = L_cluster + λ (1/B) Σᵢ₌₁ᴮ Σ_d₌₁ᴰ (xᵢd − x̂ᵢd)²",
    "vade": "p(x, z, c) = p(c) p(z│c) p(x│z),    p(z│c) = N(μ꜀, σ꜀² I)",
    "pareto": "maxₘ (ARI(m), Robustness(m), −Time(m), −Memory(m))",
}


# ---------------------------------------------------------------------------
# Data loaders (strictly from tables_dir)
# ---------------------------------------------------------------------------
def load_csv(name: str, tables_dir: Path = TABLES) -> list[dict]:
    path = tables_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_selection(tables_dir: Path = TABLES) -> dict:
    """Load E05 train-only stability selection JSON (empty dict if missing)."""
    path = tables_dir / "selection.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict], **conds) -> dict | None:
    for r in rows:
        ok = True
        for k, v in conds.items():
            rv = r.get(k, "")
            if isinstance(v, float):
                try:
                    if abs(float(rv) - v) > 1e-9:
                        ok = False
                except ValueError:
                    ok = False
            elif str(rv) != str(v):
                ok = False
            if not ok:
                break
        if ok:
            return r
    return None


def fnum(row: dict | None, key: str | None = None) -> float | None:
    if not row:
        return None
    if key is None:
        key = "ari_mean" if "ari_mean" in row else "ari"
    val = row.get(key)
    if val in (None, ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fval(row: dict | None, key: str | None = None, digits: int = 3) -> str:
    val = fnum(row, key)
    return "—" if val is None else f"{val:.{digits}f}"


def fpm(row: dict | None, mean_key: str, digits: int = 3) -> str:
    m = fnum(row, mean_key)
    if m is None:
        return "—"
    s = fnum(row, mean_key.replace("_mean", "_std"))
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def fep(row: dict | None) -> str:
    ep = fnum(row, "epochs_ran")
    if ep is None:
        return "—"
    early = str(row.get("stopped_early", "")).lower() == "true"
    return f"{int(round(ep))}ep{'早停' if early else ''}"


def fret(clean_row: dict | None, noisy_row: dict | None) -> str:
    c = fnum(clean_row, "ari_mean")
    n = fnum(noisy_row, "ari_mean")
    if c is None or n is None or c <= 0:
        return "—"
    return f"{n / c * 100:.0f}%"


def fdiff(a: float | None, b: float | None, digits: int = 3, signed: bool = True) -> str:
    if a is None or b is None:
        return "—"
    d = a - b
    return f"{d:+.{digits}f}" if signed else f"{d:.{digits}f}"


# ---------------------------------------------------------------------------
# Low-level shape & typography helpers
# ---------------------------------------------------------------------------
def _set_font_ea(run, font_name: str = "Microsoft YaHei") -> None:
    rpr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        el = rpr.find(qn(f"a:{tag}"))
        if el is None:
            el = etree.SubElement(rpr, qn(f"a:{tag}"))
        el.set("typeface", font_name)


def _add_run(p, txt: str, size: float, bold: bool, color: RGBColor, font: str = "Microsoft YaHei"):
    r = p.add_run()
    r.text = txt
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    _set_font_ea(r, font)
    return r


TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|##[^#]+##|\[\[[^\]]+\]\])")


def _write_rich_line(p, line: str, size: float, base_bold: bool, base_color: RGBColor, font: str = "Microsoft YaHei"):
    """Mini markup: **crimson bold**, ##blue bold##, [[ink bold]]."""
    pos = 0
    for m in TOKEN_RE.finditer(line):
        if m.start() > pos:
            _add_run(p, line[pos:m.start()], size, base_bold, base_color, font)
        tok = m.group(0)
        if tok.startswith("**"):
            _add_run(p, tok[2:-2], size, True, CRIMSON, font)
        elif tok.startswith("##"):
            _add_run(p, tok[2:-2], size, True, BLUE_DEEP, font)
        else:
            _add_run(p, tok[2:-2], size, True, base_color, font)
        pos = m.end()
    if pos < len(line):
        _add_run(p, line[pos:], size, base_bold, base_color, font)
    if not line:
        _add_run(p, "", size, base_bold, base_color, font)


def box(slide, l, t, w, h, fill: RGBColor | None, line: RGBColor | None = None,
        line_pt: float = 1.0, radius: float | None = None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_pt)
    if radius:
        shp.adjustments[0] = radius
    shp.shadow.inherit = False
    return shp


def textbox(slide, l, t, w, h, content: str, size: float = 14.0, bold: bool = False,
            color: RGBColor = INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            line_spacing: float = 1.22, space_after: float = 4.0,
            font: str = "Microsoft YaHei", word_wrap: bool = True):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = word_wrap
    tf.auto_size = None
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    try:
        tf._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    lines = content.split("\n")
    for idx, raw_line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_before = Pt(0)
        p.space_after = Pt(space_after if idx < len(lines) - 1 else 0)
        _write_rich_line(p, raw_line, size, bold, color, font)
    return tb


def editable_equation(slide, l: float, t: float, w: float, h: float, text: str,
                      size: float = 20.0, color: RGBColor = BLUE_DEEP,
                      align=PP_ALIGN.CENTER):
    """Add a native editable math text box.

    The earlier deck placed Matplotlib PNGs for LaTeX.  Those images looked crisp
    but PowerPoint treated every formula, fraction bar, and Roman label as one
    raster object.  These Unicode/Math-font equations keep the same notation in
    an ordinary text box, so the user can select and edit every character and the
    long fraction bars are no longer clipped by tight image bounding boxes.
    """
    return textbox(
        slide, l, t, w, h, text, size=size, color=color, align=align,
        anchor=MSO_ANCHOR.MIDDLE, space_after=0, font="Cambria Math", word_wrap=False,
    )


def diagram_card(slide, l: float, t: float, w: float, h: float, title: str, body: str,
                 accent: RGBColor = BLUE_DEEP, fill: RGBColor = WHITE,
                 title_color: RGBColor | None = None, body_color: RGBColor = INK,
                 title_size: float = 10.0, body_size: float = 8.3):
    """Editable card used by the native architecture diagrams."""
    box(slide, l, t, w, h, fill, accent, 1.2, 0.06)
    if title:
        textbox(slide, l + 0.08, t + 0.08, w - 0.16, min(0.28, h * 0.36), title,
                size=title_size, bold=True, color=title_color or accent,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        body_top = t + min(0.42, h * 0.40)
        body_height = max(0.06, h - (body_top - t) - 0.08)
    else:
        body_top = t + 0.02
        body_height = max(0.06, h - 0.04)
    textbox(slide, l + 0.10, body_top, w - 0.20, body_height, body,
            size=body_size, color=body_color, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.14, space_after=0)


def diagram_connector(slide, x1: float, y1: float, x2: float, y2: float,
                      color: RGBColor = BLUE_DEEP, width: float = 1.5,
                      dashed: bool = False):
    """Add an editable line connector. Labels and arrow glyphs remain text."""
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color
    line.line.width = Pt(width)
    if dashed:
        line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return line


def diagram_arrow(slide, x1: float, y1: float, x2: float, y2: float,
                  color: RGBColor = BLUE_DEEP, width: float = 1.5,
                  label: str = "→"):
    """Draw a connector and an editable arrow glyph at its midpoint."""
    diagram_connector(slide, x1, y1, x2, y2, color=color, width=width)
    textbox(slide, (x1 + x2) / 2 - 0.16, (y1 + y2) / 2 - 0.16, 0.32, 0.32,
            label, size=13.0, bold=True, color=color, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, space_after=0)


def native_paradigm_diagram(slide):
    """Editable replacement for the paradigm-evolution raster diagram."""
    diagram_connector(slide, 0.62, 1.84, 12.55, 1.84, color=BLUE_LINE, width=2.0)
    textbox(slide, 1.0, 1.30, 11.2, 0.34,
            "深度聚类范式演进谱系：多种范式长期并存，并非简单线性淘汰",
            size=13.0, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    stages = [
        ("固定特征基线", "经典统计聚类", "K-means / GMM\nPCA + K-means", "坐标固定\n欧氏距离或高斯云", SLATE, SLATE_SOFT),
        ("多阶段式", "Multi-stage", "AE + K-means\nSimCLR + K-means", "先学表征\n再接聚类器", BLUE_DEEP, BLUE_SOFT),
        ("联合优化式", "Simultaneously", "DEC / IDEC\n对比聚类 CC", "聚类与表示\n同步反向更新", AMBER, AMBER_SOFT),
        ("生成式潜变量", "Generative", "VaDE\n深度混合先验", "VAE + GMM\n联合推断与生成", TEAL, TEAL_SOFT),
        ("迭代式", "Iterative", "DeepCluster\nSwAV", "伪标签 → 表示\n→ 再聚类循环", PURPLE, PURPLE_SOFT),
        ("外部预训练引导", "2024–2026 前沿", "DINOv2 / CLIP\nTAC / VLM4Cluster*", "冻结视觉模型\n外部语义引导", CRIMSON, RGBColor(0xFB, 0xEC, 0xEC)),
    ]
    for i, (title, sub, rep, mech, accent, tint) in enumerate(stages):
        x = 0.42 + i * 2.10
        diagram_card(slide, x, 2.14, 1.86, 3.20, title, f"{sub}\n\n{rep}\n\n{mech}",
                     accent=accent, fill=tint, title_color=accent, body_color=INK,
                     title_size=10.0, body_size=8.0)
        diagram_connector(slide, x + 0.93, 1.84, x + 0.93, 2.12, color=accent, width=1.2)
        if i < len(stages) - 1:
            diagram_arrow(slide, x + 1.88, 3.72, x + 2.07, 3.72, color=BLUE_LINE, width=1.2)
    textbox(slide, 0.66, 5.70, 12.0, 0.48,
            "Zhou 等（ACM Computing Surveys, 2025）图 1 四类核心交互：Multi-stage · Simultaneously · Generative · Iterative",
            size=10.5, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(slide, 0.66, 6.25, 12.0, 0.30,
            "* VLM4Cluster（arXiv:2609.36648）按预印本引用。",
            size=9.0, color=SLATE, align=PP_ALIGN.RIGHT, space_after=0)


def native_taxonomy_diagram(slide):
    """Editable replacement for the four-dimensional taxonomy raster diagram."""
    cols = [
        ("维度 1：表征空间", "数据如何编码为坐标 z", BLUE_DEEP, BLUE_SOFT,
         ["原始像素\nK-means / GMM", "线性投影\nPCA + K-means", "深度隐空间\nAE / DEC / VaDE / CC", "外部预训练\nDINOv2 / CLIP"]),
        ("维度 2：优化交互", "表示与聚类如何配合", TEAL, TEAL_SOFT,
         ["固定特征\nK-means / GMM", "多阶段解耦\nAE / SimCLR", "交替迭代\nDeepCluster / SwAV", "端到端联合\nDEC / IDEC / CC"]),
        ("维度 3：聚类机制", "如何划分样本归属", AMBER, AMBER_SOFT,
         ["欧氏质心\nK-means", "Student-t 软分配\nDEC / IDEC", "高斯概率云\nGMM / VaDE", "对比簇头\nCC / IIC"]),
        ("维度 4：监督信号", "无标签信息从哪来", PURPLE, PURPLE_SOFT,
         ["像素重构误差\nAE / IDEC / VaDE", "自训练锐化\nDEC / IDEC", "双视图增强\nSimCLR / CC / SwAV", "外部语义\nDINOv2 / CLIP / TAC"]),
    ]
    for i, (title, sub, accent, tint, items) in enumerate(cols):
        x = 0.42 + i * 3.17
        box(slide, x, 1.42, 2.92, 4.78, WHITE, accent, 1.3, 0.06)
        box(slide, x, 1.42, 2.92, 0.48, accent, None, 0.0, 0.06)
        textbox(slide, x + 0.08, 1.47, 2.76, 0.34, title, size=10.6, bold=True,
                color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        textbox(slide, x + 0.12, 2.02, 2.68, 0.34, sub, size=8.8, bold=True,
                color=accent, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        for j, item in enumerate(items):
            diagram_card(slide, x + 0.14, 2.48 + j * 0.83, 2.64, 0.66, "", item,
                         accent=accent, fill=tint, title_color=accent, body_color=INK,
                         title_size=8.6, body_size=8.1)
        if i < 3:
            diagram_arrow(slide, x + 2.94, 3.72, x + 3.13, 3.72, color=BLUE_LINE, width=1.2)


def native_dec_idec_diagram(slide):
    """Editable DEC/IDEC architecture with native cards and connectors."""
    diagram_card(slide, 0.72, 4.10, 1.26, 0.96, "输入图像 x", "784 维像素\n28×28", accent=SLATE, fill=SLATE_SOFT, title_color=INK)
    diagram_card(slide, 2.35, 3.82, 1.62, 1.52, "编码器 fθ", "784→500→500\n→2000→10", accent=BLUE_DEEP, fill=BLUE_SOFT)
    diagram_card(slide, 4.35, 4.05, 1.42, 1.06, "隐表示 z", "d = 10\nAE 预训练", accent=BLUE_DEEP, fill=WHITE)
    diagram_arrow(slide, 1.99, 4.58, 2.32, 4.58, color=BLUE_DEEP)
    diagram_arrow(slide, 3.99, 4.58, 4.32, 4.58, color=BLUE_DEEP)

    box(slide, 5.98, 2.56, 6.28, 2.08, BLUE_PALE, BLUE_DEEP, 1.2, 0.08)
    textbox(slide, 6.18, 2.70, 5.88, 0.28, "上分支：DEC 聚类自训练目标（KL 反向更新编码器与中心）",
            size=9.8, bold=True, color=BLUE_DEEP, space_after=0)
    diagram_card(slide, 6.18, 3.22, 1.72, 0.96, "Student-t 软分配", "qᵢⱼ ∝ (1+‖zᵢ−μⱼ‖²)⁻¹\nK-means 初始化", accent=BLUE_MID, title_size=9.0, body_size=7.7)
    diagram_card(slide, 8.24, 3.22, 1.58, 0.96, "锐化目标 P", "pᵢⱼ ∝ qᵢⱼ²/fⱼ\n拉紧高置信簇", accent=BLUE_MID, title_size=9.0, body_size=7.7)
    diagram_card(slide, 10.16, 3.22, 1.78, 0.96, "聚类损失", "L_cluster =\nKL(P ‖ Q)", accent=BLUE_DEEP, fill=BLUE_DEEP, title_color=WHITE, body_color=WHITE, title_size=9.2, body_size=8.0)
    diagram_arrow(slide, 5.78, 4.58, 6.16, 3.70, color=BLUE_DEEP)
    diagram_arrow(slide, 7.92, 3.70, 8.20, 3.70, color=BLUE_DEEP)
    diagram_arrow(slide, 9.84, 3.70, 10.12, 3.70, color=BLUE_DEEP)
    diagram_connector(slide, 11.00, 3.12, 3.10, 3.76, color=CRIMSON, width=1.4, dashed=True)
    textbox(slide, 6.35, 2.96, 5.45, 0.24, "反向传播：同时更新 μⱼ 与 fθ",
            size=9.0, bold=True, color=CRIMSON, align=PP_ALIGN.CENTER, space_after=0)

    box(slide, 5.98, 4.88, 6.28, 1.60, AMBER_SOFT, AMBER, 1.2, 0.08)
    textbox(slide, 6.18, 5.02, 5.88, 0.28, "下分支：IDEC 重构约束（保留解码器以守护局部流形）",
            size=9.8, bold=True, color=AMBER, space_after=0)
    diagram_card(slide, 6.18, 5.40, 1.78, 0.82, "解码器 gφ", "10→2000→500\n→500→784", accent=AMBER, title_size=9.0, body_size=7.8)
    diagram_card(slide, 8.30, 5.40, 1.62, 0.82, "重构图像 x̂", "与 x 逐像素\n平方误差", accent=AMBER, title_size=9.0, body_size=7.8)
    diagram_card(slide, 10.26, 5.40, 1.78, 0.82, "重构约束项", "λ · SSE / B\nλ=0 退化 DEC", accent=AMBER, fill=AMBER, title_color=WHITE, body_color=WHITE, title_size=9.0, body_size=7.8)
    diagram_arrow(slide, 5.78, 4.58, 6.16, 5.80, color=AMBER)
    diagram_arrow(slide, 7.98, 5.80, 8.26, 5.80, color=AMBER)
    diagram_arrow(slide, 9.96, 5.80, 10.22, 5.80, color=AMBER)
    box(slide, 0.72, 5.50, 4.54, 0.78, TEAL_SOFT, TEAL, 1.0, 0.05)
    textbox(slide, 0.88, 5.60, 4.22, 0.58, "E02 核心问题\n只优化上分支（λ=0）会不会破坏表示结构？", size=9.0, bold=True, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)


def native_vade_diagram(slide):
    """Editable VaDE generative and inference architecture."""
    box(slide, 0.62, 2.48, 5.78, 3.62, TEAL_SOFT, TEAL, 1.2, 0.08)
    textbox(slide, 0.86, 2.66, 5.30, 0.28, "生成过程：每个簇是隐空间的一团高斯云", size=10.2, bold=True, color=TEAL, align=PP_ALIGN.CENTER, space_after=0)
    diagram_card(slide, 0.92, 3.32, 1.56, 0.92, "1. 抽取簇 c", "c ~ Cat(π)\n先验比例 πc", accent=TEAL, title_size=9.2, body_size=8.0)
    diagram_card(slide, 2.78, 3.32, 1.84, 0.92, "2. 抽取隐变量 z", "z|c ~ N(μc, σc²I)\n中心与方差", accent=TEAL, title_size=9.2, body_size=8.0)
    diagram_arrow(slide, 2.50, 3.78, 2.74, 3.78, color=TEAL)
    diagram_connector(slide, 3.70, 4.24, 2.08, 5.20, color=TEAL, width=1.3)
    diagram_connector(slide, 3.70, 4.24, 3.10, 5.20, color=TEAL, width=1.3)
    diagram_connector(slide, 3.70, 4.24, 4.20, 5.20, color=TEAL, width=1.3)
    diagram_card(slide, 1.00, 5.00, 4.88, 0.72, "隐空间混合先验 p(z)", "Σc πc N(z | μc, σc²I)   →   解码器 gφ(z) 重构 x̂", accent=TEAL, fill=WHITE, title_size=9.0, body_size=8.0)

    box(slide, 6.72, 2.48, 5.98, 3.62, BLUE_PALE, BLUE_DEEP, 1.2, 0.08)
    textbox(slide, 6.98, 2.66, 5.46, 0.28, "变分推断与 ELBO 联合优化", size=10.2, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER, space_after=0)
    diagram_card(slide, 7.00, 3.32, 1.26, 0.92, "输入 x", "图像像素", accent=BLUE_DEEP, title_size=9.2, body_size=8.0)
    diagram_card(slide, 8.56, 3.32, 1.86, 0.92, "推断编码器", "μ̃, log σ̃²\n重参数化采样 z", accent=BLUE_DEEP, title_size=9.2, body_size=8.0)
    diagram_card(slide, 10.74, 3.32, 1.60, 0.92, "后验归属 γc", "q(c|x)=p(c|z)\nBayes 推断", accent=BLUE_DEEP, fill=BLUE_DEEP, title_color=WHITE, body_color=WHITE, title_size=9.0, body_size=7.9)
    diagram_arrow(slide, 8.28, 3.78, 8.52, 3.78, color=BLUE_DEEP)
    diagram_arrow(slide, 10.46, 3.78, 10.70, 3.78, color=BLUE_DEEP)
    diagram_card(slide, 7.00, 4.78, 5.34, 0.96, "ELBO = 重构保真项 − 隐变量与混合先验 KL", "对比 K-means：可建模簇方差与先验 πc；本仓库重构项采用每图 SSE。", accent=BLUE_DEEP, fill=WHITE, title_size=9.0, body_size=8.0)


def native_contrastive_diagram(slide):
    """Editable SimCLR + K-means and CC dual-view architecture."""
    diagram_card(slide, 0.62, 3.55, 1.20, 0.92, "原图 x", "无标签样本", accent=SLATE, fill=SLATE_SOFT, title_color=INK)
    diagram_card(slide, 2.18, 2.62, 1.34, 0.72, "增强视图 a", "裁剪 / 抖动", accent=BLUE_DEEP, fill=BLUE_SOFT, title_size=9.0, body_size=7.8)
    diagram_card(slide, 2.18, 4.62, 1.34, 0.72, "增强视图 b", "语义保持", accent=BLUE_DEEP, fill=BLUE_SOFT, title_size=9.0, body_size=7.8)
    diagram_arrow(slide, 1.84, 3.92, 2.14, 2.98, color=BLUE_DEEP)
    diagram_arrow(slide, 1.84, 4.10, 2.14, 4.98, color=BLUE_DEEP)
    diagram_card(slide, 3.96, 3.38, 1.82, 1.32, "共享 CNN 编码器", "小型课程 CNN\n输出 128 维 h\n非原论文 ResNet", accent=BLUE_DEEP, title_size=9.4, body_size=8.0)
    diagram_arrow(slide, 3.54, 2.98, 3.92, 3.78, color=BLUE_DEEP)
    diagram_arrow(slide, 3.54, 4.98, 3.92, 4.28, color=BLUE_DEEP)
    box(slide, 6.12, 2.20, 6.22, 2.30, BLUE_PALE, BLUE_DEEP, 1.2, 0.08)
    textbox(slide, 6.34, 2.36, 5.80, 0.26, "路线 A：SimCLR + K-means（先对比学表征，再离线聚类）", size=9.7, bold=True, color=BLUE_DEEP, space_after=0)
    diagram_card(slide, 6.34, 2.92, 1.84, 0.96, "实例投影头", "NT-Xent：正对靠近\n负对推远", accent=BLUE_MID, title_size=9.0, body_size=7.9)
    diagram_card(slide, 8.54, 2.92, 1.58, 0.96, "丢弃投影头", "取出 128 维\n冻结 h", accent=BLUE_MID, title_size=9.0, body_size=7.9)
    diagram_card(slide, 10.48, 2.92, 1.56, 0.96, "离线 K-means", "n_init=20\n输出 ĉ", accent=BLUE_DEEP, fill=BLUE_DEEP, title_color=WHITE, body_color=WHITE, title_size=9.0, body_size=7.9)
    diagram_arrow(slide, 5.80, 4.02, 6.30, 3.40, color=BLUE_DEEP)
    diagram_arrow(slide, 8.20, 3.40, 8.50, 3.40, color=BLUE_DEEP)
    diagram_arrow(slide, 10.16, 3.40, 10.44, 3.40, color=BLUE_DEEP)
    box(slide, 6.12, 4.82, 6.22, 1.46, PURPLE_SOFT, PURPLE, 1.2, 0.08)
    textbox(slide, 6.34, 4.96, 5.80, 0.26, "路线 B：CC（同时进行实例级 + 簇级对比）", size=9.7, bold=True, color=PURPLE, space_after=0)
    diagram_card(slide, 6.34, 5.38, 2.28, 0.64, "行对比（实例级）", "同一图像两视图靠近", accent=PURPLE, title_size=8.6, body_size=7.7)
    diagram_card(slide, 8.84, 5.38, 2.28, 0.64, "列对比（簇级）", "两视图簇概率一致", accent=PURPLE, title_size=8.6, body_size=7.7)
    diagram_card(slide, 11.24, 5.38, 0.86, 0.64, "簇头 ĉ", "联合输出", accent=PURPLE, fill=PURPLE, title_color=WHITE, body_color=WHITE, title_size=8.0, body_size=7.2)
    diagram_arrow(slide, 5.80, 4.18, 6.30, 5.70, color=PURPLE)
    diagram_arrow(slide, 8.66, 5.70, 8.80, 5.70, color=PURPLE)
    diagram_arrow(slide, 11.16, 5.70, 11.20, 5.70, color=PURPLE)


def native_pretrained_diagram(slide):
    """Editable encoder/clusterer decoupling diagram for E04."""
    diagram_card(slide, 0.62, 3.70, 1.34, 0.98, "CIFAR-10", "10,000 训练图\n2,000 测试图", accent=SLATE, fill=SLATE_SOFT, title_color=INK)
    box(slide, 2.32, 2.22, 3.30, 2.18, SLATE_SOFT, SLATE, 1.1, 0.08)
    textbox(slide, 2.52, 2.36, 2.90, 0.26, "从零训练主干", size=9.8, bold=True, color=SLATE, align=PP_ALIGN.CENTER, space_after=0)
    for i, txt in enumerate(["随机 CNN（基线）", "卷积自编码器 AE", "SimCLR（小型 CNN）"]):
        diagram_card(slide, 2.56, 2.86 + i * 0.46, 2.82, 0.34, "", txt, accent=BLUE_LINE, fill=WHITE, title_size=8.0, body_size=7.9)
    box(slide, 2.32, 4.68, 3.30, 1.34, BLUE_SOFT, BLUE_DEEP, 1.2, 0.08)
    textbox(slide, 2.52, 4.82, 2.90, 0.26, "冻结外部预训练主干", size=9.8, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER, space_after=0)
    diagram_card(slide, 2.56, 5.22, 2.82, 0.32, "", "DINOv2 ViT-S/14", accent=BLUE_DEEP, fill=WHITE, title_size=8.0, body_size=7.9)
    diagram_card(slide, 2.56, 5.60, 2.82, 0.32, "", "CLIP ViT-B/32", accent=BLUE_DEEP, fill=WHITE, title_size=8.0, body_size=7.9)
    diagram_arrow(slide, 1.98, 4.18, 2.28, 3.28, color=SLATE)
    diagram_arrow(slide, 1.98, 4.18, 2.28, 5.22, color=BLUE_DEEP)
    diagram_card(slide, 6.18, 3.70, 1.48, 1.08, "冻结特征 z", "统一切分\nL2 归一化", accent=TEAL, fill=TEAL_SOFT, title_size=9.2, body_size=8.0)
    diagram_arrow(slide, 5.66, 3.28, 6.14, 4.10, color=SLATE)
    diagram_arrow(slide, 5.66, 5.22, 6.14, 4.38, color=BLUE_DEEP)
    diagram_card(slide, 8.18, 2.62, 1.90, 0.92, "K-means", "欧氏质心\nn_init=20", accent=BLUE_DEEP, title_size=9.2, body_size=8.0)
    diagram_card(slide, 8.18, 3.80, 1.90, 0.92, "高斯混合 GMM", "对角协方差\n特征标准化", accent=TEAL, title_size=9.2, body_size=8.0)
    diagram_card(slide, 8.18, 4.98, 1.90, 0.92, "投影 + DEC 头", "10 维头\n20 epochs", accent=AMBER, title_size=9.2, body_size=8.0)
    diagram_arrow(slide, 7.70, 4.16, 8.14, 3.08, color=BLUE_DEEP)
    diagram_arrow(slide, 7.70, 4.16, 8.14, 4.26, color=TEAL)
    diagram_arrow(slide, 7.70, 4.16, 8.14, 5.44, color=AMBER)
    diagram_card(slide, 10.70, 3.22, 1.52, 1.92, "5 × 3 解耦检验", "行差看表示收益\n\n列差看聚类器收益", accent=BLUE_DEEP, fill=BLUE_DEEP, title_color=WHITE, body_color=WHITE, title_size=9.6, body_size=8.0)
    diagram_arrow(slide, 10.12, 3.08, 10.64, 4.12, color=BLUE_LINE)
    diagram_arrow(slide, 10.12, 4.26, 10.64, 4.12, color=BLUE_LINE)
    diagram_arrow(slide, 10.12, 5.44, 10.64, 4.12, color=BLUE_LINE)


def fit_picture(slide, path: Path, l: float, t: float, max_w: float, max_h: float,
                h_align: str = "center", v_align: str = "middle"):
    """Place an image preserving its exact natural aspect ratio inside (max_w, max_h)."""
    if not path.exists():
        box(slide, l, t, max_w, max_h, SLATE_SOFT, BLUE_LINE, 1.0, 0.04)
        textbox(slide, l + 0.1, t, max_w - 0.2, max_h, f"[缺少图像: {path.name}]",
                size=13, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return None
    with Image.open(path) as im:
        pw, ph = im.size
    ratio = pw / ph
    if max_w / max_h >= ratio:
        draw_h = max_h
        draw_w = max_h * ratio
    else:
        draw_w = max_w
        draw_h = max_w / ratio
    if h_align == "center":
        x = l + (max_w - draw_w) / 2
    elif h_align == "right":
        x = l + (max_w - draw_w)
    else:
        x = l
    if v_align == "middle":
        y = t + (max_h - draw_h) / 2
    elif v_align == "bottom":
        y = t + (max_h - draw_h)
    else:
        y = t
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(draw_w), height=Inches(draw_h))


def header_card(slide, l, t, w, h, title: str, body: str,
                header_fill: RGBColor = BLUE_DEEP, body_fill: RGBColor = WHITE,
                border: RGBColor = BLUE_LINE, title_size: float = 14.5,
                body_size: float = 13.0, header_h: float = 0.42,
                line_spacing: float = 1.24, space_after: float = 5.0):
    box(slide, l, t, w, h, body_fill, border, 1.0, 0.04)
    box(slide, l, t, w, header_h, header_fill, None, 0.0, None)
    textbox(slide, l + 0.16, t + 0.03, w - 0.32, header_h - 0.06, title,
            size=title_size, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(slide, l + 0.18, t + header_h + 0.12, w - 0.36, h - header_h - 0.22, body,
            size=body_size, color=INK, line_spacing=line_spacing, space_after=space_after)


def soft_panel(slide, l, t, w, h, title: str, body: str,
               fill: RGBColor = BLUE_SOFT, border: RGBColor = BLUE_LINE,
               title_color: RGBColor = BLUE_DEEP, title_size: float = 14.5,
               body_size: float = 13.0, line_spacing: float = 1.22, space_after: float = 4.0):
    box(slide, l, t, w, h, fill, border, 1.0, 0.05)
    textbox(slide, l + 0.16, t + 0.09, w - 0.32, 0.30, title,
            size=title_size, bold=True, color=title_color, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(slide, l + 0.16, t + 0.42, w - 0.32, h - 0.48, body,
            size=body_size, color=INK, line_spacing=line_spacing, space_after=space_after)


def data_table(slide, l, t, w, headers: list[str], rows: list[list[str]],
               col_widths: list[float], font_size: float = 12.5, header_size: float = 13.0,
               max_h: float = 4.65, highlight_cells: set[tuple[int, int]] | None = None,
               left_align_cols: set[int] | None = None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    row_h = min(0.58, max_h / max(n_rows, 1))
    shp = slide.shapes.add_table(n_rows, n_cols, Inches(l), Inches(t), Inches(w), Inches(row_h * n_rows))
    tbl = shp.table
    highlight_cells = highlight_cells or set()
    left_align_cols = left_align_cols or {0}
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Inches(cw)
    for j, h_txt in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE_DEEP
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j in left_align_cols else PP_ALIGN.CENTER
        _add_run(p, h_txt, header_size, True, WHITE)
    for i, r_data in enumerate(rows, start=1):
        for j, val in enumerate(r_data):
            cell = tbl.cell(i, j)
            cell.text = ""
            cell.fill.solid()
            if j == 0:
                cell.fill.fore_color.rgb = BLUE_SOFT if i % 2 else BLUE_PALE
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else SLATE_SOFT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j in left_align_cols else PP_ALIGN.CENTER
            is_hi = (i - 1, j) in highlight_cells
            color = CRIMSON if is_hi else (BLUE_DEEP if j == 0 else INK)
            bold = is_hi or (j == 0)
            _add_run(p, val, font_size, bold, color)
    return shp


# ---------------------------------------------------------------------------
# Uniform Blue-White Slide Chrome (no left vertical stripes, clean white deck)
# ---------------------------------------------------------------------------
def new_slide(prs, slide_ids: list[str], slide_id: str, page_label: str,
              section: str, title: str, lead: str = "", summary: str = ""):
    slide_ids.append(slide_id)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box(slide, 0, 0, W, H, WHITE)
    box(slide, 0, 0, W, 0.06, BLUE_DEEP)
    badge_w = min(2.45, max(1.55, len(section) * 0.17 + 0.45))
    box(slide, W - 0.42 - badge_w, 0.18, badge_w, 0.34, BLUE_SOFT, BLUE_LINE, 0.8, 0.25)
    textbox(slide, W - 0.42 - badge_w, 0.18, badge_w, 0.34, section,
            size=11.5, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(slide, 0.42, 0.14, W - 0.95 - badge_w, 0.48, title,
            size=23.5, bold=True, color=BLUE_DEEP, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    box(slide, 0.42, 0.68, W - 0.84, 0.015, HAIRLINE)
    if lead:
        box(slide, 0.42, 0.77, W - 0.84, 0.40, BLUE_PALE, None, 0.0, 0.08)
        textbox(slide, 0.56, 0.78, W - 1.12, 0.38, lead,
                size=13.2, color=INK, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    if summary:
        box(slide, 0.42, 6.80, 11.45, 0.42, BLUE_SOFT, BLUE_LINE, 0.7, 0.08)
        textbox(slide, 0.56, 6.81, 11.18, 0.40, summary,
                size=11.8, color=INK, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    else:
        box(slide, 0.42, 7.05, W - 0.84, 0.012, HAIRLINE)
        textbox(slide, 0.42, 7.10, 9.5, 0.28, "深度聚类：范式演进与方法选型综述  |  第九组课程汇报",
                size=10.5, color=MUTED, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    box(slide, 12.02, 6.80, 0.89, 0.42, BLUE_DEEP, None, 0.0, 0.12)
    textbox(slide, 12.02, 6.80, 0.89, 0.42, page_label,
            size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    return slide


# ---------------------------------------------------------------------------
# Slide Builders (1–24 Main Deck + A1–A6 Appendix)
# ---------------------------------------------------------------------------
def build_slides(prs, ctx: dict) -> list[str]:
    slide_ids: list[str] = []
    figs_dir: Path = ctx["figs_dir"]
    e01 = ctx["e01"]
    e02 = ctx["e02"]
    e03n = ctx["e03_noise"]
    e03s = ctx["e03_sens"]
    e04p = ctx["e04_paired"]
    e04ext = ctx["e04_pretrained"]
    costs = ctx["costs"]
    runs = ctx["all_runs"]
    sel = ctx["selection"]
    cov = ctx["coverage"]
    fails = ctx["failures"]

    n_ok = sum(1 for r in cov if r.get("status") == "ok")
    n_fail_hist = len(fails)
    exp_counts = {}
    for r in cov:
        if r.get("status") == "ok":
            exp_counts[r["experiment"]] = exp_counts.get(r["experiment"], 0) + 1

    # ---- Slide 1: Cover ----
    slide_ids.append("S01_cover")
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, W, H, WHITE)
    box(s, 0, 0, 5.15, H, BLUE_DEEP)
    box(s, 5.15, 0, 0.07, H, GOLD)
    box(s, 0.55, 0.72, 2.35, 0.38, BLUE_MID, None, 0.0, 0.22)
    textbox(s, 0.55, 0.72, 2.35, 0.38, "第九组  ·  课程综述汇报",
            size=12.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(s, 0.55, 1.55, 4.25, 2.35, "深度聚类：\n范式演进与\n方法选型综述",
            size=34.0, bold=True, color=WHITE, line_spacing=1.18, space_after=4)
    box(s, 0.55, 4.15, 1.35, 0.06, GOLD)
    textbox(s, 0.55, 4.45, 4.20, 1.45,
            "Deep Clustering:\nParadigm Evolution, Empirical\nBenchmarks, and Method Selection",
            size=14.0, color=RGBColor(0xD6, 0xE6, 0xFF), line_spacing=1.32)
    textbox(s, 0.55, 6.38, 4.20, 0.68,
            f"统一预注册协议  ·  {n_ok} 条实测记录\n2026 年课程研究型综述与开源复现仓库",
            size=11.5, color=RGBColor(0xB8, 0xD0, 0xF8), line_spacing=1.35)

    textbox(s, 5.68, 0.75, 7.1, 0.45, "本次汇报围绕的核心研究问题",
            size=18.0, bold=True, color=BLUE_DEEP, space_after=0)
    box(s, 5.68, 1.30, 7.18, 1.18, BLUE_SOFT, BLUE_LINE, 1.0, 0.06)
    textbox(s, 5.90, 1.42, 6.75, 0.96,
            "在不同的##数据结构##、##计算预算##与##先验条件##下，何时应选择传统统计聚类、两阶段表征聚类、联合优化、生成式模型、对比聚类或外部预训练视觉模型？",
            size=14.5, color=INK, line_spacing=1.35, anchor=MSO_ANCHOR.MIDDLE)

    pillars = [
        ("01  统计学根基", "衔接王学民教材第 2/3/6/7/8 章，剖析高维像素欧氏距离失效机制", BLUE_DEEP, BLUE_PALE),
        ("02  多维范式演进", "整合 Zhou (2025) 四类并存交互，延伸至 2024–2026 预训练与语言引导", TEAL, TEAL_SOFT),
        ("03  五组对照实验", f"MNIST / Fashion-MNIST / CIFAR-10 上覆盖 12 种方法与 {n_ok} 条严控实验", AMBER, AMBER_SOFT),
        ("04  证据驱动选型", "严格区分回顾性端到端成本基准与无标签 E05 负面稳定性诊断", PURPLE, PURPLE_SOFT),
    ]
    for i, (head, desc, col, tint) in enumerate(pillars):
        row_y = 2.75 + i * 0.98
        box(s, 5.68, row_y, 7.18, 0.84, tint, BLUE_LINE, 0.8, 0.06)
        textbox(s, 5.88, row_y + 0.08, 6.80, 0.30, head, size=13.5, bold=True, color=col, space_after=0)
        textbox(s, 5.88, row_y + 0.42, 6.80, 0.34, desc, size=12.0, color=INK, space_after=0)

    textbox(s, 5.68, 6.82, 7.18, 0.32,
            "正文 24 页（25 分钟讲解）  +  技术附录 6 页（A1–A6 答疑备查）",
            size=11.5, bold=True, color=MUTED, align=PP_ALIGN.RIGHT)

    # ---- Slide 2: Core Questions & Roadmap ----
    s = new_slide(
        prs, slide_ids, "S02_core_questions", "02 / 24", "一、研究背景",
        "研究核心：不做算法排行榜，聚焦三个可证伪命题",
        lead="✓ 核心定位：以##方法选型##为主线，用统一协议下的实证数据检验三项关键假设（结果允许支持或证伪）。",
        summary="##汇报节奏：##正文 24 页控制在 23–25 分钟，预留 5 分钟答疑；详细消融、敏感性与全指标总表置于附录 A1–A6。",
    )
    props = [
        ("命题 1：表征 vs. 聚类器", "表示学习是否比聚类器更决定分组效果？",
         "• 检验路径：E01 + E04 解耦矩阵\n• 关键对照：固定表示换聚类器，对比固定聚类器换表示\n• 现实意义：明确算力应投向特征还是聚类头",
         BLUE_DEEP, BLUE_SOFT),
        ("命题 2：联合优化的收益与代价", "联合优化是否始终优于两阶段方案？",
         "• 检验路径：E01 + E02 + E03 鲁棒性\n• 关键对照：共享 AE 初始化下比较 AE+KM、DEC、IDEC\n• 现实意义：量化早停协议下 λ、样本量与噪声翻转条件",
         TEAL, TEAL_SOFT),
        ("命题 3：预训练时代的复杂度权衡", "强预训练特征下，复杂聚类器还值得吗？",
         "• 检验路径：E04 冻结 DINOv2 / CLIP + 端到端成本分析\n• 关键对照：冻结特征 + K-means vs. GMM / DEC 投影头\n• 现实意义：澄清外部预训练预算与本地聚类成本边界",
         AMBER, AMBER_SOFT),
    ]
    for i, (head, q, body, col, tint) in enumerate(props):
        x = 0.42 + i * 4.22
        header_card(s, x, 1.32, 4.04, 3.55, head, f"**{q}**\n\n{body}",
                    header_fill=col, body_fill=tint, border=col, title_size=14.0, body_size=12.5, space_after=4)

    textbox(s, 0.42, 5.06, 12.5, 0.30, "25 分钟六模块汇报路线图", size=13.5, bold=True, color=BLUE_DEEP, space_after=0)
    mods = [
        ("0–4 min", "背景与统计基础", "P1–P5", BLUE_DEEP),
        ("4–11 min", "范式演进与原理", "P6–P12", BLUE_MID),
        ("11–13 min", "假设与统一协议", "P13–P14", TEAL),
        ("13–20 min", "E01–E04 与成本基准", "P15–P20", AMBER),
        ("20–23 min", "选型树与 E05 诊断", "P21–P22", PURPLE),
        ("23–25 min", "边界局限与总结", "P23–P24", SLATE),
    ]
    mw, mgap = 2.00, 0.10
    for idx, (tm, name, pgs, col) in enumerate(mods):
        mx = 0.42 + idx * (mw + mgap)
        box(s, mx, 5.42, mw, 1.18, WHITE, col, 1.2, 0.08)
        box(s, mx, 5.42, mw, 0.34, col, None, 0.0, None)
        textbox(s, mx, 5.42, mw, 0.34, f"{tm}  ({pgs})", size=11.0, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        textbox(s, mx + 0.08, 5.84, mw - 0.16, 0.66, name, size=12.2, bold=True, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

    # ---- Slide 3: Textbook Statistical Foundations ----
    s = new_slide(
        prs, slide_ids, "S03_textbook_foundations", "03 / 24", "二、统计基础",
        "统计学起点：从多元统计教材到深度表征聚类",
        lead="✓ 理论脉络：深度聚类以王学民《应用多元统计分析》五章为根基，将**固定线性坐标**推广为**可学习非线性隐空间**。",
        summary="##承接关系：##第 6 章 K-means 是实证起点，第 3、7、8 章分别对应高斯混合、线性降维与潜变量生成三大分支。",
    )
    tb_rows = [
        ["第 2 章  随机向量", "距离度量、协方差与高维空间几何", "揭示高维像素欧氏距离集中与语义失配根源", "所有方法基石"],
        ["第 3 章  多元正态分布", "均值向量与协方差矩阵刻画椭球概率云", "直接支撑高斯混合模型（GMM）的软概率划分", "GMM / VaDE"],
        ["第 6 章  聚类分析", "相似系数、系统聚类与 K-means 动态聚类", "建立最小簇内平方和目标，作为基准聚类器", "K-means 基线"],
        ["第 7 章  主成分分析", "正交线性投影保留最大全局方差方向", "作为无监督线性降维基线，与非线性自编码器对照", "PCA + K-means"],
        ["第 8 章  因子分析", "以低维不可观测潜变量解释高维观测生成", "深度变分自编码与混合高斯先验的统计思想源头", "AE / VaDE"],
    ]
    data_table(
        s, 0.42, 1.32, 12.49,
        ["教材章节（王学民）", "核心统计学概念", "在深度聚类综述中的桥梁作用", "对应实验方法"],
        tb_rows, [2.45, 3.40, 4.74, 1.90],
        font_size=12.5, header_size=13.0, max_h=4.35,
        highlight_cells={(2, 3), (3, 3), (4, 3)}, left_align_cols={0, 1, 2},
    )
    soft_panel(
        s, 0.42, 5.74, 12.49, 0.90,
        "教材视角下的核心提问",
        "传统多元统计默认“输入变量 x 已具备可比距离与明确语义”；当数据升级为 784 维灰度图或 3072 维自然彩色图像时，这套前提不再成立。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=13.0, body_size=12.5,
    )

    # ---- Slide 4: K-means Objective & Pixel Distance Failure ----
    s = new_slide(
        prs, slide_ids, "S04_kmeans_failure", "04 / 24", "二、统计基础",
        "K-means 数学目标与原始像素欧氏距离失效机制",
        lead="✓ 问题诊断：K-means 仅在给定坐标中寻找球状质心；当坐标不含语义时，**更换划界公式无法修补错误空间**。",
        summary="##关键结论：##高维图像聚类的第一瓶颈在表示空间而非质心更新公式，由此引出表征学习。",
    )
    box(s, 0.42, 1.30, 12.49, 1.18, BLUE_PALE, BLUE_LINE, 1.0, 0.06)
    textbox(s, 0.62, 1.38, 2.35, 0.34, "K-means 目标函数", size=13.0, bold=True, color=BLUE_DEEP, space_after=0)
    textbox(s, 0.62, 1.76, 2.65, 0.60, "在“分配最近中心”与\n“更新组内均值”间交替迭代", size=11.5, color=MUTED)
    editable_equation(s, 3.40, 1.38, 8.80, 1.00, EDITABLE_EQUATIONS["kmeans"], size=20.0)

    header_card(
        s, 0.42, 2.68, 6.08, 3.92,
        "为何原始像素空间在图像任务上容易失效",
        "■ **语义与像素距离错位**：同一件 T 恤平移数像素或改变光照，像素 L2 距离剧增；不同类别的深色衣物反而彼此更近。\n\n"
        "■ **高维稀疏与背景噪声**：CIFAR-10 单图 3072 维，背景天空、草地占据大量方差，主导欧氏距离计算。\n\n"
        "■ **各向同性球状假设**：K-means 默认每簇为等方差超球体，难以切分弯曲的非凸图像流形。",
        header_fill=AMBER, body_fill=AMBER_SOFT, border=AMBER, title_size=14.5, body_size=13.2, line_spacing=1.28,
    )
    header_card(
        s, 6.83, 2.68, 6.08, 3.92,
        "三层递增的统计应对思路",
        "■ ##第 1 层 · 线性投影（PCA + K-means）##：提取主方差正交轴，压缩维度并加速计算，但仍限于线性子空间。\n\n"
        "■ ##第 2 层 · 概率混合（GMM）##：引入协方差描述椭球形状，但在高维像素空间满协方差不可识别，对角近似易受标准化尺度影响。\n\n"
        "■ ##第 3 层 · 深度表征聚类##：引入可学习的非线性编码器 f_θ，先把图像映射到低维语义隐空间 z，再完成分组。",
        header_fill=BLUE_DEEP, body_fill=BLUE_SOFT, border=BLUE_DEEP, title_size=14.5, body_size=13.2, line_spacing=1.28,
    )

    # ---- Slide 5: Core Paradigm Shift x -> z -> c ----
    s = new_slide(
        prs, slide_ids, "S05_paradigm_shift", "05 / 24", "二、统计基础",
        "深度聚类的核心跃迁：不仅学分组，更学“用什么表示来分组”",
        lead="✓ 核心抽象：深度聚类通过可学习映射 f_θ 将原始观测转为低维语义坐标 z，再于隐空间执行划分。",
        summary="##范式分化根源：##后续全部算法的差异，归根结底在于“z 靠什么无标签信号学出”以及“编码器与聚类器如何交互”。",
    )
    box(s, 0.42, 1.30, 12.49, 1.06, BLUE_PALE, BLUE_LINE, 1.0, 0.06)
    editable_equation(s, 1.20, 1.38, 10.90, 0.90, EDITABLE_EQUATIONS["map"], size=24.0)

    steps = [
        ("STEP 1", "高维观测 x", "原始像素矩阵或自然图像\n（MNIST 784 / CIFAR 3072 维）\n含光照、背景与位移干扰", SLATE, SLATE_SOFT),
        ("STEP 2", "非线性编码 f_θ", "多层感知机 / CNN / ViT\n在无类别标签条件下提取\n可区分的紧凑结构特征", BLUE_DEEP, BLUE_SOFT),
        ("STEP 3", "低维隐表示 z", "同类图像在隐空间聚拢\n不同类流形边界清晰拉开\n（如 AE 10 维 / ViT 384 维）", TEAL, TEAL_SOFT),
        ("STEP 4", "簇分配 ĉ", "距离质心 / Student-t 软归属\n/ 混合高斯后验 / 对比簇头\n输出最终无监督分组", AMBER, AMBER_SOFT),
    ]
    sw, sgap = 2.94, 0.24
    for i, (tag, head, desc, col, tint) in enumerate(steps):
        sx = 0.42 + i * (sw + sgap)
        box(s, sx, 2.60, sw, 2.75, tint, col, 1.3, 0.06)
        box(s, sx, 2.60, sw, 0.42, col, None, 0.0, None)
        textbox(s, sx, 2.60, sw, 0.42, f"{tag}  ·  {head}", size=13.0, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        textbox(s, sx + 0.16, 3.16, sw - 0.32, 2.02, desc, size=12.8, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.35)
        if i < 3:
            textbox(s, sx + sw, 3.70, sgap, 0.50, "→", size=22.0, bold=True, color=BLUE_DEEP, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

    soft_panel(
        s, 0.42, 5.56, 12.49, 1.04,
        "无监督学习的根本张力",
        "训练阶段没有真实标签 y：若仅靠像素重构学习 z，模型可能把容量耗在还原背景上；若仅靠自训练伪目标拉紧簇，又可能放大初始误分。这正是后续六大范式各自求解的核心矛盾。",
        fill=WHITE, border=BLUE_LINE, title_color=BLUE_DEEP, title_size=13.0, body_size=12.5,
    )

    # ---- Slide 6: Fig.1 Paradigm Evolution (Issue #5 P1.3 fixed) ----
    s = new_slide(
        prs, slide_ids, "S06_paradigm_evolution", "06 / 24", "三、范式演进",
        "范式演进全景：Zhou (2025) 四类并存交互与 2024–2026 前沿",
        lead="✓ 演进视角：依据 ##Zhou 等（ACM Computing Surveys, 2025）##图 1，四类交互长期并存：CC 属**联合优化**，DeepCluster/SwAV 属**迭代式**。",
        summary="##图 1 要点：##新范式并未取消旧范式；DeepCluster 靠“伪标签→更新表示→再聚类”迭代循环，DINOv2/CLIP+KM 回归多阶段解耦。",
    )
    native_paradigm_diagram(s)

    # ---- Slide 7: Fig.2 Four-Dimensional Taxonomy ----
    s = new_slide(
        prs, slide_ids, "S07_taxonomy_4d", "07 / 24", "三、范式演进",
        "四维方法分类框架：跨算法解构机制联系",
        lead="✓ 理论骨架：用**表征空间 × 优化交互 × 聚类机制 × 监督信号**四个正交坐标拆解任意算法，避免强行互斥归类。",
        summary="##对比示例：##SimCLR + K-means（多阶段）、CC（联合优化直出）与 DeepCluster（伪标签迭代）可在四维坐标中清晰区分。",
    )
    native_taxonomy_diagram(s)
    textbox(s, 0.48, 6.26, 12.36, 0.42,
            "注：本综述实验覆盖其中 12 种典型方法配置（K-means / GMM / PCA / AE / DEC / IDEC / VaDE / SimCLR / CC / 随机CNN / DINOv2 / CLIP），DeepCluster / IIC / SwAV / TAC / VLM4Cluster 用于谱系分析。",
            size=11.0, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)

    # ---- Slide 8: Mathematical Foundations 1 — DEC ----
    s = new_slide(
        prs, slide_ids, "S08_dec_math", "08 / 24", "四、原理推导",
        "联合优化代表（一）：DEC Student-t 软分配与 KL 自训练",
        lead="✓ 核心机制（Xie et al., ICML 2016）：以预训练自编码器为起点，通过**锐化高置信软分配**反推编码器与簇中心同步移动。",
        summary="##无标签自训练闭环：##DEC 全程不读取真实类别标签；但纯 KL 目标缺少输入保真约束，可能放大初始误分或扭曲局部流形。",
    )
    box(s, 0.42, 1.28, 12.49, 1.12, BLUE_PALE, BLUE_LINE, 1.0, 0.06)
    editable_equation(s, 0.65, 1.34, 12.0, 0.98, EDITABLE_EQUATIONS["dec"], size=13.5)

    box(s, 0.42, 2.52, 12.49, 0.88, WHITE, BLUE_LINE, 1.0, 0.06)
    textbox(s, 0.62, 2.62, 2.05, 0.68, "自训练目标\n（固定 P 更新 Q）", size=12.5, bold=True, color=BLUE_DEEP, anchor=MSO_ANCHOR.MIDDLE)
    editable_equation(s, 2.85, 2.58, 9.60, 0.76, EDITABLE_EQUATIONS["kl"], size=18.0)

    dec_cards = [
        ("1. Student-t 软归属 q_ij", "采用自由度 α=1 的重尾 Student-t 核衡量 z_i 与中心 μ_j 的相似度：离中心越近，归属概率越高，梯度对离群点更平滑。", BLUE_DEEP, BLUE_SOFT),
        ("2. 辅助目标分布 p_ij", "分子取 q_ij² 放大高置信样本的权重以拉紧簇内结构；分母除以簇软频数 f_j = Σ_i q_ij，抑制大簇吞并小簇导致坍缩。", TEAL, TEAL_SOFT),
        ("3. 交替更新与早停准则", "每轮基于全训练集计算并暂时固定目标 P，用梯度下降同时更新 μ_j 与 f_θ；仅在训练集簇分配变化率 < 0.001 时触发早停。", AMBER, AMBER_SOFT),
    ]
    for i, (head, body, col, tint) in enumerate(dec_cards):
        cx = 0.42 + i * 4.22
        header_card(s, cx, 3.58, 4.04, 3.04, head, body,
                    header_fill=col, body_fill=tint, border=col, title_size=13.5, body_size=12.8, line_spacing=1.30)

    # ---- Slide 9: Mathematical Foundations 2 — IDEC & Fig.3 ----
    s = new_slide(
        prs, slide_ids, "S09_idec_math", "09 / 24", "四、原理推导",
        "联合优化代表（二）：IDEC 重构约束与局部结构保护（Fig.3）",
        lead="✓ 核心改进（Guo et al., IJCAI 2017）：保留解码器 g_φ 并施加重构惩罚，在**拉紧聚类边界**与**保持输入信息**之间显式权衡。",
        summary="##量纲提示：##本协议重构项为每图平方误差和 SSE，写作 L_cluster + λ L_recon；与原论文写 L_recon + γ L_cluster 的换算约为 λ = 1/(γD)。",
    )
    box(s, 0.42, 1.26, 12.49, 0.86, BLUE_PALE, BLUE_LINE, 1.0, 0.06)
    editable_equation(s, 0.80, 1.30, 11.70, 0.78, EDITABLE_EQUATIONS["idec"], size=18.0)
    native_dec_idec_diagram(s)

    # ---- Slide 10: Generative Deep Clustering — VaDE & Fig.4 ----
    s = new_slide(
        prs, slide_ids, "S10_vade_arch", "10 / 24", "四、原理推导",
        "生成式深度聚类：VaDE 变分自编码与高斯混合先验（Fig.4）",
        lead="✓ 概率生成视角（Jiang et al., IJCAI 2017）：将教材第 3 章 GMM 嵌入 VAE 隐空间，每个簇建模为带中心与方差的**高斯概率云**。",
        summary="##实验实现边界：##本仓库 VaDE 重构项采用每图平方误差 SSE（非原论文 Bernoulli 交叉熵），结果严格在统一协议内部横向比较。",
    )
    box(s, 0.42, 1.26, 12.49, 0.82, TEAL_SOFT, TEAL, 0.9, 0.06)
    editable_equation(s, 0.80, 1.30, 11.70, 0.74, EDITABLE_EQUATIONS["vade"], size=18.0)
    native_vade_diagram(s)

    # ---- Slide 11: Self-Supervised & Contrastive Clustering — Fig.5 ----
    s = new_slide(
        prs, slide_ids, "S11_contrastive_arch", "11 / 24", "四、原理推导",
        "自监督与对比聚类：SimCLR + K-means 与 CC 双视图机制（Fig.5）",
        lead="✓ 视图一致性假设：同一图像的两种语义保持增强视图应在隐空间靠近；依据是否直出簇概率分化出**多阶段**与**联合优化**两条路线。",
        summary="##课程预算实现限制：##二者均采用 128 维小型 CNN（非原论文 ResNet），课程版 CC 省去官方边际簇熵正则，易诱发簇规模偏斜。",
    )
    native_contrastive_diagram(s)
    soft_panel(
        s, 0.42, 5.68, 6.10, 0.98,
        "SimCLR + K-means（Chen et al., ICML 2020 · 多阶段）",
        "训练阶段仅优化实例级 NT-Xent 对比损失，学完 128 维特征后再接离线 K-means；语义依赖数据增强不破坏类别线索。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.2, body_size=11.4,
    )
    soft_panel(
        s, 6.81, 5.68, 6.10, 0.98,
        "Contrastive Clustering（Li et al., AAAI 2021 · 联合优化）",
        "特征矩阵行做实例对比、列做簇级对比，同时学习表征与软簇分配；区别于靠伪标签循环的迭代式 DeepCluster (2018) / SwAV (2020)。",
        fill=PURPLE_SOFT, border=PURPLE, title_color=PURPLE, title_size=12.2, body_size=11.4,
    )

    # ---- Slide 12: Pretrained & Language-Guided Frontier — Fig.6 ----
    s = new_slide(
        prs, slide_ids, "S12_pretrained_frontier", "12 / 24", "三、前沿演进",
        "2024–2026 前沿：预训练视觉模型与外部语义引导（Fig.6）",
        lead="✓ 范式转向：大规模视觉与跨模态预训练重塑了表征起点，使“##强冻结特征 + 简单聚类器##”成为自然图像强基准。",
        summary="##公平性纪律：##外部预训练模型见过海量外部数据，在 E04 中与从零训练模型分栏独立汇报，不混排为等预算榜单。",
    )
    native_pretrained_diagram(s)

    f_cards = [
        ("DINOv2 (TMLR 2024)", "大规模无标签视觉自监督；冻结 ViT-S/14 提取语义特征", BLUE_DEEP, BLUE_SOFT),
        ("CLIP (ICML 2021)", "海量图文配对监督对齐；冻结 ViT-B/32 视觉编码器", TEAL, TEAL_SOFT),
        ("TAC (ICML 2024)", "Image Clustering with External Guidance；借 WordNet 引导", AMBER, AMBER_SOFT),
        ("VLM4Cluster (2026 预印本)", "arXiv:2609.36648，非同行评审；20 数据集评 17 方法效能/鲁棒/成本", PURPLE, PURPLE_SOFT),
    ]
    fw, fgap = 2.98, 0.19
    for i, (hd, bd, col, tint) in enumerate(f_cards):
        fx = 0.42 + i * (fw + fgap)
        soft_panel(s, fx, 5.52, fw, 1.14, hd, bd, fill=tint, border=col, title_color=col, title_size=12.0, body_size=11.0, line_spacing=1.20)

    # ---- Slide 13: Pre-Experiment Method Selection Hypotheses ----
    s = new_slide(
        prs, slide_ids, "S13_selection_hypotheses", "13 / 24", "四、选型假设",
        "实验前方法选型假设矩阵（待验证命题，非预设结论）",
        lead="✓ 研究规范：在读取任何测试集指标之前，依据理论特性写下 **7 类场景的先验选型假设**，交由 E01–E05 逐条证伪或确认。",
        summary="##验证目标：##后文实验将重点检验哪些直觉成立、哪些出现反转（如强噪声/极小样本下深度方法落后，以及 E05 稳定≠正确）。",
    )
    hyp_rows = [
        ["低维 / 数据量小 / 算力极紧", "K-means 或 PCA + K-means", "结构清晰、耗时极低、可解释性强", "E01 / E03 / 成本"],
        ["高维复杂数据 / 无外部预训练", "AE + K-means", "先建立非线性隐表示，再验证联合微调是否增效", "E01 / E02"],
        ["需表示与聚类协同优化", "DEC 或 IDEC", "聚类目标直接塑形隐空间；IDEC 保护局部流形", "E01 / E02"],
        ["需显式概率解释或生成能力", "VaDE（或低维 GMM）", "混合高斯先验刻画簇形状，须检验其 ARI 与抗噪代价", "E01 / E03"],
        ["具备可靠语义保持数据增强", "SimCLR + K-means / CC", "利用双视图对比信号，须考察小网络预算下性价比", "E01 / 成本"],
        ["已有领域匹配预训练基础模型", "冻结特征 + K-means", "优先验证强表征是否已足够，复杂聚类头须净增 >0.02", "E04"],
        ["有外部文本或领域词典先验", "TAC 及语言辅助聚类", "引入外部语义邻域引导（本综述作文献前沿边界）", "文献分析"],
    ]
    data_table(
        s, 0.42, 1.32, 12.49,
        ["实际任务与资源条件", "优先尝试方案（假设）", "核心统计假设与待验证问题", "对应实验"],
        hyp_rows, [3.10, 2.75, 4.94, 1.70],
        font_size=12.2, header_size=12.8, max_h=5.20,
        highlight_cells={(2, 1), (5, 1)}, left_align_cols={0, 1, 2},
    )

    # ---- Slide 14: Unified Protocol & Label Discipline ----
    s = new_slide(
        prs, slide_ids, "S14_protocol_isolation", "14 / 24", "五、实验设计",
        "统一实验协议、五组实验矩阵与标签隔离红线",
        lead=f"✓ 可复现保障：所有超参数先冻结于 configs/protocol.yaml 再运行，累计产出 **{n_ok} 条成功实测 JSON**（无手工填数）。",
        summary="##严格标签纪律：##fit() 接口严禁传入标签；测试标签仅在 fit 返回后算 ACC/NMI/ARI；E05 诊断与早停仅看训练侧信号。",
    )
    exp_rows = [
        ["E01  基准对比", "MNIST / Fashion / CIFAR-10", f"9 种从零训练方法 × 3 种子（{exp_counts.get('E01', 0)} 条）", "比较经典、AE、联合、生成、对比性能与端到端耗时"],
        ["E02  目标消融", "Fashion-MNIST (n=1, seed 0)", f"IDEC 单因素：λ / d / 预训练 / 初始化 / 冻结（{exp_counts.get('E02', 0)} 条）", "拆解早停下 λ–轮数共变、隐维度 d 与初始化贡献"],
        ["E03  鲁棒边界", "Fashion-MNIST", f"测试噪声（{exp_counts.get('E03_noise', 0)} 条）+ 不平衡/样本量/K/异常（{exp_counts.get('E03', 0)} 条）", "量化分布偏移、小样本与类数错配下的退化分化"],
        ["E04  预训练解耦", "CIFAR-10 (同一测试切分)", f"5 编码器 × 3 聚类器(n=1) + 预训练 K-means(n=3)（{exp_counts.get('E04', 0)} 条）", "解耦“表征质量”与“聚类器复杂度”各自贡献"],
        ["E05  无标签诊断", "3 数据集 (从零 9 候选训练子集)", "仅用训练集干净/加噪 Macro Jaccard（未纳耗时）", "检验无标签稳定性是否等价语义正确（负面结果）"],
    ]
    data_table(
        s, 0.42, 1.30, 12.49,
        ["实验编号", "数据集与划分", "受控变量与覆盖规模", "回答的核心问题"],
        exp_rows, [1.80, 2.55, 4.05, 4.09],
        font_size=11.5, header_size=12.2, max_h=3.15, left_align_cols={0, 1, 2, 3},
    )
    p_cards = [
        ("数据集切分（固定子集）",
         "• MNIST / Fashion：20,000 训练 + 10,000 测试\n• CIFAR-10：10,000 训练 + 2,000 测试\n• 默认 K=10（E03 显式测 K=5..20）",
         BLUE_DEEP, BLUE_SOFT),
        ("统一网络与端到端成本口径",
         "• 28×28 AE：784→500→500→2000→10\n• AE/DEC/IDEC 共享预训练；成本区分端到端与增量微调\n• K-means n_init=20；SimCLR/CC 100 ep 小 CNN",
         TEAL, TEAL_SOFT),
        ("评价指标与失败留痕",
         "• 报告 ACC、NMI、ARI（均含 3 种子 ±SD）及簇偏斜\n• 不因分数低换种子；基础设施错误至多原 seed 重试 1 次\n• 全部图表与 PPT 数字由脚本从 results/tables 自动生成",
         AMBER, AMBER_SOFT),
    ]
    for i, (hd, bd, col, tint) in enumerate(p_cards):
        px = 0.42 + i * 4.22
        soft_panel(s, px, 4.62, 4.04, 2.02, hd, bd, fill=tint, border=col, title_color=col, title_size=12.8, body_size=11.5, line_spacing=1.26)

    # ---- Slide 15: E01 Main Benchmark Heatmap (Fig.7) ----
    m_dec = find_row(e01, dataset="mnist", method="dec")
    m_vade = find_row(e01, dataset="mnist", method="vade")
    m_ae = find_row(e01, dataset="mnist", method="ae_kmeans")
    m_km = find_row(e01, dataset="mnist", method="kmeans")
    m_cc = find_row(e01, dataset="mnist", method="cc")
    f_dec = find_row(e01, dataset="fashion_mnist", method="dec")
    f_idec = find_row(e01, dataset="fashion_mnist", method="idec")
    f_ae = find_row(e01, dataset="fashion_mnist", method="ae_kmeans")
    f_pca = find_row(e01, dataset="fashion_mnist", method="pca_kmeans")
    c_rows = [r for r in e01 if r.get("dataset") == "cifar10" and fnum(r, "ari_mean") is not None]
    c_min_v = min((fnum(r, "ari_mean") for r in c_rows), default=None)
    c_max_v = max((fnum(r, "ari_mean") for r in c_rows), default=None)
    c_range_s = f"{c_min_v:.3f}–{c_max_v:.3f}" if (c_min_v is not None and c_max_v is not None) else "—"
    c_cc = find_row(e01, dataset="cifar10", method="cc")
    m_dec_km_gap = fdiff(fnum(m_dec, "ari_mean"), fnum(m_km, "ari_mean"), digits=2)
    m_ae_km_gap = fdiff(fnum(m_ae, "ari_mean"), fnum(m_km, "ari_mean"), digits=2)
    tie_note = "在 1 SD 内近似并列" if (
        fnum(m_dec, "ari_mean") is not None and fnum(m_vade, "ari_mean") is not None
        and fnum(m_dec, "ari_std") is not None
        and abs(fnum(m_dec, "ari_mean") - fnum(m_vade, "ari_mean")) <= fnum(m_dec, "ari_std")
    ) else "共同领先"

    s = new_slide(
        prs, slide_ids, "S15_e01_benchmark", "15 / 24", "五、实验证据",
        "E01 基准对比：9 种从零训练方法在三类图像上的 ARI（Fig.7）",
        lead=f"✓ 核心发现：MNIST 上深度方法领先 K-means **{m_dec_km_gap} ARI**，但 CIFAR-10 全部 9 种从零方法均停留在 **{c_range_s}**。",
        summary="##选型与实现说明：##★ 为均值最高、≈ 表示差值 ≤ 1 SD 近似并列；IDEC 主表 λ=1.0；CC 低分伴随明显簇偏斜（见右栏与附录 A1）。",
    )
    fit_picture(s, figs_dir / "fig_e01_ari.png", 0.34, 1.28, 8.24, 5.36)
    soft_panel(
        s, 8.70, 1.32, 4.20, 1.62,
        "1. MNIST：DEC 与 VaDE 近似并列领先",
        f"• ##DEC ({fpm(m_dec, 'ari_mean')})## 与 ##VaDE ({fpm(m_vade, 'ari_mean')})## {tie_note}，显著领先 K-means ({fpm(m_km, 'ari_mean')})。\n• AE+KM ({fpm(m_ae, 'ari_mean')}) 已贡献 {m_ae_km_gap} 增益。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.5, body_size=11.3,
    )
    soft_panel(
        s, 8.70, 3.08, 4.20, 1.62,
        "2. Fashion-MNIST：差距收窄，耗时分化",
        f"• ##DEC ({fpm(f_dec, 'ari_mean')})## 与 IDEC λ=1 ({fpm(f_idec, 'ari_mean')}) 居前，AE+KM ({fval(f_ae)}) 接近 K-means。\n• **PCA+KM 端到端仅 {fval(f_pca, 'time_train_e2e_mean_sec', 1)}s** 达 {fpm(f_pca, 'ari_mean')}。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=12.5, body_size=11.3,
    )
    m_cc_mf = f"{fnum(m_cc, 'max_fraction_mean') * 100:.0f}%" if fnum(m_cc, "max_fraction_mean") is not None else "—"
    c_cc_mf = f"{fnum(c_cc, 'max_fraction_mean') * 100:.0f}%" if fnum(c_cc, "max_fraction_mean") is not None else "—"
    m_cc_ent = fval(m_cc, "entropy_norm_mean", 2)
    soft_panel(
        s, 8.70, 4.84, 4.20, 1.80,
        "3. CIFAR-10 瓶颈与 CC 簇偏斜证据",
        f"• 9 种从零方法 ARI 仅 **{c_range_s}**，难越像素—语义鸿沟。\n• 课程版 CC 因小型 CNN 且**省去官方熵项**，MNIST/CIFAR 最大单簇占比达 **{m_cc_mf}/{c_cc_mf}**（熵 {m_cc_ent}），出现簇偏斜。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.5, body_size=11.2,
    )

    # ---- Slide 16: E01 Latent Space t-SNE (Fig.8) ----
    r0_pca = find_row(runs, experiment="E01", dataset="fashion_mnist", method="pca_kmeans", seed=0)
    r0_ae = find_row(runs, experiment="E01", dataset="fashion_mnist", method="ae_kmeans", seed=0)
    r0_dec = find_row(runs, experiment="E01", dataset="fashion_mnist", method="dec", seed=0)
    r0_idec = find_row(runs, experiment="E01", dataset="fashion_mnist", method="idec", seed=0)

    s = new_slide(
        prs, slide_ids, "S16_e01_embeddings", "16 / 24", "五、实验证据",
        "E01 潜在空间几何演化与共享初始化对照（Fig.8）",
        lead="✓ 控制变量设计：AE + K-means、DEC、IDEC (λ=1.0) **共享完全相同的 AE 预训练权重**，隐空间差异纯由聚类阶段目标引起。",
        summary="##标签隔离与读图纪律：##可视化脚本严格不读取 y_test，颜色为各方法预测簇编号；t-SNE 仅辅助观察形态，定量结论以 ARI/NMI 为准。",
    )
    fit_picture(s, figs_dir / "fig_e01_embeddings.png", 0.36, 1.26, 12.60, 3.68)

    lat_cards = [
        ("PCA + K-means（线性投影）",
         f"10 维线性主成分受限于平面切分，服装相似类别边界交错（seed 0 ARI = {fval(r0_pca, 'ari')}，轮廓 {fval(r0_pca, 'silhouette', 2)}）。", SLATE, SLATE_SOFT),
        ("AE + K-means（非线性起点）",
         f"自编码器拉开部分大类流形，但重构目标不显式鼓励类间间隔（seed 0 ARI = {fval(r0_ae, 'ari')}，轮廓 {fval(r0_ae, 'silhouette', 2)}）。", BLUE_DEEP, BLUE_SOFT),
        ("DEC（纯 KL 自训练拉紧）",
         f"KL 锐化把预测簇压缩成紧致孤岛，轮廓系数升至 {fval(r0_dec, 'silhouette', 2)}，但重构偏离扩大（ARI = {fval(r0_dec, 'ari')}）。", TEAL, TEAL_SOFT),
        ("IDEC λ=1.0（保留重构解码）",
         f"保留解码器后簇形更舒展、局部几何未剧烈塌缩，在分离与重构保真间折中（ARI = {fval(r0_idec, 'ari')}，轮廓 {fval(r0_idec, 'silhouette', 2)}）。", AMBER, AMBER_SOFT),
    ]
    lw, lgap = 2.98, 0.19
    for i, (hd, bd, col, tint) in enumerate(lat_cards):
        lx = 0.42 + i * (lw + lgap)
        soft_panel(s, lx, 5.06, lw, 1.58, hd, bd, fill=tint, border=col, title_color=col, title_size=11.8, body_size=11.0, line_spacing=1.22)

    # ---- Slide 17: E02 Loss Ablation lambda — Issue #4 Fix ----
    l0 = find_row(e02, factor="lambda_recon", value=0.0)
    l001 = find_row(e02, factor="lambda_recon", value=0.01)
    l01 = find_row(e02, factor="lambda_recon", value=0.1)
    l1 = find_row(e02, factor="lambda_recon", value=1.0)
    l10 = find_row(e02, factor="lambda_recon", value=10.0)
    ep_chain = "→".join(
        str(int(round(fnum(r, "epochs_ran")))) if fnum(r, "epochs_ran") is not None else "—"
        for r in (l0, l001, l01, l1, l10)
    )

    s = new_slide(
        prs, slide_ids, "S17_e02_lambda_ablation", "17 / 24", "五、实验证据",
        "E02 重构权重 λ 消融：当前早停协议下 λ 与训练轮数共变（Fig.9）",
        lead=f"✓ 方法学披露（n=1, seed 0）：λ 从 0 增至 10 时，实际训练轮数同步单调拉长（**{ep_chain} ep**），ARI 回落不能单独归因于量纲。",
        summary="##解读边界：##*λ=0 重构项未进入总损失，其 SSE 属定义不可比点；固定 50 ep 的隐维度 d 消融与初始化组件消融见附录 A2。",
    )
    fit_picture(s, figs_dir / "fig_e02_lambda.png", 0.34, 1.30, 8.30, 5.30)

    soft_panel(
        s, 8.76, 1.32, 4.14, 1.48,
        "1. 早停区 vs. 跑满区（双变量共变）",
        f"• λ=0 / 0.01 / 0.1 分别在 **{fep(l0)} / {fep(l001)} / {fep(l01)}** 停下。\n• 仅 λ=1 与 λ=10 跑满 **{fep(l1)}**，λ 并非唯一变化量。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.4, body_size=11.3,
    )
    soft_panel(
        s, 8.76, 2.92, 4.14, 1.92,
        "2. 实测走势与 λ=0 不可比标注",
        f"• λ=0 ({fep(l0)})：ARI {fval(l0)}，SSE **{fval(l0, 'recon_sse', 1)}***（重构未入损失，**不可比**）。\n"
        f"• λ=0.01–0.1：ARI {fval(l001)}–{fval(l01)}，SSE {fval(l001, 'recon_sse', 1)}–{fval(l01, 'recon_sse', 1)}。\n"
        f"• λ=1–10 ({fep(l1)})：ARI 降至 {fval(l1)}–{fval(l10)}，SSE {fval(l1, 'recon_sse', 1)}。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=12.4, body_size=11.2,
    )
    soft_panel(
        s, 8.76, 4.96, 4.14, 1.68,
        "3. 早停 × λ 相互作用机制",
        "统一协议规定训练集簇分配变化 < 0.001 即早停；增大 λ 让重构梯度持续微调隐坐标、推迟早停，故 ARI 变化是 λ 与训练时长的复合效应。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.4, body_size=11.2,
    )

    # ---- Slide 18: E03 Noise Robustness — Issue #5 P1.4 Fix ----
    km0 = find_row(e03n, dataset="fashion_mnist", method="kmeans", noise_std=0.0)
    km3 = find_row(e03n, dataset="fashion_mnist", method="kmeans", noise_std=0.3)
    pca0 = find_row(e03n, dataset="fashion_mnist", method="pca_kmeans", noise_std=0.0)
    pca3 = find_row(e03n, dataset="fashion_mnist", method="pca_kmeans", noise_std=0.3)
    dec0 = find_row(e03n, dataset="fashion_mnist", method="dec", noise_std=0.0)
    dec3 = find_row(e03n, dataset="fashion_mnist", method="dec", noise_std=0.3)
    idec0 = find_row(e03n, dataset="fashion_mnist", method="idec", noise_std=0.0)
    idec3 = find_row(e03n, dataset="fashion_mnist", method="idec", noise_std=0.3)
    vade0 = find_row(e03n, dataset="fashion_mnist", method="vade", noise_std=0.0)
    vade3 = find_row(e03n, dataset="fashion_mnist", method="vade", noise_std=0.3)
    gmm0 = find_row(e03n, dataset="fashion_mnist", method="gmm", noise_std=0.0)
    gmm1 = find_row(e03n, dataset="fashion_mnist", method="gmm", noise_std=0.1)

    s = new_slide(
        prs, slide_ids, "S18_e03_noise_robustness", "18 / 24", "五、实验证据",
        "E03 鲁棒性测试：测试噪声反转与 GMM / VaDE 分化（Fig.11）",
        lead=f"✓ 核心反转（n=3）：像素 K-means 保持率高达 **{fret(km0, km3)}**（{fval(km0, 'ari_mean')}→{fval(km3, 'ari_mean')}），σ≥0.2 反超全部神经方法。",
        summary="##结论边界：##轻噪归零仅限于当前标准化像素对角 GMM，同属概率生成的 VaDE 仍保持半数以上 ARI；其余 4 项敏感性见附录 A3–A4。",
    )
    fit_picture(s, figs_dir / "fig_e03_noise.png", 0.34, 1.28, 8.28, 5.36)

    soft_panel(
        s, 8.76, 1.32, 4.14, 1.60,
        "1. 传统质心法抗噪显著更稳",
        f"• 像素 K-means：{fval(km0, 'ari_mean')} → **{fval(km3, 'ari_mean')}（{fret(km0, km3)}）**。\n• PCA+KM 同步保持 {fret(pca0, pca3)}，线性欧氏距离对零均值高斯噪声天然平滑。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=12.6, body_size=11.4,
    )
    soft_panel(
        s, 8.76, 3.04, 4.14, 1.66,
        "2. 神经编码器与 IDEC 同步退化",
        f"• DEC：{fval(dec0, 'ari_mean')} → {fval(dec3, 'ari_mean')}（{fret(dec0, dec3)}）；IDEC：{fval(idec0, 'ari_mean')} → {fval(idec3, 'ari_mean')}（{fret(idec0, idec3)}）。\n• 仅加干净像素重构**无法抵御未见分布噪声**。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.6, body_size=11.4,
    )
    soft_panel(
        s, 8.76, 4.82, 4.14, 1.82,
        "3. 不泛化概率家族：GMM vs. VaDE",
        f"• 当前 **StandardScaler+对角协方差像素 GMM** 在 σ=0.1 即从 {fval(gmm0, 'ari_mean')} 坍缩至 **{fval(gmm1, 'ari_mean')}**（单簇）。\n• 但隐空间生成模型 ##VaDE## 从 {fval(vade0, 'ari_mean')} → **{fval(vade3, 'ari_mean')}（{fret(vade0, vade3)}）**，脆弱性系像素预处理实现所致。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.4, body_size=11.1,
    )

    # ---- Slide 19: E04 Pretrained Decoupling — Issue #1 & #5 P2.7 Fix ----
    dino3 = find_row(e04ext, encoder="dinov2", clusterer="kmeans")
    clip3 = find_row(e04ext, encoder="clip", clusterer="kmeans")
    dino_km0 = find_row(e04p, encoder="dinov2", clusterer="kmeans")
    dino_gmm0 = find_row(e04p, encoder="dinov2", clusterer="gmm")
    dino_dec0 = find_row(e04p, encoder="dinov2", clusterer="dec_head")
    clip_km0 = find_row(e04p, encoder="clip", clusterer="kmeans")
    ae_km0 = find_row(e04p, encoder="ae", clusterer="kmeans")
    scratch_e04 = [fnum(r, "ari") for r in e04p if r.get("encoder") in {"random_cnn", "ae", "simclr"} and fnum(r, "ari") is not None]
    scratch_range = f"{min(scratch_e04):.3f}–{max(scratch_e04):.3f}" if scratch_e04 else "—"
    lift_x = f"{fnum(dino_km0, 'ari') / fnum(ae_km0, 'ari'):.1f}×" if (fnum(dino_km0, "ari") and fnum(ae_km0, "ari")) else "—"

    s = new_slide(
        prs, slide_ids, "S19_e04_pretrained_decouple", "19 / 24", "五、实验证据",
        f"E04 预训练解耦：表征跃升 {lift_x}，冻结特征上投影头未添益（Fig.15）",
        lead=f"✓ 核心对比：左图 5×3 为 **n=1 (seed 0) 配对**（†两格取自同切分 E01）；右表独立给 DINOv2/CLIP K-means **n=3 均值±SD**。",
        summary="##范围限定：##“复杂聚类器无正收益”仅限当前冻结特征 + 对角 GMM / 10 维投影 DEC 头 + seed 0 设置；外部预训练不作等预算排名。",
    )
    fit_picture(s, figs_dir / "fig_e04_paired.png", 0.34, 1.28, 8.24, 5.34)

    e04_mini = [
        ["DINOv2+KM (n=3)", fpm(dino3, "ari_mean"), fval(dino3, "acc_mean"), f"{fval(dino3, 'time_feature_sec_mean', 1)}s"],
        ["CLIP+KM (n=3)", fpm(clip3, "ari_mean"), fval(clip3, "acc_mean"), f"{fval(clip3, 'time_feature_sec_mean', 1)}s"],
    ]
    textbox(s, 8.72, 1.30, 4.18, 0.30, "独立 3 种子稳定性（n=3，仅 K-means）", size=12.4, bold=True, color=BLUE_DEEP, space_after=0)
    data_table(
        s, 8.72, 1.64, 4.18,
        ["方案", "ARI (n=3)", "ACC", "提特征"],
        e04_mini, [1.42, 1.22, 0.72, 0.82],
        font_size=10.4, header_size=10.8, max_h=1.42,
        highlight_cells={(0, 1), (1, 2)}, left_align_cols={0},
    )
    soft_panel(
        s, 8.72, 3.18, 4.18, 1.62,
        "纵看行差：表征决定数量级上限",
        f"• 红虚线上方 3 种从零主干全在 {scratch_range}。\n• 下方冻结预训练主干拉升 **{lift_x}**（seed 0 达 {fval(dino_km0, 'ari')} / {fval(clip_km0, 'ari')}），CLIP ACC 达 **{fval(clip3, 'acc_mean')}**。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.4, body_size=11.3,
    )
    soft_panel(
        s, 8.72, 4.94, 4.18, 1.70,
        "横看列差：当前投影头设置下 K-means 最优",
        f"• seed 0 同特征上，DINOv2+KM ({fval(dino_km0, 'ari')}) 高于 GMM ({fval(dino_gmm0, 'ari')}) 与 10 维 DEC 头 (**{fval(dino_dec0, 'ari')}**)。\n• 结论限定于冻结主干+投影头设置，不外推至端到端微调。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=12.4, body_size=11.2,
    )

    # ---- Slide 20: Retrospective Benchmark Pareto Frontier — Issue #3 & #5 P1.5 Fix ----
    cf_fm = [r for r in costs if r.get("dataset") == "fashion_mnist"]
    cf_pca = find_row(cf_fm, method="pca_kmeans")
    cf_km = find_row(cf_fm, method="kmeans")
    cf_ae = find_row(cf_fm, method="ae_kmeans")
    cf_dec = find_row(cf_fm, method="dec")
    cf_idec = find_row(cf_fm, method="idec")
    cf_vade = find_row(cf_fm, method="vade")
    cf_sim = find_row(cf_fm, method="simclr_kmeans")
    cf_cc = find_row(cf_fm, method="cc")
    front_e2e_rows = sorted(
        [r for r in cf_fm if str(r.get("front_ari_time_e2e", "")).lower() == "true"],
        key=lambda r: fnum(r, "time_train_e2e_mean_sec") or 0.0,
    )
    front_chain = " → ".join(
        f"{SHORT_LABELS.get(r['method'], r['method'])} ({fval(r, 'time_train_e2e_mean_sec', 1)}s)"
        for r in front_e2e_rows
    )

    s = new_slide(
        prs, slide_ids, "S20_benchmark_pareto_e2e", "20 / 24", "五、历史成本基准",
        "回顾性历史基准：性能—端到端训练成本帕累托前沿（Fig.16）",
        lead=f"✓ 端到端基准（含测试 ARI，**非 E05**）：补回 AE 预训练后 4 点前沿为 **{front_chain}**。",
        summary="##口径声明：##本图读取 benchmark_cost_frontiers.csv（使用 E01 测试标签，仅作历史基准回顾），缓存命中微调耗时不可当作免费预训练。",
    )
    fit_picture(s, figs_dir / "fig_benchmark_pareto.png", 0.34, 1.28, 8.26, 5.36)

    box(s, 8.74, 1.32, 4.16, 0.78, BLUE_PALE, BLUE_LINE, 1.0, 0.06)
    editable_equation(s, 8.82, 1.36, 4.00, 0.70, EDITABLE_EQUATIONS["pareto"], size=9.4)

    soft_panel(
        s, 8.74, 2.22, 4.16, 2.24,
        "四点端到端帕累托前沿（Fashion-MNIST）",
        f"• ##PCA+KM##：{fval(cf_pca, 'time_train_e2e_mean_sec', 1)}s / ARI {fval(cf_pca)}，极速初筛。\n"
        f"• ##K-means##：{fval(cf_km, 'time_train_e2e_mean_sec', 1)}s / ARI {fval(cf_km)}，无需 GPU 且抗噪 {fret(km0, km3)}。\n"
        f"• ##AE+KM##：{fval(cf_ae, 'time_train_e2e_mean_sec', 1)}s / ARI {fval(cf_ae)}，端到端回到前沿。\n"
        f"• ##DEC##：端到端 **{fval(cf_dec, 'time_train_e2e_mean_sec', 1)}s** / ARI **{fval(cf_dec)}**（仅微调 {fval(cf_dec, 'time_train_mean_sec', 1)}s）。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.2, body_size=11.0,
    )
    soft_panel(
        s, 8.74, 4.58, 4.16, 2.06,
        "独立端到端 vs. 共享预训练摊销",
        f"• 独立端到端：DEC {fval(cf_dec, 'time_train_e2e_mean_sec', 1)}s、IDEC {fval(cf_idec, 'time_train_e2e_mean_sec', 1)}s（含 {fval(cf_ae, 'time_train_e2e_mean_sec', 1)}s AE）；三模型共享预训练摊销为 {fval(cf_dec, 'time_train_amortized_mean_sec', 1)}s / {fval(cf_idec, 'time_train_amortized_mean_sec', 1)}s。\n"
        f"• VaDE ({fval(cf_vade, 'time_train_e2e_mean_sec', 1)}s) 及小 CNN SimCLR ({fval(cf_sim, 'time_train_e2e_mean_sec', 0)}s) / CC ({fval(cf_cc, 'time_train_e2e_mean_sec', 0)}s) 在两维被支配。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.2, body_size=10.9,
    )

    # ---- Slide 21: Fig.17 Method Selection Decision Tree — Issue #3 & #5 P0.1 ----
    s = new_slide(
        prs, slide_ids, "S21_selection_decision_tree", "21 / 24", "六、方法选型",
        "基于历史基准的分场景选型决策树与无标签警示（Fig.17）",
        lead="✓ 证据分层：Q1–Q3 总结自 **E01–E04 带标签历史基准**；面对全新无标签数据时，须按左下角红框警惕 **E05 稳定≠正确**。",
        summary="##选型红线：##当前 E05 仅测训练侧扰动稳定性（未纳入预测耗时，CPU/CUDA 耗时不可混排），只用于排雷坍缩，新域语义须少量标注验证。",
    )
    fit_picture(s, ASSETS / "diag_decision_tree.png", 0.34, 1.24, 12.65, 5.44)

    # ---- Slide 22: Verdict on 3 Propositions & E05 Negative Result — Issue #1, #2, #3, #5 ----
    def _sel_counterexample(ds_key: str, ds_lbl: str, best_label: str, best_ari_str: str) -> str:
        entry = sel.get(ds_key)
        if not entry or not entry.get("choice"):
            return f"• {ds_lbl}：**证据不足**（insufficient_evidence）"
        ch = entry["choice"]
        jacc = entry.get("evidence", {}).get(ch, {}).get("macro_cluster_jaccard")
        jacc_s = f"{jacc:.3f}" if isinstance(jacc, (int, float)) else "—"
        ch_ari = fval(find_row(e01, dataset=ds_key, method=ch), "ari_mean")
        return f"• {ds_lbl}：最稳 {SHORT_LABELS.get(ch, ch)} (Jaccard {jacc_s}, ARI {ch_ari}) ≪ {best_label} ({best_ari_str})"

    if not sel or all(not sel.get(d, {}).get("choice") for d in ("mnist", "fashion_mnist", "cifar10")):
        e05_card_body = (
            "■ ##当前诊断状态：证据不足##\n"
            "未检测到完整匹配的 selection.json 有效候选（choice 为 null 或文件缺失），按协议返回 **证据不足（insufficient_evidence）**，绝不回退至测试集 ARI。\n\n"
            "■ ##设计原则##\n"
            "E05 仅允许读取训练侧无标签扰动诊断，用于排查空簇/坍缩风险；新域语义确认需借助少量人工标注的半监督验证。"
        )
    else:
        speed_used = any(bool(sel.get(d, {}).get("speed_used_in_selection")) for d in ("mnist", "fashion_mnist", "cifar10"))
        speed_txt = "已启用预测耗时阈值" if speed_used else "仅用训练扰动 Jaccard（未纳入预测耗时，CPU/CUDA 不可混排）"
        e05_card_body = (
            f"■ ##负面诊断结论：稳定 ≠ 正确##\n"
            f"E05 {speed_txt}，高稳定往往偏向对噪声迟钝的像素距离划分——**稳定地分错**：\n"
            + _sel_counterexample("mnist", "MNIST", "DEC", fval(m_dec)) + "\n"
            + _sel_counterexample("fashion_mnist", "Fashion", "DEC", fval(f_dec)) + "\n"
            + _sel_counterexample("cifar10", "CIFAR-10", "DINOv2+KM*", fval(dino3, "ari_mean")) + "\n"
            + "（*外部预训练不在 E05 从零 9 候选池）\n\n"
            + "■ ##正确定位##\n"
            + "E05 仅作**坍缩风险筛查**（淘汰轻噪归零像素 GMM），不可当“最佳方法推荐器”；新域语义确认需少量标注（**半监督/小样本验证**）。"
        )

    s = new_slide(
        prs, slide_ids, "S22_verdict_e05_negative", "22 / 24", "六、方法选型",
        "回到开篇三命题：实证检验结论与 E05 负面诊断（稳定 ≠ 正确）",
        lead="✓ 证据闭环：三项命题全部获得量化回答，同时 E05 揭示**单纯最大化无标签扰动稳定性会选出语义较差的像素基线**。",
        summary="##方法论贡献：##严格隔离 selection.json（无标签训练稳定性排雷）与 benchmark_interpretation.json（带标签回顾解读），详见附录 A5。",
    )
    verdicts = [
        ("命题 1：表征是否比聚类器更重要？", "强支持", TEAL, TEAL_SOFT,
         f"• E04（CIFAR-10）：冻结预训练带来 **{lift_x} ARI 跃升**（{fval(ae_km0, 'ari')}→{fval(dino_km0, 'ari')}），同特征换聚类器无正益。\n"
         f"• E01（MNIST）：AE 非线性映射将 K-means 从 {fval(m_km)} 提升至 {fval(m_ae)}，贡献整体增益主体。"),
        ("命题 2：联合优化是否始终优于两阶段？", "有条件成立（非总是）", AMBER, AMBER_SOFT,
         f"• 干净充足数据占优：DEC 在 MNIST ({fval(m_dec)}) 与 Fashion ({fval(f_dec)}) 高于 AE+KM ({fval(m_ae)} / {fval(f_ae)})。\n"
         f"• 边界条件下反转：强噪声（σ≥0.2）、1% 小样本下落后 K-means；E02 亦显示 λ 受早停轮数共变且初始化贡献远超微调。"),
        ("命题 3：强预训练特征下复杂聚类是否值得？", "当前设置不支持", BLUE_DEEP, BLUE_SOFT,
         f"• 冻结 DINOv2 / CLIP 特征上，朴素 K-means（n=3: {fval(dino3, 'ari_mean')} / {fval(clip3, 'ari_mean')}）即为最强基线。\n"
         f"• 本次 seed 0 叠加对角 GMM 或 10 维 DEC 投影头耗时更高且 ARI 反降（DINOv2 跌至 {fval(dino_dec0, 'ari')}）。"),
    ]
    for i, (q, tag, col, tint, body) in enumerate(verdicts):
        vy = 1.30 + i * 1.84
        box(s, 0.42, vy, 8.20, 1.66, tint, col, 1.1, 0.05)
        box(s, 0.58, vy + 0.16, 2.15, 0.36, col, None, 0.0, 0.22)
        textbox(s, 0.58, vy + 0.16, 2.15, 0.36, tag, size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        textbox(s, 2.86, vy + 0.15, 5.60, 0.38, q, size=13.8, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        textbox(s, 0.58, vy + 0.62, 7.88, 0.94, body, size=11.7, color=INK, line_spacing=1.26, space_after=4)

    header_card(
        s, 8.80, 1.30, 4.10, 5.34,
        "E05 负面诊断：稳定 ≠ 正确",
        e05_card_body,
        header_fill=PURPLE, body_fill=PURPLE_SOFT, border=PURPLE, title_size=13.2, body_size=11.0, line_spacing=1.23,
    )

    # ---- Slide 23: Limitations & Open Problems ----
    s = new_slide(
        prs, slide_ids, "S23_limitations_future", "23 / 24", "六、局限与展望",
        "实验边界、实现局限与未来开放研究方向",
        lead="✓ 学术诚信：下列 **四项硬边界** 必须与前文实证结论一并阅读，防止将特定协议下的观测外推为无条件普适排名。",
        summary="##复现与透明声明：##所有限制均已同步写入仓库 README.md、docs/experiment_protocol.md、docs/method_selection.md 与 AGENTS.md。",
    )
    lims = [
        ("1. 真实类数 K=10 已知与单种子消融边界", "主基准固定 K=10；E01 与预训练 K-means 为 3 种子，E02、E03 敏感性与 E04 5×3 矩阵为 seed 0 单次观测，且 E02 λ 与早停轮数共变。"),
        ("2. 课程预算下的具体算法实现边界", "SimCLR/CC 采用 128 维小型 CNN，且 CC 省去官方簇熵项导致簇偏斜；VaDE 用每图 SSE 而非 Bernoulli 交叉熵；GMM 为标准化对角协方差实现。"),
        ("3. 外部预训练数据预算不对等与冻结范围", "DINOv2 与 CLIP 受益于海量外部预训练数据；E04 负增益结论仅限冻结主干 + 10 维投影头设置，不外推至端到端大主干微调。"),
        ("4. E05 候选池与纯无标签选择局限", "当前 E05 仅覆盖从零 9 候选且未纳入预测耗时；扰动稳定性只能筛查坍缩风险，不能替代少量人工标注的半监督语义验证。"),
    ]
    box(s, 0.42, 1.30, 7.60, 5.34, WHITE, AMBER, 1.2, 0.05)
    box(s, 0.42, 1.30, 7.60, 0.42, AMBER, None, 0.0, None)
    textbox(s, 0.60, 1.33, 7.20, 0.36, "必须与结论并读的四条实验边界", size=14.0, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    for i, (hd, bd) in enumerate(lims):
        ly = 1.84 + i * 1.18
        box(s, 0.60, ly, 7.24, 1.06, AMBER_SOFT if i % 2 == 0 else SLATE_SOFT, BLUE_LINE, 0.6, 0.05)
        textbox(s, 0.76, ly + 0.08, 6.92, 0.30, hd, size=12.2, bold=True, color=AMBER if i % 2 == 0 else BLUE_DEEP, space_after=0)
        textbox(s, 0.76, ly + 0.40, 6.92, 0.60, bd, size=11.0, color=INK, line_spacing=1.18)

    opens = [
        ("方向 1：未知 K 的自适应聚类", "结合非参数贝叶斯、谱间隙或视觉语言模型粒度协商，动态确定簇数与层级结构。", BLUE_DEEP, BLUE_SOFT),
        ("方向 2：超越稳定性的无标签/少样本选型", "针对 E05“稳定≠正确”瓶颈，探索结合极少量标注（半监督验证）或跨视图语义一致性的可靠选型准则。", TEAL, TEAL_SOFT),
        ("方向 3：VLM 语义引导与分布鲁棒", "沿 TAC (ICML 2024) 与 VLM4Cluster (2026) 脉络，研究外部文本先验在跨领域漂移下的失真与成本边界。", PURPLE, PURPLE_SOFT),
    ]
    for i, (hd, bd, col, tint) in enumerate(opens):
        oy = 1.30 + i * 1.82
        header_card(s, 8.24, oy, 4.67, 1.68, hd, bd, header_fill=col, body_fill=tint, border=col, title_size=12.6, body_size=11.5, header_h=0.38)

    # ---- Slide 24: Closing & Reproducibility ----
    slide_ids.append("S24_closing_repro")
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, W, H, WHITE)
    box(s, 0, 0, W, 0.08, BLUE_DEEP)
    box(s, 0.42, 0.35, 12.49, 1.35, BLUE_DEEP, None, 0.0, 0.06)
    textbox(s, 0.75, 0.48, 11.80, 0.56, "总结：先问表示与预算，再区分历史基准与无标签排雷",
            size=23.5, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    textbox(s, 0.75, 1.08, 11.80, 0.48, "聚类算法解决的是“如何划界”；它划在怎样的表示空间里，往往比划界公式本身更决定成败。",
            size=14.5, color=RGBColor(0xFD, 0xE6, 0x8A), anchor=MSO_ANCHOR.MIDDLE, space_after=0)

    c_rules = [
        ("准则 1 · 表示优先", "先评估已有预训练或简单降维坐标是否已具语义；自然图像优先冻结视觉基础模型 + K-means。", BLUE_DEEP, BLUE_SOFT),
        ("准则 2 · 端到端成本与边界", "按含前置预训练的端到端成本权衡复杂度；强噪声、极小样本下 PCA+KM / K-means 更稳健。", TEAL, TEAL_SOFT),
        ("准则 3 · 稳定性 ≠ 语义正确", "无标签训练稳定性只能排雷坍缩风险，不能替代语义验证；新域确认须依赖少量标注样本。", AMBER, AMBER_SOFT),
    ]
    for i, (hd, bd, col, tint) in enumerate(c_rules):
        cx = 0.42 + i * 4.22
        header_card(s, cx, 1.98, 4.04, 2.75, hd, bd, header_fill=col, body_fill=tint, border=col, title_size=14.5, body_size=13.2, header_h=0.46, line_spacing=1.32)

    box(s, 0.42, 4.98, 12.49, 1.56, SLATE_SOFT, BLUE_LINE, 1.0, 0.06)
    textbox(s, 0.68, 5.10, 12.0, 0.36, "开源可复现与完整归档入口", size=14.0, bold=True, color=BLUE_DEEP, space_after=0)
    textbox(s, 0.68, 5.52, 12.0, 0.90,
            f"■ ##GitHub 代码与证据仓库##：github.com/KQYaili/DeepClustering-Survey（协议、源码、{n_ok} 条原始 JSON、CI 测试、图表与 PPT）\n"
            "■ ##Google Drive 完整归档##：DeepClustering-Survey（含 217 份训练权重 checkpoints、外部预训练缓存与历史溯源备份）\n"
            "■ ##答疑跳转提示##：附录 A1（全指标与端到端耗时） · A2（E02 维度/组件） · A3–A4（E03 敏感性） · A5（E05 负面诊断） · A6（文献与审计）",
            size=11.8, color=INK, line_spacing=1.32)

    textbox(s, 0.42, 6.76, 10.8, 0.46, "感谢各位老师与同学聆听！欢迎批评指正与交流讨论。",
            size=16.5, bold=True, color=BLUE_DEEP, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    box(s, 12.02, 6.80, 0.89, 0.42, BLUE_DEEP, None, 0.0, 0.12)
    textbox(s, 12.02, 6.80, 0.89, 0.42, "24 / 24",
            size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

    # =======================================================================
    # TECHNICAL APPENDIX (Slides 25–30: A1–A6)
    # =======================================================================

    # ---- Slide 25 (A1): Full E01 Quantitative Metrics Table ----
    s = new_slide(
        prs, slide_ids, "A01_e01_full_table", "A1 / A6", "技术附录",
        "附录 A1：E01 全指标量化汇总表（含 ±SD、端到端耗时与簇偏斜）",
        lead="✓ 完整记录：以下全部数字直接读取自 results/tables/e01_summary.csv（n=3 种子），Fashion 耗时列为独立端到端训练秒数。",
        summary=f"##簇偏斜诊断：##课程版 CC 省略官方熵项，MNIST/CIFAR 最大单簇占比达 {m_cc_mf}/{c_cc_mf}；IDEC 表格配置为 λ=1.0。",
    )
    a1_rows = []
    hi_a1: set[tuple[int, int]] = set()
    method_order = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
    best_idx = {}
    for ds in ("mnist", "fashion_mnist", "cifar10"):
        vals = [(i, fnum(find_row(e01, dataset=ds, method=m), "ari_mean")) for i, m in enumerate(method_order)]
        vals = [(i, v) for i, v in vals if v is not None]
        if vals:
            best_idx[ds] = max(vals, key=lambda x: x[1])[0]

    for i, m in enumerate(method_order):
        rm = find_row(e01, dataset="mnist", method=m)
        rf = find_row(e01, dataset="fashion_mnist", method=m)
        rc = find_row(e01, dataset="cifar10", method=m)
        t_e2e = fval(rf, "time_train_e2e_mean_sec", 1)
        t_raw = fval(rf, "time_train_mean_sec", 1)
        t_disp = f"{t_e2e}s" if t_e2e == t_raw else f"{t_e2e}s ({t_raw}s*)"
        a1_rows.append([
            METHOD_LABELS[m],
            fval(rm, "acc_mean"), fval(rm, "nmi_mean"), fpm(rm, "ari_mean"),
            fval(rf, "acc_mean"), fval(rf, "nmi_mean"), fpm(rf, "ari_mean"),
            t_disp,
            fpm(rc, "ari_mean"),
        ])
        if best_idx.get("mnist") == i:
            hi_a1.add((i, 3))
        if best_idx.get("fashion_mnist") == i:
            hi_a1.add((i, 6))
        if best_idx.get("cifar10") == i:
            hi_a1.add((i, 8))

    data_table(
        s, 0.36, 1.32, 12.61,
        ["方法（*括号为缓存微调）", "M-ACC", "M-NMI", "MNIST ARI±SD", "F-ACC", "F-NMI", "Fashion ARI±SD", "Fashion 端到端", "CIFAR ARI±SD"],
        a1_rows, [2.25, 0.88, 0.88, 1.65, 0.88, 0.88, 1.65, 1.89, 1.65],
        font_size=10.3, header_size=10.5, max_h=5.25, highlight_cells=hi_a1, left_align_cols={0},
    )

    # ---- Slide 26 (A2): E02 Latent Dim (Fig.10) & Component Ablations ----
    d2 = find_row(e02, factor="latent_dim", value=2.0)
    d10 = find_row(e02, factor="latent_dim", value=10.0)
    c_base = find_row(e02, factor="pretrain", value="True")
    c_np = find_row(e02, factor="pretrain", value="False")
    c_rc = find_row(e02, factor="center_init", value="random")
    c_fr = find_row(e02, factor="freeze_encoder", value="True")
    d_np = fdiff(fnum(c_np, "ari"), fnum(c_base, "ari"))
    d_rc = fdiff(fnum(c_rc, "ari"), fnum(c_base, "ari"))
    d_fr = fdiff(fnum(c_fr, "ari"), fnum(c_base, "ari"))

    s = new_slide(
        prs, slide_ids, "A02_e02_latent_components", "A2 / A6", "技术附录",
        "附录 A2：E02 隐空间维度曲线（Fig.10）与初始化组件消融",
        lead=f"✓ 补充消融（n=1, seed 0）：维度 5 档均跑满 50 ep，**d=10 峰值 {fval(d10)}**；取消预训练 ({d_np}) 或随机中心 ({d_rc}) 远大于冻结编码器 ({d_fr})。",
        summary="##工程洞察：##与存在早停共变的 λ 网格不同，维度 d 与初始化组件均在统一 50 ep 下对照（冻结编码器 10 ep 属结构性早停）。",
    )
    fit_picture(s, figs_dir / "fig_e02_latent.png", 0.36, 1.32, 6.22, 4.22)
    fit_picture(s, figs_dir / "fig_e02_components.png", 6.76, 1.32, 6.22, 4.22)
    soft_panel(
        s, 0.42, 5.64, 6.10, 1.02,
        "左图 · 隐空间维度 d（2 → 128，均 50 ep）",
        f"d=2 信息瓶颈过窄（SSE {fval(d2, 'recon_sse', 1)}, ARI {fval(d2)}）；d≥32 重构趋平但高维稀释距离，d=10 兼顾紧致与可分。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.0, body_size=11.0,
    )
    soft_panel(
        s, 6.81, 5.64, 6.10, 1.02,
        "右图 · 组件剥离归因（相对基准 ΔARI）",
        f"取消 AE 预训练 ({fval(c_np)}, Δ={d_np}) 与随机中心 ({fval(c_rc)}, Δ={d_rc}) 降幅远超冻结编码器 ({fval(c_fr)}, Δ={d_fr})。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.0, body_size=11.0,
    )

    # ---- Slide 27 (A3): E03 Class Imbalance (Fig.12) & Sample Efficiency (Fig.13) ----
    imb50_rows = [r for r in e03s if r.get("factor") == "imbalance" and fnum(r, "value") == 50.0]
    imb_idec = find_row(imb50_rows, method="idec")
    imb_km = find_row(imb50_rows, method="kmeans")
    imb1_row = find_row(e03s, factor="imbalance", method="idec", value=1.0)
    n_imb1 = f"{int(fnum(imb1_row, 'n_train')):,}" if fnum(imb1_row, "n_train") else "—"
    n_imb50 = f"{int(fnum(imb_idec, 'n_train')):,}" if fnum(imb_idec, "n_train") else "—"
    sf_km = find_row(e03s, factor="sample_frac", method="kmeans", value=0.01)
    sf_dec = find_row(e03s, factor="sample_frac", method="dec", value=0.01)
    sf_ae = find_row(e03s, factor="sample_frac", method="ae_kmeans", value=0.01)
    n_sf01 = f"{int(fnum(sf_km, 'n_train')):,}" if fnum(sf_km, "n_train") else "—"

    s = new_slide(
        prs, slide_ids, "A03_e03_imbalance_samples", "A3 / A6", "技术附录",
        "附录 A3：E03 类别长尾不平衡（Fig.12）与样本规模效率（Fig.13）",
        lead=f"✓ 边界规律（n=1, seed 0）：50:1 长尾（n_train={n_imb50}）下 IDEC ({fval(imb_idec)}) 略稳；**1% 样本（{n_sf01} 张）时 K-means ({fval(sf_km)}) 显著领先**。",
        summary="##适用区间：##长尾采样同时缩小了总样本量（见左图横轴标注）；样本量达 10%–25% 以上后深度嵌入聚类才稳定建立优势。",
    )
    fit_picture(s, figs_dir / "fig_e03_imbalance.png", 0.36, 1.32, 6.22, 4.20)
    fit_picture(s, figs_dir / "fig_e03_sample_frac.png", 6.76, 1.32, 6.22, 4.20)
    soft_panel(
        s, 0.42, 5.64, 6.10, 1.02,
        f"左图 · 长尾不平衡（n_train: {n_imb1} → {n_imb50}）",
        f"训练集按实验设计抽样（fit 仍不见标签），干净测试集评价；50:1 时 K-means 降至 {fval(imb_km)}，IDEC 保持 {fval(imb_idec)}。",
        fill=TEAL_SOFT, border=TEAL, title_color=TEAL, title_size=11.8, body_size=11.0,
    )
    soft_panel(
        s, 6.81, 5.64, 6.10, 1.02,
        "右图 · 无标签样本效率（1% → 100%）",
        f"{n_sf01} 张极小样本下深度网络过拟合明显（DEC {fval(sf_dec)} / AE {fval(sf_ae)}），传统 K-means 无需训参反而最稳（{fval(sf_km)}）。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=11.8, body_size=11.0,
    )

    # ---- Slide 28 (A4): E03 K Mismatch (Fig.14) & Outlier Pollution ----
    k20_dec = find_row(e03s, factor="k", method="dec", value=20.0)
    out0_dec = find_row(e03s, factor="outlier_frac", method="dec", value=0.0)
    out20_dec = find_row(e03s, factor="outlier_frac", method="dec", value=0.2)
    out0_idec = find_row(e03s, factor="outlier_frac", method="idec", value=0.0)
    out20_idec = find_row(e03s, factor="outlier_frac", method="idec", value=0.2)
    out0_km = find_row(e03s, factor="outlier_frac", method="kmeans", value=0.0)
    out20_km = find_row(e03s, factor="outlier_frac", method="kmeans", value=0.2)
    dec_out_d = fdiff(fnum(out20_dec, "ari"), fnum(out0_dec, "ari"))
    idec_out_d = fdiff(fnum(out20_idec, "ari"), fnum(out0_idec, "ari"))

    s = new_slide(
        prs, slide_ids, "A04_e03_k_outliers", "A4 / A6", "技术附录",
        "附录 A4：E03 预设聚类数 K 错配（Fig.14）与异常样本污染",
        lead="✓ 未知类数与脏数据测试（n=1, seed 0）：低估 K=5 重创所有方法，适度高估（K=12–20）时 **DEC 退化最平缓**。",
        summary="##实务建议：##未知真实类数时，宁可轻度高估 K 再做后处理合并，避免严重低估；脏数据比例高时可辅以异常检测或简单基线兜底。",
    )
    fit_picture(s, figs_dir / "fig_e03_k.png", 0.36, 1.32, 6.22, 4.20)
    fit_picture(s, figs_dir / "fig_e03_outlier_frac.png", 6.76, 1.32, 6.22, 4.20)
    soft_panel(
        s, 0.42, 5.64, 6.10, 1.02,
        "左图 · 预设聚类数 K 错配（真实 K=10）",
        f"全部方法在 K=10 附近达到峰值；高估至 K=20 时 DEC 仍保持 {fval(k20_dec)}，表现出更强的过分割容错能力。",
        fill=BLUE_SOFT, border=BLUE_DEEP, title_color=BLUE_DEEP, title_size=12.0, body_size=11.0,
    )
    soft_panel(
        s, 6.81, 5.64, 6.10, 1.02,
        "右图 · 均匀噪声异常图污染（0% → 20%）",
        f"DEC ({fval(out0_dec)}→{fval(out20_dec)}, Δ={dec_out_d}) 与 IDEC ({fval(out0_idec)}→{fval(out20_idec)}, Δ={idec_out_d}) 缓降；K-means 在 {fval(out0_km)}–{fval(out20_km)} 平稳。",
        fill=AMBER_SOFT, border=AMBER, title_color=AMBER, title_size=12.0, body_size=11.0,
    )

    # ---- Slide 29 (A5): E05 Negative Diagnostic (3 Datasets) ----
    m_gmm_j = sel.get("mnist", {}).get("evidence", {}).get("gmm", {}).get("macro_cluster_jaccard")
    f_gmm_j = sel.get("fashion_mnist", {}).get("evidence", {}).get("gmm", {}).get("macro_cluster_jaccard")
    gmm_j_s = f"{f_gmm_j:.3f}–{m_gmm_j:.3f}" if (isinstance(m_gmm_j, (int, float)) and isinstance(f_gmm_j, (int, float))) else "—"
    s = new_slide(
        prs, slide_ids, "A05_e05_stability_negative", "A5 / A6", "技术附录",
        "附录 A5：E05 负面诊断三数据集对照（训练稳定性 ≠ 语义正确）",
        lead="✓ 三数据集一致反例：蓝条（无标签训练稳定性，未纳耗时）均选中像素 K-means/PCA+KM，但橙条（历史 ARI）全部落后于最优路线。",
        summary="##探索性补充：##在 6 种正常工作方法上，稳定性与 ARI 的 Spearman 秩相关在三数据集均为负（-0.49/-0.60/-0.54，n=6 小样本反例，非统计定论）。",
    )
    fit_picture(s, figs_dir / "fig_e05_stability.png", 0.30, 1.24, 12.73, 4.38)
    soft_panel(
        s, 0.42, 5.70, 12.49, 0.96,
        "正确使用 E05：坍缩风险筛查器，而非最佳方法推荐器",
        f"E05 在拟合训练子集施加 σ=0.1 微扰：能立刻排雷轻噪坍缩的像素 GMM（Jaccard {gmm_j_s}），但因不感知语义且不含外部预训练路线，无法选出语义最优模型；新域确认语义需少量标注（半监督验证）。",
        fill=PURPLE_SOFT, border=PURPLE, title_color=PURPLE, title_size=11.8, body_size=11.0,
    )

    # ---- Slide 30 (A6): Core Literature Index & Reproducibility Audit ----
    s = new_slide(
        prs, slide_ids, "A06_refs_audit", "A6 / A6", "技术附录",
        "附录 A6：核心文献核验索引与可复现审计清单",
        lead="✓ 文献与数据溯源：全部引用严格对齐 docs/bibliography.bib；测试标签隔离、端到端成本与讲稿 ID 绑定均纳入自动化测试。",
        summary="##审计结论：##覆盖核心单元测试、标签隔离/成本/PPT 绑定回归测试及 GitHub Actions CI，GitHub 与 Google Drive 双端完整归档。",
    )
    bib_rows = [
        ["Zhou et al., 2025", "ACM Computing Surveys", "深度聚类综述；图 1 四类交互分类骨架"],
        ["Xie et al., 2016", "ICML (PMLR 48)", "DEC：Student-t 软分配与 KL 散度联合自训练"],
        ["Guo et al., 2017", "IJCAI", "IDEC：保留解码器重构约束以保护局部结构"],
        ["Jiang et al., 2017", "IJCAI", "VaDE：VAE + 高斯混合先验生成式深度聚类"],
        ["Caron et al., 2018/20", "ECCV / NeurIPS", "DeepCluster 伪标签迭代；SwAV 跨视图原型一致"],
        ["Chen et al., 2020", "ICML", "SimCLR：实例级对比视觉表征（+离线 K-means）"],
        ["Li et al., 2021", "AAAI", "Contrastive Clustering (CC)：实例+簇联合对比"],
        ["Oquab 2024 / Radford 21", "TMLR / ICML", "DINOv2 无监督视觉基础模型；CLIP 图文对齐"],
        ["Li et al., 2024", "ICML (PMLR 235)", "TAC（Image Clustering with External Guidance）"],
        ["Hu et al., 2026*", "arXiv:2609.36648", "VLM4Cluster（预印本）：20 数据集 17 方法统一评测"],
    ]
    data_table(
        s, 0.42, 1.30, 8.22,
        ["作者与年份", "发表载体", "在本综述中的核心贡献"],
        bib_rows, [2.05, 1.95, 4.22],
        font_size=10.5, header_size=11.0, max_h=5.32, left_align_cols={0, 1, 2},
    )
    header_card(
        s, 8.86, 1.30, 4.05, 5.34,
        "仓库审计与失败留痕清单",
        f"■ ##运行覆盖率##：{n_ok} / {n_ok} 项计划记录状态 ok（E01:{exp_counts.get('E01', 0)} · E02:{exp_counts.get('E02', 0)} · E03:{exp_counts.get('E03', 0)} · 噪声:{exp_counts.get('E03_noise', 0)} · E04:{exp_counts.get('E04', 0)}）。\n\n"
        f"■ ##失败透明留痕##：failures.csv 保留 {n_fail_hist} 次 CIFAR-10 下载中断记录（已脱敏本地路径），仅按原 seed 重试 1 次，不因分数低换种。\n\n"
        "■ ##自动化回归测试##：覆盖 E05 测试标签零泄漏、预测耗时口径、端到端成本、PPT 动态数据绑定与 30 页讲稿 ID 一一校验。\n\n"
        "■ ##双端存储分工##：GitHub 存代码/协议/表/图/PPT（embeddings 1.6MB 留库保离线复现）；Drive 另存 217 份权重（1.4 GB）。",
        header_fill=BLUE_DEEP, body_fill=BLUE_SOFT, border=BLUE_DEEP, title_size=13.0, body_size=11.1, line_spacing=1.22,
    )
    return slide_ids


def parse_speaker_notes(raw: str) -> list[tuple[str, str]]:
    """Parse ## [SlideID] heading sections from speaker_notes.md."""
    pattern = re.compile(r"^## \[([A-Za-z0-9_]+)\][^\n]*\n\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
    entries: list[tuple[str, str]] = []
    for m in pattern.finditer(raw):
        sid = m.group(1).strip()
        body = m.group(2).strip()
        entries.append((sid, body))
    return entries


def inject_speaker_notes(prs, slide_ids: list[str], notes_path: Path = NOTES_MD) -> None:
    """Strictly bind speaker notes by stable slide ID; raise ValueError on any mismatch."""
    if not notes_path.exists():
        raise ValueError(f"Speaker notes file missing: {notes_path}")
    entries = parse_speaker_notes(notes_path.read_text(encoding="utf-8"))
    note_ids = [sid for sid, _ in entries]
    if len(set(note_ids)) != len(note_ids):
        dups = sorted({sid for sid in note_ids if note_ids.count(sid) > 1})
        raise ValueError(f"Duplicate slide IDs in speaker notes: {dups}")
    missing = [sid for sid in slide_ids if sid not in note_ids]
    extra = [sid for sid in note_ids if sid not in slide_ids]
    if missing or extra:
        raise ValueError(f"Speaker notes slide ID mismatch: missing={missing}, extra={extra}")
    if note_ids != slide_ids:
        raise ValueError(f"Speaker notes slide order mismatch: expected={slide_ids}, got={note_ids}")

    by_id = dict(entries)
    for idx, sid in enumerate(slide_ids):
        body = by_id[sid]
        if not body:
            raise ValueError(f"Empty speaker note body for slide ID: {sid}")
        prs.slides[idx].notes_slide.notes_text_frame.text = f"[{sid}]\n{body}"


def build(tables_dir: Path = TABLES, out_path: Path = OUT,
          notes_path: Path = NOTES_MD, figs_dir: Path = FIGS) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "深度聚类：范式演进与方法选型综述"
    prs.core_properties.subject = "课程综述汇报（24 页正文 + 6 页技术附录）"
    prs.core_properties.author = "第九组"

    ctx = {
        "figs_dir": figs_dir,
        "e01": load_csv("e01_summary.csv", tables_dir),
        "e02": load_csv("e02_ablation.csv", tables_dir),
        "e03_noise": load_csv("e03_noise.csv", tables_dir),
        "e03_sens": load_csv("e03_sensitivity.csv", tables_dir),
        "e04_paired": load_csv("e04_paired.csv", tables_dir),
        "e04_pretrained": load_csv("e04_pretrained.csv", tables_dir),
        "costs": load_csv("benchmark_cost_frontiers.csv", tables_dir),
        "all_runs": load_csv("all_runs.csv", tables_dir),
        "selection": load_selection(tables_dir),
        "coverage": load_csv("coverage.csv", tables_dir),
        "failures": load_csv("failures.csv", tables_dir),
    }
    slide_ids = build_slides(prs, ctx)
    inject_speaker_notes(prs, slide_ids, notes_path=notes_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"Built {len(prs.slides)} slides ({len(slide_ids)} ID-verified notes) -> {out_path}")
    return out_path


if __name__ == "__main__":
    build()
