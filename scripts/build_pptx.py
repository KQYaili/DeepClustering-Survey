"""Build the 24-slide deck. Result numbers are read from tables, never typed in."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
OUT = ROOT / "presentation" / "DeepClustering-Survey.pptx"

NAVY = RGBColor(0x1E, 0x3A, 0x5F)
AMBER = RGBColor(0xB4, 0x53, 0x09)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
BG = RGBColor(0xF4, 0xF7, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD6, 0xDE, 0xE8)
SOFT = RGBColor(0xE7, 0xEE, 0xF6)

W, H = 13.333, 7.5


def _ea(run, name: str) -> None:
    rpr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        node = rpr.find(qn(f"a:{tag}"))
        if node is None:
            node = etree.SubElement(rpr, qn(f"a:{tag}"))
        node.set("typeface", name)


def _run(paragraph, text, size, bold, color, font="Microsoft YaHei"):
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    _ea(run, font)
    return run


def rect(slide, l, t, w, h, fill, line=None, radius=None):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    box = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(1)
    if radius:
        box.adjustments[0] = radius
    box.shadow.inherit = False
    return box


def text(slide, l, t, w, h, content, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    frame.margin_left = Emu(0)
    frame.margin_right = Emu(0)
    frame.margin_top = Emu(0)
    frame.margin_bottom = Emu(0)
    try:
        frame._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    first = True
    for line in content.split("\n"):
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.alignment = align
        paragraph.space_after = Pt(0)
        paragraph.space_before = Pt(0)
        _run(paragraph, line, size, bold, color)
    return box


def notes(slide, content: str) -> None:
    slide.notes_slide.notes_text_frame.text = content


def chrome(slide, page: int, title: str, kicker: str = ""):
    rect(slide, 0, 0, W, H, BG)
    rect(slide, 0, 0, 0.08, H, NAVY)
    rect(slide, 0, 7.28, W, 0.22, NAVY)
    text(slide, 0.38, 7.30, 9.2, 0.18, "深度聚类：范式演进与方法选型", size=10, color=WHITE)
    text(slide, 11.4, 7.30, 1.6, 0.18, f"{page}  /  24", size=10, color=WHITE, align=PP_ALIGN.RIGHT)
    if kicker:
        text(slide, 0.42, 0.18, 8, 0.24, kicker, size=12, bold=True, color=AMBER)
        text(slide, 0.40, 0.40, 12.4, 0.48, title, size=28, bold=True, color=NAVY)
    else:
        text(slide, 0.40, 0.28, 12.4, 0.5, title, size=28, bold=True, color=NAVY)


def new(prs, page, title, kicker=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    chrome(slide, page, title, kicker)
    return slide


def load_e01():
    path = ROOT / "results" / "tables" / "e01_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs():
    path = ROOT / "results" / "tables" / "all_runs.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_selection():
    path = ROOT / "results" / "tables" / "selection.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(name):
    path = ROOT / "results" / "tables" / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_of(rows, key):
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        values.append(float(raw))
    if not values:
        return None
    return sum(values) / len(values)


def lookup_ari(summary, dataset, method):
    for row in summary:
        if row["dataset"] == dataset and row["method"] == method:
            return float(row["ari_mean"])
    return None


def lookup_summary_row(summary, dataset, method):
    for row in summary:
        if row["dataset"] == dataset and row["method"] == method:
            return row
    return None


def value_from_rows(rows, predicate, key="ari"):
    for row in rows:
        if predicate(row):
            raw = row.get(key, "")
            if raw not in (None, ""):
                return float(raw)
    return None


def fmt_value(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def fmt(row, key):
    value = row.get(key)
    if value in (None, ""):
        return "—"
    std_key = key.replace("_mean", "_std")
    std = row.get(std_key)
    try:
        mean = float(value)
        if std in (None, ""):
            return f"{mean:.3f}"
        return f"{mean:.3f}±{float(std):.3f}"
    except ValueError:
        return str(value)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "深度聚类：范式演进与方法选型综述"
    prs.core_properties.subject = "课程汇报"
    cover(prs)
    question(prs)
    textbook(prs)
    kmeans(prs)
    representation(prs)
    taxonomy(prs)
    axes(prs)
    dec(prs)
    idec(prs)
    vade(prs)
    contrast(prs)
    pretrained(prs)
    hypotheses(prs)
    protocol(prs)
    e01(prs)
    shared_init(prs)
    e02(prs)
    e03(prs)
    e04(prs)
    pareto(prs)
    rule(prs)
    propositions(prs)
    limits(prs)
    close(prs)
    note_text = (ROOT / "presentation" / "speaker_notes.md").read_text(encoding="utf-8")
    for match in re.finditer(r"^## 第 (\d+) 页[^\n]*\n\n(.*?)(?=^## |\Z)", note_text, re.MULTILINE | re.DOTALL):
        page = int(match.group(1))
        if 1 <= page <= len(prs.slides):
            prs.slides[page - 1].notes_slide.notes_text_frame.text = match.group(2).strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


NAMES = {
    "kmeans": "直接对像素做 K 均值",
    "pca_kmeans": "先做主成分，再 K 均值",
    "gmm": "高斯混合",
    "ae_kmeans": "自编码器之后再 K 均值",
    "dec": "深度嵌入聚类",
    "idec": "带重构的深度嵌入",
    "vade": "生成式聚类",
    "simclr_kmeans": "对比学习之后再 K 均值",
    "cc": "对比聚类",
}


def cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, H, NAVY)
    rect(slide, 0, 0, 0.18, H, AMBER)
    text(slide, 0.7, 1.45, 11, 0.3, "第九组  ·  课程汇报", size=16, bold=True, color=RGBColor(0xF5, 0xC1, 0x6C))
    text(slide, 0.7, 1.95, 11.5, 1.5, "没有标签时，\n怎么把图片分成组", size=40, bold=True, color=WHITE)
    text(slide, 0.72, 4.35, 11, 1.1, "手写数字、服装、自然小图。\n什么时候用课本里的方法就够了，什么时候才值得训练一个神经网络。", size=20, color=RGBColor(0xD6, 0xE2, 0xF0))
    notes(slide, "先用一句人话说明今天要讲的事。不要念副标题里的术语。")


def question(prs):
    slide = new(prs, 2, "今天只回答三件事", "听完就能带走")
    items = [
        ("1", "坐标", "是“图长得像不像”重要，还是“用哪种分组算法”重要？"),
        ("2", "训练方式", "边学坐标边分组，一定比先学完再分组更好吗？"),
        ("3", "现成模型", "别人已经训练好一个很强的视觉模型时，还要不要再做一个复杂分组器？"),
    ]
    for index, (num, label, body) in enumerate(items):
        left = 0.42 + index * 4.2
        rect(slide, left, 1.45, 3.95, 4.4, WHITE, LINE, 0.08)
        rect(slide, left, 1.45, 3.95, 0.1, AMBER if index == 2 else NAVY)
        text(slide, left + 0.25, 1.8, 3.4, 0.4, num, size=20, bold=True, color=AMBER)
        text(slide, left + 0.25, 2.35, 3.4, 0.55, label, size=26, bold=True, color=NAVY)
        text(slide, left + 0.25, 3.2, 3.4, 2.1, body, size=18, color=INK)
    notes(slide, "三件事彼此独立。讲每一件时不要提前剧透后面的实验。")


def textbook(prs):
    slide = new(prs, 3, "课本里的聚类，先假设点已经能比远近", "从教材出发")
    rows = [
        ("第2章  随机向量", "两个东西像不像，要先有一个可以算距离的坐标"),
        ("第3章  多元正态", "每一组可以想成一团云，这就是后面高斯混合的来源"),
        ("第6章  聚类分析", "今天的起点：K 均值。离哪个中心近，就进哪一组"),
        ("第7章  主成分", "先把坐标压成少数几维，再分组。这是线性的做法"),
        ("第8章  因子分析", "观测背后还有看不见的变量。生成式方法走的是这条路"),
    ]
    for index, (head, body) in enumerate(rows):
        top = 1.25 + index * 1.05
        rect(slide, 0.42, top, 12.4, 0.92, WHITE, LINE, 0.08)
        rect(slide, 0.42, top, 0.1, 0.92, TEAL)
        text(slide, 0.75, top + 0.12, 4.2, 0.68, head, size=18, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 5.1, top + 0.12, 7.3, 0.68, body, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "点明第6章是起点，第7、8章解释后来的表征和潜变量。")


def kmeans(prs):
    slide = new(prs, 4, "K 均值：离哪个中心近，就进哪一组", "把每张图看成一长串像素")
    _equation(slide, "eq_kmeans.png", 0.45, 1.28, 7.2, 1.05)
    rect(slide, 0.45, 2.7, 6.15, 4.0, WHITE, LINE, 0.08)
    text(slide, 0.7, 2.95, 5.7, 0.4, "用在图片上会碰到什么", size=18, bold=True, color=NAVY)
    text(slide, 0.7, 3.55, 5.7, 2.8, "一张图被拉成几千个明暗数字。\n两张图的像素差小，不等于它们是同一类东西。\n\n一件衬衫和一件 T 恤，像素可以差得很远，人看起来却很像。", size=16, color=INK)
    rect(slide, 6.85, 2.7, 5.95, 4.0, NAVY, radius=0.08)
    text(slide, 7.15, 2.95, 5.4, 0.45, "所以问题先不在分组公式", size=18, bold=True, color=WHITE)
    text(slide, 7.15, 3.6, 5.4, 2.6, "公式只是在已有坐标里找中心。\n\n坐标如果没有语义，换一种更复杂的分组，仍然是在错误的空间里切。", size=16, color=RGBColor(0xE8, 0xEE, 0xF6))
    notes(slide, "用衬衫和 T 恤举一个例子。公式只停留十秒。")


def representation(prs):
    slide = new(prs, 5, "先把图变成少数几个有意义的数", "深度聚类多出来的一步")
    labels = [
        ("原来的图", "一张手写数字\n或一张衣服"),
        ("压缩", "神经网络把它\n收成十个左右的数"),
        ("新坐标", "像的图，在这组数里离得近"),
        ("分组", "再在这组数上\n找中心、划组"),
    ]
    for index, (title, body) in enumerate(labels):
        left = 0.45 + index * 3.2
        rect(slide, left, 1.7, 3.0, 3.6, WHITE, LINE, 0.08)
        text(slide, left + 0.2, 2.0, 2.6, 0.8, title, size=22, bold=True, color=NAVY)
        text(slide, left + 0.2, 3.1, 2.6, 1.6, body, size=16, color=INK)
    text(slide, 0.5, 5.7, 12.2, 1.0, "后面所有方法的差别，就是这一步怎么学：只先压缩，还是边压缩边分组；坐标是自己学的，还是用别人训好的模型。", size=16, color=INK)
    notes(slide, "这一页只建立“先压缩再分组”。不要展开具体算法。")


def taxonomy(prs):
    slide = new(prs, 6, "人们用四种方式把“压缩”和“分组”接在一起", "这不是一条淘汰史")
    cards = [
        ("先学完，再分组", "坐标学完就不再改，接着用普通方法分组。", "自编码器\n对比学习\n现成视觉模型"),
        ("一起学", "分组变好的同时，把坐标也改一改。", "深度嵌入聚类\n再加重构约束"),
        ("当成生成过程", "每一组是一团云，图是从云里抽出来的。", "高斯混合\n生成式聚类"),
        ("来回修正", "先猜一个组，再用这个猜测继续教网络。", "对比聚类"),
    ]
    for index, (title, body, examples) in enumerate(cards):
        left = 0.4 + index * 3.22
        rect(slide, left, 1.4, 3.05, 4.7, WHITE, LINE, 0.08)
        rect(slide, left, 1.4, 3.05, 0.1, [NAVY, AMBER, TEAL, RGBColor(0x6D, 0x28, 0xD9)][index])
        text(slide, left + 0.18, 1.7, 2.7, 0.9, title, size=18, bold=True, color=NAVY)
        text(slide, left + 0.18, 2.7, 2.7, 1.5, body, size=14, color=INK)
        text(slide, left + 0.18, 4.35, 2.7, 1.4, examples, size=14, color=MUTED)
    text(slide, 0.45, 6.25, 12.3, 0.7, "四种可以同时存在。新方法没有取消旧方法。", size=15, color=MUTED)
    notes(slide, "英文分类名放在备注里：multi-stage、simultaneous、generative、iterative。Zhou 等，ACM Computing Surveys，2025。台上用中文。")


def axes(prs):
    slide = new(prs, 7, "比较两种方法时，只问四个白话问题", "不用先记住论文名字")
    rows = [
        ("坐标从哪来", "原始像素，主成分，自己训练的网络，还是别人训好的模型"),
        ("分组什么时候做", "学完再分，还是学的过程中就在分"),
        ("一组怎么定", "离中心近，还是更像某一团概率云"),
        ("网络靠什么学习", "把图还原出来，两张相似图靠拢，或者借助文字"),
    ]
    for index, (head, body) in enumerate(rows):
        top = 1.4 + index * 1.25
        rect(slide, 0.45, top, 12.4, 1.08, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.14, 2.6, 0.8, head, size=18, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 3.5, top + 0.14, 9.0, 0.8, body, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "这套坐标解释联系，不把算法塞进互斥格子。")


def dec(prs):
    slide = new(prs, 8, "深度嵌入聚类：边分组，边改坐标", "2016 年的一个代表方法")
    _equation(slide, "eq_dec.png", 0.45, 1.25, 12.2, 1.05)
    points = [
        "每个点对每一组都有一个软的归属：离这组中心越近，归属越高。",
        "训练时让归属变得更干脆，同时把网络的坐标也拉向对应的中心。",
        "这里不用图片的真实类别。网络只看到图本身。",
    ]
    for index, line in enumerate(points):
        top = 2.7 + index * 1.25
        rect(slide, 0.45, top, 12.4, 1.08, WHITE, LINE, 0.08)
        text(slide, 0.75, top + 0.25, 11.8, 0.6, line, size=18, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "公式是学生 t 分布的软分配。台上说“离得近，归属就高”即可。论文是 Xie、Girshick、Farhadi，ICML 2016。")


def idec(prs):
    slide = new(prs, 9, "再加上一条要求：还要能把图画回去", "带重构的深度嵌入")
    _equation(slide, "eq_idec.png", 0.4, 1.25, 12.4, 1.05)
    rect(slide, 0.45, 2.6, 6.2, 4.05, WHITE, LINE, 0.08)
    text(slide, 0.7, 2.85, 5.7, 0.45, "多出来的那一项是什么", size=18, bold=True, color=NAVY)
    text(slide, 0.7, 3.5, 5.7, 2.7, "网络除了把组分开，还要把压缩后的数还原成原来的图。\n\n还原得越像，说明这些数还留着图的细节。", size=16, color=INK)
    rect(slide, 6.9, 2.6, 5.9, 4.05, SOFT, radius=0.08)
    text(slide, 7.15, 2.85, 5.4, 0.45, "权重只是一个旋钮", size=18, bold=True, color=NAVY)
    text(slide, 7.15, 3.5, 5.4, 2.7, "旋钮为 0：完全不要求还原。\n旋钮很大：几乎只顾还原。\n\n我们把 0 到 10 的五档都做了，不是只挑最好看的一档。", size=16, color=INK)
    notes(slide, "不要在台上对比论文里的希腊字母。问的就是：顾着还原，分组会不会变差。论文是 Guo 等，IJCAI 2017。")


def vade(prs):
    slide = new(prs, 10, "生成式聚类：每一组是一团云", "图被看成从这团云里抽出来的")
    cols = [
        ("怎么想", "压缩后的坐标不是随便散落的。每一组对应一团云，图更像从哪团云里来，就进哪一组。"),
        ("和 K 均值的差别", "K 均值只找一个中心。这里还描述这团云有多宽，所以一组可以有形状。"),
        ("今天怎么比", "它也能把图画回去。我们用平方误差来衡量像不像。数字只和今天同一张表里的方法比。"),
    ]
    for index, (title, body) in enumerate(cols):
        left = 0.45 + index * 4.2
        rect(slide, left, 1.45, 3.95, 4.6, WHITE, LINE, 0.08)
        text(slide, left + 0.25, 1.75, 3.45, 0.5, title, size=22, bold=True, color=NAVY)
        text(slide, left + 0.25, 2.55, 3.45, 3.0, body, size=16, color=INK)
    notes(slide, "论文是 Jiang 等，IJCAI 2017。台上用“一团云”讲完即可。")


def contrast(prs):
    slide = new(prs, 11, "对比学习：同一张图的两种看法应该靠近", "分组可以后做，也可以一起做")
    rect(slide, 0.45, 1.4, 6.05, 5.2, WHITE, LINE, 0.08)
    text(slide, 0.7, 1.65, 5.6, 0.7, "先学会比较，再分组", size=22, bold=True, color=NAVY)
    text(slide, 0.7, 2.55, 5.5, 3.5, "把一张图做两种轻微改动。\n网络要让这两种改动在坐标里靠近，和其他图拉开。\n\n这一步还没有组。组是学完之后，用普通 K 均值划的。", size=16, color=INK)
    rect(slide, 6.75, 1.4, 6.05, 5.2, WHITE, LINE, 0.08)
    text(slide, 7.0, 1.65, 5.6, 0.7, "比较和分组一起学", size=22, bold=True, color=NAVY)
    text(slide, 7.0, 2.55, 5.5, 3.5, "同样要求两种改动靠近。\n同时还要求两种改动被分进同一组。\n\n用的是小网络。它说明这条路的思路，不是论文里大网络的成绩。", size=16, color=INK)
    notes(slide, "左是 SimCLR，Chen 等，ICML 2020。右是对比聚类，Li 等，AAAI 2021。台上不必念损失函数名字。")


def pretrained(prs):
    slide = new(prs, 12, "也可以直接用别人已经训好的视觉模型", "我们不再训练这个大模型")
    blocks = [
        ("只取坐标", "模型已经在海量图片上看过很多东西。我们把它冻住，只问它：这张小图对应哪一串数。"),
        ("两种来源", "一种只看图，叫 DINOv2。另一种学过图和文字的对应，叫 CLIP。"),
        ("文字还能再帮一把", "也有方法用词典里的词来引导分组。今天不重做那一项，只说明还有这条路。"),
    ]
    for index, (title, body) in enumerate(blocks):
        top = 1.35 + index * 1.75
        rect(slide, 0.45, top, 12.4, 1.58, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.16, 11.8, 0.4, title, size=18, bold=True, color=NAVY)
        text(slide, 0.7, top + 0.62, 11.8, 0.75, body, size=15, color=INK)
    notes(slide, "不要把预训练模型和从零训练的模型放进同一张公平榜。")


def hypotheses(prs):
    slide = new(prs, 13, "做实验之前，先写下会怎么选", "这些只是猜测")
    headers = ["如果手里是这样的数据", "先试"]
    data = [
        ("图不多，又要快", "直接 K 均值，或先做主成分"),
        ("像素太乱，没有现成模型", "先训练一个自编码器"),
        ("希望分组和压缩互相帮忙", "深度嵌入聚类，并可要求能还原"),
        ("希望每组像一团有形状的云", "高斯混合或生成式聚类"),
        ("同一张图可以做出两种可靠的改动", "对比学习"),
        ("已经有一个对这类图很强的模型", "用它的坐标，再做 K 均值"),
    ]
    table(slide, 0.45, 1.3, 12.4, headers, data, [6.3, 6.1], font=15)
    text(slide, 0.5, 6.5, 12.2, 0.45, "下一页开始用三套图检验这些猜测。这里不写谁会赢。", size=15, color=MUTED)
    notes(slide, "念两行就够。强调这是实验前的猜测。")


def protocol(prs):
    slide = new(prs, 14, "三套大家都认识的图", "训练时不看答案")
    items = [
        ("手写数字", "黑白，10 种数字。结构比较清楚。"),
        ("服装", "同样是黑白小图，但衬衫和外套更容易混。"),
        ("自然小图", "彩色的飞机、汽车、动物。从像素里很难直接看出类别。"),
        ("怎么打分", "训练时不看答案。ARI 接近 1，表示分组和真实类别一致；接近 0，表示和碰运气差不多。E01 做 3 个种子，E02/E03 用 seed 0，E04 在表里写明 n。"),
    ]
    for index, (title, body) in enumerate(items):
        column = index % 2
        row = index // 2
        left = 0.42 + column * 6.45
        top = 1.35 + row * 2.6
        rect(slide, left, top, 6.2, 2.4, WHITE, LINE, 0.08)
        text(slide, left + 0.25, top + 0.2, 5.7, 0.45, title, size=18, bold=True, color=NAVY)
        text(slide, left + 0.25, top + 0.8, 5.7, 1.3, body, size=15, color=INK)
    notes(slide, "不平衡实验是例外：训练标签只用来构造比例。")


def e01(prs):
    slide = new(prs, 15, "自己从头训练时，谁更接近真实类别", "数字是 3 次的平均 ARI")
    rows = load_e01()
    methods = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    labels = {"mnist": "数字", "fashion_mnist": "服装", "cifar10": "自然图"}
    present = [method for method in methods if any(lookup_ari(rows, dataset, method) is not None for dataset in datasets)]
    if not present:
        rect(slide, 0.45, 1.5, 12.4, 4.8, WHITE, LINE, 0.08)
        text(slide, 0.8, 3.2, 11.6, 1.0, "当前分析没有可比记录，保留为空。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
        return
    data = []
    for method in present:
        line = [NAMES[method]]
        for dataset in datasets:
            row = lookup_summary_row(rows, dataset, method)
            line.append("—" if row is None else fmt(row, "ari_mean"))
        data.append(line)
    table(slide, 0.4, 1.18, 12.5, ["方法", "手写数字", "服装", "自然小图"], data, [4.6, 2.6, 2.6, 2.7], font=12)
    bests = []
    for dataset in datasets:
        scored = [(NAMES[method], lookup_ari(rows, dataset, method)) for method in present]
        scored = [item for item in scored if item[1] is not None]
        if scored:
            name, value = max(scored, key=lambda item: item[1])
            bests.append(f"{labels[dataset]}最高 {value:.2f}")
    text(slide, 0.45, 6.48, 12.4, 0.45, "1 分是和真实类别一致，0 分接近碰运气。" + "，".join(bests) + "。表中 ± 是跨种子的标准差。", size=13, color=MUTED)
    notes(slide, "先指自然小图整列都很低，再说数字和服装上深度嵌入聚类最高。不要念英文缩写。")


def shared_init(prs):
    slide = new(prs, 16, "这三种方法从同一份压缩出发", "差别只来自后面怎么训练")
    steps = [
        ("1", "先把图压成十个数", "自编码器先学一次。三种方法共用这一次的结果。"),
        ("2", "学完就分组", "网络不再改。只在这十个数上做 K 均值。"),
        ("3", "边分组边改", "深度嵌入聚类继续改这十个数。带重构的版本还要求能把图画回去。"),
    ]
    for index, (num, title, body) in enumerate(steps):
        left = 0.45 + index * 4.2
        rect(slide, left, 1.55, 3.95, 4.5, WHITE, LINE, 0.08)
        text(slide, left + 0.25, 1.85, 1, 0.5, num, size=28, bold=True, color=AMBER)
        text(slide, left + 0.25, 2.55, 3.4, 0.8, title, size=20, bold=True, color=NAVY)
        text(slide, left + 0.25, 3.5, 3.4, 2.0, body, size=16, color=INK)
    notes(slide, "生成式和对比学习不是从这个共同起点出发的。如果有人问，再补一句。")


def e02(prs):
    slide = new(prs, 17, "重构权重有一个量纲合适的甜点区", "E02：服装图，seed 0")
    image = ROOT / "results" / "figures" / "fig_e02_lambda.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.35), Inches(1.35), Inches(8.3), Inches(5.3))
    else:
        rect(slide, 0.4, 1.4, 8.2, 5.2, WHITE, LINE, 0.08)
        text(slide, 0.7, 3.4, 7.6, 1.0, "当前分析没有可比记录，保留为空。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    rect(slide, 8.85, 1.4, 4.0, 5.2, NAVY, radius=0.08)
    text(slide, 9.1, 1.7, 3.55, 0.8, "怎么读这张图", size=20, bold=True, color=WHITE)
    e02 = load_table("e02_ablation.csv")
    def e02_ari(token):
        return value_from_rows(e02, lambda row: token in row.get("source", ""))
    ari_zero = e02_ari("lam0__")
    ari_best = max((float(row["ari"]) for row in e02 if "lat10" in row.get("source", "") and "pre1" in row.get("source", "") and "initkmeans" in row.get("source", "") and "fr0" in row.get("source", "") and "lam" in row.get("source", "")), default=None)
    text(slide, 9.1, 2.65, 3.55, 3.5, f"横轴是重构权重。\n蓝线是分得准不准。\n橙线是每张图的平方误差，越低越好。\n\nλ=0 时 ARI 是 {fmt_value(ari_zero)}。λ=0.01 到 0.1 仍在高位，λ≥1 后开始下降，说明先匹配损失量纲，再谈权重大小。", size=16, color=WHITE)
    notes(slide, "只讲这一件事：顾还原，分组略变差。不要展开其他消融。")


def e03(prs):
    slide = new(prs, 18, "测试图加了雪花之后，谁掉得快", "训练时没见过这些噪声")
    image = ROOT / "results" / "figures" / "fig_e03_noise.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.3), Inches(1.3), Inches(8.4), Inches(5.4))
    else:
        rect(slide, 0.4, 1.4, 8.2, 5.2, WHITE, LINE, 0.08)
        text(slide, 0.7, 3.4, 7.6, 1.0, "当前分析没有可比记录，保留为空。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    rect(slide, 8.85, 1.4, 4.0, 5.2, WHITE, LINE, 0.08)
    text(slide, 9.1, 1.65, 3.55, 0.7, "这页只看一件事", size=18, bold=True, color=NAVY)
    noise = load_table("e03_noise.csv")
    clean = value_from_rows(noise, lambda row: row.get("dataset") == "fashion_mnist" and row.get("method") == "idec" and row.get("noise_std") == "0.0", "ari_mean")
    noisy = value_from_rows(noise, lambda row: row.get("dataset") == "fashion_mnist" and row.get("method") == "idec" and row.get("noise_std") == "0.3", "ari_mean")
    gmm_noisy = value_from_rows(noise, lambda row: row.get("dataset") == "fashion_mnist" and row.get("method") == "gmm" and row.get("noise_std") == "0.1", "ari_mean")
    text(slide, 9.1, 2.5, 3.55, 3.6, f"横轴是噪声标准差。\n纵轴仍是 ARI。\n\nIDEC 在干净服装图上是 {fmt_value(clean)}，噪声升到 0.3 时是 {fmt_value(noisy)}。GMM 在轻微噪声下是 {fmt_value(gmm_noisy)}，说明概率模型对尺度和协方差更敏感。", size=16, color=INK)
    notes(slide, "图例如果是英文缩写，口头翻译：idec 是带重构的深度嵌入。其他鲁棒性实验不在这一页展开。")


def e04(prs):
    slide = new(prs, 19, "自然小图：现成模型的坐标，比自己学的有用", "数字仍是 ARI，1 最好")
    summary = load_e01()
    runs = load_runs()

    def e04_cell(method, source="e04"):
        if source == "e01":
            row = lookup_summary_row(summary, "cifar10", method)
            return "—" if row is None else f"{float(row['ari_mean']):.3f} (n={row['n']})"
        selected = [row for row in runs if row["experiment"] == "E04" and row["method"] == method and row["ari"] not in ("", None)]
        if not selected:
            return "—"
        value = sum(float(row["ari"]) for row in selected) / len(selected)
        return f"{value:.3f} (n={len(selected)})"

    data = [
        ["随机 CNN", e04_cell("random_cnn_kmeans"), e04_cell("random_cnn_gmm"), e04_cell("random_cnn_dec_head")],
        ["AE", e04_cell("ae_kmeans", "e01"), e04_cell("ae_gmm"), e04_cell("ae_dec_head")],
        ["SimCLR", e04_cell("simclr_kmeans", "e01"), e04_cell("simclr_gmm"), e04_cell("simclr_dec_head")],
        ["DINOv2", e04_cell("dinov2_kmeans"), e04_cell("dinov2_gmm"), e04_cell("dinov2_dec_head")],
        ["CLIP", e04_cell("clip_kmeans"), e04_cell("clip_gmm"), e04_cell("clip_dec_head")],
    ]
    table(slide, 0.55, 1.3, 12.2, ["坐标从哪来", "再做 K 均值", "再做高斯混合", "再训一个小分组头"], data, [2.7, 3.1, 3.2, 3.2], font=15)
    text(slide, 0.55, 5.5, 12.2, 1.3, "每格都标 n 和种子来源。冻结的 DINOv2/CLIP 坐标明显高于从零训练的坐标；在这些现成坐标上，换成 GMM 或再训一个小分组头也没有超过 K 均值。预训练模型见过外部数据，不能和从头训练的方法当作同一条公平赛道。", size=15, color=INK)
    notes(slide, "先指上面三行，再指下面两行。结论是：坐标变了，分组算法没再添多少。")


def pareto(prs):
    slide = new(prs, 20, "又要准、又要快，往往不是同一个方法", "服装图")
    image = ROOT / "results" / "figures" / "fig_e05_pareto.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.3), Inches(1.25), Inches(8.2), Inches(5.5))
        text(slide, 8.6, 1.55, 4.3, 5.0, "横轴是训练秒数，越左越快。纵轴是 ARI，越高越准。\n\n没有另一种方法既更准又更快的，才被单独标出。\n\n这页只看速度和准确，不看抗噪声。", size=16, color=INK)
    else:
        rect(slide, 0.5, 1.6, 12.3, 4.6, WHITE, LINE, 0.08)
        text(slide, 0.9, 3.2, 11.5, 1.0, "当前分析没有可比记录，保留为空。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    notes(slide, "如果图上的名字是英文，按前面的中文对照念。不要把这页说成唯一正确答案。")


def rule(prs):
    slide = new(prs, 21, "如果现在就要选一个做法", "按刚才三套图，各给一句")
    selection = load_selection()
    choices = []
    for dataset, label in [("cifar10", "自然小图"), ("mnist", "手写数字"), ("fashion_mnist", "服装")]:
        choice = selection.get(dataset, {}).get("choice")
        if choice:
            choices.append((label, NAMES.get(choice, choice), "这是训练子集上的无标签稳定性诊断，不是拿测试集 ARI 反选出来的冠军。新数据没有标签时，仍要结合稳定性、速度和业务约束。"))
        else:
            choices.append((label, "证据不足，先保留多个候选", "当前配置没有输出可用的无标签稳定性选择。"))
    for index, (dataset, choice, body) in enumerate(choices):
        top = 1.25 + index * 1.9
        rect(slide, 0.45, top, 12.4, 1.75, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.12, 11.8, 0.35, dataset, size=14, bold=True, color=AMBER)
        text(slide, 0.7, top + 0.48, 11.8, 0.45, choice, size=22, bold=True, color=NAVY)
        text(slide, 0.7, top + 1.05, 11.8, 0.55, body, size=15, color=INK)
    notes(slide, "三句各自成立，不要合成一个“深度方法总是更好”。带重构的方法抗噪声更差，有人问再补。")


def propositions(prs):
    slide = new(prs, 22, "回到开头的三件事", "每件单独回答")
    cards = [
        ("坐标比分组算法要紧", "自然小图上，冻结的外部视觉坐标明显强于从零训练坐标。"),
        ("边学边分，有时更好，但不是总是", "深度嵌入聚类在部分基准上占优。E02 显示重构项存在量纲匹配的甜点区。"),
        ("有了好坐标，不必再加一个复杂分组器", "在冻结 DINOv2 和 CLIP 坐标上，K 均值已经是强基线。"),
    ]
    for index, (title, body) in enumerate(cards):
        top = 1.3 + index * 1.85
        rect(slide, 0.45, top, 12.4, 1.7, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.14, 11.9, 0.4, title, size=18, bold=True, color=NAVY)
        text(slide, 0.7, top + 0.6, 11.9, 0.9, body, size=15, color=INK)
    notes(slide, "Fashion 和 MNIST 的规则输出不是 ARI 最高的方法。下一页之前要把这个差别说清。规则页已经写了。")


def limits(prs):
    slide = new(prs, 23, "这些话要和结论一起听", "否则会把结果说大")
    limits_ = [
        "我们事先知道一共有 10 类。真正没标签的任务，常常连有几类都不知道。",
        "每套图只用了一部分训练图片。这不是把数据用到极致之后的最好成绩。",
        "对比学习用的是小网络。它说明这条思路在今天这个设置下怎么样，不是论文里大网络的成绩。",
        "DINOv2 和 CLIP 见过我们没有的海量图片。它们赢，不代表分组公式本身更聪明。",
    ]
    for index, line in enumerate(limits_):
        top = 1.4 + index * 1.3
        rect(slide, 0.45, top, 12.4, 1.12, WHITE, LINE, 0.08)
        text(slide, 0.75, top + 0.28, 11.8, 0.6, line, size=18, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "四条都要说。尤其是外部数据和小型 CNN。")


def close(prs):
    slide = new(prs, 24, "先看表示和预算", "收束")
    rect(slide, 0.45, 1.6, 12.4, 4.4, NAVY, radius=0.06)
    text(slide, 0.9, 2.15, 11.5, 2.2, "先问坐标有没有语义。\n再问要不要边学边分。\n最后问现成模型是不是已经够用。", size=28, bold=True, color=WHITE)
    text(slide, 0.9, 4.85, 11.5, 0.7, "分组算法要解决的是划界。它划在什么空间里，比算法本身更决定结果。", size=18, color=RGBColor(0xF5, 0xC1, 0x6C))
    notes(slide, "停在这三问。答疑时回到对应的那一套图，不要回到仓库。")


def table(slide, left, top, width, headers, data, widths, font=14):
    rows = len(data) + 1
    cols = len(headers)
    row_h = min(0.56, 5.05 / rows)
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(row_h * rows))
    grid = shape.table
    for index, value in enumerate(widths):
        grid.columns[index].width = Inches(value)
    for j, header in enumerate(headers):
        cell = grid.cell(0, j)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        paragraph = cell.text_frame.paragraphs[0]
        _run(paragraph, header, font, True, WHITE)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(data, start=1):
        for j, value in enumerate(row):
            cell = grid.cell(i, j)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else SOFT
            paragraph = cell.text_frame.paragraphs[0]
            _run(paragraph, value, font, False, INK)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.02)
    shape.name = "data_table"
    return shape


def _equation(slide, filename, l, t, w, h):
    path = ASSETS / filename
    if path.exists():
        picture = slide.shapes.add_picture(str(path), Inches(l), Inches(t), height=Inches(h))
        if picture.width > Inches(w):
            scale = Inches(w) / picture.width
            picture.width = int(picture.width * scale)
            picture.height = int(picture.height * scale)
    else:
        rect(slide, l, t, w, h, WHITE, LINE, 0.06)
        text(slide, l + 0.2, t + 0.25, w - 0.4, h - 0.4, filename, size=14, color=MUTED)


if __name__ == "__main__":
    print(build())
