"""Build the 24-slide deck. Result numbers are read from tables, never typed in."""

from __future__ import annotations

import csv
import json
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, H, NAVY)
    rect(slide, 0, 0, 0.18, H, AMBER)
    text(slide, 0.7, 1.35, 11, 0.3, "第九组  ·  课程汇报", size=16, bold=True, color=RGBColor(0xF5, 0xC1, 0x6C))
    text(slide, 0.7, 1.85, 11.5, 1.6, "深度聚类：\n范式演进与方法选型", size=40, bold=True, color=WHITE)
    text(slide, 0.72, 4.15, 10.5, 0.9, "在不同的数据结构、计算预算和先验下，\n什么时候用经典聚类、联合优化、生成模型、对比学习或预训练特征。", size=18, color=RGBColor(0xD6, 0xE2, 0xF0))
    text(slide, 0.72, 6.35, 10, 0.3, "结果只来自本仓库实测。没有数字的格子保持空白。", size=14, color=RGBColor(0xF5, 0xC1, 0x6C))
    notes(slide, "开场只抛问题。不要预告谁会赢。")


def question(prs):
    slide = new(prs, 2, "三个命题，都允许被否定", "问题")
    items = [
        ("01", "表征", "表征学习是否比聚类器本身的复杂度更重要？"),
        ("02", "目标", "联合优化是否始终优于先学表征、再聚类？"),
        ("03", "预算", "高质量预训练特征可用时，复杂聚类还值不值得？"),
    ]
    for index, (num, label, body) in enumerate(items):
        left = 0.42 + index * 4.2
        rect(slide, left, 1.45, 3.95, 4.3, WHITE, LINE, 0.08)
        rect(slide, left, 1.45, 3.95, 0.1, AMBER if index == 2 else NAVY)
        text(slide, left + 0.25, 1.75, 3.4, 0.4, num, size=20, bold=True, color=AMBER)
        text(slide, left + 0.25, 2.25, 3.4, 0.5, label, size=26, bold=True, color=NAVY)
        text(slide, left + 0.25, 3.1, 3.4, 2.0, body, size=18, color=INK)
    notes(slide, "强调三个命题都可以被实验反对。")


def textbook(prs):
    slide = new(prs, 3, "教材提供的是坐标，不是算法清单", "统计学起点")
    rows = [
        ("第2章  随机向量", "距离、协方差，以及高维里欧氏距离变钝"),
        ("第3章  多元正态", "高斯混合的概率来源"),
        ("第6章  聚类分析", "K-means、系统聚类；今天从 K-means 出发"),
        ("第7章  主成分", "线性降维。深度编码器是它的非线性对照"),
        ("第8章  因子分析", "潜变量。生成式聚类沿这条线往下走"),
    ]
    for index, (head, body) in enumerate(rows):
        top = 1.25 + index * 1.05
        rect(slide, 0.42, top, 12.4, 0.92, WHITE, LINE, 0.08)
        rect(slide, 0.42, top, 0.1, 0.92, TEAL)
        text(slide, 0.75, top + 0.12, 4.2, 0.68, head, size=18, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 5.1, top + 0.12, 7.3, 0.68, body, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "点明第6章是起点，第7、8章解释后来的表征和潜变量。")


def kmeans(prs):
    slide = new(prs, 4, "K-means 假设距离已经有意义", "目标函数")
    _equation(slide, "eq_kmeans.png", 0.5, 1.35, 8.2, 1.15)
    rect(slide, 0.45, 2.8, 6.1, 3.9, WHITE, LINE, 0.08)
    text(slide, 0.7, 3.05, 5.6, 0.4, "它在做什么", size=18, bold=True, color=NAVY)
    text(slide, 0.7, 3.6, 5.6, 2.7, "固定坐标，交替更新中心和分配。\n类数 K 要事先给定。\n中心是该类样本的均值。\n\n像素上的欧氏距离，常常不是我们关心的语义差别。", size=16, color=INK)
    rect(slide, 6.8, 2.8, 5.95, 3.9, NAVY, radius=0.08)
    text(slide, 7.1, 3.05, 5.4, 0.4, "因此下一步不是换一个 K", size=18, bold=True, color=WHITE)
    text(slide, 7.1, 3.65, 5.4, 2.6, "如果坐标不对，更复杂的分配规则也是在错的空间里切分。\n\n深度聚类改变的是：用什么表示来分组。", size=16, color=RGBColor(0xE8, 0xEE, 0xF6))
    notes(slide, "先写目标，再指出像素距离的问题。")


def representation(prs):
    slide = new(prs, 5, "分组之前，先学习表示", "从固定特征到可学习特征")
    _equation(slide, "eq_map.png", 0.55, 1.45, 8.5, 1.2)
    labels = [("x", "原始观测\n像素或向量"), ("fθ", "可学习编码器\n线性或非线性"), ("z", "用来聚类的坐标"), ("ĉ", "簇分配")]
    for index, (title, body) in enumerate(labels):
        left = 0.5 + index * 3.2
        rect(slide, left, 3.15, 2.9, 2.7, WHITE, LINE, 0.08)
        text(slide, left + 0.2, 3.4, 2.5, 0.6, title, size=26, bold=True, color=NAVY)
        text(slide, left + 0.2, 4.2, 2.5, 1.2, body, size=16, color=INK)
    notes(slide, "编码器是可学习的。z 的维度和训练目标后面会分开比较。")


def taxonomy(prs):
    slide = new(prs, 6, "四类交互，不是四段历史", "Zhou 等，ACM Comput. Surv. 2025")
    cards = [
        ("Multi-stage", "先固定表征，再聚类", "AE + K-means\nSimCLR + K-means\nDINOv2 / CLIP + K-means"),
        ("Simultaneously", "表征和簇分配一起训练", "DEC\nIDEC"),
        ("Generative", "显式的潜变量混合", "VaDE\nGMM"),
        ("Iterative", "伪标签或视图一致性来回更新", "CC\n文献中的 DeepCluster"),
    ]
    for index, (title, body, examples) in enumerate(cards):
        left = 0.4 + (index % 4) * 3.22
        rect(slide, left, 1.45, 3.05, 4.15, WHITE, LINE, 0.08)
        rect(slide, left, 1.45, 3.05, 0.1, [NAVY, AMBER, TEAL, RGBColor(0x6D, 0x28, 0xD9)][index])
        text(slide, left + 0.18, 1.7, 2.7, 0.7, title, size=16, bold=True, color=NAVY)
        text(slide, left + 0.18, 2.45, 2.7, 0.9, body, size=14, color=INK)
        text(slide, left + 0.18, 3.5, 2.7, 1.7, examples, size=14, color=MUTED)
    text(slide, 0.45, 5.85, 12.3, 0.9, "Simultaneously 是综述 Figure 1 的原标签。四支可以并存，后出现的方法并不取代前面的方法。", size=14, color=MUTED)
    notes(slide, "DOI 10.1145/3689036。类别名来自预印本 Figure 1，正式版同一篇综述。")


def axes(prs):
    slide = new(prs, 7, "再用四个坐标看方法，避免硬分类", "比较框架")
    rows = [
        ("表征空间", "原始像素  ·  线性子空间  ·  从零训练的隐空间  ·  外部预训练"),
        ("优化方式", "分阶段  ·  交替  ·  联合"),
        ("如何分配", "距离  ·  Student-t 软分配  ·  混合概率"),
        ("信息从哪来", "重构  ·  伪标签  ·  数据增强  ·  外部文本"),
    ]
    for index, (head, body) in enumerate(rows):
        top = 1.4 + index * 1.25
        rect(slide, 0.45, top, 12.4, 1.08, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.14, 2.6, 0.8, head, size=18, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        text(slide, 3.5, top + 0.14, 9.0, 0.8, body, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "这套坐标解释联系，不把算法塞进互斥格子。")


def dec(prs):
    slide = new(prs, 8, "DEC：用聚类目标改表示", "Xie, Girshick, Farhadi, ICML 2016")
    _equation(slide, "eq_dec.png", 0.45, 1.3, 12.2, 1.15)
    _equation(slide, "eq_kl.png", 0.45, 2.55, 7.5, 0.95)
    points = [
        "q 是样本到聚类中心的软分配，α = 1。",
        "p 由 q² / 簇频率得到，每个 epoch 用全体训练嵌入重算一次。",
        "反传只经过 q。p 不带梯度。",
        "早停看训练集分配变化，不看测试标签。",
    ]
    for index, line in enumerate(points):
        top = 3.7 + index * 0.72
        rect(slide, 0.45, top, 0.12, 0.48, AMBER)
        text(slide, 0.75, top, 12, 0.5, line, size=16, color=INK)
    notes(slide, "这是全场要推导的第一段。强调 p 的计算不用标签。")


def idec(prs):
    slide = new(prs, 9, "IDEC 把重构加回来", "Guo 等，IJCAI 2017")
    _equation(slide, "eq_idec.png", 0.4, 1.28, 12.4, 1.2)
    rect(slide, 0.45, 2.75, 6.15, 3.9, WHITE, LINE, 0.08)
    text(slide, 0.7, 2.95, 5.7, 0.4, "这个 λ 不是原文的 γ", size=18, bold=True, color=NAVY)
    text(slide, 0.7, 3.55, 5.7, 2.8, "原文大致是重构加 γ 倍聚类损失。\n我们把系数放在重构项上，因为要问：加强聚类目标会不会破坏表示。\n\nλ = 0 就是拿掉重构梯度。", size=16, color=INK)
    rect(slide, 6.85, 2.75, 5.95, 3.9, SOFT, radius=0.08)
    text(slide, 7.1, 2.95, 5.5, 0.4, "预先指定的五个值", size=18, bold=True, color=NAVY)
    text(slide, 7.1, 3.55, 5.5, 2.7, "0，0.01，0.1，1，10。\n\n不是搜出来的最优值。\nE02 把五档都报出来，包括不好的那档。", size=16, color=INK)
    notes(slide, "一定要口头区分 λ 和 Guo 等人的 γ。重构是每图平方和。")


def vade(prs):
    slide = new(prs, 10, "VaDE：混合先验下的生成聚类", "Jiang 等，IJCAI 2017")
    cols = [
        ("先验", "z 来自一个对角高斯混合。簇是成分，不是事后贴上的标签。"),
        ("推断", "编码器给出 q(z|x)。簇责任由各成分的后验得到。"),
        ("本次实现", "重构用每图平方误差，不用原论文的 Bernoulli 解码。数字不能和原表并列。"),
    ]
    for index, (title, body) in enumerate(cols):
        left = 0.45 + index * 4.2
        rect(slide, left, 1.45, 3.95, 4.6, WHITE, LINE, 0.08)
        text(slide, left + 0.25, 1.75, 3.45, 0.5, title, size=22, bold=True, color=NAVY)
        text(slide, left + 0.25, 2.55, 3.45, 3.0, body, size=16, color=INK)
    notes(slide, "生成式的价值是解释，不一定是更高的 ACC。等实测再说话。")


def contrast(prs):
    slide = new(prs, 11, "对比学习：先表示，或同时做簇", "两条路线，同一个小网络家族")
    rect(slide, 0.45, 1.4, 6.05, 4.7, WHITE, LINE, 0.08)
    text(slide, 0.7, 1.65, 5.6, 0.45, "SimCLR，然后 K-means", size=20, bold=True, color=NAVY)
    text(slide, 0.7, 2.3, 5.5, 3.4, "两个视图，NT-Xent，温度 0.5。\n聚类用投影头之前的 128 维。\n\n这是多阶段：对比损失里没有簇。\nChen 等，ICML 2020。", size=16, color=INK)
    rect(slide, 6.75, 1.4, 6.05, 4.7, WHITE, LINE, 0.08)
    text(slide, 7.0, 1.65, 5.6, 0.45, "Contrastive Clustering", size=20, bold=True, color=NAVY)
    text(slide, 7.0, 2.3, 5.5, 3.4, "实例对比之外，把两个视图的簇分配按列对比。\n簇温度 1.0。\n\n网络是小型 CNN，不是原文 ResNet。\nLi 等，AAAI 2021。", size=16, color=INK)
    notes(slide, "主动声明不是官方实现。MNIST 不做水平翻转。")


def pretrained(prs):
    slide = new(prs, 12, "外部表征把数据预算也带进来了", "预训练与文本")
    blocks = [
        ("冻结视觉特征", "DINOv2 ViT-S/14 来自大规模无标注图像。CLIP ViT-B/32 来自图文对。主干不更新。"),
        ("文本引导", "TAC 用 WordNet 名词做外部语义。Li 等，ICML 2024。本次不重训，只作为条件。"),
        ("统一评测", "VLM4Cluster，arXiv:2609.36648，2026-09-29 预印本。17 个方法，20 个数据集。尚未同行评审。"),
    ]
    for index, (title, body) in enumerate(blocks):
        top = 1.35 + index * 1.75
        rect(slide, 0.45, top, 12.4, 1.58, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.16, 11.8, 0.4, title, size=18, bold=True, color=NAVY)
        text(slide, 0.7, top + 0.62, 11.8, 0.75, body, size=15, color=INK)
    notes(slide, "不要把预训练模型和从零训练的模型放进同一张公平榜。")


def hypotheses(prs):
    slide = new(prs, 13, "选型表是假设", "实验前写定")
    headers = ["条件", "先考虑"]
    data = [
        ("低维、类数已知、算力紧", "K-means 或 PCA + K-means"),
        ("高维、没有外部模型", "AE，然后 K-means"),
        ("嵌入要直接服务聚类", "IDEC，而不是只有 KL"),
        ("需要混合解释", "VaDE 或 GMM"),
        ("增强可靠", "SimCLR 或 CC"),
        ("预训练特征与领域匹配", "冻结特征 + K-means"),
    ]
    table(slide, 0.45, 1.35, 12.4, headers, data, [4.8, 7.6])
    text(slide, 0.5, 6.55, 12, 0.45, "哪一条成立，由后面的实测决定。这里不填预期准确率。", size=15, color=MUTED)
    notes(slide, "这是假设页。不要讲成结论。")


def protocol(prs):
    slide = new(prs, 14, "协议在看测试集之前就固定了", "实验纪律")
    items = [
        ("子集", "MNIST 与 Fashion 各 2 万训练、1 万测试。CIFAR-10 为 1 万 / 2 千。种子 0、1、2。"),
        ("网络", "灰度图用 784-500-500-2000-10。CIFAR 用卷积自编码器。SimCLR 维度是 128。"),
        ("标签", "测试标签只做最终 ACC、NMI、ARI。不参与早停，不参与挑种子。"),
        ("预算", "AE 与 DEC/IDEC 各 50 epoch。训练损失仍明显下降时最多再加 10 epoch，只加一次。"),
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
    slide = new(prs, 15, "E01  九种方法的平均 ARI", "3 个种子")
    rows = load_e01()
    methods = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    labels = {"mnist": "MNIST", "fashion_mnist": "Fashion", "cifar10": "CIFAR-10"}
    present = [method for method in methods if any(lookup_ari(rows, dataset, method) is not None for dataset in datasets)]
    if not present:
        rect(slide, 0.45, 1.5, 12.4, 4.8, WHITE, LINE, 0.08)
        text(slide, 0.8, 3.2, 11.6, 1.0, "这一页的表还没有生成。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
        return
    data = []
    for method in present:
        line = [method]
        for dataset in datasets:
            value = lookup_ari(rows, dataset, method)
            line.append("—" if value is None else f"{value:.3f}")
        data.append(line)
    table(slide, 0.55, 1.22, 12.2, ["方法", "MNIST", "Fashion", "CIFAR-10"], data, [3.4, 2.9, 2.9, 3.0], font=13)
    bests = []
    for dataset in datasets:
        scored = [(method, lookup_ari(rows, dataset, method)) for method in present]
        scored = [item for item in scored if item[1] is not None]
        if scored:
            method, value = max(scored, key=lambda item: item[1])
            bests.append(f"{labels[dataset]} {method} {value:.3f}")
    text(slide, 0.55, 6.45, 12.2, 0.4, "平均 ARI。最高：" + "；".join(bests) + "。", size=13, color=MUTED)
    notes(slide, "CIFAR 上从零训练的方法都挤在 0.05 附近。MNIST 最高是 DEC，Fashion 最高也是 DEC。")


def shared_init(prs):
    slide = new(prs, 16, "AE、DEC、IDEC 从同一份预训练出发", "对照解释")
    steps = [
        ("1", "预训练", "同一子集、同一隐维，只训一次自编码器。"),
        ("2", "AE + K-means", "编码器冻结。中心是嵌入上的 K-means。"),
        ("3", "DEC / IDEC", "从这份权重继续微调。差别来自目标，不来自另一次初始化。"),
    ]
    for index, (num, title, body) in enumerate(steps):
        left = 0.45 + index * 4.2
        rect(slide, left, 1.6, 3.95, 4.2, WHITE, LINE, 0.08)
        text(slide, left + 0.25, 1.9, 1, 0.5, num, size=28, bold=True, color=AMBER)
        text(slide, left + 0.25, 2.6, 3.4, 0.6, title, size=20, bold=True, color=NAVY)
        text(slide, left + 0.25, 3.45, 3.4, 1.8, body, size=16, color=INK)
    notes(slide, "VaDE 和 SimCLR 不在这个共享初始化里。口头说一次。")


def e02(prs):
    slide = new(prs, 17, "E02  重构权重", "Fashion-MNIST，隐维 10，种子 0")
    image = ROOT / "results" / "figures" / "fig_e02_lambda.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.5), Inches(1.35), Inches(8.2), Inches(5.4))
    else:
        rect(slide, 0.45, 1.4, 8.3, 5.2, WHITE, LINE, 0.08)
        text(slide, 0.7, 3.4, 7.8, 1.2, "λ 曲线尚未生成。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    rect(slide, 8.95, 1.4, 3.9, 5.2, NAVY, radius=0.08)
    text(slide, 9.2, 1.7, 3.45, 0.7, "测到的权衡", size=20, bold=True, color=WHITE)
    text(slide, 9.2, 2.55, 3.45, 3.6, "λ 从 0 加到 10，测试重构误差下降，ARI 也从约 0.47 降到 0.44。\n\n加重构没有提高 ARI。", size=16, color=WHITE)
    notes(slide, "五档都报了。重构变好的同时 ARI 变差，不要只讲其中一档。")


def e03(prs):
    slide = new(prs, 18, "E03  噪声加在测试图上", "模型不重新训练")
    image = ROOT / "results" / "figures" / "fig_e03_noise.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.45), Inches(1.35), Inches(8.4), Inches(5.4))
    else:
        rect(slide, 0.45, 1.4, 8.3, 5.2, WHITE, LINE, 0.08)
        text(slide, 0.7, 3.4, 7.8, 1.0, "噪声曲线尚未生成。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    rect(slide, 8.95, 1.4, 3.9, 5.2, WHITE, LINE, 0.08)
    text(slide, 9.2, 1.65, 3.45, 0.8, "协议", size=18, bold=True, color=NAVY)
    text(slide, 9.2, 2.5, 3.45, 3.6, "标准差 0、0.1、0.2、0.3。\n像素截断到 [0, 1]。\n\n这是分布变化，不是训练时的数据增强。", size=15, color=INK)
    notes(slide, "K 设错、不平衡和样本量在仓库的 E03 JSON 里，正文只放噪声，除非那些曲线已经画出来。")


def e04(prs):
    slide = new(prs, 19, "E04  同一特征上，换聚类器", "CIFAR-10，平均 ARI")
    summary = load_e01()
    runs = load_runs()

    def e04_ari(method):
        values = [float(row["ari"]) for row in runs if row["experiment"] == "E04" and row["method"] == method and row["ari"] not in ("", None)]
        if not values:
            return None
        return sum(values) / len(values)

    def cell(value):
        return "—" if value is None else f"{value:.3f}"

    data = [
        ["随机 CNN", cell(e04_ari("random_cnn_kmeans")), cell(e04_ari("random_cnn_gmm")), cell(e04_ari("random_cnn_dec_head"))],
        ["AE", cell(lookup_ari(summary, "cifar10", "ae_kmeans")), cell(e04_ari("ae_gmm")), cell(e04_ari("ae_dec_head"))],
        ["SimCLR", cell(lookup_ari(summary, "cifar10", "simclr_kmeans")), cell(e04_ari("simclr_gmm")), cell(e04_ari("simclr_dec_head"))],
        ["DINOv2", cell(e04_ari("dinov2_kmeans")), cell(e04_ari("dinov2_gmm")), cell(e04_ari("dinov2_dec_head"))],
        ["CLIP", cell(e04_ari("clip_kmeans")), cell(e04_ari("clip_gmm")), cell(e04_ari("clip_dec_head"))],
    ]
    table(slide, 0.7, 1.35, 11.9, ["编码器", "K-means", "GMM", "DEC 头"], data, [2.8, 3.0, 3.0, 3.1], font=16)
    text(slide, 0.7, 5.55, 11.9, 1.2, "随机 CNN、AE、SimCLR 都在 0.06 以下。DINOv2 + K-means 是 0.622，CLIP + K-means 是 0.615。\n同一冻结特征上，GMM 和 DEC 头都没有超过 K-means。DINOv2 和 CLIP 用了外部数据，不和从零训练的模型排在一张公平榜上。", size=15, color=INK)
    notes(slide, "第二阶段没有赚到 0.02。规则因此停在 DINOv2 + K-means。")


def pareto(prs):
    slide = new(prs, 20, "不找唯一赢家", "ARI 与训练时间")
    image = ROOT / "results" / "figures" / "fig_e05_pareto.png"
    if image.exists():
        slide.shapes.add_picture(str(image), Inches(0.4), Inches(1.3), Inches(8.0), Inches(5.5))
        text(slide, 8.6, 1.6, 4.3, 4.5, "琥珀点是当前测到的非支配方法：没有另一个方法同时 ARI 更高、时间更短。\n\n灰色点被支配。\n\n显存和噪声还没齐时，不把这张图叫成四目标帕累托。", size=16, color=INK)
    else:
        rect(slide, 0.5, 1.6, 12.3, 4.6, WHITE, LINE, 0.08)
        text(slide, 0.9, 3.2, 11.5, 1.2, "前沿图会在 E01 的时间和 ARI 都写入之后生成。\n现在不画示意图充当结果。", size=20, color=MUTED, align=PP_ALIGN.CENTER)
    notes(slide, "只解释已经画出来的点。")


def rule(prs):
    slide = new(prs, 21, "规则的输出，不是 ARI 最高者", "selection.json")
    choices = [
        ("CIFAR-10", "DINOv2 + K-means", "预训练 K-means 比从零训练的最好方法高过 0.05。第二阶段没有再高 0.02。"),
        ("MNIST", "AE + K-means", "IDEC 没保住 90% 的噪声 ARI，规则退回 AE。实测最高仍是 DEC，0.669。"),
        ("Fashion", "PCA + K-means", "AE 相对 PCA 的差不到 0.02，两者接近时规则取更快的。实测最高是 DEC，0.435。"),
    ]
    for index, (dataset, choice, body) in enumerate(choices):
        top = 1.28 + index * 1.85
        rect(slide, 0.45, top, 12.4, 1.7, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.16, 2.6, 0.4, dataset, size=14, bold=True, color=AMBER)
        text(slide, 0.7, top + 0.52, 11.8, 0.4, choice, size=22, bold=True, color=NAVY)
        text(slide, 0.7, top + 1.05, 11.8, 0.5, body, size=15, color=INK)
    notes(slide, "规则没有 DEC 分支。所以 MNIST 和 Fashion 上它不会选出 ARI 更高的 DEC。要把这两句话分开说。")


def propositions(prs):
    slide = new(prs, 22, "三个命题，按测到的数说", "结论")
    cards = [
        ("表征更要紧", "CIFAR-10 上，从零训练的九种方法 ARI 都在 0.03 到 0.05。DINOv2 配 K-means 是 0.622。同一特征上再换 GMM 或 DEC 头，都没有更高。"),
        ("联合优化不总是更好", "MNIST 上 DEC 为 0.669，高于先训练自编码器再做 K-means 的 0.587。Fashion 上 DEC 为 0.435。λ 加大后 ARI 下降，IDEC 在噪声下掉得更多。"),
        ("复杂头不值这个成本", "预注册规则在 CIFAR-10 上选择 DINOv2 配 K-means。DEC 头和 GMM 都没有再高出 0.02。"),
    ]
    for index, (title, body) in enumerate(cards):
        top = 1.3 + index * 1.85
        rect(slide, 0.45, top, 12.4, 1.7, WHITE, LINE, 0.08)
        text(slide, 0.7, top + 0.14, 11.9, 0.4, title, size=18, bold=True, color=NAVY)
        text(slide, 0.7, top + 0.6, 11.9, 0.9, body, size=15, color=INK)
    notes(slide, "Fashion 和 MNIST 的规则输出不是 ARI 最高的方法。下一页之前要把这个差别说清。规则页已经写了。")


def limits(prs):
    slide = new(prs, 23, "这些限制留在结论旁边", "不要放进附录才说")
    limits_ = [
        "K = 10 是已知类数。E03 会故意设错，但主表不是。",
        "训练只用官方训练集的一个固定子集，不是完整训练集上的最好成绩。",
        "SimCLR 和 CC 是小型 CNN。VaDE 的重构也不是原文的 Bernoulli 解码。",
        "DINOv2 和 CLIP 的优势包含它们见过的外部数据，不只是聚类算法。",
    ]
    for index, line in enumerate(limits_):
        top = 1.4 + index * 1.3
        rect(slide, 0.45, top, 12.4, 1.12, WHITE, LINE, 0.08)
        text(slide, 0.75, top + 0.28, 11.8, 0.6, line, size=18, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(slide, "四条都要说。尤其是外部数据和小型 CNN。")


def close(prs):
    slide = new(prs, 24, "先看表示和预算", "收束")
    rect(slide, 0.45, 1.6, 12.4, 4.4, NAVY, radius=0.06)
    text(slide, 0.9, 2.1, 11.5, 2.4, "聚类器要解决的是分配。\n表示决定分配发生在什么空间里。\n预训练把别人的数据预算带了进来。", size=26, bold=True, color=WHITE)
    text(slide, 0.9, 4.7, 11.5, 0.8, "复杂目标和复杂网络，只有在这个空间里还能提高指标、并且付得起成本时才值得。", size=18, color=RGBColor(0xF5, 0xC1, 0x6C))
    notes(slide, "停在这句话。答疑时回到空着的表，而不是回到印象。")


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
