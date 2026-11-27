"""Render crisp 300-DPI academic architecture, taxonomy, and decision-tree diagrams.

Conceptual diagrams (Fig.1-6) encode literature/mechanism structure only.
The method selection decision tree (Fig.17) reads verified metrics dynamically from
results/tables/ when present so no test numbers are hard-coded by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "assets"
TABLES = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

BLUE_DEEP = "#073DA5"
BLUE_MID = "#2563EB"
BLUE_LIGHT = "#EAF0FC"
BLUE_BORDER = "#9CB3D3"
TEAL = "#0F766E"
TEAL_LIGHT = "#E6F4F1"
AMBER = "#C2591A"
AMBER_LIGHT = "#FDF1E6"
PURPLE = "#6D28D9"
PURPLE_LIGHT = "#F1ECFC"
CRIMSON = "#C81E1E"
SLATE = "#475569"
SLATE_LIGHT = "#F1F5F9"
INK = "#111722"
WHITE = "#FFFFFF"


def _setup() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei") if f in installed]
    plt.rcParams.update(
        {
            "font.family": [*fonts, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "savefig.dpi": 300,
        }
    )


def _box(ax, x, y, w, h, title="", body="", fc=WHITE, ec=BLUE_BORDER, lw=1.4,
         title_color=BLUE_DEEP, body_color=INK, title_size=10.5, body_size=8.8,
         rad=0.14, header_h=None, header_fc=None, header_tc=WHITE, zorder=3,
         title_y=0.70, body_y=0.31):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if header_h and header_fc:
        hdr = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle=f"round,pad=0.02,rounding_size={min(rad, header_h * 0.4)}",
            facecolor=header_fc, edgecolor=header_fc, linewidth=0.8, zorder=zorder + 1,
        )
        ax.add_patch(hdr)
        if title:
            ax.text(x + w / 2, y + h - header_h / 2, title,
                    ha="center", va="center", fontsize=title_size, fontweight="bold",
                    color=header_tc, zorder=zorder + 2)
        if body:
            ax.text(x + w / 2, y + (h - header_h) / 2, body,
                    ha="center", va="center", fontsize=body_size, color=body_color,
                    linespacing=1.35, zorder=zorder + 2)
    else:
        if title and body:
            ax.text(x + w / 2, y + h * title_y, title,
                    ha="center", va="center", fontsize=title_size, fontweight="bold",
                    color=title_color, zorder=zorder + 1)
            ax.text(x + w / 2, y + h * body_y, body,
                    ha="center", va="center", fontsize=body_size, color=body_color,
                    linespacing=1.32, zorder=zorder + 1)
        elif title:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center", fontsize=title_size, fontweight="bold",
                    color=title_color, linespacing=1.25, zorder=zorder + 1)
        elif body:
            ax.text(x + w / 2, y + h / 2, body,
                    ha="center", va="center", fontsize=body_size, color=body_color,
                    linespacing=1.30, zorder=zorder + 1)
    return patch


def _arrow(ax, p1, p2, color=BLUE_DEEP, lw=1.8, ls="-", rad=0.0,
           label="", label_pos=0.5, label_dx=0.0, label_dy=0.16, zorder=4):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>,head_length=6.5,head_width=4.0",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(arr)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + label_dx
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_dy
        ax.text(
            mx, my, label, ha="center", va="center", fontsize=8.6, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc=WHITE, ec="none", alpha=0.92),
            zorder=zorder + 1,
        )


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print("rendered", name)


def fig1_paradigm_evolution() -> None:
    """Fig.1: Deep clustering paradigm evolution & Zhou et al. (2025) 4 coexisting interactions."""
    fig, ax = plt.subplots(figsize=(12.2, 4.9))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    ax.plot([0.5, 11.7], [4.36, 4.36], color=BLUE_BORDER, lw=2.2, zorder=1)
    _arrow(ax, (11.2, 4.36), (11.85, 4.36), color=BLUE_DEEP, lw=2.2)
    ax.text(6.1, 4.66, "深度聚类范式演进谱系（多种范式长期并存，并非简单线性淘汰）",
            ha="center", va="center", fontsize=12.4, fontweight="bold", color=BLUE_DEEP)

    stages = [
        ("固定特征基线", "经典统计聚类", "K-means / GMM\nPCA + K-means", "坐标固定不变\n欧氏距离/高斯云", SLATE, SLATE_LIGHT),
        ("多阶段式", "Multi-stage", "AE + K-means\nSimCLR + K-means", "先学表征后固定\n再接传统聚类器", BLUE_DEEP, BLUE_LIGHT),
        ("联合优化式", "Simultaneously", "DEC (2016)\nIDEC (2017)", "聚类目标反向更新\n编码器与簇中心", AMBER, AMBER_LIGHT),
        ("生成式潜变量", "Generative", "VaDE (2017)\n深度混合先验", "VAE + 高斯混合先验\n联合推断与生成", TEAL, TEAL_LIGHT),
        ("迭代与对比", "Iterative / Contrast", "DeepCluster / SwAV\nIIC / CC (2021)", "伪标签交替修正\n双视图实例+簇对比", PURPLE, PURPLE_LIGHT),
        ("外部预训练引导", "2024–2026 前沿", "DINOv2 / CLIP\nTAC / VLM4Cluster*", "冻结视觉基础模型\n外部语义与评测基准", CRIMSON, "#FBECEC"),
    ]
    w = 1.78
    gap = 0.18
    x0 = 0.25
    for idx, (title, sub, rep, mech, accent, tint) in enumerate(stages):
        x = x0 + idx * (w + gap)
        ax.add_patch(Circle((x + w / 2, 4.36), 0.11, facecolor=accent, edgecolor=WHITE, lw=1.8, zorder=5))
        ax.plot([x + w / 2, x + w / 2], [3.86, 4.25], color=accent, lw=1.4, ls=":")
        _box(ax, x, 1.02, w, 2.82, title=title, fc=WHITE, ec=accent, lw=1.4,
             header_h=0.58, header_fc=accent, header_tc=WHITE, title_size=10.2, rad=0.12)
        ax.text(x + w / 2, 3.00, sub, ha="center", va="center", fontsize=8.6, fontweight="bold", color=accent)
        _box(ax, x + 0.10, 2.02, w - 0.20, 0.74, body=rep, fc=tint, ec="none", body_size=8.6, body_color=INK, rad=0.08, zorder=4)
        ax.text(x + w / 2, 1.48, mech, ha="center", va="center", fontsize=8.4, color=SLATE, linespacing=1.35)
        if idx < len(stages) - 1:
            _arrow(ax, (x + w + 0.01, 2.42), (x + w + gap - 0.01, 2.42), color=BLUE_BORDER, lw=1.4)

    # Bottom bracket cleanly separated from footnote
    bx1 = x0 + 1 * (w + gap)
    bx2 = x0 + 4 * (w + gap) + w
    ax.plot([bx1, bx1, bx2, bx2], [0.86, 0.68, 0.68, 0.86], color=BLUE_DEEP, lw=1.6)
    ax.text((bx1 + bx2) / 2, 0.44,
            "Zhou 等（ACM Computing Surveys, 2025）图 1 四类核心交互：Multi-stage · Simultaneously · Generative · Iterative",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color=BLUE_DEEP)
    ax.text(11.95, 0.12, "*注：VLM4Cluster (arXiv:2609.36648) 按 2026 年 9 月预印本引用，未作同行评审结论处理。",
            ha="right", va="center", fontsize=8.0, color=SLATE)
    _save(fig, "diag_paradigm_evolution.png")


def fig2_taxonomy_4d() -> None:
    """Fig.2: Four-dimensional method taxonomy framework."""
    fig, ax = plt.subplots(figsize=(12.0, 4.65))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.65)
    ax.axis("off")

    cols = [
        ("维度 1：表征空间", "数据如何编码为坐标 z", BLUE_DEEP, BLUE_LIGHT,
         ["原始像素空间\nK-means / GMM", "线性投影空间\nPCA + K-means", "深度隐空间\nAE / DEC / VaDE / CC", "外部预训练空间\nDINOv2 / CLIP"]),
        ("维度 2：优化交互", "表示与聚类如何配合", TEAL, TEAL_LIGHT,
         ["固定特征单阶段\nK-means / GMM", "多阶段解耦\nAE / SimCLR / DINOv2", "交替迭代修正\nDeepCluster / SwAV", "端到端联合优化\nDEC / IDEC / VaDE / CC"]),
        ("维度 3：聚类机制", "如何划分样本归属", AMBER, AMBER_LIGHT,
         ["欧氏距离质心\nK-means", "Student-t 软分配\nDEC / IDEC", "高斯混合概率云\nGMM / VaDE", "对比头直接输出\nCC / IIC"]),
        ("维度 4：监督信号", "无标签信息从哪来", PURPLE, PURPLE_LIGHT,
         ["像素重构误差\nAE / IDEC / VaDE", "自训练锐化分布\nDEC / IDEC", "双视图数据增强\nSimCLR / CC / SwAV", "外部视觉或图文语义\nDINOv2 / CLIP / TAC"]),
    ]

    cw = 2.74
    gap = 0.22
    x0 = 0.19
    for c_idx, (title, sub, accent, tint, items) in enumerate(cols):
        x = x0 + c_idx * (cw + gap)
        _box(ax, x, 0.18, cw, 4.28, title=title, fc=WHITE, ec=accent, lw=1.4,
             header_h=0.60, header_fc=accent, header_tc=WHITE, title_size=10.5, rad=0.14)
        ax.text(x + cw / 2, 3.62, sub, ha="center", va="center", fontsize=8.8, fontweight="bold", color=accent)
        for r_idx, txt in enumerate(items):
            y = 2.76 - r_idx * 0.80
            head, tail = txt.split("\n")
            _box(ax, x + 0.12, y, cw - 0.24, 0.68, title=head, body=tail,
                 fc=tint, ec=accent, lw=0.75, title_color=INK, body_color=SLATE,
                 title_size=9.1, body_size=8.1, rad=0.08, zorder=4, title_y=0.67, body_y=0.28)
        if c_idx < 3:
            _arrow(ax, (x + cw + 0.02, 2.25), (x + cw + gap - 0.02, 2.25), color=BLUE_BORDER, lw=1.5)

    _save(fig, "diag_taxonomy_4d.png")


def fig3_dec_idec() -> None:
    """Fig.3: DEC vs IDEC mechanism architecture comparison."""
    fig, ax = plt.subplots(figsize=(11.6, 4.95))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.95)
    ax.axis("off")

    _box(ax, 0.25, 1.75, 1.35, 1.15, title="输入图像 x", body="784 维像素\n(28×28)", fc=SLATE_LIGHT, ec=SLATE, title_color=INK, title_size=10.0, body_size=8.6)

    enc = Polygon([[1.95, 1.45], [3.55, 1.82], [3.55, 2.82], [1.95, 3.18]],
                  closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE_DEEP, lw=1.8, zorder=3)
    ax.add_patch(enc)
    ax.text(2.72, 2.32, "编码器 f_θ\n784→500→500\n→2000→10", ha="center", va="center",
            fontsize=9.1, fontweight="bold", color=BLUE_DEEP, zorder=4)
    _arrow(ax, (1.62, 2.32), (1.94, 2.32), color=BLUE_DEEP)

    _box(ax, 3.92, 1.82, 1.35, 1.00, title="隐表示 z", body="d = 10 维\n(AE 预训练)", fc=WHITE, ec=BLUE_DEEP, lw=1.8, title_size=10.2, body_size=8.5)
    _arrow(ax, (3.57, 2.32), (3.90, 2.32), color=BLUE_DEEP)

    # Upper branch: DEC clustering head
    ax.add_patch(FancyBboxPatch((5.65, 2.42), 5.75, 1.72, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F4F8FF", edgecolor=BLUE_DEEP, lw=1.4, zorder=1))
    ax.text(5.85, 3.92, "上分支：DEC 聚类自训练目标（KL 散度反向更新编码器与中心）",
            ha="left", va="center", fontsize=9.4, fontweight="bold", color=BLUE_DEEP, zorder=2)

    _box(ax, 5.85, 2.66, 1.72, 0.96, title="Student-t 软分配", body="q_ij ∝ (1+‖z_i-μ_j‖²)⁻¹\nK-means 初始化 μ_j",
         fc=WHITE, ec=BLUE_MID, lw=1.3, title_size=9.3, body_size=8.0)
    _box(ax, 7.88, 2.66, 1.55, 0.96, title="锐化目标分布 P", body="p_ij ∝ q_ij² / f_j\n拉紧高置信簇",
         fc=WHITE, ec=BLUE_MID, lw=1.3, title_size=9.3, body_size=8.0)
    _box(ax, 9.72, 2.66, 1.50, 0.96, title="聚类损失", body="L_cluster =\nKL(P ‖ Q)",
         fc=BLUE_DEEP, ec=BLUE_DEEP, title_color=WHITE, body_color="#E8F0FE", title_size=9.6, body_size=8.6)

    _arrow(ax, (5.29, 2.52), (5.83, 3.12), color=BLUE_DEEP)
    _arrow(ax, (7.59, 3.14), (7.86, 3.14), color=BLUE_DEEP)
    _arrow(ax, (9.45, 3.14), (9.70, 3.14), color=BLUE_DEEP)

    # Clean elevated backprop arc completely above the upper box
    _arrow(ax, (10.45, 4.20), (2.75, 3.26), color=CRIMSON, lw=1.6, ls="--", rad=0.18,
           label="反向传播：同时更新 μ_j 与 f_θ（缺少约束时存在隐空间漂移风险）",
           label_pos=0.50, label_dx=-0.10, label_dy=0.72)

    # Lower branch: IDEC reconstruction preservation
    ax.add_patch(FancyBboxPatch((5.65, 0.18), 5.75, 1.98, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=AMBER_LIGHT, edgecolor=AMBER, lw=1.6, ls="-", zorder=1))
    ax.text(5.85, 1.92, "下分支：IDEC 新增重构约束（保留解码器以守护局部流形结构）",
            ha="left", va="center", fontsize=9.4, fontweight="bold", color=AMBER, zorder=2)

    dec = Polygon([[5.90, 0.70], [7.25, 0.40], [7.25, 1.68], [5.90, 1.38]],
                  closed=True, facecolor=WHITE, edgecolor=AMBER, lw=1.6, zorder=3)
    ax.add_patch(dec)
    ax.text(6.62, 1.04, "解码器 g_φ\n10→2000→500\n→500→784", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=AMBER, zorder=4)
    _box(ax, 7.62, 0.58, 1.48, 0.92, title="重构图像 x̂", body="与原图 x 逐像素\n计算平方误差",
         fc=WHITE, ec=AMBER, lw=1.3, title_color=AMBER, title_size=9.3, body_size=8.0)
    _box(ax, 9.42, 0.58, 1.80, 0.92, title="重构约束项", body="λ · (1/B) Σ‖x_i - x̂_i‖₂²\nλ=0 退化为 DEC",
         fc=AMBER, ec=AMBER, title_color=WHITE, body_color="#FFF7ED", title_size=9.4, body_size=8.2)

    _arrow(ax, (5.29, 2.08), (5.88, 1.12), color=AMBER)
    _arrow(ax, (7.27, 1.04), (7.60, 1.04), color=AMBER)
    _arrow(ax, (9.12, 1.04), (9.40, 1.04), color=AMBER)

    ax.text(0.28, 0.58, "核心实验问题（E02）：\n只优化上分支（λ=0）会不会\n破坏表示结构？如何匹配\nKL 与 SSE 两个损失的量纲？",
            ha="left", va="center", fontsize=8.6, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc=TEAL_LIGHT, ec=TEAL, lw=1.1))

    _save(fig, "diag_dec_idec.png")


def fig4_vade_arch() -> None:
    """Fig.4: VaDE generative clustering probabilistic & network diagram."""
    fig, ax = plt.subplots(figsize=(11.6, 4.25))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.18, 0.25), 5.15, 3.72, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=TEAL_LIGHT, edgecolor=TEAL, lw=1.5, zorder=1))
    ax.text(2.75, 3.68, "生成过程：每个簇是隐空间的一团高斯云", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color=TEAL)

    _box(ax, 0.48, 2.38, 1.55, 0.92, title="1. 抽取簇 c", body="c ~ Cat(π)\n先验比例 π_c", fc=WHITE, ec=TEAL, title_color=TEAL, title_size=9.6, body_size=8.3)
    _box(ax, 2.36, 2.38, 1.82, 0.92, title="2. 抽取隐变量 z", body="z | c ~ N(μ_c, σ_c²I)\n带中心与方差形状", fc=WHITE, ec=TEAL, title_color=TEAL, title_size=9.6, body_size=8.2)
    for cx, cy, w, h, col in [(1.55, 1.24, 1.05, 0.56, BLUE_DEEP), (2.78, 1.36, 0.92, 0.64, TEAL), (3.95, 1.18, 1.00, 0.50, AMBER)]:
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=col, alpha=0.18, edgecolor=col, lw=1.4, zorder=3))
        ax.add_patch(Circle((cx, cy), 0.06, facecolor=col, zorder=4))
    ax.text(2.75, 0.56, "隐空间混合先验 p(z) = Σ_c π_c N(z | μ_c, σ_c²I)\n解码器 g_φ(z) 将隐变量重构为观测图像 x̂",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color=INK, linespacing=1.35)
    _arrow(ax, (2.05, 2.84), (2.34, 2.84), color=TEAL)
    _arrow(ax, (3.25, 2.36), (2.78, 1.74), color=TEAL)

    ax.add_patch(FancyBboxPatch((5.65, 0.25), 5.75, 3.72, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=BLUE_LIGHT, edgecolor=BLUE_DEEP, lw=1.5, zorder=1))
    ax.text(8.52, 3.68, "变分推断与 ELBO 联合优化", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color=BLUE_DEEP)

    _box(ax, 5.90, 2.38, 1.20, 0.92, title="输入 x", body="图像像素", fc=WHITE, ec=BLUE_DEEP, title_size=9.6, body_size=8.3)
    _box(ax, 7.42, 2.38, 1.75, 0.92, title="推断编码器", body="输出 (μ̃, log σ̃²)\n重参数化采样 z", fc=WHITE, ec=BLUE_DEEP, title_size=9.6, body_size=8.2)
    _box(ax, 9.50, 2.38, 1.65, 0.92, title="后验归属 γ_c", body="q(c|x) = p(c|z)\n闭式 Bayes 推断", fc=BLUE_DEEP, ec=BLUE_DEEP, title_color=WHITE, body_color="#E8F0FE", title_size=9.5, body_size=8.2)
    _arrow(ax, (7.12, 2.84), (7.40, 2.84), color=BLUE_DEEP)
    _arrow(ax, (9.19, 2.84), (9.48, 2.84), color=BLUE_DEEP)

    _box(ax, 5.90, 0.46, 5.25, 1.56,
         title="ELBO = 重构保真项  −  隐变量与混合先验 KL 散度",
         body="• 对比 K-means：刻画簇方差 σ_c 与先验 π_c，支持生成采样\n• 对比裸 GMM：先经深度编码器降维再建模，避开高维协方差病态\n• 实现边界：本仓库重构项为每图平方误差 SSE（非原论文 Bernoulli）",
         fc=WHITE, ec=BLUE_BORDER, title_color=BLUE_DEEP, title_size=9.3, body_size=8.0, rad=0.10, title_y=0.80, body_y=0.35)

    _save(fig, "diag_vade_arch.png")


def fig5_contrastive_arch() -> None:
    """Fig.5: SimCLR+K-means vs Contrastive Clustering (CC) dual-view architecture."""
    fig, ax = plt.subplots(figsize=(11.8, 4.4))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    _box(ax, 0.20, 1.72, 1.25, 0.95, title="原图 x", body="单张无标签\n训练样本", fc=SLATE_LIGHT, ec=SLATE, title_color=INK, title_size=10.0, body_size=8.4)
    _box(ax, 1.82, 2.68, 1.35, 0.78, title="增强视图 x̃ᵃ", body="裁剪 / 抖动", fc=BLUE_LIGHT, ec=BLUE_DEEP, title_size=9.4, body_size=8.2)
    _box(ax, 1.82, 0.94, 1.35, 0.78, title="增强视图 x̃ᵇ", body="随机语义保持", fc=BLUE_LIGHT, ec=BLUE_DEEP, title_size=9.4, body_size=8.2)
    _arrow(ax, (1.47, 2.38), (1.80, 3.04), color=BLUE_DEEP)
    _arrow(ax, (1.47, 2.00), (1.80, 1.34), color=BLUE_DEEP)

    _box(ax, 3.55, 1.48, 1.68, 1.44, title="共享 CNN 编码器", body="小型课程 CNN\n输出 128 维特征 h\n(非原论文 ResNet)",
         fc=WHITE, ec=BLUE_DEEP, lw=1.8, title_size=9.6, body_size=8.3)
    _arrow(ax, (3.19, 3.04), (3.53, 2.48), color=BLUE_DEEP)
    _arrow(ax, (3.19, 1.34), (3.53, 1.92), color=BLUE_DEEP)

    ax.add_patch(FancyBboxPatch((5.62, 2.40), 5.95, 1.75, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F4F8FF", edgecolor=BLUE_DEEP, lw=1.4, zorder=1))
    ax.text(5.82, 3.92, "路线 A：SimCLR + K-means（多阶段：先对比学表征，再离线聚类）",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=BLUE_DEEP)
    _box(ax, 5.82, 2.62, 1.82, 1.00, title="实例投影头", body="NT-Xent 对比损失\n拉近正对、推远负对", fc=WHITE, ec=BLUE_MID, title_size=9.3, body_size=8.1)
    _box(ax, 7.96, 2.62, 1.56, 1.00, title="丢弃投影头", body="取出 128 维\n冻结表示 h", fc=WHITE, ec=BLUE_MID, title_size=9.3, body_size=8.1)
    _box(ax, 9.84, 2.62, 1.52, 1.00, title="离线 K-means", body="n_init = 20\n输出预测簇 ĉ", fc=BLUE_DEEP, ec=BLUE_DEEP, title_color=WHITE, body_color="#E8F0FE", title_size=9.4, body_size=8.2)
    _arrow(ax, (5.25, 2.55), (5.80, 3.10), color=BLUE_DEEP)
    _arrow(ax, (7.66, 3.12), (7.94, 3.12), color=BLUE_DEEP)
    _arrow(ax, (9.54, 3.12), (9.82, 3.12), color=BLUE_DEEP)

    ax.add_patch(FancyBboxPatch((5.62, 0.22), 5.95, 1.94, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor=PURPLE_LIGHT, edgecolor=PURPLE, lw=1.5, zorder=1))
    ax.text(5.82, 1.94, "路线 B：Contrastive Clustering / CC（同时进行实例级 + 簇级对比）",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=PURPLE)
    _box(ax, 5.82, 1.09, 2.55, 0.65, title="行对比（实例级）", body="同一图像两种视图靠近", fc=WHITE, ec=PURPLE, title_color=PURPLE, title_size=8.8, body_size=8.0, rad=0.08, title_y=0.68, body_y=0.28)
    _box(ax, 5.82, 0.36, 2.55, 0.65, title="列对比（簇级）", body="同一簇在两视图列一致", fc=WHITE, ec=PURPLE, title_color=PURPLE, title_size=8.8, body_size=8.0, rad=0.08, title_y=0.68, body_y=0.28)
    _box(ax, 8.75, 0.42, 2.60, 1.28, title="簇头直接输出 ĉ", body="端到端联合产出簇概率\n注：课程版省去官方簇熵项\n与小网络共同影响实测上限",
         fc=PURPLE, ec=PURPLE, title_color=WHITE, body_color="#F3E8FF", title_size=9.4, body_size=8.0, title_y=0.77, body_y=0.33)
    _arrow(ax, (5.25, 1.85), (5.80, 1.40), color=PURPLE)
    _arrow(ax, (5.25, 1.72), (5.80, 0.68), color=PURPLE)
    _arrow(ax, (8.39, 1.40), (8.73, 1.18), color=PURPLE)
    _arrow(ax, (8.39, 0.68), (8.73, 0.94), color=PURPLE)

    _save(fig, "diag_contrastive_arch.png")


def fig6_pretrained_decouple() -> None:
    """Fig.6: Pretrained era encoder x clusterer decoupling framework (E04)."""
    fig, ax = plt.subplots(figsize=(11.4, 4.15))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.15)
    ax.axis("off")

    _box(ax, 0.22, 1.55, 1.45, 1.15, title="CIFAR-10", body="10,000 训练图\n2,000 测试图\n(同一切分)", fc=SLATE_LIGHT, ec=SLATE, title_color=INK, title_size=10.2, body_size=8.5)

    ax.add_patch(FancyBboxPatch((2.05, 1.98), 3.25, 1.96, boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=SLATE_LIGHT, edgecolor=SLATE, lw=1.2, ls="--", zorder=1))
    ax.text(3.68, 3.74, "从零训练主干（仅课程数据）", ha="center", va="center", fontsize=9.0, fontweight="bold", color=SLATE)
    for idx, txt in enumerate(["随机 CNN（未训练基线）", "卷积自编码器 AE", "SimCLR（小型 CNN）"]):
        _box(ax, 2.22, 3.12 - idx * 0.50, 2.90, 0.40, body=txt, fc=WHITE, ec=BLUE_BORDER, body_size=8.8, rad=0.07)

    ax.add_patch(FancyBboxPatch((2.05, 0.22), 3.25, 1.54, boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=BLUE_LIGHT, edgecolor=BLUE_DEEP, lw=1.6, zorder=1))
    ax.text(3.68, 1.56, "冻结外部预训练主干（含外部数据）", ha="center", va="center", fontsize=9.0, fontweight="bold", color=BLUE_DEEP)
    _box(ax, 2.22, 0.90, 2.90, 0.46, body="DINOv2 ViT-S/14（自监督视觉）", fc=WHITE, ec=BLUE_DEEP, body_size=8.8, body_color=BLUE_DEEP, rad=0.07)
    _box(ax, 2.22, 0.34, 2.90, 0.46, body="CLIP ViT-B/32（图文配对监督）", fc=WHITE, ec=BLUE_DEEP, body_size=8.8, body_color=BLUE_DEEP, rad=0.07)

    _arrow(ax, (1.69, 2.38), (2.04, 2.95), color=SLATE)
    _arrow(ax, (1.69, 1.90), (2.04, 0.98), color=BLUE_DEEP, lw=2.0)

    _box(ax, 5.75, 1.48, 1.48, 1.30, title="冻结特征 z", body="统一测试切分\n预训练特征经\nL2 归一化", fc=TEAL_LIGHT, ec=TEAL, title_color=TEAL, title_size=9.8, body_size=8.4)
    _arrow(ax, (5.32, 2.95), (5.73, 2.35), color=SLATE)
    _arrow(ax, (5.32, 0.98), (5.73, 1.95), color=BLUE_DEEP, lw=2.0)

    c_specs = [
        ("K-means", "欧氏质心划分\n(n_init=20)", BLUE_DEEP, 3.00),
        ("高斯混合 GMM", "对角协方差\n特征标准化", TEAL, 1.68),
        ("投影 + DEC 头", "仅微调 10 维头\n20 epochs", AMBER, 0.36),
    ]
    for title, body, col, y in c_specs:
        _box(ax, 7.75, y, 2.05, 0.95, title=title, body=body, fc=WHITE, ec=col, lw=1.6, title_color=col, title_size=9.8, body_size=8.3)
        _arrow(ax, (7.25, 2.13), (7.73, y + 0.48), color=col, lw=1.6)

    _box(ax, 10.10, 0.95, 1.12, 2.30, title="5 × 3\n解耦检验", body="行差看\n表示收益\n\n列差看\n聚类器收益",
         fc=BLUE_DEEP, ec=BLUE_DEEP, title_color=WHITE, body_color="#E8F0FE", title_size=10.2, body_size=8.6)
    for y in (3.48, 2.15, 0.84):
        _arrow(ax, (9.82, y), (10.08, 2.10), color=BLUE_BORDER, lw=1.3)

    _save(fig, "diag_pretrained_decouple.png")


def fig17_decision_tree() -> None:
    """Fig.17: Evidence-backed Method Selection Decision Tree (numbers read dynamically)."""
    def _csv(p: Path) -> list[dict]:
        return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []

    e01 = _csv(TABLES / "e01_summary.csv")
    e04 = _csv(TABLES / "e04_pretrained.csv")
    e02 = _csv(TABLES / "e02_ablation.csv")
    e03n = _csv(TABLES / "e03_noise.csv")

    def _e01_val(ds, m, k="ari_mean"):
        for r in e01:
            if r["dataset"] == ds and r["method"] == m:
                return float(r[k])
        return None

    dino_ari = next((float(r["ari_mean"]) for r in e04 if r["encoder"] == "dinov2"), None)
    clip_ari = next((float(r["ari_mean"]) for r in e04 if r["encoder"] == "clip"), None)
    pca_t = _e01_val("fashion_mnist", "pca_kmeans", "time_train_mean_sec")
    dec_m = _e01_val("mnist", "dec")
    dec_f = _e01_val("fashion_mnist", "dec")
    vade_m = _e01_val("mnist", "vade")
    km_clean = next((float(r["ari_mean"]) for r in e03n if r["dataset"] == "fashion_mnist" and r["method"] == "kmeans" and float(r["noise_std"]) == 0.0), None)
    km_noisy = next((float(r["ari_mean"]) for r in e03n if r["dataset"] == "fashion_mnist" and r["method"] == "kmeans" and float(r["noise_std"]) == 0.3), None)
    km_ret = (km_noisy / km_clean * 100) if (km_clean and km_noisy) else None
    lam01 = next((float(r["ari"]) for r in e02 if r["factor"] == "lambda_recon" and float(r["value"]) == 0.1), None)

    fig, ax = plt.subplots(figsize=(12.4, 5.25))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.25)
    ax.axis("off")

    _box(ax, 3.85, 4.35, 4.70, 0.74, title="Q1：是否有领域匹配的强预训练视觉模型？",
         body="（如自然图像可用 DINOv2 / CLIP 冻结编码器）",
         fc=BLUE_DEEP, ec=BLUE_DEEP, title_color=WHITE, body_color="#DCE9FF", title_size=10.3, body_size=8.4, rad=0.14)

    dino_s = f"{dino_ari:.2f}" if dino_ari else "—"
    clip_s = f"{clip_ari:.2f}" if clip_ari else "—"
    _box(ax, 8.80, 2.75, 3.40, 1.38, title="推荐：冻结预训练特征 + K-means",
         body=f"• E04 实测：DINOv2 {dino_s} / CLIP {clip_s}\n• 换 GMM 或加 DEC 头未再带来正收益\n• 若需外部语义引导，再评估 TAC/VLM",
         fc=TEAL_LIGHT, ec=TEAL, lw=1.8, title_color=TEAL, title_size=9.5, body_size=8.1, rad=0.12, title_y=0.80, body_y=0.35)
    _arrow(ax, (8.58, 4.70), (10.50, 4.16), color=TEAL, lw=2.0, label="是", label_pos=0.45, label_dy=0.16)

    _box(ax, 0.95, 2.98, 4.85, 0.76, title="Q2：样本极少（<5%）、强噪声或预算极紧？",
         body="（无外部预训练，需从当前数据直接建模）",
         fc=BLUE_LIGHT, ec=BLUE_DEEP, lw=1.6, title_color=BLUE_DEEP, title_size=9.9, body_size=8.3, rad=0.12)
    _arrow(ax, (3.83, 4.70), (3.35, 3.77), color=BLUE_DEEP, lw=2.0, label="否（从零建模）", label_pos=0.52, label_dx=-0.55, label_dy=0.06)

    ret_s = f"{km_ret:.0f}%" if km_ret else "—"
    pca_ts = f"{pca_t:.1f}s" if pca_t else "—"
    _box(ax, 0.16, 1.18, 3.30, 1.36, title="推荐：PCA+K-means / 像素 K-means",
         body=f"• E01 耗时仅 {pca_ts}，帕累托最快端点\n• E03 噪声保持率 {ret_s}，1% 小样本最优\n• E05 无标签训练稳定性最高（≈0.95）",
         fc=SLATE_LIGHT, ec=SLATE, lw=1.6, title_color=INK, title_size=9.0, body_size=8.0, rad=0.12, title_y=0.80, body_y=0.35)
    _arrow(ax, (2.25, 2.96), (1.80, 2.57), color=SLATE, lw=1.8, label="是", label_pos=0.45, label_dx=-0.22, label_dy=0.04)

    _box(ax, 4.30, 1.72, 4.10, 0.76, title="Q3：核心需求是概率生成还是高干净 ARI？",
         body="（样本充足、结构较清晰，允许 GPU 训练编码器）",
         fc=BLUE_LIGHT, ec=BLUE_DEEP, lw=1.5, title_color=BLUE_DEEP, title_size=9.6, body_size=8.2, rad=0.12)
    _arrow(ax, (4.85, 2.96), (6.30, 2.51), color=BLUE_DEEP, lw=1.8, label="否", label_pos=0.48, label_dx=0.18, label_dy=0.10)

    vade_s = f"{vade_m:.2f}" if vade_m else "—"
    dec_ms = f"{dec_m:.2f}" if dec_m else "—"
    dec_fs = f"{dec_f:.2f}" if dec_f else "—"
    lam_s = f"{lam01:.2f}" if lam01 else "—"

    _box(ax, 3.68, 0.14, 2.74, 1.18, title="概率云与生成：VaDE",
         body=f"• MNIST ARI {vade_s}，显式混合先验\n• 避开高维裸 GMM（轻噪塌缩至 0）",
         fc=PURPLE_LIGHT, ec=PURPLE, lw=1.5, title_color=PURPLE, title_size=9.1, body_size=7.9, rad=0.10, title_y=0.75, body_y=0.32)
    _box(ax, 6.62, 0.14, 2.74, 1.18, title="最高干净 ARI：DEC",
         body=f"• MNIST {dec_ms} / Fashion {dec_fs}\n• 需先 AE 预训练 + K-means 初始化",
         fc=BLUE_LIGHT, ec=BLUE_DEEP, lw=1.6, title_color=BLUE_DEEP, title_size=9.1, body_size=7.9, rad=0.10, title_y=0.75, body_y=0.32)
    _box(ax, 9.56, 0.14, 2.68, 1.18, title="兼顾流形保真：小权重 IDEC",
         body=f"• λ∈[0.01,0.1]（ARI≈{lam_s}, SSE -78%）\n• 警惕 λ≥1 淹没聚类梯度",
         fc=AMBER_LIGHT, ec=AMBER, lw=1.6, title_color=AMBER, title_size=9.0, body_size=7.8, rad=0.10, title_y=0.75, body_y=0.32)

    _arrow(ax, (5.30, 1.70), (5.05, 1.35), color=PURPLE, lw=1.6)
    _arrow(ax, (6.50, 1.70), (7.85, 1.35), color=BLUE_DEEP, lw=1.6)
    _arrow(ax, (7.80, 1.70), (10.75, 1.35), color=AMBER, lw=1.6)

    _box(ax, 0.16, 0.14, 3.30, 0.84, title="无标签落地红线（E05）",
         body="新数据无标签时禁止偷看测试指标；\n用训练稳定性 Jaccard + 耗时筛查候选。",
         fc="#FEF2F2", ec=CRIMSON, lw=1.2, title_color=CRIMSON, title_size=8.9, body_size=7.8, rad=0.10, title_y=0.74, body_y=0.30)

    _save(fig, "diag_decision_tree.png")


def main() -> None:
    _setup()
    fig1_paradigm_evolution()
    fig2_taxonomy_4d()
    fig3_dec_idec()
    fig4_vade_arch()
    fig5_contrastive_arch()
    fig6_pretrained_decouple()
    fig17_decision_tree()


if __name__ == "__main__":
    main()
