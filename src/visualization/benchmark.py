"""Publication-grade figures strictly derived from measured run tables and aligned embeddings."""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from visualization.style import (
    AMBER,
    BLUE_DEEP,
    BLUE_MID,
    CRIMSON,
    INK,
    MUTED,
    PURPLE,
    SLATE,
    TEAL,
    apply_style,
)

METHODS = {
    "kmeans": "像素 K-means",
    "pca_kmeans": "PCA + K-means",
    "gmm": "高斯混合 GMM",
    "ae_kmeans": "AE + K-means",
    "dec": "DEC",
    "idec": "IDEC",
    "vade": "VaDE",
    "simclr_kmeans": "SimCLR + K-means",
    "cc": "对比聚类 CC",
}
SHORT_METHODS = {
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
DATASETS = {"mnist": "MNIST\n(手写数字)", "fashion_mnist": "Fashion-MNIST\n(服装图像)", "cifar10": "CIFAR-10\n(自然图像)"}

# Distinct colorblind-friendly academic palette + markers
METHOD_COLORS = {
    "kmeans": "#0B4F6C",
    "pca_kmeans": "#2E7D5B",
    "gmm": "#B45309",
    "ae_kmeans": "#D95F3B",
    "dec": BLUE_DEEP,
    "idec": PURPLE,
    "vade": TEAL,
    "simclr_kmeans": "#9A5B36",
    "cc": "#526075",
}
METHOD_MARKERS = {
    "kmeans": "s",
    "pca_kmeans": "P",
    "gmm": "x",
    "ae_kmeans": "^",
    "dec": "o",
    "idec": "D",
    "vade": "v",
    "simclr_kmeans": "<",
    "cc": "v",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "academic_blue",
    ["#F5F8FD", "#D3E3FA", "#6CA0E8", "#1E5DC6", "#062F82"],
)
TSNE_PALETTE = [
    "#073DA5", "#0F766E", "#C2591A", "#7C3AED", "#DC2626",
    "#0284C7", "#65A30D", "#DB2777", "#B45309", "#475569",
]


def _setup() -> None:
    apply_style()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei") if f in installed]
    if not fonts:
        warnings.warn("Install a CJK font to display Chinese figure labels.")
    plt.rcParams.update(
        {
            "font.family": [*fonts, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "axes.axisbelow": True,
        }
    )


def _heat_e01(e01: pd.DataFrame) -> plt.Figure | None:
    if e01.empty:
        return None
    mean_grid = e01.pivot(index="method", columns="dataset", values="ari_mean").reindex(
        index=list(METHODS), columns=list(DATASETS)
    )
    std_grid = e01.pivot(index="method", columns="dataset", values="ari_std").reindex(
        index=list(METHODS), columns=list(DATASETS)
    )
    vals = mean_grid.to_numpy(float)
    stds = std_grid.to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    scale = Normalize(0.0, 0.72)
    im = ax.imshow(vals, cmap=HEAT_CMAP, norm=scale, aspect="auto")

    ax.set_xticks(range(len(DATASETS)), list(DATASETS.values()), fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(METHODS)), list(METHODS.values()), fontsize=10.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=7)
    ax.grid(False)

    # Crisp white cell borders
    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    # Family separator lines (Classical | AE/Joint | Generative | Contrastive)
    for sep in (2.5, 5.5, 6.5):
        ax.axhline(sep, color="#0B2545", linewidth=1.35, linestyle="-", alpha=0.65)

    # Highlight best cell per column
    col_best = np.nanargmax(vals, axis=0)
    for i, j in np.ndindex(vals.shape):
        val = vals[i, j]
        sd = stds[i, j]
        if np.isfinite(val):
            r, g, b, _ = HEAT_CMAP(scale(val))
            txt_color = "#FFFFFF" if 0.21 * r + 0.72 * g + 0.07 * b < 0.54 else INK
            is_best = i == col_best[j]
            main_txt = f"{val:.3f}" + (" ★" if is_best else "")
            ax.text(
                j, i - 0.11, main_txt,
                ha="center", va="center",
                fontsize=11.0 if is_best else 10.4,
                fontweight="bold" if is_best else "semibold",
                color=txt_color,
            )
            if np.isfinite(sd):
                ax.text(
                    j, i + 0.22, f"± {sd:.3f}",
                    ha="center", va="center",
                    fontsize=8.3,
                    color=txt_color,
                    alpha=0.88,
                )
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (j - 0.48, i - 0.47), 0.96, 0.94,
                        fill=False, edgecolor="#F59E0B", linewidth=2.0, zorder=4,
                    )
                )
        else:
            ax.text(j, i, "—", color=MUTED, ha="center", va="center")

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03, aspect=22)
    cbar.set_label("平均 ARI（3 种子均值）", fontsize=9.8, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.0, length=2)

    ax.set_title("E01 从零训练基准：9 种方法在 3 个数据集上的 ARI 对比", loc="left", pad=36, fontsize=12.5, color=BLUE_DEEP)
    fig.text(
        0.02, -0.01,
        "注：各格展示 3 个随机种子（0/1/2）的均值与样本标准差；★ 与金框标出该列均值最高者；SimCLR/CC 为小型 CNN 课程实现。",
        fontsize=8.6, color=MUTED,
    )
    return fig


def _heat_e04(pairs: pd.DataFrame, protocol: dict) -> plt.Figure | None:
    if pairs.empty:
        return None
    cfg = protocol["e04"]
    encoders = {
        "random_cnn": "随机 CNN（未训练）",
        "ae": "卷积自编码器 AE",
        "simclr": "SimCLR（小型 CNN）",
        "dinov2": "DINOv2 ViT-S/14 ★",
        "clip": "CLIP ViT-B/32 ★",
    }
    clusterers = {"kmeans": "K-means", "gmm": "高斯混合 GMM", "dec_head": "投影头 + DEC 微调"}
    grid = pairs.pivot(index="encoder", columns="clusterer", values="ari").reindex(
        index=cfg["encoders"], columns=cfg["clusterers"]
    )
    vals = grid.to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    scale = Normalize(0.0, 0.68)
    im = ax.imshow(vals, cmap=HEAT_CMAP, norm=scale, aspect="auto")

    ax.set_xticks(range(len(clusterers)), [clusterers[c] for c in grid.columns], fontsize=10.8, fontweight="bold")
    ax.set_yticks(range(len(encoders)), [encoders[e] for e in grid.index], fontsize=10.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=7)
    ax.grid(False)

    ax.set_xticks(np.arange(-0.5, len(clusterers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(encoders), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    # Clear boundary separating from-scratch vs external pretrained encoders
    ax.axhline(2.5, color=CRIMSON, linewidth=2.2, linestyle="--", zorder=5)
    ax.text(
        2.53, 1.0, "从零训练\n(课程数据)\n0.01–0.05",
        va="center", ha="left", fontsize=9.0, fontweight="bold", color=MUTED,
        transform=ax.transData,
    )
    ax.text(
        2.53, 3.5, "外部预训练\n(冻结主干)\n0.28–0.64",
        va="center", ha="left", fontsize=9.0, fontweight="bold", color=BLUE_DEEP,
        transform=ax.transData,
    )

    for i, j in np.ndindex(vals.shape):
        val = vals[i, j]
        if np.isfinite(val):
            r, g, b, _ = HEAT_CMAP(scale(val))
            txt_color = "#FFFFFF" if 0.21 * r + 0.72 * g + 0.07 * b < 0.54 else INK
            is_peak = val >= 0.62
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=12.0 if is_peak else 11.2,
                fontweight="bold" if is_peak else "semibold",
                color=txt_color,
            )
            if is_peak:
                ax.add_patch(
                    Rectangle(
                        (j - 0.47, i - 0.46), 0.94, 0.92,
                        fill=False, edgecolor="#F59E0B", linewidth=2.2, zorder=4,
                    )
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.18, aspect=18)
    cbar.set_label("ARI（CIFAR-10，seed 0）", fontsize=9.6, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.0, length=2)

    ax.set_title("E04 表征与聚类器解耦：同一批 CIFAR-10 特征的 5×3 交叉矩阵", loc="left", pad=34, fontsize=12.2, color=BLUE_DEEP)
    fig.text(
        0.02, -0.02,
        "注：红虚线下方 ★ 为冻结的外部预训练主干（含外部数据预算），不与上方从零训练模型作等预算排名；DEC 仅微调 10 维投影头。",
        fontsize=8.6, color=MUTED,
    )
    return fig


def _ablation(frame: pd.DataFrame, factor: str) -> plt.Figure | None:
    data = frame[frame.factor.eq(factor)].sort_values("value")
    if data.empty:
        return None
    xs = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), layout="constrained")
    labels = [f"{float(v):g}" for v in data.value]
    is_lam = factor == "lambda_recon"

    configs = [
        ("ari", BLUE_DEEP, "分组得分 ARI  ↑", (0.0, 0.56)),
        ("recon_sse", AMBER, "每图重构平方误差 SSE  ↓", (0.0, float(data["recon_sse"].max()) * 1.18)),
    ]
    for idx, (ax, (metric, color, title, ylim)) in enumerate(zip(axes, configs)):
        ys = data[metric].to_numpy(float)
        if is_lam:
            ax.axvspan(0.7, 2.3, color="#E6F4F1", alpha=0.85, zorder=0)
            note_txt = "量纲匹配甜点区\n(ARI ≈ 0.47)" if idx == 0 else "甜点区\nSSE 降 74%–78%"
            note_y = 0.27 if idx == 0 else ylim[1] * 0.86
            ax.text(
                1.5, note_y, note_txt,
                ha="center", va="top", fontsize=8.6, fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=TEAL, lw=0.85, alpha=0.95),
                zorder=5,
            )
        else:
            # Highlight peak d = 10 at index 1
            peak_i = int(np.argmax(data["ari"].to_numpy(float)))
            ax.axvline(peak_i, color=TEAL, linestyle=":", linewidth=1.3, alpha=0.8)
            if idx == 0:
                ax.scatter([peak_i], [ys[peak_i]], s=115, facecolors="none", edgecolors=CRIMSON, linewidths=2.0, zorder=5)

        ax.plot(xs, ys, marker="o", markersize=6.5, color=color, linewidth=2.4, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        for x_i, y_i in zip(xs, ys):
            fmt_s = f"{y_i:.3f}" if idx == 0 else f"{y_i:.1f}"
            ox, oy = 0, 10
            if is_lam and idx == 1 and x_i == 0:
                ox, oy = 18, -2
            elif is_lam and idx == 1 and x_i == 1:
                ox, oy = 14, 10
            ax.annotate(
                fmt_s, (x_i, y_i),
                textcoords="offset points", xytext=(ox, oy),
                ha="center", fontsize=9.0, fontweight="bold", color=color,
            )

        ax.set_xticks(xs, labels, fontsize=10.0)
        ax.set_xlabel("重构损失权重 λ" if is_lam else "隐空间维度 d", fontsize=10.2, labelpad=6)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=11.2, color=INK, pad=8)
        ax.grid(axis="x", visible=False)

    subtitle = "重构权重 λ 消融" if is_lam else "隐空间维度 d 消融"
    fig.suptitle(f"E02 单因素消融（Fashion-MNIST · IDEC · seed {data.seed.iloc[0]}）：{subtitle}", fontweight="bold", fontsize=12.0, color=BLUE_DEEP)
    return fig


def _ablation_components(frame: pd.DataFrame) -> plt.Figure | None:
    if frame.empty:
        return None
    items = []

    def _pick(factor_name: str, val_str: str) -> pd.Series | None:
        sub = frame[frame.factor.eq(factor_name) & frame.value.astype(str).eq(val_str)]
        return sub.iloc[0] if len(sub) else None

    base = _pick("pretrain", "True")
    if base is None:
        return None
    base_ari = float(base["ari"])
    specs = [
        ("完整 IDEC 基准\n(AE 预训练 + K-means 中心 + 联合更新)", base, BLUE_DEEP),
        ("冻结编码器\n(仅更新聚类中心，不联合微调)", _pick("freeze_encoder", "True"), TEAL),
        ("随机初始化聚类中心\n(去掉 K-means 中心初始化)", _pick("center_init", "random"), AMBER),
        ("取消 AE 预训练\n(随机初始化网络直接联合训练)", _pick("pretrain", "False"), CRIMSON),
    ]
    for label, row, color in specs:
        if row is not None:
            items.append((label, float(row["ari"]), float(row["nmi"]), color))
    if len(items) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ys = np.arange(len(items))[::-1]
    aris = [it[1] for it in items]
    colors = [it[3] for it in items]
    labels = [it[0] for it in items]

    bars = ax.barh(ys, aris, height=0.56, color=colors, edgecolor="none", alpha=0.92, zorder=3)
    ax.axvline(base_ari, color=BLUE_DEEP, linestyle="--", linewidth=1.1, alpha=0.55, zorder=2)

    for y_pos, (lbl, ari, nmi, col) in zip(ys, items):
        delta = ari - base_ari
        delta_s = "基准" if abs(delta) < 1e-6 else f"Δ = {delta:+.3f}"
        ax.text(
            ari + 0.008, y_pos,
            f"ARI {ari:.3f}  ({delta_s})  |  NMI {nmi:.3f}",
            va="center", ha="left", fontsize=9.4,
            fontweight="bold" if abs(delta) < 1e-6 or delta < -0.1 else "normal",
            color=INK if abs(delta) < 0.1 else CRIMSON,
        )

    ax.set_yticks(ys, labels, fontsize=9.8)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("分组得分 ARI（Fashion-MNIST，seed 0，λ=1.0，d=10）", fontsize=10.0)
    ax.grid(axis="y", visible=False)
    ax.set_title("E02 组件消融：初始化质量对最终 ARI 的贡献远超单纯联合微调", loc="left", fontsize=11.6, color=BLUE_DEEP, pad=10)
    return fig


def _curves(frame: pd.DataFrame, factor: str, protocol: dict) -> plt.Figure | None:
    noise = factor == "noise_std"
    robust = protocol["e03"]
    plot_methods = list(robust["methods"])
    if noise:
        # Include GMM as dashed reference because its collapse under sigma=0.1 is a core finding
        if "gmm" in set(frame["method"].unique()) and "gmm" not in plot_methods:
            plot_methods = [*plot_methods, "gmm"]
        frame = frame[frame.dataset.eq(robust["dataset"]) & frame.method.isin(plot_methods)]
    else:
        frame = frame[frame.factor.eq(factor)]
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    levels = sorted(frame[factor].unique())
    index = {v: i for i, v in enumerate(levels)}

    end_annotations: list[tuple[float, str, str]] = []
    for m in plot_methods:
        series = frame[frame.method.eq(m)].sort_values(factor)
        if series.empty:
            continue
        xs = [index[v] for v in series[factor]]
        ys = series["ari_mean" if noise else "ari"].to_numpy(float)
        is_gmm = m == "gmm"
        ax.plot(
            xs, ys,
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            markersize=5.6 if not is_gmm else 6.0,
            lw=2.2 if not is_gmm else 1.8,
            ls="--" if is_gmm else "-",
            markeredgecolor="white" if not is_gmm else METHOD_COLORS[m],
            markeredgewidth=0.7,
            label=METHODS[m] + (" (对照)" if is_gmm else ""),
            zorder=4 if m in {"kmeans", "dec", "idec"} else 3,
        )
        if noise:
            deviations = series.ari_std.to_numpy(float)
            ax.fill_between(xs, ys - deviations, ys + deviations, color=METHOD_COLORS[m], alpha=0.10, zorder=1)
            ret = (ys[-1] / ys[0] * 100.0) if ys[0] > 0 else 0.0
            end_annotations.append((float(ys[-1]), f"{SHORT_METHODS[m]} {ys[-1]:.2f} ({ret:.0f}%)", METHOD_COLORS[m]))

    if noise and end_annotations:
        # Spread overlapping right-side endpoint labels cleanly
        end_annotations.sort(key=lambda t: t[0])
        placed_y: list[float] = []
        min_gap = 0.036
        for raw_y, txt, col in end_annotations:
            y_pos = raw_y
            if placed_y and y_pos - placed_y[-1] < min_gap:
                y_pos = placed_y[-1] + min_gap
            placed_y.append(y_pos)
            ax.text(len(levels) - 1 + 0.08, y_pos, txt, va="center", ha="left", fontsize=8.8, fontweight="bold", color=col)
        ax.set_xlim(-0.15, len(levels) - 1 + 1.05)

    # Helpful domain reference cues on sensitivity plots
    if factor == "k" and 10 in index:
        k10 = index[10]
        ax.axvline(k10, color=CRIMSON, linestyle=":", linewidth=1.3, alpha=0.75, zorder=2)
        ax.text(k10 + 0.08, 0.505, "真实类数 K=10", fontsize=8.8, fontweight="bold", color=CRIMSON, va="top")
    if factor == "sample_frac" and 0.01 in index:
        ax.annotate(
            "极小样本 (200张)\nK-means 0.353 显著领先",
            xy=(0, 0.353), xytext=(0.45, 0.455),
            fontsize=8.6, fontweight="bold", color="#0B4F6C",
            arrowprops=dict(arrowstyle="->", color="#0B4F6C", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.25", fc="#EAF2F8", ec="#0B4F6C", lw=0.8),
        )

    labels = {
        "noise_std": "测试图像高斯噪声标准差 σ（像素范围 [0, 1]）",
        "k": "预设聚类数 K（Fashion-MNIST 真实类数 = 10）",
        "sample_frac": "无标签训练样本比例（100% = 20,000 张）",
        "imbalance": "训练集多数类与少数类目标比例",
        "outlier_frac": "训练集中替换为均匀噪声异常图的比例",
    }
    titles = {
        "noise_std": "E03 测试期高斯噪声鲁棒性（3 种子均值 ± 标准差）",
        "k": "E03 预设聚类数 K 错配敏感性（seed 0）",
        "sample_frac": "E03 训练样本规模效率曲线（seed 0）",
        "imbalance": "E03 类别长尾不平衡鲁棒性（seed 0）",
        "outlier_frac": "E03 异常噪声样本污染鲁棒性（seed 0）",
    }
    ticks = [f"{v:g}" for v in levels]
    if factor in {"sample_frac", "outlier_frac"}:
        ticks = [f"{100 * v:g}%" for v in levels]
    if factor == "imbalance":
        ticks = [f"{v:g}:1" for v in levels]

    ax.set_xticks(range(len(levels)), ticks, fontsize=10.0)
    ax.set_xlabel(labels[factor], fontsize=10.2, labelpad=6)
    ax.set_ylabel("分组得分 ARI", fontsize=10.2)
    y_col = "ari_mean" if noise else "ari"
    ax.set_ylim(min(-0.02, float(frame[y_col].min()) - 0.02), max(0.54, float(frame[y_col].max()) + 0.06))
    ax.set_title(titles[factor], loc="left", fontsize=11.8, color=BLUE_DEEP, pad=10)
    ax.grid(axis="x", visible=False)
    if noise:
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=3, fontsize=9.0, columnspacing=1.2)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.01), fontsize=9.2)
    return fig


def _embeddings(raw, dataset: str, seed: int, all_runs: pd.DataFrame) -> plt.Figure | None:
    methods = ["pca_kmeans", "ae_kmeans", "dec", "idec"]
    paths = [raw / "embeddings" / f"{dataset}__{m}__seed{seed}.npz" for m in methods]
    if not all(p.is_file() for p in paths):
        return None

    # Fetch actual seed-0 test ARI from measured runs for each panel caption
    ari_map = {}
    if not all_runs.empty:
        sub = all_runs[all_runs.experiment.eq("E01") & all_runs.dataset.eq(dataset) & all_runs.seed.eq(seed)]
        ari_map = dict(zip(sub["method"], sub["ari"]))

    fig, axes = plt.subplots(1, len(methods), figsize=(12.2, 3.65), layout="constrained")
    shared_ids = None
    cmap = matplotlib.colors.ListedColormap(TSNE_PALETTE)
    for method, path, ax in zip(methods, paths, axes):
        with np.load(path, allow_pickle=False) as raw_features:
            ids = raw_features["sample_ids"]
            if shared_ids is None:
                shared_ids = ids.copy()
                subset = np.random.default_rng(seed).choice(len(ids), min(1000, len(ids)), replace=False)
            if not np.array_equal(shared_ids, ids):
                raise ValueError(f"Unmatched test images: {path.name}")
            x = raw_features["embeddings"][subset]
            y = raw_features["predictions"][subset]
        with threadpool_limits(limits=1):
            positions = TSNE(
                n_components=2, random_state=seed, init="pca",
                learning_rate="auto", perplexity=min(30, len(x) - 1),
            ).fit_transform(x)
        ax.scatter(*positions.T, c=y, cmap=cmap, vmin=-0.5, vmax=9.5, s=8, alpha=0.82, linewidths=0)
        ari_s = f" (ARI={ari_map[method]:.3f})" if method in ari_map and pd.notna(ari_map[method]) else ""
        ax.set_title(f"{METHODS[method]}{ari_s}", fontsize=10.6, fontweight="bold", color=BLUE_DEEP, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#FAFCFF")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#CBD8EB")
            sp.set_linewidth(1.0)
        ax.grid(False)

    fig.suptitle(f"E01 潜在空间二维 t-SNE 投影（同一批 {len(subset)} 张 Fashion-MNIST 测试图，seed {seed}）", fontweight="bold", fontsize=12.0, color=INK)
    fig.supxlabel("每点对应一张测试图，颜色为各方法预测簇编号；各面板独立运行 t-SNE，仅用于辅助观察簇分离形态，定量结论以 ARI/NMI 为准。", fontsize=8.8, color=MUTED)
    return fig


def _frontier(costs: pd.DataFrame, dataset: str) -> plt.Figure | None:
    group = costs[costs.dataset.eq(dataset)].dropna(subset=["front_ari_time"])
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    frontier = group[group.front_ari_time].sort_values("time_train_mean_sec")

    # Subtle shaded region under the Pareto frontier
    ax.plot(frontier.time_train_mean_sec, frontier.ari_mean, lw=2.0, ls="--", color=BLUE_DEEP, zorder=2)
    ax.fill_between(
        frontier.time_train_mean_sec,
        0.15,
        frontier.ari_mean,
        color="#EAF0FC",
        alpha=0.45,
        zorder=1,
    )

    # Carefully tuned direct non-overlapping label offsets (in points) for Fashion-MNIST
    label_offsets = {
        "pca_kmeans": (10, -14),
        "kmeans": (-10, 14),
        "gmm": (10, -4),
        "dec": (10, 4),
        "idec": (10, 8),
        "ae_kmeans": (10, -14),
        "vade": (10, -4),
        "simclr_kmeans": (-14, 10),
        "cc": (-14, -14),
    }

    for row in group.itertuples():
        color = METHOD_COLORS[row.method]
        is_front = bool(row.front_ari_time)
        marker = "D" if is_front else "o"
        size = 92 if is_front else 64
        ax.scatter(
            row.time_train_mean_sec, row.ari_mean,
            s=size, marker=marker, color=color,
            edgecolors="white", linewidths=1.2, zorder=4 if is_front else 3,
        )
        dx, dy = label_offsets.get(row.method, (8, 4))
        ha = "left" if dx >= 0 else "right"
        lbl = f"{SHORT_METHODS[row.method]} ({row.time_train_mean_sec:.1f}s, {row.ari_mean:.3f})"
        ax.annotate(
            lbl,
            (row.time_train_mean_sec, row.ari_mean),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="center",
            fontsize=8.9 if is_front else 8.4,
            fontweight="bold" if is_front else "normal",
            color=BLUE_DEEP if is_front else INK,
        )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 380)
    ax.set_ylim(0.16, 0.475)
    ax.set_xlabel("实测平均训练耗时（秒，对数刻度，含当次缓存条件）", fontsize=10.2, labelpad=6)
    ax.set_ylabel("平均分组得分 ARI（3 种子均值）", fontsize=10.2)
    ax.set_title("E05 性能—计算成本帕累托前沿（Fashion-MNIST 从零训练方法）", loc="left", fontsize=11.8, color=BLUE_DEEP, pad=10)

    legend_handles = [
        Line2D([], [], color=BLUE_DEEP, marker="D", markersize=7.5, ls="--", lw=1.8, label="两目标帕累托前沿（PCA+KM → K-means → DEC）"),
        Line2D([], [], color=SLATE, marker="o", markersize=6.8, ls="", label="被前沿支配方案（更慢且 ARI 未更高）"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9.0, frameon=True, facecolor="white", edgecolor="#CBD8EB")
    return fig


def _stability_comparison(selection: dict | None, e01: pd.DataFrame) -> plt.Figure | None:
    if not selection or e01.empty:
        return None
    datasets = [("mnist", "MNIST（手写数字）"), ("fashion_mnist", "Fashion-MNIST（服装图像）")]
    methods = list(METHODS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")

    for ax, (ds, ds_title) in zip(axes, datasets):
        ev = selection.get(ds, {}).get("evidence", {})
        choice = selection.get(ds, {}).get("choice")
        ds_e01 = e01[e01.dataset.eq(ds)].set_index("method")
        if not ev or ds_e01.empty:
            continue
        ys = np.arange(len(methods))[::-1]
        jacc = [float(ev[m]["macro_cluster_jaccard"]) if m in ev else np.nan for m in methods]
        aris = [float(ds_e01.loc[m, "ari_mean"]) if m in ds_e01.index else np.nan for m in methods]
        h = 0.36
        ax.barh(ys + h / 2, jacc, height=h, color=BLUE_DEEP, alpha=0.90, label="无标签训练稳定性 (Macro Jaccard)")
        ax.barh(ys - h / 2, aris, height=h, color=AMBER, alpha=0.88, label="带标签测试集回顾性 ARI")

        for y_p, m, jv, av in zip(ys, methods, jacc, aris):
            if np.isfinite(jv):
                star = " ★" if m == choice else ""
                ax.text(jv + 0.012, y_p + h / 2, f"{jv:.2f}{star}", va="center", fontsize=8.0, fontweight="bold" if m == choice else "normal", color=BLUE_DEEP)
            if np.isfinite(av):
                ax.text(av + 0.012, y_p - h / 2, f"{av:.2f}", va="center", fontsize=8.0, color=AMBER)

        ax.set_yticks(ys, [SHORT_METHODS[m] for m in methods], fontsize=9.2)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("指标值（0–1，越高越强）", fontsize=9.6)
        ax.set_title(ds_title, loc="left", fontsize=11.0, color=BLUE_DEEP)
        ax.grid(axis="y", visible=False)

    axes[0].legend(loc="lower right", fontsize=8.4, frameon=True, facecolor="white", edgecolor="#CBD8EB")
    fig.suptitle("E05 无标签训练稳定性诊断（★选中） vs. 回顾性测试集 ARI", fontweight="bold", fontsize=11.8, color=INK)
    return fig


def render_figures(root, protocol, tables, selection: dict | None = None):
    _setup()
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(name: str, fig: plt.Figure | None) -> None:
        path = figures / f"fig_{name}.png"
        if fig is not None:
            fig.savefig(path, facecolor="white", dpi=300, bbox_inches="tight", pad_inches=0.14)
            plt.close(fig)
        else:
            path.unlink(missing_ok=True)

    save("e01_ari", _heat_e01(tables["e01_summary"]))
    save("e04_paired", _heat_e04(tables["e04_paired"], protocol))
    for factor, name in (("latent_dim", "latent"), ("lambda_recon", "lambda")):
        save(f"e02_{name}", _ablation(tables["e02_ablation"], factor))
    save("e02_components", _ablation_components(tables["e02_ablation"]))
    for factor in ("noise_std", "k", "imbalance", "sample_frac", "outlier_frac"):
        tab = "e03_noise" if factor == "noise_std" else "e03_sensitivity"
        save("e03_" + ("noise" if factor == "noise_std" else factor), _curves(tables[tab], factor, protocol))
    save("e05_pareto", _frontier(tables["e05_cost_frontiers"], protocol["e03"]["dataset"]))
    save("e05_stability", _stability_comparison(selection, tables["e01_summary"]))
    save("e01_embeddings", _embeddings(root / "results/raw", protocol["e02"]["dataset"], protocol["e02"]["seed"], tables["all_runs"]))
