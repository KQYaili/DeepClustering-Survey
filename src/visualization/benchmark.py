"""Publication-grade figures strictly derived from measured run tables and aligned embeddings."""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from visualization.style import (
    AMBER,
    BLUE_DEEP,
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
    "idec": "IDEC (λ=1.0)",
    "vade": "VaDE",
    "simclr_kmeans": "SimCLR + K-means (小 CNN)",
    "cc": "CC (小 CNN, 省熵项)",
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
DATASETS = {
    "mnist": "MNIST\n(手写数字)",
    "fashion_mnist": "Fashion-MNIST\n(服装图像)",
    "cifar10": "CIFAR-10\n(自然图像)",
}

# Distinct colorblind-friendly academic palette + unique markers per method
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
    "vade": "p",
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

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    scale = Normalize(0.0, 0.72)
    im = ax.imshow(vals, cmap=HEAT_CMAP, norm=scale, aspect="auto")

    ax.set_xticks(range(len(DATASETS)), list(DATASETS.values()), fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(METHODS)), list(METHODS.values()), fontsize=10.0)
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

    col_best = np.nanargmax(vals, axis=0)
    for i, j in np.ndindex(vals.shape):
        val = vals[i, j]
        sd = stds[i, j]
        if np.isfinite(val):
            r, g, b, _ = HEAT_CMAP(scale(val))
            txt_color = "#FFFFFF" if 0.21 * r + 0.72 * g + 0.07 * b < 0.54 else INK
            best_i = int(col_best[j])
            is_best = i == best_i
            best_sd = stds[best_i, j]
            near_tie = (not is_best) and np.isfinite(best_sd) and (vals[best_i, j] - val <= best_sd)
            suffix = " ★" if is_best else (" ≈" if near_tie else "")
            main_txt = f"{val:.3f}{suffix}"
            ax.text(
                j, i - 0.11, main_txt,
                ha="center", va="center",
                fontsize=10.8 if is_best else 10.2,
                fontweight="bold" if is_best else "semibold",
                color=txt_color,
            )
            if np.isfinite(sd):
                ax.text(
                    j, i + 0.22, f"± {sd:.3f}",
                    ha="center", va="center",
                    fontsize=8.2,
                    color=txt_color,
                    alpha=0.90,
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

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, aspect=22)
    cbar.set_label("平均 ARI（3 种子均值）", fontsize=9.8, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.0, length=2)

    ax.set_title("E01 从零训练基准：9 种方法在 3 个数据集上的 ARI 对比（n=3 种子）", loc="left", pad=36, fontsize=12.2, color=BLUE_DEEP)
    fig.text(
        0.02, -0.02,
        "注：各格为 3 种子均值 ± 样本标准差；★ 为列均值最高者，≈ 表示与最高者差值 ≤ 1 SD（近似并列）；IDEC λ=1.0，SimCLR/CC 为小型 CNN（CC 省略熵项）。",
        fontsize=8.3, color=MUTED,
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
    src_grid = pairs.pivot(index="encoder", columns="clusterer", values="source_experiment").reindex(
        index=cfg["encoders"], columns=cfg["clusterers"]
    )
    vals = grid.to_numpy(float)

    scratch_vals = vals[:3, :]
    ext_vals = vals[3:, :]
    scratch_range = f"{np.nanmin(scratch_vals):.2f}–{np.nanmax(scratch_vals):.2f}"
    ext_range = f"{np.nanmin(ext_vals):.2f}–{np.nanmax(ext_vals):.2f}"

    fig, ax = plt.subplots(figsize=(9.0, 4.55))
    scale = Normalize(0.0, 0.68)
    im = ax.imshow(vals, cmap=HEAT_CMAP, norm=scale, aspect="auto")

    ax.set_xticks(range(len(clusterers)), [clusterers[c] for c in grid.columns], fontsize=10.6, fontweight="bold")
    ax.set_yticks(range(len(encoders)), [encoders[e] for e in grid.index], fontsize=10.3)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=7)
    ax.grid(False)

    ax.set_xticks(np.arange(-0.5, len(clusterers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(encoders), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    ax.axhline(2.5, color=CRIMSON, linewidth=2.2, linestyle="--", zorder=5)
    ax.text(
        2.54, 1.0, f"从零训练\n(课程数据)\n{scratch_range}",
        va="center", ha="left", fontsize=8.9, fontweight="bold", color=MUTED,
        transform=ax.transData,
    )
    ax.text(
        2.54, 3.5, f"外部预训练\n(冻结主干)\n{ext_range}",
        va="center", ha="left", fontsize=8.9, fontweight="bold", color=BLUE_DEEP,
        transform=ax.transData,
    )

    peak_val = float(np.nanmax(vals))
    for i, j in np.ndindex(vals.shape):
        val = vals[i, j]
        if np.isfinite(val):
            r, g, b, _ = HEAT_CMAP(scale(val))
            txt_color = "#FFFFFF" if 0.21 * r + 0.72 * g + 0.07 * b < 0.54 else INK
            src_exp = str(src_grid.iloc[i, j])
            is_e01 = src_exp == "E01"
            is_peak = val >= peak_val - 0.015
            ax.text(
                j, i - (0.08 if is_e01 else 0.0),
                f"{val:.3f}" + (" †" if is_e01 else ""),
                ha="center", va="center",
                fontsize=11.8 if is_peak else 11.0,
                fontweight="bold" if is_peak else "semibold",
                color=txt_color,
            )
            if is_e01:
                ax.text(
                    j, i + 0.24, "同切分 E01",
                    ha="center", va="center",
                    fontsize=7.6, color=txt_color, alpha=0.88,
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

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.20, aspect=18)
    cbar.set_label(f"ARI（CIFAR-10，n=1，seed {cfg['seed']}）", fontsize=9.4, color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.0, length=2)

    ax.set_title(
        f"E04 表征与聚类器解耦：CIFAR-10 5×3 交叉矩阵（n=1 配对，seed {cfg['seed']}）",
        loc="left", pad=34, fontsize=12.0, color=BLUE_DEEP,
    )
    fig.text(
        0.02, -0.03,
        "注：全表为 seed 0 单次配对（n=1）；标 † 两格（AE/SimCLR+K-means）取自同切分 E01，其余 13 格来自 E04；红虚线下 ★ 为冻结外部主干（DEC 仅微调 10 维头）。",
        fontsize=8.2, color=MUTED,
    )
    return fig


def _ablation(frame: pd.DataFrame, factor: str) -> plt.Figure | None:
    data = frame[frame.factor.eq(factor)].sort_values("value")
    if data.empty:
        return None
    xs = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.15), layout="constrained")
    is_lam = factor == "lambda_recon"

    if is_lam:
        labels = [
            f"{float(r.value):g}\n({int(r.epochs_ran)}ep{'早停' if bool(r.stopped_early) else ''})"
            for r in data.itertuples()
        ]
    else:
        labels = [f"{float(r.value):g}" for r in data.itertuples()]

    configs = [
        ("ari", BLUE_DEEP, "分组得分 ARI  ↑", (0.0, 0.57)),
        ("recon_sse", AMBER, "每图重构平方误差 SSE  ↓", (0.0, float(data["recon_sse"].max()) * 1.18)),
    ]
    for idx, (ax, (metric, color, title, ylim)) in enumerate(zip(axes, configs)):
        ys = data[metric].to_numpy(float)
        if is_lam:
            early_mask = data["stopped_early"].astype(bool).to_numpy()
            if early_mask.any() and (~early_mask).any():
                split_x = float(np.where(early_mask)[0][-1]) + 0.5
                ax.axvspan(-0.4, split_x, color="#EBF3FF", alpha=0.65, zorder=0)
                ax.axvspan(split_x, len(xs) - 0.6, color="#FDF3E7", alpha=0.65, zorder=0)
                ax.axvline(split_x, color=SLATE, linestyle="--", linewidth=1.1, alpha=0.7, zorder=2)
                if idx == 0:
                    ax.text(split_x / 2 - 0.15, 0.06, "早停区\n(20→35 ep)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=BLUE_DEEP)
                    ax.text((split_x + len(xs) - 0.6) / 2, 0.06, "跑满区\n(50 ep)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=AMBER)
                else:
                    ax.text(2.5, ylim[1] * 0.62, "仅 λ>0 可比：\n重构项纳入总损失", ha="center", va="center", fontsize=8.3, fontweight="bold", color=AMBER,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=AMBER, lw=0.8, alpha=0.92), zorder=5)

            if idx == 1 and float(data.value.iloc[0]) == 0.0:
                # λ=0 excludes reconstruction from the loss: plot as non-comparable hollow point
                ax.plot(xs[1:], ys[1:], marker="o", markersize=6.5, color=color, linewidth=2.4,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=4)
                ax.plot(xs[:2], ys[:2], linestyle=":", color=MUTED, linewidth=1.4, zorder=3)
                ax.scatter([xs[0]], [ys[0]], s=68, facecolors="white", edgecolors=MUTED, linewidths=1.9, zorder=5)
            else:
                ax.plot(xs, ys, marker="o", markersize=6.5, color=color, linewidth=2.4,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        else:
            peak_i = int(np.argmax(data["ari"].to_numpy(float)))
            ax.axvline(peak_i, color=TEAL, linestyle=":", linewidth=1.3, alpha=0.8)
            if idx == 0:
                ax.scatter([peak_i], [ys[peak_i]], s=115, facecolors="none", edgecolors=CRIMSON, linewidths=2.0, zorder=5)
            ax.plot(xs, ys, marker="o", markersize=6.5, color=color, linewidth=2.4,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=4)

        for x_i, y_i in zip(xs, ys):
            if is_lam and idx == 1 and x_i == 0 and float(data.value.iloc[0]) == 0.0:
                ax.annotate(
                    f"{y_i:.1f}*\n(重构未入损失\n不可比)", (x_i, y_i),
                    textcoords="offset points", xytext=(16, -8),
                    ha="left", va="center", fontsize=8.1, fontweight="bold", color=MUTED,
                )
                continue
            fmt_s = f"{y_i:.3f}" if idx == 0 else f"{y_i:.1f}"
            ox, oy = 0, 10
            if is_lam and idx == 1 and x_i == 1:
                ox, oy = 8, 12
            ax.annotate(
                fmt_s, (x_i, y_i),
                textcoords="offset points", xytext=(ox, oy),
                ha="center", fontsize=8.8, fontweight="bold", color=color,
            )

        ax.set_xticks(xs, labels, fontsize=8.8 if is_lam else 9.8)
        ax.set_xlabel("重构损失权重 λ（括号为实际训练轮数）" if is_lam else "隐空间维度 d（各档均跑满 50 ep）", fontsize=9.8, labelpad=6)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=11.0, color=INK, pad=8)
        ax.grid(axis="x", visible=False)

    seed_val = int(data.seed.iloc[0])
    if is_lam:
        fig.suptitle(
            f"E02 重构权重 λ 消融（Fashion-MNIST · IDEC · n=1, seed {seed_val}）：当前早停协议下 λ 与轮数共变",
            fontweight="bold", fontsize=11.4, color=BLUE_DEEP,
        )
        fig.supxlabel(
            "注：早停按训练集簇分配变化 < 0.001 触发，λ 越大越晚早停；ARI 变化由 λ 与训练时长共同作用，不能单独归因于权重/量纲；*λ=0 重构项不入损失。",
            fontsize=8.2, color=MUTED,
        )
    else:
        fig.suptitle(
            f"E02 隐空间维度 d 消融（Fashion-MNIST · IDEC · n=1, seed {seed_val}，λ=1.0）",
            fontweight="bold", fontsize=11.6, color=BLUE_DEEP,
        )
        fig.supxlabel(
            "注：单种子受控消融（seed 0），全部 5 档维度均完整运行 50 epoch。",
            fontsize=8.3, color=MUTED,
        )
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
        ("冻结编码器\n(仅更新簇中心；10ep 结构性早停)", _pick("freeze_encoder", "True"), TEAL),
        ("随机初始化聚类中心\n(去掉 K-means 中心初始化)", _pick("center_init", "random"), AMBER),
        ("取消 AE 预训练\n(随机网络直接联合训练)", _pick("pretrain", "False"), CRIMSON),
    ]
    for label, row, color in specs:
        if row is not None:
            items.append((label, float(row["ari"]), float(row["nmi"]), int(row["epochs_ran"]), color))
    if len(items) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8.6, 3.85))
    ys = np.arange(len(items))[::-1]
    aris = [it[1] for it in items]
    colors = [it[4] for it in items]
    labels = [it[0] for it in items]

    ax.barh(ys, aris, height=0.56, color=colors, edgecolor="none", alpha=0.92, zorder=3)
    ax.axvline(base_ari, color=BLUE_DEEP, linestyle="--", linewidth=1.1, alpha=0.55, zorder=2)

    for y_pos, (_, ari, nmi, ep, _) in zip(ys, items):
        delta = ari - base_ari
        delta_s = "基准" if abs(delta) < 1e-6 else f"Δ = {delta:+.3f}"
        ax.text(
            ari + 0.008, y_pos,
            f"ARI {ari:.3f} ({delta_s}) | NMI {nmi:.3f} | {ep}ep",
            va="center", ha="left", fontsize=9.1,
            fontweight="bold" if abs(delta) < 1e-6 or delta < -0.1 else "normal",
            color=INK if abs(delta) < 0.1 else CRIMSON,
        )

    ax.set_yticks(ys, labels, fontsize=9.5)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("分组得分 ARI（Fashion-MNIST，n=1, seed 0，λ=1.0，d=10）", fontsize=9.8)
    ax.grid(axis="y", visible=False)
    ax.set_title("E02 组件消融：初始化质量对最终 ARI 的贡献远超单纯联合微调", loc="left", fontsize=11.4, color=BLUE_DEEP, pad=10)
    return fig


def _curves(frame: pd.DataFrame, factor: str, protocol: dict) -> plt.Figure | None:
    noise = factor == "noise_std"
    robust = protocol["e03"]
    plot_methods = list(robust["methods"])
    if noise:
        # Plot VaDE alongside GMM so probabilistic-family behavior is not overgeneralized from GMM alone.
        for extra in ("vade", "gmm"):
            if extra in set(frame["method"].unique()) and extra not in plot_methods:
                plot_methods = [*plot_methods, extra]
        frame = frame[frame.dataset.eq(robust["dataset"]) & frame.method.isin(plot_methods)]
    else:
        frame = frame[frame.factor.eq(factor)]
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.8 if noise else 8.4, 4.6 if noise else 4.35))
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
        legend_lbl = METHODS[m]
        if is_gmm:
            legend_lbl = "GMM (标准化+对角协方差)"
        elif m == "vade":
            legend_lbl = "VaDE (隐空间生成对照)"
        ax.plot(
            xs, ys,
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            markersize=5.8 if m != "vade" else 6.6,
            lw=2.2 if not is_gmm else 1.9,
            ls="--" if is_gmm else "-",
            markeredgecolor="white" if not is_gmm else METHOD_COLORS[m],
            markeredgewidth=0.75,
            label=legend_lbl,
            zorder=5 if m in {"kmeans", "vade", "dec"} else 3,
        )
        if noise:
            deviations = series.ari_std.to_numpy(float)
            ax.fill_between(xs, ys - deviations, ys + deviations, color=METHOD_COLORS[m], alpha=0.09, zorder=1)
            ret = (ys[-1] / ys[0] * 100.0) if ys[0] > 0 else 0.0
            end_annotations.append((float(ys[-1]), f"{SHORT_METHODS[m]} {ys[-1]:.3f} ({ret:.0f}%)", METHOD_COLORS[m]))

    if noise and end_annotations:
        end_annotations.sort(key=lambda t: t[0])
        placed_y: list[float] = []
        min_gap = 0.044
        for raw_y, txt, col in end_annotations:
            y_pos = raw_y
            if placed_y and y_pos - placed_y[-1] < min_gap:
                y_pos = placed_y[-1] + min_gap
            placed_y.append(y_pos)
            x_end = len(levels) - 1
            ax.plot([x_end + 0.03, x_end + 0.14], [raw_y, y_pos], color=col, lw=0.85, ls=":", alpha=0.75)
            ax.text(x_end + 0.16, y_pos, txt, va="center", ha="left", fontsize=8.4, fontweight="bold", color=col)
        ax.set_xlim(-0.15, len(levels) - 1 + 1.28)

    if factor == "k" and 10 in index:
        k10 = index[10]
        ax.axvline(k10, color=CRIMSON, linestyle=":", linewidth=1.3, alpha=0.75, zorder=2)
        ax.text(k10 + 0.08, 0.505, "真实类数 K=10", fontsize=8.8, fontweight="bold", color=CRIMSON, va="top")
    if factor == "sample_frac" and 0.01 in index:
        km_small = frame[frame.method.eq("kmeans") & frame[factor].eq(0.01)]
        if len(km_small):
            km_v = float(km_small["ari"].iloc[0])
            n_small = int(km_small["n_train"].iloc[0])
            ax.annotate(
                f"极小样本 ({n_small}张)\nK-means {km_v:.3f} 显著领先",
                xy=(0, km_v), xytext=(0.45, 0.455),
                fontsize=8.6, fontweight="bold", color="#0B4F6C",
                arrowprops=dict(arrowstyle="->", color="#0B4F6C", lw=1.1),
                bbox=dict(boxstyle="round,pad=0.25", fc="#EAF2F8", ec="#0B4F6C", lw=0.8),
            )

    labels = {
        "noise_std": "测试图像高斯噪声标准差 σ（像素范围 [0, 1]）",
        "k": "预设聚类数 K（Fashion-MNIST 真实类数 = 10）",
        "sample_frac": "无标签训练样本比例（100% = 20,000 张）",
        "imbalance": "训练集多数类与少数类比例（括号为实际训练样本总量 n_train）",
        "outlier_frac": "训练集中替换为均匀噪声异常图的比例",
    }
    titles = {
        "noise_std": "E03 测试期高斯噪声鲁棒性（含 VaDE 与当前 GMM 对照，n=3 种子）",
        "k": "E03 预设聚类数 K 错配敏感性（n=1, seed 0）",
        "sample_frac": "E03 训练样本规模效率曲线（n=1, seed 0）",
        "imbalance": "E03 类别长尾不平衡鲁棒性（n=1, seed 0）",
        "outlier_frac": "E03 异常噪声样本污染鲁棒性（n=1, seed 0）",
    }
    ticks = [f"{v:g}" for v in levels]
    if factor in {"sample_frac", "outlier_frac"}:
        ticks = [f"{100 * v:g}%" for v in levels]
    if factor == "imbalance":
        tick_list = []
        for v in levels:
            n_sub = frame[frame[factor].eq(v)]["n_train"].dropna()
            n_str = f"\n(n={int(n_sub.iloc[0]):,})" if len(n_sub) else ""
            tick_list.append(f"{v:g}:1{n_str}")
        ticks = tick_list

    ax.set_xticks(range(len(levels)), ticks, fontsize=9.6 if factor == "imbalance" else 10.0)
    ax.set_xlabel(labels[factor], fontsize=9.9, labelpad=6)
    ax.set_ylabel("分组得分 ARI", fontsize=10.2)
    y_col = "ari_mean" if noise else "ari"
    y_top = max(0.54, float(frame[y_col].max()) + 0.06)
    if noise and end_annotations:
        y_top = max(y_top, max(placed_y) + 0.04)
    ax.set_ylim(min(-0.02, float(frame[y_col].min()) - 0.02), y_top)
    ax.set_title(titles[factor], loc="left", fontsize=11.4, color=BLUE_DEEP, pad=10)
    ax.grid(axis="x", visible=False)
    if noise:
        ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=4, fontsize=8.1, columnspacing=0.9, handlelength=1.8)
        fig.text(
            0.02, -0.18,
            "注：GMM 坍缩仅限于当前 StandardScaler+对角协方差像素实现（σ=0.1 退化为 1 簇）；隐空间生成模型 VaDE 保持率与 DEC/IDEC 接近。",
            fontsize=8.0, color=MUTED,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.01), fontsize=8.8)
        if factor == "imbalance":
            fig.text(
                0.02, -0.11,
                "注：按指数衰减比例从训练集抽样构造长尾分布，不平衡加剧的同时总训练样本量也由 20,000 降至 2,412（单种子 n=1）。",
                fontsize=8.1, color=MUTED,
            )
    return fig


def _embeddings(raw, dataset: str, seed: int, all_runs: pd.DataFrame) -> plt.Figure | None:
    methods = ["pca_kmeans", "ae_kmeans", "dec", "idec"]
    paths = [raw / "embeddings" / f"{dataset}__{m}__seed{seed}.npz" for m in methods]
    if not all(p.is_file() for p in paths):
        return None

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
        ax.set_title(f"{SHORT_METHODS[method]}{ari_s}", fontsize=10.8, fontweight="bold", color=BLUE_DEEP, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#FAFCFF")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#CBD8EB")
            sp.set_linewidth(1.0)
        ax.grid(False)

    fig.suptitle(f"E01 潜在空间二维 t-SNE 投影（同一批 {len(subset)} 张 Fashion-MNIST 测试图，seed {seed}）", fontweight="bold", fontsize=12.0, color=INK)
    fig.supxlabel(
        "注：颜色为各方法预测簇编号（依实验隔离约束，可视化脚本不读取 y_test）；各面板独立 t-SNE，仅作形态辅助，定量结论以 ARI/NMI 为准。",
        fontsize=8.6, color=MUTED,
    )
    return fig


def _frontier(costs: pd.DataFrame, dataset: str) -> plt.Figure | None:
    """Retrospective labeled ARI vs. end-to-end training cost Pareto frontier (not label-free E05)."""
    group = costs[costs.dataset.eq(dataset)].dropna(subset=["front_ari_time_e2e", "time_train_e2e_mean_sec"])
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    frontier = group[group.front_ari_time_e2e].sort_values("time_train_e2e_mean_sec")

    ax.plot(frontier.time_train_e2e_mean_sec, frontier.ari_mean, lw=2.1, ls="--", color=BLUE_DEEP, zorder=2)
    ax.fill_between(
        frontier.time_train_e2e_mean_sec,
        0.15,
        frontier.ari_mean,
        color="#EAF0FC",
        alpha=0.45,
        zorder=1,
    )

    label_offsets = {
        "pca_kmeans": (10, -14),
        "kmeans": (-10, 14),
        "gmm": (10, -4),
        "ae_kmeans": (0, -18),
        "dec": (10, 6),
        "idec": (10, 5),
        "vade": (10, -5),
        "simclr_kmeans": (-12, 12),
        "cc": (-12, -14),
    }

    ae_row = group[group.method.eq("ae_kmeans")]
    ae_t = float(ae_row["time_train_e2e_mean_sec"].iloc[0]) if len(ae_row) else np.nan

    for row in group.itertuples():
        color = METHOD_COLORS[row.method]
        is_front = bool(row.front_ari_time_e2e)
        marker = "D" if is_front else "o"
        size = 95 if is_front else 64
        t_e2e = float(row.time_train_e2e_mean_sec)
        t_cached = float(row.time_train_mean_sec)

        # Show cached incremental fine-tuning ghost marker when E2E differs from cached time
        if np.isfinite(t_cached) and abs(t_e2e - t_cached) > 1.0:
            ax.scatter(
                t_cached, row.ari_mean,
                s=58, marker=marker, facecolors="white", edgecolors=color,
                linewidths=1.5, alpha=0.82, zorder=3,
            )
            ax.annotate(
                "", xy=(t_e2e * 0.94, row.ari_mean), xytext=(t_cached * 1.06, row.ari_mean),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, ls=":", alpha=0.85),
                zorder=2,
            )
            if row.method == "dec":
                g_off, g_ha = (0, 11), "center"
            else:
                g_off, g_ha = (-10, 0), "right"
            ax.annotate(
                f"仅微调 {t_cached:.1f}s",
                (t_cached, row.ari_mean),
                textcoords="offset points",
                xytext=g_off,
                ha=g_ha, va="center", fontsize=7.6, color=SLATE,
            )

        ax.scatter(
            t_e2e, row.ari_mean,
            s=size, marker=marker, color=color,
            edgecolors="white", linewidths=1.2, zorder=4 if is_front else 3,
        )
        dx, dy = label_offsets.get(row.method, (8, 4))
        ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
        lbl = f"{SHORT_METHODS[row.method]} ({t_e2e:.1f}s, {row.ari_mean:.3f})"
        ax.annotate(
            lbl,
            (t_e2e, row.ari_mean),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="center",
            fontsize=8.7 if is_front else 8.2,
            fontweight="bold" if is_front else "normal",
            color=BLUE_DEEP if is_front else INK,
        )

    front_names = " → ".join(SHORT_METHODS[m] for m in frontier.method)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 420)
    ax.set_ylim(0.16, 0.48)
    ax.set_xlabel("端到端平均训练耗时（秒，对数刻度；DEC/IDEC 已补回前置 AE 预训练）", fontsize=9.8, labelpad=6)
    ax.set_ylabel("回顾性测试集平均 ARI（E01，3 种子均值）", fontsize=9.8)
    ax.set_title(
        "回顾性历史基准：性能—端到端训练成本帕累托前沿（Fashion-MNIST · 非无标签 E05）",
        loc="left", fontsize=11.4, color=BLUE_DEEP, pad=10,
    )

    legend_handles = [
        Line2D([], [], color=BLUE_DEEP, marker="D", markersize=7.2, ls="--", lw=1.8, label=f"端到端两目标帕累托前沿（{front_names}）"),
        Line2D([], [], color=SLATE, marker="o", markersize=6.6, ls="", label="被前沿支配方案（更慢且 ARI 未更高）"),
        Line2D([], [], color=BLUE_DEEP, marker="D", mfc="white", markersize=6.6, ls=":", lw=1.2, label="缓存命中增量微调耗时（未含前置 AE，仅作参考）"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.3, frameon=True, facecolor="white", edgecolor="#CBD8EB")
    ae_note = f"（{ae_t:.1f}s）" if np.isfinite(ae_t) else ""
    fig.text(
        0.02, -0.02,
        f"注：纵轴 ARI 使用测试集标签，仅作历史基准回顾，不参与无标签 E05；DEC/IDEC 独立端到端耗时 = 同配置 AE 预训练{ae_note} + 微调耗时。",
        fontsize=8.1, color=MUTED,
    )
    return fig


def _stability_comparison(selection: dict | None, e01: pd.DataFrame, e04_pre: pd.DataFrame | None = None) -> plt.Figure | None:
    """Three-dataset negative diagnostic: E05 train stability (label-free) vs. retrospective test ARI."""
    if not selection or e01.empty:
        return None
    datasets = [
        ("mnist", "MNIST（手写数字）"),
        ("fashion_mnist", "Fashion-MNIST（服装图像）"),
        ("cifar10", "CIFAR-10（自然图像）"),
    ]
    methods = list(METHODS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.15), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.77, bottom=0.17, wspace=0.08)

    dino_ari = None
    if e04_pre is not None and not e04_pre.empty:
        d_sub = e04_pre[e04_pre.encoder.eq("dinov2")]
        if len(d_sub):
            dino_ari = float(d_sub["ari_mean"].iloc[0])

    for col_idx, (ax, (ds, ds_title)) in enumerate(zip(axes, datasets)):
        ev = selection.get(ds, {}).get("evidence", {})
        choice = selection.get(ds, {}).get("choice")
        ds_e01 = e01[e01.dataset.eq(ds)].set_index("method")
        if not ev or ds_e01.empty:
            continue
        ys = np.arange(len(methods))[::-1]
        jacc = [float(ev[m]["macro_cluster_jaccard"]) if m in ev else np.nan for m in methods]
        aris = [float(ds_e01.loc[m, "ari_mean"]) if m in ds_e01.index else np.nan for m in methods]
        h = 0.37
        ax.barh(ys + h / 2, jacc, height=h, color=BLUE_DEEP, alpha=0.90, label="E05 训练侧稳定性 (Macro Jaccard，无标签)")
        ax.barh(ys - h / 2, aris, height=h, color=AMBER, alpha=0.88, label="历史 E01 测试集 ARI（事后对照，不入 E05）")

        best_ari_m = ds_e01["ari_mean"].idxmax()
        for y_p, m, jv, av in zip(ys, methods, jacc, aris):
            if np.isfinite(jv):
                star = " ★" if m == choice else ""
                ax.text(jv + 0.015, y_p + h / 2, f"{jv:.2f}{star}", va="center", fontsize=7.8,
                        fontweight="bold" if m == choice else "normal", color=BLUE_DEEP)
            if np.isfinite(av):
                mark = " ▲" if m == best_ari_m else ""
                ax.text(av + 0.015, y_p - h / 2, f"{av:.2f}{mark}", va="center", fontsize=7.8,
                        fontweight="bold" if m == best_ari_m else "normal", color=AMBER)

        if ds == "cifar10" and dino_ari is not None:
            ax.axvline(dino_ari, color=TEAL, linestyle="--", linewidth=1.35, zorder=4)
            ax.text(
                dino_ari + 0.03, 1.05,
                f"DINOv2+KM\nARI={dino_ari:.3f}\n(不在从零池)",
                ha="left", va="center", fontsize=7.6, fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=0.22", fc="#E6F4F1", ec=TEAL, lw=0.85, alpha=0.96),
                zorder=5,
            )

        ax.set_yticks(ys, [SHORT_METHODS[m] for m in methods], fontsize=9.2)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("指标值（0–1）", fontsize=9.2)
        ax.set_title(ds_title, loc="left", fontsize=10.8, fontweight="bold", color=BLUE_DEEP, pad=8)
        ax.grid(axis="y", visible=False)
        if col_idx > 0:
            ax.tick_params(axis="y", length=0)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.52, 0.905), ncol=2, fontsize=9.0, frameon=True, facecolor="white", edgecolor="#CBD8EB")
    fig.suptitle("E05 负面诊断：训练扰动稳定性（★最高）≠ 语义正确（▲从零最高 ARI）", fontweight="bold", fontsize=12.2, color=CRIMSON, y=0.975)
    fig.text(
        0.52, 0.03,
        "注：E05 仅用训练集干净/加噪（σ=0.1）预测且未纳入预测耗时；像素 K-means/PCA+KM 稳定性均 ≈0.95，但在三数据集上 ARI 均落后于最优路线（稳定地分错）。",
        ha="center", va="center", fontsize=8.3, color=MUTED,
    )
    return fig


def render_figures(root, protocol, tables, selection: dict | None = None):
    _setup()
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(name: str, fig: plt.Figure | None) -> None:
        path = figures / f"fig_{name}.png"
        if fig is not None:
            tmp = figures / f".tmp_fig_{name}.png"
            fig.savefig(tmp, facecolor="white", dpi=300, bbox_inches="tight", pad_inches=0.14)
            plt.close(fig)
            tmp.replace(path)
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
    save("benchmark_pareto", _frontier(tables["benchmark_cost_frontiers"], protocol["e03"]["dataset"]))
    save("e05_stability", _stability_comparison(selection, tables["e01_summary"], tables.get("e04_pretrained")))
    save("e01_embeddings", _embeddings(root / "results/raw", protocol["e02"]["dataset"], protocol["e02"]["seed"], tables["all_runs"]))
