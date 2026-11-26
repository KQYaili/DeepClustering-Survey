"""Classroom figures derived from measured run tables and aligned image features."""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from visualization.style import apply_style

METHODS = {"kmeans": "像素 K 均值", "pca_kmeans": "主成分 + K 均值", "gmm": "高斯混合",
           "ae_kmeans": "自编码器 + K 均值", "dec": "DEC", "idec": "IDEC", "vade": "VaDE",
           "simclr_kmeans": "SimCLR + K 均值", "cc": "对比聚类 CC"}
DATASETS = {"mnist": "手写数字", "fashion_mnist": "服装", "cifar10": "自然小图"}
PALETTE = ["#174D60", "#54835E", "#D08A26", "#D0614C", "#436FAD", "#7A5F9F",
           "#1B8985", "#AA6947", "#576378"]
COLORS = dict(zip(METHODS, PALETTE))
INK = "#1D3840"
TEAL = "#187B80"
MAP = LinearSegmentedColormap.from_list("evidence", ["#F4F6F0", "#9BD4C4", "#357F7C", "#124956"])


def _setup():
    apply_style()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei") if f in installed]
    if not fonts:
        warnings.warn("Install a CJK font to display Chinese figure labels.")
    plt.rcParams.update({"font.family": [*fonts, "DejaVu Sans"], "axes.unicode_minus": False,
                         "savefig.dpi": 185, "axes.labelcolor": INK, "axes.titlecolor": INK,
                         "text.color": INK, "axes.axisbelow": True, "font.size": 11})


def _heat(grid, rows, columns, title, note):
    fig, ax = plt.subplots(figsize=(8.1, max(3.3, len(rows) * .45 + 1.25)))
    scale = Normalize(0, 1)
    ax.imshow(grid, cmap=MAP, norm=scale, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, loc="left", pad=13)
    ax.grid(False)
    ax.tick_params(length=0)
    for i, j in np.ndindex(grid.shape):
        val = grid[i, j]
        if np.isfinite(val):
            red, green, blue, _ = MAP(scale(val))
            color = "#FFFFFF" if .21 * red + .72 * green + .07 * blue < .50 else "#163A3B"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, color=color)
        else:
            ax.text(j, i, "—", color=INK, ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(note, fontsize=9, labelpad=11)
    return fig


def _ablation(frame, factor):
    data = frame[frame.factor.eq(factor)].sort_values("value")
    if data.empty:
        return None
    xs = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), layout="constrained")
    labels = [f"{float(v):g}" for v in data.value]
    for ax, metric, color, title in zip(axes, ["ari", "recon_sse"], [TEAL, "#AA6039"],
                                        ["分组得分 ARI  ↑", "每张图的重构平方误差  ↓"]):
        ax.plot(xs, data[metric], marker="o", color=color, linewidth=2.2)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("重构权重 λ" if factor == "lambda_recon" else "表示的维数")
        if metric == "ari":
            ax.set_ylim(min(0, data.ari.min()), max(.55, data.ari.max() * 1.12))
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"服装图像 · IDEC · 随机种子 {data.seed.iloc[0]}", fontweight="bold")
    return fig


def _curves(frame, factor, protocol):
    noise = factor == "noise_std"
    robust = protocol["e03"]
    if noise:
        frame = frame[frame.dataset.eq(robust["dataset"]) & frame.method.isin(robust["methods"])]
    else:
        frame = frame[frame.factor.eq(factor)]
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.9, 4.25))
    levels = sorted(frame[factor].unique())
    # The number of examples varies by orders of magnitude; each measured condition gets a tick.
    index = {v: i for i, v in enumerate(levels)}
    for m in robust["methods"]:
        series = frame[frame.method.eq(m)].sort_values(factor)
        if series.empty:
            continue
        xs = [index[v] for v in series[factor]]
        ys = series["ari_mean" if noise else "ari"].to_numpy(float)
        ax.plot(xs, ys, color=COLORS[m], marker="o", markersize=4.4, lw=1.9, label=METHODS[m])
        if noise:
            deviations = series.ari_std.to_numpy(float)
            ax.fill_between(xs, ys-deviations, ys+deviations, color=COLORS[m], alpha=.09)
    labels = {"noise_std": "测试图像的噪声标准差（像素范围 0–1）", "k": "要求分成几组（真实类别数为 10）",
              "sample_frac": "训练图像占原训练子集的比例", "imbalance": "训练集中多数类与少数类的目标比例",
              "outlier_frac": "训练图像中替换为随机噪声的比例"}
    ticks = [f"{v:g}" for v in levels]
    if factor in {"sample_frac", "outlier_frac"}:
        ticks = [f"{100*v:g}%" for v in levels]
    if factor == "imbalance":
        ticks = [f"{v:g}:1" for v in levels]
    ax.set_xticks(range(len(levels)), ticks)
    ax.set_xlabel(labels[factor])
    ax.set_ylabel("分组得分 ARI")
    ax.set_ylim(min(-.015, frame["ari_mean" if noise else "ari"].min()),
                max(.53, frame["ari_mean" if noise else "ari"].max()+.07))
    if noise:
        counts = sorted(frame.n.unique())
        ntext = str(counts[0]) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
        note = f"各条件种子数 {ntext}，色带为样本标准差"
    else:
        note = f"单一划分，随机种子 {robust['seed']}"
    ax.set_title("服装图像 · " + note, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), fontsize=9)
    return fig


def _embeddings(raw, dataset, seed):
    methods = ["pca_kmeans", "ae_kmeans", "dec", "idec"]
    paths = [raw / "embeddings" / f"{dataset}__{m}__seed{seed}.npz" for m in methods]
    if not all(p.is_file() for p in paths):
        return None
    fig, axes = plt.subplots(1, len(methods), figsize=(12.2, 3.65), layout="constrained")
    shared_ids = None
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
            positions = TSNE(n_components=2, random_state=seed, init="pca",
                             learning_rate="auto", perplexity=min(30, len(x)-1)).fit_transform(x)
        ax.scatter(*positions.T, c=y, cmap="tab10", vmin=-.5, vmax=9.5, s=5, alpha=.78, linewidths=0)
        ax.set_title(METHODS[method], fontsize=12)
        ax.set_axis_off()
    fig.suptitle(f"同一批 {len(subset)} 张服装图像的二维投影", fontweight="bold")
    fig.supxlabel("每点是一张图，颜色取自各方法的预测簇。各图独立拟合 t-SNE，颜色与坐标跨图无固定对应。", fontsize=9)
    return fig


def _frontier(costs, dataset):
    group = costs[costs.dataset.eq(dataset)].dropna(subset=["front_ari_time"])
    if group.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.3, 4.45))
    frontier = group[group.front_ari_time].sort_values("time_train_mean_sec")
    ax.plot(frontier.time_train_mean_sec, frontier.ari_mean, lw=1.5, ls="--", color=TEAL)
    handles = []
    for row in group.itertuples():
        # A side key keeps names legible for near-identical cost/score pairs.
        color = COLORS[row.method]
        marker = "D" if row.front_ari_time else "o"
        ax.scatter(row.time_train_mean_sec, row.ari_mean, s=62, marker=marker, color=color, zorder=3)
        handles.append(Line2D([], [], color=color, marker=marker, ls="", label=METHODS[row.method]))
    ax.set_xscale("log")
    ax.set_xlabel("实测拟合秒数（对数刻度，含当次缓存条件）")
    ax.set_ylabel("平均分组得分 ARI")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title("服装图像 · 菱形为速度和得分的两目标前沿", loc="left")
    return fig


def render_figures(root, protocol, tables):
    _setup()
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(name, fig):
        path = figures / f"fig_{name}.png"
        if fig is not None:
            fig.savefig(path, facecolor="white", pad_inches=.15)
            plt.close(fig)
        else:
            path.unlink(missing_ok=True)

    e01 = tables["e01_summary"]
    grid = e01.pivot(index="method", columns="dataset", values="ari_mean")
    grid = grid.reindex(index=list(METHODS), columns=list(DATASETS))
    save("e01_ari", _heat(grid.to_numpy(float), list(METHODS.values()), list(DATASETS.values()),
                           "三类图像上的平均分组得分 ARI", "分数越高，与人工类别越接近。每格种子数和标准差见 E01 汇总表。") if len(e01) else None)
    cfg = protocol["e04"]
    encoders = {"random_cnn": "随机卷积网络", "ae": "自编码器", "simclr": "SimCLR",
                "dinov2": "DINOv2", "clip": "CLIP"}
    clusterers = {"kmeans": "K 均值", "gmm": "高斯混合", "dec_head": "投影 + DEC"}
    pairs = tables["e04_paired"]
    grid = pairs.pivot(index="encoder", columns="clusterer", values="ari")
    grid = grid.reindex(index=cfg["encoders"], columns=cfg["clusterers"])
    save("e04_paired", _heat(grid.to_numpy(float), [encoders[x] for x in grid.index],
                             [clusterers[x] for x in grid.columns], "同一批自然图像 · 更换表示或分组方法",
                             f"每格均为种子 {cfg['seed']} 的 ARI；末两行使用外部预训练。") if len(pairs) else None)
    for factor, name in (("latent_dim", "latent"), ("lambda_recon", "lambda")):
        save(f"e02_{name}", _ablation(tables["e02_ablation"], factor))
    for factor in ("noise_std", "k", "imbalance", "sample_frac", "outlier_frac"):
        tab = "e03_noise" if factor == "noise_std" else "e03_sensitivity"
        save("e03_" + ("noise" if factor == "noise_std" else factor), _curves(tables[tab], factor, protocol))
    save("e05_pareto", _frontier(tables["e05_cost_frontiers"], protocol["e03"]["dataset"]))
    save("e01_embeddings", _embeddings(root / "results/raw", protocol["e02"]["dataset"], protocol["e02"]["seed"]))
