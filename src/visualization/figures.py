"""Result figures, drawn from results/tables at the size they occupy on a slide.

Every figure is sized in inches to its slot on the 13.333 x 7.5 inch deck, so the
point sizes below are the point sizes the audience sees. Color encodes the method
family only (see style.py); text stays in neutral ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from visualization import style as S

FROM_SCRATCH = ["kmeans", "pca_kmeans", "gmm", "ae_kmeans", "dec", "idec", "vade", "simclr_kmeans", "cc"]
DATASETS = ["mnist", "fashion_mnist", "cifar10"]
DATASET_NOTE = {"mnist": "手写数字", "fashion_mnist": "服饰", "cifar10": "自然图像"}
CLUSTER_COLORS = ["#2A78D6", "#EB6834", "#16A07A", "#A83E9E", "#D6A100",
                  "#5560C2", "#D9534F", "#36A3C9", "#8C6D46", "#77A33A"]


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


def _value_grid(ax, axis="x"):
    ax.grid(True, axis=axis, color=S.GRID, linewidth=1.0)
    ax.grid(False, axis="y" if axis == "x" else "x")
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def _heading(ax, title: str, note: str | None = None, y: float = 1.035) -> None:
    """Bold panel title with an optional gray note on the same baseline."""
    fig = ax.figure
    head = ax.text(0.0, y, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=15,
                   fontweight="bold", color=S.INK)
    if note:
        renderer = fig.canvas.get_renderer()
        box = head.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        ax.text(box.x1 + 0.03, y, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
                color=S.SECONDARY)


def _family_handles(families):
    names = {S.CLASSICAL: "经典方法", S.DEEP: "自编码器系", S.GENERATIVE: "生成式",
             S.CONTRASTIVE: "对比学习", S.PRETRAINED: "外部预训练"}
    return [Line2D([], [], marker="s", linestyle="", markersize=10, color=c, label=names[c]) for c in families]


# --------------------------------------------------------------------------- benchmark
def benchmark(e01: pd.DataFrame, path: Path) -> None:
    gaps = {"gmm": 0.5, "idec": 0.5, "vade": 0.5}  # space after each family group, top-down
    y, ypos = 0.0, {}
    for method in FROM_SCRATCH:
        ypos[method] = -y
        y += 1.0 + gaps.get(method, 0.0)
    fig, axes = S.figure(12.03, 4.85, ncols=3, sharey=True)
    fig.subplots_adjust(left=0.148, right=0.985, top=0.90, bottom=0.12, wspace=0.12)
    for ax, dataset in zip(axes, DATASETS):
        rows = e01[e01.dataset.eq(dataset)].set_index("method")
        best = rows.ari_mean.idxmax()
        for method in FROM_SCRATCH:
            if method not in rows.index:
                continue
            mean, sd = _num(rows.at[method, "ari_mean"]), _num(rows.at[method, "ari_std"])
            if mean is None:
                continue
            ax.barh(ypos[method], mean, height=0.62, color=S.FAMILY[method], linewidth=0)
            if sd:
                ax.plot([max(mean - sd, 0), mean + sd], [ypos[method]] * 2, color=S.BODY, linewidth=1.1)
            label_x = mean + (sd or 0) + 0.012
            weight, color = ("bold", S.INK) if method == best else ("normal", S.SECONDARY)
            ax.text(label_x, ypos[method], f"{mean:.3f}", va="center", ha="left", fontsize=11,
                    color=color, fontweight=weight)
        ax.set_xlim(0, 0.78)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
        _value_grid(ax, "x")
        _heading(ax, S.DATASET_LABEL[dataset], DATASET_NOTE[dataset])
        if dataset == "cifar10" and len(rows):
            hi = float(rows.ari_mean.max())
            ax.text(0.45, -5.0, f"全部 ≤ {hi:.3f}\n（接近随机乱分）", ha="center", va="center",
                    fontsize=12, color=S.SECONDARY,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor=S.PANEL, edgecolor="none"))
    axes[0].set_yticks([ypos[m] for m in FROM_SCRATCH])
    axes[0].set_yticklabels([S.LABEL[m] for m in FROM_SCRATCH], fontsize=12.5, color=S.BODY)
    axes[1].set_xlabel("ARI（3 个随机种子均值，误差线为标准差）", fontsize=12, color=S.SECONDARY, labelpad=8)
    S.save(fig, path)


# --------------------------------------------------------------------------- embeddings
def embeddings(raw: Path, dataset: str, seed: int, runs: pd.DataFrame, path: Path, n_points: int = 2000) -> None:
    from sklearn.manifold import TSNE

    methods = ["pca_kmeans", "ae_kmeans", "dec", "idec"]
    loaded = {}
    for method in methods:
        source = raw / "embeddings" / f"{dataset}__{method}__seed{seed}.npz"
        if not source.exists():
            return
        with np.load(source, allow_pickle=False) as data:
            loaded[method] = (data["embeddings"], data["predictions"])
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(next(iter(loaded.values()))[0]), size=n_points, replace=False))
    fig, axes = S.figure(12.03, 4.0, ncols=4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.2, wspace=0.06)
    e01 = runs[runs.experiment.eq("E01") & runs.dataset.eq(dataset) & runs.seed.eq(seed)].set_index("method")
    for ax, method in zip(axes, methods):
        z, pred = loaded[method]
        xy = TSNE(n_components=2, perplexity=35, init="pca", random_state=0).fit_transform(z[pick])
        colors = np.array(CLUSTER_COLORS)[pred[pick] % len(CLUSTER_COLORS)]
        ax.scatter(xy[:, 0], xy[:, 1], s=5.5, c=colors, linewidths=0, alpha=0.9, rasterized=True)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ari = _num(e01.at[method, "ari"]) if method in e01.index else None
        ax.text(0.5, -0.04, S.LABEL[method], transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color=S.INK)
        if ari is not None:
            ax.text(0.5, -0.14, f"ARI {ari:.3f}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, color=S.SECONDARY)
        ax.set_aspect("equal", adjustable="datalim")
    S.save(fig, path)


# --------------------------------------------------------------------------- encoder x clusterer
ENCODERS = ["random_cnn", "ae", "simclr", "dinov2", "clip"]
ENCODER_COLOR = {"random_cnn": S.CLASSICAL, "ae": S.DEEP, "simclr": S.CONTRASTIVE,
                 "dinov2": S.PRETRAINED, "clip": S.PRETRAINED}
CLUSTERER_MARKER = {"kmeans": "o", "gmm": "s", "dec_head": "^"}


def representation_vs_clusterer(summary: pd.DataFrame, path: Path) -> None:
    datasets = [d for d in DATASETS if d in set(summary.dataset)]
    fig, axes = S.figure(12.03, 4.6, ncols=len(datasets), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.86, bottom=0.13, wspace=0.10)
    ypos = {"random_cnn": 0.0, "ae": -1.0, "simclr": -2.0, "dinov2": -3.45, "clip": -4.45}
    dodge = {"kmeans": 0.24, "gmm": 0.0, "dec_head": -0.24}
    for ax, dataset in zip(axes, datasets):
        rows = summary[summary.dataset.eq(dataset)]
        ax.axhspan(-4.95, -2.95, color=S.PANEL, zorder=0, linewidth=0)
        km_scores = {}
        for encoder in ENCODERS:
            sub = rows[rows.encoder.eq(encoder)].set_index("clusterer")
            if "kmeans" in sub.index:
                v = _num(sub.at["kmeans", "ari_mean"])
                if v is not None:
                    km_scores[encoder] = v
        best_enc = max(km_scores, key=km_scores.get) if km_scores else None
        for encoder in ENCODERS:
            sub = rows[rows.encoder.eq(encoder)].set_index("clusterer")
            vals = {c: _num(sub.at[c, "ari_mean"]) for c in CLUSTERER_MARKER if c in sub.index}
            vals = {c: v for c, v in vals.items() if v is not None}
            if not vals:
                continue
            hi = max(vals.values())
            y0 = ypos[encoder]
            for clusterer, value in vals.items():
                ax.scatter(value, y0 + dodge[clusterer], marker=CLUSTERER_MARKER[clusterer], s=62,
                           color=ENCODER_COLOR[encoder], edgecolors="white", linewidths=1.3, zorder=3)
            km = vals.get("kmeans")
            if km is not None and (encoder in {"dinov2", "clip"} or encoder == best_enc):
                ax.text(hi + 0.035, y0, f"{km:.2f}", va="center", ha="left", fontsize=12,
                        color=S.INK, fontweight="bold")
        ax.set_xlim(-0.02, 0.92)
        ax.set_ylim(-4.95, 0.45)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
        _value_grid(ax, "x")
        _heading(ax, S.DATASET_LABEL[dataset], y=1.025)
    axes[0].set_yticks([ypos[e] for e in ENCODERS])
    axes[0].set_yticklabels([S.ENCODER_LABEL[e] for e in ENCODERS], fontsize=12.5, color=S.BODY)
    for label, y in (("从零训练", -1.0), ("冻结预训练", -3.95)):
        axes[0].text(-0.42, y, label, transform=axes[0].get_yaxis_transform(), rotation=90,
                     ha="center", va="center", fontsize=11.5, fontweight="bold", color=S.SECONDARY)
    handles = [Line2D([], [], marker=CLUSTERER_MARKER[c], linestyle="", markersize=8.5, color=S.BODY,
                      markeredgecolor="white", label=S.CLUSTERER_LABEL[c]) for c in ("kmeans", "gmm", "dec_head")]
    legend = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 1.005), ncol=3, fontsize=12,
                        handletextpad=0.3, columnspacing=1.4, frameon=False)
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(box.x0 + 0.004, (box.y0 + box.y1) / 2, "聚类器", ha="right", va="center", fontsize=12,
             color=S.SECONDARY)
    axes[len(axes) // 2].set_xlabel("ARI（3 个随机种子均值）", fontsize=12, color=S.SECONDARY, labelpad=8)
    S.save(fig, path)


# --------------------------------------------------------------------------- noise (dumbbell)
def noise(noise_table: pd.DataFrame, dataset: str, path: Path) -> None:
    rows = noise_table[noise_table.dataset.eq(dataset)]
    hi_sigma = rows.noise_std.max()
    clean = rows[rows.noise_std.eq(0)].set_index("method").ari_mean
    noisy = rows[rows.noise_std.eq(hi_sigma)].set_index("method").ari_mean
    methods = [m for m in FROM_SCRATCH if m in clean.index and m in noisy.index]
    methods.sort(key=lambda m: float(noisy[m]))
    fig, ax = S.figure(7.55, 4.85)
    fig.subplots_adjust(left=0.232, right=0.865, top=0.90, bottom=0.12)
    for i, m in enumerate(methods):
        c0, c1 = float(clean[m]), float(noisy[m])
        color = S.FAMILY[m]
        ax.plot([c1, c0], [i, i], color=color, alpha=0.35, linewidth=3, zorder=1)
        ax.scatter(c0, i, s=60, facecolors="white", edgecolors=color, linewidths=2, zorder=3)
        ax.scatter(c1, i, s=70, color=color, edgecolors="white", linewidths=1.5, zorder=4)
        keep = c1 / c0 if c0 > 0 else np.nan
        emphasis = keep >= 0.85
        ax.text(0.665, i, f"{keep:.0%}", va="center", ha="left", fontsize=12.5,
                color=S.INK if emphasis else S.SECONDARY, fontweight="bold" if emphasis else "normal")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([S.LABEL[m] for m in methods], fontsize=12.5, color=S.BODY)
    ax.set_xlim(-0.01, 0.62)
    ax.set_ylim(-0.7, len(methods) - 0.3)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    _value_grid(ax, "x")
    ax.text(0.665, len(methods) - 0.35, "保留", ha="left", va="bottom", fontsize=11.5, color=S.SECONDARY)
    ax.text(0.58, len(methods) - 0.35, "← 线段越短，越抗噪", ha="right", va="bottom", fontsize=10.5,
            color=S.SECONDARY)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
                      markeredgecolor=S.BODY, markeredgewidth=1.8, label="干净测试图"),
               Line2D([], [], marker="o", linestyle="", markersize=8, color=S.BODY,
                      label=f"加噪 σ = {hi_sigma:g}")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=12,
              handletextpad=0.3, columnspacing=1.4)
    ax.set_xlabel("ARI（Fashion-MNIST，3 个随机种子均值）", fontsize=12, color=S.SECONDARY, labelpad=8)
    S.save(fig, path)


# --------------------------------------------------------------------------- sensitivity lines
LINE_STYLE = {
    "kmeans": dict(color=S.CLASSICAL_DARK, lw=2.6, z=4),
    "ae_kmeans": dict(color=S.DEEP_LIGHT, lw=2.0, z=2),
    "dec": dict(color=S.DEEP, lw=2.8, z=5),
    "idec": dict(color=S.DEEP_DARK, lw=2.0, z=3),
    "cc": dict(color=S.CONTRASTIVE, lw=1.8, z=1),
}


def _factor_lines(ax, table, factor, methods, xmap=None):
    rows = table[table.factor.eq(factor)]
    for m in methods:
        sub = rows[rows.method.eq(m)].copy()
        if sub.empty:
            continue
        sub["x"] = sub["value"].astype(float)
        sub = sub.sort_values("x")
        x = sub["x"].map(xmap) if xmap else sub["x"]
        y = sub.ari_mean.astype(float)
        st = LINE_STYLE[m]
        ax.plot(x, y, color=st["color"], linewidth=st["lw"], zorder=st["z"], marker="o", markersize=5.5,
                markeredgecolor="white", markeredgewidth=1.2)
        sd = pd.to_numeric(sub.ari_std, errors="coerce")
        if sd.notna().all():
            ax.fill_between(x, y - sd, y + sd, color=st["color"], alpha=0.10, linewidth=0, zorder=0)


def sample_size(sens: pd.DataFrame, path: Path) -> None:
    fig, ax = S.figure(7.55, 4.85)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.93, bottom=0.17)
    fracs = sorted(sens[sens.factor.eq("sample_frac")]["value"].astype(float).unique())
    xmap = {f: i for i, f in enumerate(fracs)}
    rows = sens[sens.factor.eq("sample_frac")]
    methods = ["cc", "ae_kmeans", "idec", "kmeans", "dec"]
    ends = {}
    for m in methods:
        sub = rows[rows.method.eq(m)].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
        if len(sub):
            ends[m] = float(sub.ari_mean.iloc[-1])
    _factor_lines(ax, sens, "sample_frac", methods, xmap=xmap)
    _end_labels(ax, ends, x=len(fracs) - 1 + 0.12)
    n_train = rows.groupby(rows["value"].astype(float)).n_train_mean.first() if "n_train_mean" in rows else None
    ticks = []
    for f in fracs:
        n = int(round(float(n_train[f]))) if n_train is not None and f in n_train.index else None
        ticks.append(f"{f:.0%}\n{n:,} 张" if n else f"{f:.0%}")
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(ticks, fontsize=11.5)
    ax.set_xlim(-0.25, len(fracs) - 1 + 0.15)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    _value_grid(ax, "y")
    ax.set_ylabel("ARI", fontsize=12, color=S.SECONDARY)
    ax.set_xlabel("训练集规模（Fashion-MNIST，占 2 万张训练图的比例）", fontsize=12, color=S.SECONDARY, labelpad=6)
    km_sub = rows[rows.method.eq("kmeans")].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    dec_sub = rows[rows.method.eq("dec")].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(km_sub) == len(dec_sub) and len(km_sub) >= 3:
        diff = dec_sub.ari_mean.to_numpy(float) - km_sub.ari_mean.to_numpy(float)
        if diff[0] < 0 and diff[-1] > 0:
            cross_idx = int(np.argmax(diff > 0))
            ax.annotate("样本够多后\nDEC 反超", xy=(cross_idx, float(dec_sub.ari_mean.iloc[cross_idx])),
                        xytext=(cross_idx - 0.35, 0.455), ha="center", va="bottom", fontsize=10.5,
                        color=S.DEEP, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=S.DEEP, lw=1.2, shrinkB=6))
    S.save(fig, path)


def _end_labels(ax, ends: dict, x: float, min_gap: float = 0.028):
    """Direct labels at line ends, nudged apart only as far as needed and joined by leader lines."""
    items = sorted(ends.items(), key=lambda kv: kv[1])
    placed = []
    for m, y in items:
        target = y
        if placed and target - placed[-1][1] < min_gap:
            target = placed[-1][1] + min_gap
        placed.append((m, target, y))
    for m, ty, y in placed:
        ax.annotate(S.LABEL[m], xy=(x - 0.1, y), xytext=(x + 0.08, ty), va="center", ha="left", fontsize=12,
                    color=S.BODY, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, linewidth=0.8, shrinkA=0, shrinkB=2)
                    if abs(ty - y) > 0.004 else None)


# --------------------------------------------------------------------------- components + lambda
def components_and_lambda(e02: pd.DataFrame, fixed: pd.DataFrame, path: Path, runs: pd.DataFrame | None = None) -> None:
    S.apply_style()
    fig = plt.figure(figsize=(12.03, 4.5))
    left = fig.add_axes([0.205, 0.14, 0.30, 0.72])
    right = fig.add_axes([0.63, 0.14, 0.35, 0.72])
    pick = [("pretrain", "False", "不做 AE 预训练"), ("center_init", "random", "簇中心随机初始化"),
            ("freeze_encoder", "True", "冻结编码器，只学簇中心"), ("freeze_encoder", "False", "完整 IDEC")]
    vals = []
    for factor, value, label in pick:
        row = e02[e02.factor.eq(factor) & e02["value"].astype(str).eq(value)]
        if len(row):
            vals.append((label, _num(row.ari_mean.iloc[0]), _num(row.ari_std.iloc[0])))
    full_mean = next((m for lab, m, _ in vals if lab == "完整 IDEC"), None)
    for i, (label, mean, sd) in enumerate(vals):
        full = label == "完整 IDEC"
        color = S.DEEP if full else "#A9C9F2"
        left.barh(i, mean, height=0.58, color=color, linewidth=0)
        if sd:
            left.plot([mean - sd, mean + sd], [i, i], color=S.BODY, linewidth=1.1)
        if not full and full_mean is not None and mean is not None:
            delta = mean - full_mean
            txt = f"{mean:.3f} ({delta:+.2f})"
        else:
            txt = f"{mean:.3f}"
        left.text(mean + (sd or 0) + 0.012, i, txt, va="center", fontsize=11.5,
                  color=S.INK if full else S.SECONDARY, fontweight="bold" if full else "normal")
    left.set_yticks(range(len(vals)))
    left.set_yticklabels([v[0] for v in vals], fontsize=12.5, color=S.BODY)
    left.set_xlim(0, 0.62)
    left.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    left.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    _value_grid(left, "x")
    _heading(left, "逐个去掉组件（看谁功劳最大）", y=1.04)
    left.set_xlabel("ARI", fontsize=12, color=S.SECONDARY)

    # Right: the lambda effect as paired differences from lambda = 0 (same seed, same pretrained AE)
    lambdas = [0.01, 0.1, 1.0, 10.0]
    xs = list(range(len(lambdas)))
    right.axhline(0, color=S.MUTED, linewidth=1.0, zorder=1)
    handles = []
    series = [("固定训练 50 轮", "E02_fixed", S.DEEP, "o", True), ("按原协议早停", "E02", S.CLASSICAL_DARK, "o", False)]
    for label, experiment, color, marker, filled in series:
        if runs is None or runs.empty:
            break
        sub = runs[runs.experiment.eq(experiment) & runs.method.eq("idec") & runs.latent_dim.eq(10)
                   & runs.pretrain.eq(True) & runs.center_init.eq("kmeans") & runs.freeze_encoder.eq(False)]
        table = sub.pivot_table(index="seed", columns="lambda_recon", values="ari")
        if 0.0 not in table:
            continue
        diffs = table.sub(table[0.0], axis=0)
        mean = np.array([diffs[l].mean() if l in diffs else np.nan for l in lambdas])
        sd = np.array([diffs[l].std(ddof=1) if l in diffs and diffs[l].count() > 1 else np.nan for l in lambdas])
        shift = -0.07 if filled else 0.07
        if filled:
            right.plot([x + shift for x in xs], mean, color=color, linewidth=2.4, zorder=3)
        right.errorbar([x + shift for x in xs], mean, yerr=sd, fmt=marker, color=color, ecolor=color,
                       elinewidth=1.2, capsize=0, markersize=7, zorder=4,
                       markerfacecolor=color if filled else "white", markeredgewidth=1.6)
        handles.append(Line2D([], [], color=color, marker=marker, linewidth=2.4 if filled else 0,
                              markerfacecolor=color if filled else "white", markersize=7, label=label))
    right.legend(handles=handles, loc="lower left", fontsize=12, handlelength=1.8)
    right.set_xticks(xs)
    right.set_xticklabels(["0.01", "0.1", "1", "10"])
    right.set_xlim(-0.4, len(lambdas) - 0.6)
    right.set_ylim(-0.085, 0.025)
    right.set_yticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02])
    right.set_yticklabels(["−0.08", "−0.06", "−0.04", "−0.02", "0", "+0.02"])
    right.text(len(lambdas) - 0.65, 0.003, "↑ 高于 DEC", ha="right", va="bottom", fontsize=10.5, color=S.SECONDARY)
    right.text(len(lambdas) - 0.65, -0.004, "↓ 低于 DEC", ha="right", va="top", fontsize=10.5, color=S.SECONDARY)
    _value_grid(right, "y")
    _heading(right, "重构权重 λ：相对 λ = 0（DEC）的 ARI 变化", y=1.04)
    right.set_xlabel("λ", fontsize=12, color=S.SECONDARY)
    right.set_ylabel("同一种子上的 ARI 差", fontsize=12, color=S.SECONDARY)
    S.save(fig, path)


# --------------------------------------------------------------------------- cost frontier
COST_OFFSETS = {
    "pca_kmeans": (8, -16), "kmeans": (6, -15), "gmm": (8, -14), "ae_kmeans": (-12, 13), "dec": (-10, 9),
    "idec": (9, 0), "vade": (9, -9), "simclr_kmeans": (-10, 9), "cc": (9, -2),
}


def cost_frontier(costs: pd.DataFrame, dataset: str, path: Path) -> None:
    rows = costs[costs.dataset.eq(dataset)].copy()
    rows["t"] = rows.time_train_e2e_mean_sec.astype(float)
    rows["a"] = rows.ari_mean.astype(float)
    fig, ax = S.figure(7.55, 4.85)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.14)
    front = rows[rows.front_ari_time_e2e.astype(str).str.lower().eq("true")].sort_values("t")
    ax.plot(front.t, front.a, color=S.INK, linewidth=1.4, zorder=2)
    for r in rows.itertuples():
        on_front = r.method in set(front.method)
        ax.scatter(r.t, r.a, s=95 if on_front else 70, color=S.FAMILY[r.method], edgecolors="white",
                   linewidths=1.8, zorder=4)
        dx, dy = COST_OFFSETS.get(r.method, (8, 0))
        ax.annotate(f"{S.LABEL[r.method]}", (r.t, r.a), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx >= 0 else "right", va="center", fontsize=11.5,
                    color=S.INK if on_front else S.SECONDARY, fontweight="bold" if on_front else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.9, 450)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1 秒", "3", "10", "30", "100", "300 秒"])
    ax.set_ylim(0.15, 0.48)
    _value_grid(ax, "y")
    ax.grid(True, axis="x", color=S.GRID, linewidth=1.0)
    ax.set_ylabel("ARI", fontsize=12, color=S.SECONDARY)
    ax.set_xlabel("端到端训练时间（对数刻度，含 AE 预训练）", fontsize=12, color=S.SECONDARY, labelpad=6)
    ax.text(1.05, 0.468, "↖ 越往左上越划算（越快且越准）", ha="left", va="top", fontsize=11,
            color=S.SECONDARY)
    S.save(fig, path)


# --------------------------------------------------------------------------- stability vs correctness
def stability(selection: dict | None, e01: pd.DataFrame, path: Path) -> None:
    if not selection:
        return
    fig, axes = S.figure(12.03, 4.45, ncols=3, sharey=False)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.14, wspace=0.2)
    for ax, dataset in zip(axes, DATASETS):
        evidence = (selection.get(dataset) or {}).get("evidence") or {}
        rows = e01[e01.dataset.eq(dataset)].set_index("method")
        pts = [(m, float(evidence[m]["macro_cluster_jaccard"]), float(rows.at[m, "ari_mean"]))
               for m in FROM_SCRATCH if m in evidence and m in rows.index
               and evidence[m].get("macro_cluster_jaccard") is not None]
        if not pts:
            continue
        stable = max(pts, key=lambda p: p[1])
        accurate = max(pts, key=lambda p: p[2])
        for m, s, a in pts:
            ax.scatter(s, a, s=72, color=S.FAMILY[m], edgecolors="white", linewidths=1.5, zorder=3)
        ax.scatter(stable[1], stable[2], s=330, facecolors="none", edgecolors=S.INK, linewidths=1.6, zorder=4)
        ax.scatter(accurate[1], accurate[2], s=330, facecolors="none", edgecolors=S.INK, linewidths=1.6,
                   linestyle=(0, (2, 2)), zorder=4)
        top = max(p[2] for p in pts)
        ax.set_xlim(-0.03, 1.06)
        ax.set_ylim(-0.02 * top, top * 1.18)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        _value_grid(ax, "y")
        ax.grid(True, axis="x", color=S.GRID, linewidth=1.0)
        _heading(ax, S.DATASET_LABEL[dataset], y=1.17)
        ax.text(0.0, 1.095, f"实线圈 最稳定：{S.LABEL[stable[0]]}（ARI {stable[2]:.3f}）", transform=ax.transAxes,
                fontsize=11.5, color=S.BODY, ha="left", va="bottom")
        ax.text(0.0, 1.025, f"虚线圈 最准确：{S.LABEL[accurate[0]]}（ARI {accurate[2]:.3f}）", transform=ax.transAxes,
                fontsize=11.5, color=S.BODY, ha="left", va="bottom")
    axes[0].set_ylabel("真实 ARI（越靠上 = 分得越准）", fontsize=12, color=S.SECONDARY)
    axes[1].set_xlabel("无标签稳定性（越靠右 = 加噪前后分组越不易变）", fontsize=12, color=S.SECONDARY,
                       labelpad=6)
    S.save(fig, path)


# --------------------------------------------------------------------------- appendix panels
def sensitivity_panels(sens: pd.DataFrame, path: Path) -> None:
    fig, axes = S.figure(12.03, 4.4, ncols=3)
    fig.subplots_adjust(left=0.055, right=0.835, top=0.90, bottom=0.17, wspace=0.22)
    methods = ["cc", "ae_kmeans", "idec", "kmeans", "dec"]
    specs = [("k", "预设簇数 K（真实为 10）", None), ("outlier_frac", "训练集中均匀噪声图的比例", None),
             ("imbalance", "多数类 : 少数类", None)]
    for ax, (factor, xlabel, _) in zip(axes, specs):
        vals = sorted(sens[sens.factor.eq(factor)]["value"].astype(float).unique())
        xmap = {v: i for i, v in enumerate(vals)}
        _factor_lines(ax, sens, factor, methods, xmap=xmap)
        ax.set_xticks(range(len(vals)))
        if factor == "outlier_frac":
            ax.set_xticklabels([f"{v:.0%}" for v in vals])
        elif factor == "imbalance":
            ax.set_xticklabels([f"{v:g}:1" for v in vals])
        else:
            ax.set_xticklabels([f"{v:g}" for v in vals])
        ax.set_ylim(0, 0.52)
        _value_grid(ax, "y")
        ax.set_xlabel(xlabel, fontsize=12, color=S.SECONDARY)
    handles = [Line2D([], [], color=LINE_STYLE[m]["color"], linewidth=2.6, label=S.LABEL[m])
               for m in ["dec", "idec", "ae_kmeans", "kmeans", "cc"]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.845, 0.55), fontsize=12)
    axes[0].set_ylabel("ARI", fontsize=12, color=S.SECONDARY)
    S.save(fig, path)


def latent_dim(e02: pd.DataFrame, path: Path) -> None:
    rows = e02[e02.factor.eq("latent_dim")].copy()
    rows["d"] = rows["value"].astype(float)
    rows = rows.sort_values("d")
    fig, ax = S.figure(5.6, 4.2)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.92, bottom=0.17)
    x = range(len(rows))
    y = rows.ari_mean.astype(float).values
    sd = pd.to_numeric(rows.ari_std, errors="coerce").values
    ax.plot(list(x), y, color=S.DEEP, linewidth=2.6, marker="o", markersize=6, markeredgecolor="white")
    if np.isfinite(sd).all():
        ax.fill_between(list(x), y - sd, y + sd, color=S.DEEP, alpha=0.12, linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(d)}" for d in rows.d])
    ax.set_ylim(0.2, 0.52)
    _value_grid(ax, "y")
    if len(y):
        best_i = int(np.argmax(y))
        best_d = int(rows.d.iloc[best_i])
        ax.annotate(f"{best_d} 维最优（{y[best_i]:.3f}）", xy=(best_i, y[best_i]),
                    xytext=(best_i + 0.55, min(y[best_i] + 0.065, 0.495)), ha="left", va="center",
                    fontsize=11, fontweight="bold", color=S.DEEP,
                    arrowprops=dict(arrowstyle="->", color=S.DEEP, lw=1.2, shrinkB=6))
    ax.set_xlabel("隐空间维度 d（IDEC，Fashion-MNIST）", fontsize=12, color=S.SECONDARY)
    ax.set_ylabel("ARI", fontsize=12, color=S.SECONDARY)
    S.save(fig, path)


# --------------------------------------------------------------------------- entry point
def render_figures(root: Path, protocol: dict, tables: dict, selection: dict | None = None) -> None:
    figures = root / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    for old in figures.glob("fig_*.png"):
        old.unlink()
    out = lambda name: figures / f"{name}.png"  # noqa: E731
    benchmark(tables["e01_summary"], out("benchmark_ari"))
    embeddings(root / "results/raw", protocol["e02"]["dataset"], int(protocol["e02"]["seeds"][0]),
               tables["all_runs"], out("embeddings_tsne"))
    representation_vs_clusterer(tables["e04_summary"], out("representation_vs_clusterer"))
    noise(tables["e03_noise"], protocol["e03"]["dataset"], out("noise_robustness"))
    sample_size(tables["e03_sensitivity_summary"], out("sample_size"))
    components_and_lambda(tables["e02_summary"], tables.get("e02_fixed_epochs"), out("components_lambda"),
                          tables["all_runs"])
    cost_frontier(tables["benchmark_cost_frontiers"], protocol["e03"]["dataset"], out("cost_frontier"))
    stability(selection, tables["e01_summary"], out("stability_vs_ari"))
    sensitivity_panels(tables["e03_sensitivity_summary"], out("sensitivity_k_outliers_imbalance"))
    latent_dim(tables["e02_summary"], out("latent_dim"))
