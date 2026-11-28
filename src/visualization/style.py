"""Design tokens shared by the result figures and the slide deck.

Color carries one meaning everywhere: the method family. Text and chart chrome stay
neutral so that the data is the only colored element on a slide.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Neutral ink and chrome
INK = "#17171A"
BODY = "#3C3C43"
SECONDARY = "#6B6B73"
MUTED = "#9A9AA1"
HAIRLINE = "#E4E4E7"
GRID = "#EDEDF0"
PANEL = "#F5F5F4"
SURFACE = "#FFFFFF"
NIGHT = "#0F1115"

# Method families (validated all-pairs for color-vision deficiency on white)
DEEP = "#2A78D6"         # AE + K-means, DEC, IDEC
PRETRAINED = "#EB6834"   # DINOv2, CLIP (frozen)
CONTRASTIVE = "#16A07A"  # SimCLR + K-means, CC
GENERATIVE = "#A83E9E"   # VaDE
CLASSICAL = "#8E8E93"    # K-means, PCA + K-means, GMM
CLASSICAL_DARK = "#55555C"

# Lighter and darker steps of the deep-family blue for within-family contrast
DEEP_DARK = "#1C5CAB"
DEEP_LIGHT = "#86B6EF"

FAMILY = {
    "kmeans": CLASSICAL, "pca_kmeans": CLASSICAL, "gmm": CLASSICAL,
    "ae_kmeans": DEEP, "dec": DEEP, "idec": DEEP,
    "vade": GENERATIVE,
    "simclr_kmeans": CONTRASTIVE, "cc": CONTRASTIVE,
    "dinov2_kmeans": PRETRAINED, "clip_kmeans": PRETRAINED,
}

LABEL = {
    "kmeans": "K-means", "pca_kmeans": "PCA + K-means", "gmm": "GMM",
    "ae_kmeans": "AE + K-means", "dec": "DEC", "idec": "IDEC", "vade": "VaDE",
    "simclr_kmeans": "SimCLR + K-means", "cc": "CC",
    "dinov2_kmeans": "DINOv2 + K-means", "clip_kmeans": "CLIP + K-means",
}

ENCODER_LABEL = {
    "random_cnn": "随机 CNN", "ae": "自编码器", "simclr": "SimCLR",
    "dinov2": "DINOv2", "clip": "CLIP",
}
CLUSTERER_LABEL = {"kmeans": "K-means", "gmm": "GMM", "dec_head": "DEC 头"}
DATASET_LABEL = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}

FONT = "Noto Sans CJK SC"


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": HAIRLINE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": BODY,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "font.size": 12,
        "savefig.dpi": 300,
    })


def figure(width_in: float, height_in: float, **kwargs):
    """A figure sized to its slot on a 13.333 x 7.5 inch slide, so 12 pt renders as 12 pt."""
    apply_style()
    return plt.subplots(figsize=(width_in, height_in), **kwargs)


def save(fig, path) -> None:
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
