"""Draw the image-based illustrations of the deck from the datasets and cached features.

Needs the dataset caches under DATA_ROOT ({dataset}_full.npz) and the CIFAR-10 seed-0
feature caches in results/cache/features (written by the E04 runs). Test labels are
never read: example images are chosen with training labels, and the mosaics are
arranged by the representations alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.data import add_gaussian_noise, evaluation_images  # noqa: E402
from runtime.paths import data_root  # noqa: E402
from visualization import style as S  # noqa: E402

ASSETS = ROOT / "presentation" / "assets"
FASHION_NAMES = ["T 恤", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _to_image(array: np.ndarray, size: int) -> Image.Image:
    pixels = np.clip(np.transpose(array, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    mode = "L" if pixels.shape[2] == 1 else "RGB"
    image = Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels, mode=mode).convert("RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS if size > 64 else Image.Resampling.BICUBIC)


def _train_arrays(dataset: str):
    with np.load(data_root() / f"{dataset}_full.npz", allow_pickle=False) as arrays:
        return arrays["train_x"], arrays["train_y"]


# ----------------------------------------------------------------------------- strips
def dataset_strips(per_row: int = 10, size: int = 112, gap: int = 10) -> None:
    for dataset in ("mnist", "fashion_mnist", "cifar10"):
        images, labels = _train_arrays(dataset)
        rng = np.random.default_rng(7)
        picks = [int(rng.choice(np.flatnonzero(labels == c))) for c in range(per_row)]
        width = per_row * size + (per_row - 1) * gap
        canvas = Image.new("RGB", (width, size), "white")
        for i, index in enumerate(picks):
            canvas.paste(_to_image(images[index], size), (i * (size + gap), 0))
        canvas.save(ASSETS / f"strip_{dataset}.png")


# ----------------------------------------------------------------------------- pixel distance
def _shift(image: np.ndarray, dx: int) -> np.ndarray:
    out = np.zeros_like(image)
    if dx > 0:
        out[..., dx:] = image[..., :-dx]
    else:
        out[..., :dx] = image[..., -dx:]
    return out


def pixel_distance_example(size: int = 300, shift: int = 4, n: int = 1000) -> dict:
    """How often does a small shift move a T-shirt farther than the nearest other-class image?"""
    images, labels = _train_arrays("fashion_mnist")
    flat = images.reshape(len(images), -1).astype(np.float32)
    rng = np.random.default_rng(0)
    shirts = rng.choice(np.flatnonzero(labels == 0), size=n, replace=False)
    others = np.flatnonzero(labels != 0)
    a = flat[shirts]
    d2 = (a ** 2).sum(1)[:, None] + (flat[others] ** 2).sum(1)[None, :] - 2.0 * a @ flat[others].T
    nearest = others[d2.argmin(1)]
    d_other = np.sqrt(np.maximum(d2.min(1), 0.0))
    stats = {}
    for dx in (2, shift):
        d_move = np.array([np.linalg.norm(images[i] - _shift(images[i], dx)) for i in shirts])
        stats[dx] = (d_move, float((d_move > d_other).mean()))
    d_move = stats[shift][0]
    median = int(np.argsort(d_move / d_other)[n // 2])
    anchor, other = images[shirts[median]], images[nearest[median]]
    for name, array in (("anchor", anchor), ("moved", _shift(anchor, shift)), ("other", other)):
        _to_image(array, size).save(ASSETS / f"pixel_{name}.png")
    info = {"shift": shift, "n_shirts": n, "d_moved": float(d_move[median]), "d_other": float(d_other[median]),
            "other_class": FASHION_NAMES[int(labels[nearest[median]])],
            "fraction_farther": stats[shift][1], "fraction_farther_shift2": stats[2][1]}
    (ASSETS / "pixel_distance.yaml").write_text(yaml.safe_dump(info, allow_unicode=True), encoding="utf-8")
    return info


# ----------------------------------------------------------------------------- noise strip
def noise_strip(levels=(0.0, 0.1, 0.2, 0.3), size: int = 220) -> None:
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    images, _ = evaluation_images("fashion_mnist", protocol, 0)
    sample = images[11:12]
    for std in levels:
        noisy = add_gaussian_noise(sample, std, 0)[0]
        _to_image(noisy, size).save(ASSETS / f"noise_{str(std).replace('.', 'p')}.png")


# ----------------------------------------------------------------------------- mosaics
def _grid_layout(points: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Assign each 2-D point to one grid cell, keeping neighbours together."""
    from scipy.optimize import linear_sum_assignment

    span = points.max(0) - points.min(0)
    p = (points - points.min(0)) / np.where(span > 0, span, 1)
    gx, gy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    cost = ((p[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty(len(points), dtype=int)
    order[row_ind] = col_ind
    return order


def _mosaic(images: np.ndarray, coords: np.ndarray, cols: int, rows: int, cell: int, gap: int,
            background=(255, 255, 255)) -> Image.Image:
    slots = _grid_layout(coords, cols, rows)
    width, height = cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), background)
    for i, slot in enumerate(slots):
        r, c = divmod(int(slot), cols)
        canvas.paste(_to_image(images[i], cell), (c * (cell + gap), r * (cell + gap)))
    return canvas


def _tsne(matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(matrix)


def representation_mosaics(cols: int = 20, rows: int = 15, cell: int = 64, gap: int = 4) -> None:
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    images, _ids = evaluation_images("cifar10", protocol, 0)
    feature_file = ROOT / "results/cache/features/cifar10__dinov2__seed0__sf1__imb1__out0.npz"
    with np.load(feature_file) as stored:
        dino = stored["test"]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(images), size=cols * rows, replace=False))
    pixel_xy = _tsne(images[pick].reshape(len(pick), -1))
    dino_xy = _tsne(dino[pick])
    _mosaic(images[pick], pixel_xy, cols, rows, cell, gap).save(ASSETS / "mosaic_pixels.png")
    _mosaic(images[pick], dino_xy, cols, rows, cell, gap).save(ASSETS / "mosaic_dinov2.png")


def cover_mosaic(cols: int = 26, rows: int = 22, cell: int = 56, gap: int = 4) -> None:
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    images, _ids = evaluation_images("cifar10", protocol, 0)
    with np.load(ROOT / "results/cache/features/cifar10__dinov2__seed0__sf1__imb1__out0.npz") as stored:
        dino = stored["test"]
    rng = np.random.default_rng(1)
    pick = np.sort(rng.choice(len(images), size=cols * rows, replace=False))
    night = _rgb(S.NIGHT)
    mosaic = _mosaic(images[pick], _tsne(dino[pick]), cols, rows, cell, gap, background=night)
    # Fade the left edge into the slide background so the title can sit beside it.
    width, height = mosaic.size
    ramp = np.clip(np.linspace(-0.15, 1.0, width) / 0.55, 0, 1) ** 1.6
    alpha = Image.fromarray((np.tile(ramp, (height, 1)) * 255).astype(np.uint8), mode="L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(2))
    base = Image.new("RGB", mosaic.size, night)
    Image.composite(mosaic, base, alpha).save(ASSETS / "cover_mosaic.png", quality=95)


def readme_mosaics() -> None:
    """The two mosaics side by side with their K-means ARI, for the repository front page."""
    import matplotlib.pyplot as plt
    import pandas as pd

    tables = ROOT / "results/tables"
    e01 = pd.read_csv(tables / "e01_summary.csv")
    e04 = pd.read_csv(tables / "e04_summary.csv")
    pixel = e01[(e01.dataset == "cifar10") & (e01.method == "kmeans")].ari_mean.iloc[0]
    dino = e04[(e04.dataset == "cifar10") & (e04.encoder == "dinov2") & (e04.clusterer == "kmeans")].ari_mean.iloc[0]
    S.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
    for ax, name, head, ari in zip(axes, ("mosaic_pixels", "mosaic_dinov2"), ("像素空间", "DINOv2 表示空间"),
                                   (pixel, dino)):
        ax.imshow(Image.open(ASSETS / f"{name}.png"))
        ax.set_axis_off()
        ax.set_title(f"{head}　K-means ARI {ari:.2f}", loc="left", fontsize=15, color=S.INK, pad=8)
    fig.savefig(ASSETS / "readme_mosaics.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    dataset_strips()
    info = pixel_distance_example()
    print("pixel distance example:", info)
    noise_strip()
    representation_mosaics()
    cover_mosaic()
    readme_mosaics()


if __name__ == "__main__":
    main()
