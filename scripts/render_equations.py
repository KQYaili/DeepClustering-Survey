"""Render the few displayed equations. Labels are mathtext, not data."""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "presentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

EQUATIONS = {
    "eq_kmeans.png": r"$\min \sum_{i=1}^{N} \|x_i-\mu_{c_i}\|_2^2$",
    "eq_map.png": r"$x \;\rightarrow\; f_\theta(x)=z \;\rightarrow\; \hat{c}$",
    "eq_dec.png": r"$q_{ij}=\frac{(1+\|z_i-\mu_j\|^2)^{-1}}{\sum_{j'}(1+\|z_i-\mu_{j'}\|^2)^{-1}}$",
    "eq_kl.png": r"$\mathcal{L}_{\mathrm{DEC}}=\mathrm{KL}(P\|Q)$",
    "eq_idec.png": r"$\mathcal{L}=\mathrm{KL}(P\|Q)+\lambda\,\frac{1}{B}\sum_i\|x_i-\hat{x}_i\|_2^2$",
}


def main() -> None:
    for name, equation in EQUATIONS.items():
        fig = plt.figure(figsize=(10.2, 1.15), dpi=200)
        fig.patch.set_facecolor("white")
        fig.text(0.02, 0.48, equation, fontsize=22, color="#1E3A5F", va="center", ha="left")
        fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", pad_inches=0.12)
        plt.close(fig)
        print(name)


if __name__ == "__main__":
    main()
