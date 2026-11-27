"""Render publication-grade equations at 300 DPI for the blue-white slide deck."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "presentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

EQUATIONS = {
    "eq_kmeans.png": r"$\min_{\{\mu_k\},\,\{c_i\}} \sum_{i=1}^{N} \|x_i - \mu_{c_i}\|_2^2$",
    "eq_map.png": r"$x \;\longrightarrow\; z = f_\theta(x) \;\longrightarrow\; \hat{c}$",
    "eq_dec.png": r"$q_{ij}=\frac{(1+\|z_i-\mu_j\|^2)^{-1}}{\sum_{j'}(1+\|z_i-\mu_{j'}\|^2)^{-1}},\qquad p_{ij}=\frac{q_{ij}^2 / f_j}{\sum_{j'} q_{ij'}^2 / f_{j'}}$",
    "eq_kl.png": r"$\mathcal{L}_{\mathrm{DEC}} = \mathrm{KL}(P\,\|\,Q) = \frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{K} p_{ij}\log\frac{p_{ij}}{q_{ij}}$",
    "eq_idec.png": r"$\mathcal{L}_{\mathrm{IDEC}} = \mathcal{L}_{\mathrm{cluster}} + \lambda\,\frac{1}{B}\sum_{i=1}^{B}\sum_{d=1}^{D}(x_{id}-\hat{x}_{id})^2$",
    "eq_vade.png": r"$p(x,z,c) = p(c)\,p(z\mid c)\,p(x\mid z),\qquad p(z\mid c)=\mathcal{N}(\mu_c,\sigma_c^2 I)$",
    "eq_pareto.png": r"$\max_{m}\left(\mathrm{ARI}(m),\;\mathrm{Robustness}(m),\;-\mathrm{Time}(m),\;-\mathrm{Memory}(m)\right)$",
}


def main() -> None:
    plt.rcParams.update({"mathtext.fontset": "stix"})
    for name, equation in EQUATIONS.items():
        fig = plt.figure(figsize=(11.0, 1.1), dpi=300)
        fig.patch.set_alpha(0.0)
        fig.text(0.5, 0.5, equation, fontsize=22, color="#073DA5", va="center", ha="center")
        fig.savefig(OUT / name, bbox_inches="tight", transparent=True, pad_inches=0.08, dpi=300)
        plt.close(fig)
        print("rendered", name)


if __name__ == "__main__":
    main()
