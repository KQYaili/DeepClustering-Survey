"""Typeset the deck's equations with LaTeX into transparent 600-dpi PNGs.

All equations use the same font size, so placing a PNG at its natural size
(pixels / 600 inches) keeps the type consistent from slide to slide.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
DPI = 600
INK = "17171A"

EQUATIONS = {
    "kmeans": r"\min_{\{C_k\},\,\{\mu_k\}}\ \sum_{k=1}^{K}\ \sum_{x_i\in C_k}\ \lVert x_i-\mu_k\rVert_2^{2}",
    "pipeline": r"z_i=f_\theta(x_i),\qquad \hat c_i=\arg\min_{k}\ \lVert z_i-\mu_k\rVert_2",
    "dec_q": r"q_{ij}=\frac{\left(1+\lVert z_i-\mu_j\rVert^{2}\right)^{-1}}"
             r"{\sum_{j'}\left(1+\lVert z_i-\mu_{j'}\rVert^{2}\right)^{-1}}",
    "dec_p": r"p_{ij}=\frac{q_{ij}^{2}/f_j}{\sum_{j'}q_{ij'}^{2}/f_{j'}},\qquad f_j=\sum_i q_{ij}",
    "dec_loss": r"L_{\mathrm{DEC}}=\mathrm{KL}(P\,\Vert\,Q)=\sum_i\sum_j p_{ij}\log\frac{p_{ij}}{q_{ij}}",
    "idec_loss": r"L_{\mathrm{IDEC}}=\mathrm{KL}(P\,\Vert\,Q)"
                 r"+\lambda\,\frac{1}{n}\sum_i\lVert x_i-g_\phi(z_i)\rVert_2^{2}",
    "vade_gen": r"c\sim\mathrm{Cat}(\pi),\qquad z\mid c\sim\mathcal{N}\!\left(\mu_c,\ \sigma_c^{2}I\right),"
                r"\qquad x\mid z\sim p_\phi(x\mid z)",
    "vade_elbo": r"\log p(x)\ \ge\ \mathbb{E}_{q(z,c\mid x)}\!\left[\log p_\phi(x\mid z)\right]"
                 r"-\mathrm{KL}\!\left(q(z,c\mid x)\,\Vert\,p(z,c)\right)",
    "nt_xent": r"\ell_i=-\log\frac{\exp\!\left(\mathrm{sim}(h_i,h_i^{+})/\tau\right)}"
               r"{\sum_{k\neq i}\exp\!\left(\mathrm{sim}(h_i,h_k)/\tau\right)}",
    "jaccard": r"J=\frac{1}{K}\sum_{k=1}^{K}\frac{\lvert A_k\cap B_k\rvert}{\lvert A_k\cup B_k\rvert}",
}

TEMPLATE = r"""\documentclass[border=2pt]{standalone}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{exscale}
\usepackage{xcolor}
\definecolor{ink}{HTML}{%s}
\begin{document}
\fontsize{22}{26}\selectfont\color{ink}$\displaystyle %s$
\end{document}
"""


def render(name: str, body: str, workdir: Path) -> None:
    tex = workdir / f"{name}.tex"
    tex.write_text(TEMPLATE % (INK, body), encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=workdir,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = ASSETS / f"eq_{name}"
    subprocess.run(["pdftocairo", "-png", "-r", str(DPI), "-transp", "-singlefile",
                    str(workdir / f"{name}.pdf"), str(out)], check=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for old in ASSETS.glob("eq_*.png"):
        old.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in EQUATIONS.items():
            render(name, body, Path(tmp))
    print(f"{len(EQUATIONS)} equations -> {ASSETS}")


if __name__ == "__main__":
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is required")
    main()
