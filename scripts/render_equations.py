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
    # Original mathematical derivations
    "kmeans_noise": r"\mathbb{E}_{\varepsilon}\!\left[\lVert x+\varepsilon-\mu_k\rVert_2^2\right]"
                    r"=\lVert x-\mu_k\rVert_2^2 + D\sigma^2",
    "pca_equiv": r"\min_{W^\top W=I_d}\ \frac{1}{N}\sum_{i}\lVert \tilde{x}_i-WW^\top\tilde{x}_i\rVert_2^2"
                 r"\ \Longleftrightarrow\ \max_{W^\top W=I_d}\ \mathrm{tr}(W^\top S W)",
    "gmm_em": r"\gamma_{ik}=\frac{\pi_k\,\mathcal{N}(z_i\mid\mu_k,\Sigma_k)}"
              r"{\sum_{\ell}\pi_\ell\,\mathcal{N}(z_i\mid\mu_\ell,\Sigma_\ell)},\qquad"
              r"\mu_k'=\frac{\sum_i \gamma_{ik}z_i}{\sum_i \gamma_{ik}}",
    "dec_grad": r"\frac{\partial L_{\mathrm{DEC}}}{\partial z_i}=\frac{\alpha+1}{\alpha}\sum_{j=1}^{K}"
                r"\!\left(1+\frac{\lVert z_i-\mu_j\rVert_2^2}{\alpha}\right)^{-1}(p_{ij}-q_{ij})(z_i-\mu_j)",
    "idec_equiv": r"\frac{1}{\gamma}\!\left(\frac{1}{BD}\sum_{i}\lVert x_i-\hat{x}_i\rVert_2^2+\gamma L_{\mathrm{KL}}\right)"
                  r"=L_{\mathrm{KL}}+\underbrace{\frac{1}{\gamma D}}_{\lambda}\frac{1}{B}\sum_{i}\lVert x_i-\hat{x}_i\rVert_2^2",
    "vade_qstar": r"q^{*}(c\mid x)=p(c\mid z)=\frac{\pi_c\,\mathcal{N}(z\mid\mu_c,\mathrm{diag}(\sigma_c^2))}"
                  r"{\sum_{c'}\pi_{c'}\,\mathcal{N}(z\mid\mu_{c'},\mathrm{diag}(\sigma_{c'}^2))}",
    "vade_kl_closed": r"\mathrm{KL}(q(z\mid x)\Vert p(z\mid c))=\frac{1}{2}\sum_{j=1}^{d}"
                      r"\!\left(\log\sigma_{cj}^2+\frac{\tilde{\sigma}_j^2+(\tilde{\mu}_j-\mu_{cj})^2}{\sigma_{cj}^2}"
                      r"-1-\log\tilde{\sigma}_j^2\right)",
    "infonce_mi": r"I(x^{a};x^{b})\ \ge\ \log(2N-1)-\mathcal{L}_{\mathrm{NT\text{-}Xent}}",
    "cc_entropy": r"\mathcal{L}_{\mathrm{CC}}=\mathcal{L}_{\mathrm{inst}}+\mathcal{L}_{\mathrm{clus}}-H(Y),\quad "
                  r"H(Y)=-\sum_{k=1}^{K}\big[P(y_k^a)\log P(y_k^a)+P(y_k^b)\log P(y_k^b)\big]",
    "l2_cosine": r"\lVert \hat{u}_i-\hat{u}_j\rVert_2^2=\lVert \hat{u}_i\rVert_2^2+\lVert \hat{u}_j\rVert_2^2"
                 r"-2\hat{u}_i^\top\hat{u}_j=2\big(1-\cos(u_i,u_j)\big)",
    # Evaluation metrics
    "metric_acc": r"\mathrm{ACC}=\max_{m\in\mathcal{P}_K}\ \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left[y_i=m(\hat{c}_i)\right]",
    "metric_nmi": r"\mathrm{NMI}(Y,\hat{C})=\frac{2\,I(Y;\hat{C})}{H(Y)+H(\hat{C})}",
    "metric_ari": r"\mathrm{ARI}=\frac{\sum_{j,k}\binom{n_{jk}}{2}-\left[\sum_j\binom{a_j}{2}\sum_k\binom{b_k}{2}\right]/\binom{N}{2}}"
                  r"{\frac{1}{2}\left[\sum_j\binom{a_j}{2}+\sum_k\binom{b_k}{2}\right]-\left[\sum_j\binom{a_j}{2}\sum_k\binom{b_k}{2}\right]/\binom{N}{2}}",
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
