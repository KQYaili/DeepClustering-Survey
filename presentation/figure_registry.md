# 图表登记表（Figure Registry）

所有**概念架构与原理图（Fig.1–Fig.6）**由 `scripts/render_diagrams.py` 生成；所有**实证数据图（Fig.7–Fig.16b）**严格由 `scripts/analyze.py` 从 `results/raw/` 计算并调用 `src/visualization/benchmark.py` 在 300 DPI 下生成；**选型决策树（Fig.17）**动态读取 `results/tables/` 实测统计量。仓库与 PPT 中不出现任何模拟或手填结果数字。

| 编号 | 图表名称 | 类型 / 实验编号 | 文件路径 | 数据源 / 生成脚本 | 所在幻灯片 (Slide ID) |
| --- | --- | --- | --- | --- | --- |
| Fig.1 | 深度聚类范式演进谱系图 | 概念图（文献综合） | `presentation/assets/diag_paradigm_evolution.png` | `scripts/render_diagrams.py`（Zhou 2025 图 1 四类交互） | 第 6 页 (`S06_paradigm_evolution`) |
| Fig.2 | 四维方法分类框架图 | 概念图（理论整理） | `presentation/assets/diag_taxonomy_4d.png` | `scripts/render_diagrams.py`（对齐 `docs/taxonomy.md`） | 第 7 页 (`S07_taxonomy_4d`) |
| Fig.3 | DEC 与 IDEC 原理对照架构图 | 概念图（数学原理） | `presentation/assets/diag_dec_idec.png` | `scripts/render_diagrams.py` | 第 9 页 (`S09_idec_math`) |
| Fig.4 | VaDE 深度生成式聚类架构图 | 概念图（数学原理） | `presentation/assets/diag_vade_arch.png` | `scripts/render_diagrams.py` | 第 10 页 (`S10_vade_arch`) |
| Fig.5 | SimCLR 与 CC 双视图对比原理图 | 概念图（数学原理） | `presentation/assets/diag_contrastive_arch.png` | `scripts/render_diagrams.py` | 第 11 页 (`S11_contrastive_arch`) |
| Fig.6 | 预训练特征与聚类器解耦框架图 | 概念图（E04 设计） | `presentation/assets/diag_pretrained_decouple.png` | `scripts/render_diagrams.py` | 第 12 页 (`S12_pretrained_frontier`) |
| Fig.7 | E01 多数据集 9 方法 ARI 热力图（±SD 与近似并列） | 实证数据图（E01，n=3） | `results/figures/fig_e01_ari.png` | `scripts/analyze.py` ← `e01_summary.csv` | 第 15 页 (`S15_e01_benchmark`) |
| Fig.8 | E01 潜在空间二维 t-SNE 投影图（预测簇上色） | 实证数据图（E01，seed 0） | `results/figures/fig_e01_embeddings.png` | `scripts/analyze.py` ← `results/raw/embeddings/` | 第 16 页 (`S16_e01_embeddings`) |
| Fig.9 | E02 重构权重 λ 消融曲线（含早停 `epochs_ran` 与 `λ=0` 不可比注） | 实证数据图（E02，n=1） | `results/figures/fig_e02_lambda.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 第 17 页 (`S17_e02_lambda_ablation`) |
| Fig.10 | E02 隐空间维度 d 双指标消融曲线（均 50 ep） | 实证数据图（E02，n=1） | `results/figures/fig_e02_latent.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 附录 A2 (`A02_e02_latent_components`) |
| Fig.10b | E02 初始化与微调组件消融条形图 | 实证数据图（E02，n=1） | `results/figures/fig_e02_components.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 附录 A2 (`A02_e02_latent_components`) |
| Fig.11 | E03 测试期高斯噪声鲁棒性曲线（含 VaDE 与当前 GMM 对照） | 实证数据图（E03_noise，n=3） | `results/figures/fig_e03_noise.png` | `scripts/analyze.py` ← `e03_noise.csv` | 第 18 页 (`S18_e03_noise_robustness`) |
| Fig.12 | E03 类别长尾不平衡敏感性曲线（标明实际 `n_train`） | 实证数据图（E03，n=1） | `results/figures/fig_e03_imbalance.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A3 (`A03_e03_imbalance_samples`) |
| Fig.13 | E03 训练样本规模效率曲线 | 实证数据图（E03，n=1） | `results/figures/fig_e03_sample_frac.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A3 (`A03_e03_imbalance_samples`) |
| Fig.14 | E03 预设聚类数 K 错配敏感性曲线 | 实证数据图（E03，n=1） | `results/figures/fig_e03_k.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A4 (`A04_e03_k_outliers`) |
| Fig.14b | E03 异常噪声样本污染鲁棒性曲线 | 实证数据图（E03，n=1） | `results/figures/fig_e03_outlier_frac.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A4 (`A04_e03_k_outliers`) |
| Fig.15 | E04 编码器 × 聚类器 5×3 解耦热力图（n=1 配对，标注 † E01 来源） | 实证数据图（E04/E01，n=1） | `results/figures/fig_e04_paired.png` | `scripts/analyze.py` ← `e04_paired.csv` | 第 19 页 (`S19_e04_pretrained_decouple`) |
| Fig.16 | 回顾性历史基准：性能—端到端训练成本帕累托前沿（非 E05） | 实证数据图（带标签 E01 回顾，n=3） | `results/figures/fig_benchmark_pareto.png` | `scripts/analyze.py` ← `benchmark_cost_frontiers.csv` | 第 20 页 (`S20_benchmark_pareto_e2e`) |
| Fig.16b | E05 负面诊断：三数据集无标签训练稳定性 vs. 历史测试 ARI | 实证数据图（E05 诊断 + E01 对照） | `results/figures/fig_e05_stability.png` | `scripts/analyze.py` ← `selection.json` + `e01_summary.csv` + `e04_pretrained.csv` | 附录 A5 (`A05_e05_stability_negative`) |
| Fig.17 | 基于历史基准的分场景选型决策树与无标签警示 | 综合决策图（动态读表） | `presentation/assets/diag_decision_tree.png` | `scripts/render_diagrams.py` ← `results/tables/` | 第 21 页 (`S21_selection_decision_tree`) |
