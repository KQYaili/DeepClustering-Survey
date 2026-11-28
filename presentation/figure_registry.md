# 图表来源

实验图由 `scripts/analyze.py` 调用 `src/visualization/figures.py` 从 `results/tables` 生成，尺寸与它们在幻灯片上的位置一致。插图由 `scripts/render_illustrations.py` 从数据集和缓存特征生成，全部 23 组公式由 `scripts/render_equations.py` 用 LaTeX 600 dpi 透明底排版。概念示意图与逐模型分层网络架构图在 `scripts/build_pptx.py` 中用 PowerPoint 原生形状绘制，可以直接编辑。

| 文件 | 内容 | 数据来源 | 页 |
| --- | --- | --- | --- |
| `presentation/assets/cover_mosaic.png` | CIFAR-10 测试图按 DINOv2 表示的 t-SNE 排列 | 种子 0 测试切分、DINOv2 特征缓存 | 1 |
| `presentation/assets/pixel_*.png`、`pixel_distance.yaml` | T 恤、右移 4 像素的同一图、最近的另一类图；1000 张 T 恤的统计 | Fashion-MNIST 训练集 | 8 |
| `presentation/assets/eq_*.png`（共 23 张） | 流程与 K-means/PCA/GMM/DEC/IDEC/VaDE/SimCLR/CC/余弦等价与 ACC/NMI/ARI/Jaccard 原始 LaTeX 公式 | `scripts/render_equations.py` | 2, 8–12, 15, 16, 19, A05 |
| `presentation/assets/strip_*.png` | 三个数据集的样例图 | 各数据集训练集 | 17 |
| `results/figures/benchmark_ari.png` | 9 种从零训练方法在三个数据集上的 ARI | `e01_summary.csv` | 20 |
| `results/figures/embeddings_tsne.png` | PCA、AE、DEC、IDEC 表示空间的 t-SNE | `results/raw/embeddings/` | 21 |
| `results/figures/representation_vs_clusterer.png` | 5 种编码器 × 3 种聚类器 × 3 个数据集 | `e04_summary.csv` | 22 |
| `presentation/assets/mosaic_pixels.png`、`mosaic_dinov2.png` | 同一批 300 张图在像素空间与 DINOv2 空间的排列 | 种子 0 测试切分、DINOv2 特征缓存 | 23 |
| `results/figures/noise_robustness.png`、`presentation/assets/noise_*.png` | 干净与加噪测试图上的 ARI；噪声示例 | `e03_noise.csv` | 24 |
| `results/figures/sample_size.png` | 训练样本量与 ARI | `e03_sensitivity_summary.csv` | 25 |
| `results/figures/components_lambda.png` | IDEC 组件消融；λ 在早停与固定 50 轮下的对照 | `e02_summary.csv`、`e02_fixed_epochs.csv` | 26 |
| `results/figures/cost_frontier.png` | 端到端训练时间与 ARI 的帕累托前沿 | `benchmark_cost_frontiers.csv` | 27 |
| `results/figures/stability_vs_ari.png` | 无标签稳定性与 ARI | `selection.json`、`e01_summary.csv` | 28 |
| `results/figures/latent_dim.png` | 隐空间维度消融 | `e02_summary.csv` | A02 |
| `results/figures/sensitivity_k_outliers_imbalance.png` | 簇数、噪声图比例、类别不平衡 | `e03_sensitivity_summary.csv` | A03 |
