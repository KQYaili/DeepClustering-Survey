# Deep Clustering: A Survey of Paradigm Evolution, Empirical Benchmarks, and Method Selection

深度聚类：范式演进、实验基准与方法选型综述。

这是一份面向课程汇报的可复现实验仓库。它回答一个实际问题：当数据结构、计算预算和先验条件不同，应该优先尝试经典聚类、联合优化、生成式聚类、对比聚类，还是外部预训练特征。

## 当前状态

协议先写入 `configs/protocol.yaml`，再读取测试集指标。当前 `results/raw` 有 809 条成功 JSON 记录，包含 E01 基准、E02 消融、E03 敏感性、E03 噪声和 E04 表征/聚类器比较。E05 是训练侧稳定性诊断，结果以 `results/raw/E05_train/*.npz` 保存，不把测试标签送入拟合或选型。

本轮协议修复以 `8eb43fc` 为前一状态。受影响的旧文件保存在 `results/history/pre-repair-8eb43fc/`，修复说明、来源哈希和运行日志保存在 `results/provenance/protocol-repair-20261126/`。其中修复了 SimCLR 初始化顺序和小样本 E03 的 epoch 追加逻辑，并按原 seed 重跑受影响记录。一次失败的 CIFAR-10 GMM 运行保留在 `results/attempts/`，随后按同一 seed 只重试一次。

汇总表和图全部由 `scripts/analyze.py` 从 `results/raw` 生成。`results/tables/failures.csv` 保留失败记录，不能用换 seed 的方式掩盖失败。模型权重、外部预训练缓存和完整运行归档不进入 Git，另存于 Google Drive 的 `DeepClustering-Survey` 文件夹。

## 怎么跑

在 WSL2 中：

```bash
bash scripts/setup_wsl.sh
bash scripts/launch_wsl.sh
```

`launch_wsl.sh` 会先跑核心测试，再按队列、噪声评估和分析脚本执行。只查看队列而不运行：

```bash
python scripts/run_queue.py --dry-run
```

当前实验使用本机 WSL2 的 RTX 5070 Ti Laptop GPU。数据缓存在 `~/.cache/dc-survey/data`，不放进 Git。

## 证据目录

| 路径 | 内容 |
| --- | --- |
| `results/raw/**/*.json` | 每次运行的指标、训练轨迹、耗时和状态 |
| `results/tables/` | 由原始 JSON 生成的 CSV/JSON 汇总 |
| `results/figures/` | 与汇总表对应的图 |
| `results/raw/E05_train/*.npz` | 不看测试标签的训练稳定性诊断 |
| `results/checkpoints/` | 模型权重，提交到 Drive 归档 |
| `results/history/` | 协议修复前的历史结果备份 |
| `results/provenance/` | 协议、代码哈希和修复运行日志 |
| `presentation/DeepClustering-Survey.pptx` | 24 页面向零背景听众的汇报 |
| `presentation/speaker_notes.md` | 对应的逐页讲稿 |

## 读结果前先知道

- `fit` 不接收标签。测试集标签只在评价步骤加载。类别不平衡子集使用训练集标签构造，这是实验设计，不是损失函数或选种子。
- 早停只观察训练集簇分配变化。追加 epoch 只在训练损失继续下降时触发，两个阈值固定在 `configs/protocol.yaml`。
- 重构项是每张图的平方误差和再对 batch 取平均。IDEC 写成 `L_cluster + λ L_recon`，这里的 λ 不能直接当作 Guo 等人论文中的 γ。
- 28×28 输入的自编码器是 `784 → 500 → 500 → 2000 → 10`。CIFAR-10 使用单独的卷积网络。
- SimCLR 和 Contrastive Clustering 使用课程版小型 CNN，不是原论文的 ResNet。课程版 CC 省略了原论文的熵项，因此只在本仓库实现内部比较。
- VaDE 的重构误差按图计算平方误差，不是 Bernoulli 交叉熵。GMM 使用对角协方差，并在聚类前对特征做标准化。
- DINOv2 和 CLIP 是冻结的外部预训练模型。它们与从零训练的模型分开展示，不能解释成同一公平排行榜。

## 汇报入口

直接打开 `presentation/DeepClustering-Survey.pptx`。图表来源可以沿着每页脚注回到 `results/tables/` 和 `results/figures/`，方法、数据和限制说明见 `docs/experiment_protocol.md`、`docs/taxonomy.md` 与 `docs/literature_review.md`。
