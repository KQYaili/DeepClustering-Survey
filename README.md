# Deep Clustering: A Survey of Paradigm Evolution, Empirical Benchmarks, and Method Selection

深度聚类：范式演进、实验基准与方法选型综述。

这是一份课程研究仓库。它回答的问题是：在不同的数据结构、计算预算和先验条件下，什么时候用经典聚类、联合优化、生成式聚类、对比聚类或预训练特征。

## 当前状态

实验队列在 WSL2 里跑，机器是 RTX 5070 Ti Laptop（12 GB）。协议写在 `configs/protocol.yaml`，并且在看到任何测试集指标之前已经固定。测试集标签只在 `src/evaluation/metrics.py` 里使用。没有填过预期数字。

仓库里的表是队列中途的快照。Fashion-MNIST 上 K-means、PCA + K-means、GMM、AE + K-means、DEC、IDEC 已各有 3 个种子。MNIST、对比学习、CIFAR-10 和后面的消融还在跑，那些格子保持空白。

`scripts/run_queue.py --dry-run` 会列出全部作业。已完成的指标在 `results/raw/`，汇总表和图由 `scripts/analyze.py` 从这些 JSON 生成。失败的作业保留错误信息，不会被换一个种子再挑。

## 怎么跑

在 WSL 里：

```bash
bash scripts/setup_wsl.sh   # 只补 umap 和 open_clip，不替换已有的 CUDA 版 PyTorch
bash scripts/launch_wsl.sh  # 先跑 tests/test_core.py，再跑队列、噪声评估和作图
```

本机使用 `/home/lgd/anaconda3/bin/python`（PyTorch 2.14 + CUDA 13.2）。数据缓存在 `~/.cache/dc-survey/data`，不放进 Git。

只看作业清单：

```bash
python scripts/run_queue.py --dry-run
```

## 仓库里什么是证据

| 路径 | 内容 |
| --- | --- |
| `results/raw/**/*.json` | 一次运行的指标、耗时、训练损失和状态 |
| `results/tables/` | 由原始 JSON 汇总的表，不手改 |
| `results/figures/` | 由同一批 JSON 画的图 |
| `results/checkpoints/` | 模型权重，不提交 |
| `presentation/DeepClustering-Survey.pptx` | 汇报。结果页只引用已经写进表里的数 |

## 协议里需要先知道的三件事

重构项是每张图的平方误差之和，再对 batch 取平均，不是逐像素平均。IDEC 写成 `L_cluster + λ L_recon`。这不是 Guo 等人 2017 年论文里的 `γ`：他们把权重放在聚类项上。

28×28 的自编码器是 `784 → 500 → 500 → 2000 → 10`。CIFAR-10 用单独的卷积神经网络，不用这条全连接结构。

SimCLR 和 Contrastive Clustering 用的是小型 CNN，不是原论文的 ResNet。表里的数字是这条课程实现的结果，不拿去和论文报告的准确率并列。

DINOv2 和 CLIP 是冻结的外部预训练模型。它们和从零训练的模型不放在同一张“公平排行榜”里。
