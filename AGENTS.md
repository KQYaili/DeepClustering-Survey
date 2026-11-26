# 实验约束

## 标签

`fit` 不接收标签。测试集标签只允许出现在 `src/evaluation/metrics.py`，以及 `scripts/run_one.py`、`scripts/eval_noise.py` 在 `fit` 返回之后的评价步骤。

类别不平衡子集用训练集标签构造比例。这是实验设计，不是损失函数、早停或选种子。`docs/experiment_protocol.md` 写明了这一点。

早停只看训练集上簇分配的变化。追加 epoch 只看训练损失是否还在下降。两项规则都写在 `configs/protocol.yaml`。看到测试集结果之后不要改这些数。

## 结果

不要把模拟数、预期数或手填数写进 `results/` 或 PPT。图和表只能由 `scripts/analyze.py` 从 `results/raw` 生成。

一次运行失败就保留 `status: failed` 和 traceback。基础设施错误最多再跑一次。不准因为准确率低而换种子。

SimCLR、CC 的网络不是原论文网络。VaDE 的重构是每图平方误差，不是 Bernoulli 交叉熵。写结论时带上这些限制。

## 引用

文献条目以 `docs/bibliography.bib` 为准。Zhou 等人的四类交互是 multi-stage、iterative、generative、simultaneously，来自综述 Figure 1，不是互斥的历史阶段。VLM4Cluster（arXiv:2609.36648）按预印本引用。TAC 是 ICML 2024 的 Text-Aided Clustering，论文题名是 Image Clustering with External Guidance。

## 代码

新算法实现 `predict(x)` 和 `embeddings(x)`，超参数来自 `configs/protocol.yaml` 和作业 JSON，不在脚本里另写一套。改协议必须改 yaml，并在还没有对应测试集结果时改。
