# 文献

正式条目在 `bibliography.bib`。这里只保留和汇报直接有关的定位。

Zhou 等人（ACM Comput. Surv. 2025, DOI 10.1145/3689036）按表征与聚类的交互分类，不是按年份。Figure 1 的四支是 multi-stage、iterative、generative、simultaneously。

DEC（Xie et al., ICML 2016）用聚类损失更新嵌入。IDEC（Guo et al., IJCAI 2017）加回局部重构，避免只优化 KL 时把已经学到的结构压扁。VaDE（Jiang et al., IJCAI 2017）把混合先验放进变分下界。

DeepCluster（Caron et al., ECCV 2018）和 SwAV（Caron et al., NeurIPS 2020）代表用聚类伪标签或在线分配做表征学习。Contrastive Clustering（Li et al., AAAI 2021）同时做实例对比和簇对比。本仓库用小型 CNN 实现后一种思路，用来跑实验，不声称复现他们的 ResNet 结果。SimCLR（Chen et al., ICML 2020）本身不输出簇，这里只把它当作“先对比、再 K-means”的多阶段对照。

TAC（Li et al., ICML 2024, PMLR 235:27890–27902）用 WordNet 文本做外部引导，论文里的方法名是 Text-Aided Clustering。它出现在演进讨论里，这次没有重训。

DINOv2 和 CLIP 是冻结特征的来源。VLM4Cluster（Hu et al., arXiv:2609.36648，2026-09-29 预印本）在 20 个数据集上比较 17 种方法，并同时报效果、分布偏移和计算。它还没有经过同行评审。我们的 CIFAR 子集实验只回答课程规模下的同一个问题，不重复他们的基准。
