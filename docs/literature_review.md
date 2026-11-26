# 怎样为一张没有标签的图片找到同伴

把靴子和凉鞋放在一起，可以依据鞋的用途，也可以依据轮廓；把深色球鞋和深色上衣放在一起，则可能只是依据颜色。数据给出了图片，研究者还要说明“相似”对应任务中的哪种关系。深度聚类的各条路线给出了不同的建模答案。文献条目的作者、年份和出版信息见[参考文献库](bibliography.bib)。

## 从固定距离到可学习的相似性

教材中的 K-means 接受给定坐标，以到簇中心的平方距离决定分配。先用 PCA 降维，是在相同分组方法前更换坐标。自编码器将线性投影推广为可学习的非线性编码，训练信号来自重构输入。重构得好的坐标可能精细记录了背景，语义相近的照片仍可能相隔很远。这解释了为什么要实际比较像素、PCA 和 AE 三条基线。[教材目录](https://www.sufep.com/e/action/ShowInfo.php?classid=31&id=19670)

DEC 让分组结果回过头来更新表示。初始表示决定初始簇中心，模型再强化较有把握的软分配。IDEC 在更新时要求保留重构能力，使压紧某些簇与保留输入信息形成可观测的权衡。VaDE 把簇建模为隐空间中的混合成分，同时优化生成与推断。它多提供了一种解释簇的方式；实际 ARI、重构误差和耗时仍须分别测量。[DEC](https://proceedings.mlr.press/v48/xieb16.html) · [IDEC](https://doi.org/10.24963/ijcai.2017/243) · [VaDE](https://doi.org/10.24963/ijcai.2017/273)

## 当同一张图有两个视图

DeepCluster 轮流聚类和更新特征；前一轮的簇编号充当下一轮的临时标签。IIC 最大化成对视图分配的互信息。SwAV 让一个视图预测另一个视图的原型分配。三者都利用无标签数据构造训练信号，但信号的单位各异，分别是离线分配、成对分配关系与跨视图原型。[DeepCluster](https://arxiv.org/abs/1807.05520) · [IIC](https://arxiv.org/abs/1807.06653) · [SwAV](https://arxiv.org/abs/2006.09882)

SimCLR 从同一图像的增强视图学习表征，分组是本课程另接的 K-means 操作。CC 同时进行实例级与簇级对比，预测头直接产生簇分配。二者都依赖“增强后仍保留类别线索”这一假设。本项目用了小型 CNN，课程版 CC 有两项对比损失，省略官方实现的簇熵项。这些差别限制了与论文结果的横向比较。[SimCLR](https://proceedings.mlr.press/v119/chen20j.html) · [CC](https://ojs.aaai.org/index.php/AAAI/article/view/17037) · [CC 官方损失实现](https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/modules/contrastive_loss.py)

## 当已有模型带来外部知识

DINOv2 通过大规模视觉自监督学习表示。CLIP 用图文配对学习对齐表示。本项目冻结两种视觉编码器，用同一批自然图片比较不同分组器。即使本地没有动过编码器的权重，模型在课程开始之前已经接触过大量外部数据。特征提取时间、聚类耗时和历史训练成本因此分别解释。[DINOv2](https://arxiv.org/abs/2304.07193) · [CLIP](https://proceedings.mlr.press/v139/radford21a.html)

TAC 的论文题名为 *Image Clustering with External Guidance*。它从 WordNet 的名词中选择有判别力的词语，以跨模态邻域信息辅助图像聚类。课程里使用的 CLIP 图像特征没有执行这一步。Hu 等人的 VLM4Cluster（2026 年预印本）在 20 个数据集上比较 17 种方法，把效果、分布漂移、扰动和成本纳入评价。我们的三个数据集和高斯噪声实验只覆盖其中一部分问题。[TAC，ICML 2024](https://proceedings.mlr.press/v235/li24aa.html) · [VLM4Cluster 预印本](https://arxiv.org/abs/2609.36648)

Zhou 等人的综述按表征模块和聚类模块的交互组织方法；Hou 等人的 2026 年图像聚类综述提供图像场景的补充背景。本项目用前者组织机制，用预算、先验来源及扰动条件组织选型。[Zhou 等](https://doi.org/10.1145/3689036) · [Hou 等](https://doi.org/10.1016/j.patcog.2025.112590)
