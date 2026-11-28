# 汇报结构

正文 29 页，约 28 分钟；附录 6 页用于答疑。全篇按 **“统计基础 → 模型架构与原始数学推导 → 数据集与实验设计 → 各实验结果对比 → 结论与选型”** 顺序组织，所有实验结果对比统一放在后半部分。每页只讲一个结论：标题写结论，正文给证据。页面 ID 与 `presentation/speaker_notes.md` 一一对应，构建时校验。

| 页 | 页面 ID | 部分 | 标题（结论） | 证据 |
| --- | --- | --- | --- | --- |
| 1 | `S01_cover` | 封面 | 深度聚类：范式演进、对照实验与方法选择 | CIFAR-10 测试图按 DINOv2 表示排列 |
| 2 | `S02_question` | 引言 | 先改表示，还是先换聚类器？ | 三个可检验的命题与五步汇报路线 |
| 3 | `S03_statistics_roots` | 统计基础 | 深度聚类建立在多元统计的几块基石上 | 教材第 2、3、6、7、8 章与深度聚类的对应 |
| 4 | `S04_pixel_distance` | 统计基础 | K-means 在像素空间里量错了距离 | 平移 4 像素的 T 恤与另一类图像的像素距离 |
| 5 | `S05_core_idea` | 统计基础 | 深度聚类：先学一个让距离有意义的表示 | 编码器—表示—聚类流程与两个设计问题 |
| 6 | `S06_paradigms` | 方法谱系 | 表示与聚类的四种配合方式长期并存 | 多阶段、联合优化、生成式、迭代式与外部预训练 |
| 7 | `S07_dec` | 方法谱系 | DEC：用自己最有把握的预测训练自己 | 两阶段网络架构、软分配、目标分布与平方锐化示意 |
| 8 | `S08_idec` | 方法谱系 | IDEC：保留解码器，防止表示被聚类目标拉坏 | 对称自编码器 + 双分支联合优化架构（拔河比喻） |
| 9 | `S09_vade` | 方法谱系 | VaDE：把高斯混合模型搬进隐空间 | 变分编码器 + 重参数采样 + 隐空间 GMM 先验架构与 ELBO |
| 10 | `S10_contrastive` | 方法谱系 | 对比学习：同一张图的两张变形照应当靠近 | 共享 SmallCNN 孪生架构、实例/簇双对比头与 NT-Xent |
| 11 | `S11_pretrained` | 方法谱系 | 预训练模型把“学表示”交给了海量外部数据 | 冻结 ViT-S/14 & ViT-B/32 + L2 归一 + 三种下游聚类器架构 |
| 12 | `S11b_architectures` | 方法谱系 | 各模型网络架构总览：从像素统计到预训练骨干 | 5 大家族 11 种模型的分层结构、目标函数与实测参数量对照 |
| 13 | `S11c_deriv_classical_dec` | 原始数学推导 | 原始数学推导（一）：经典统计基础与 DEC / IDEC 梯度 | K-means 抗噪分解、PCA 等价、GMM 像素塌缩、DEC Eq.(4)–(5) 梯度与 IDEC λ=1/(γD) |
| 14 | `S11d_deriv_vade_contrast` | 原始数学推导 | 原始数学推导（二）：VaDE 变分下界与对比学习 / 余弦几何 | VaDE 闭式后验 q*(c\|x)=p(c\|z) 与 KL、InfoNCE 互信息下界、CC 熵项边界与 L2 余弦等价 |
| 15 | `S12a_datasets` | 数据与实验设计 | 三个基准数据集构成从灰度笔画到自然图像的难度阶梯 | MNIST / Fashion-MNIST / CIFAR-10 样例条、维度划分、10 类全称与难度梯度 |
| 16 | `S12_setup` | 数据与实验设计 | 七组对照实验（E01–E05）：基准、消融、鲁棒性与特征分解 | 5 家族色标、E01–E05 动态记录规模矩阵与 4 类受控扰动构造 |
| 17 | `S12b_metrics_rules` | 数据与实验设计 | 评价指标定义与四项防泄密实验铁律 | ARI / ACC / NMI / 宏观 Jaccard 精确公式、Silhouette 边界与四项硬协议 |
| 18 | `S13_benchmark` | 实验结果对比 | 从零训练：DEC 与 VaDE 领先，CIFAR-10 上全部失效 | `results/figures/benchmark_ari.png` |
| 19 | `S14_geometry` | 实验结果对比 | 非线性表示先把类拉开，DEC 再把每个簇压紧 | `results/figures/embeddings_tsne.png` |
| 20 | `S15_representation_vs_clusterer` | 实验结果对比 | 换表示与换聚类器带来的 ARI 变化（数值随结果表更新） | `results/figures/representation_vs_clusterer.png` |
| 21 | `S16_pixels_vs_dinov2` | 实验结果对比 | 同一批 CIFAR-10 图像：像素空间按颜色排，DINOv2 空间按物体排 | `presentation/assets/mosaic_*.png` |
| 22 | `S17_noise` | 实验结果对比 | 加噪后排名反转 | `results/figures/noise_robustness.png` |
| 23 | `S18_sample_size` | 实验结果对比 | 训练样本很少时，联合优化不如 K-means | `results/figures/sample_size.png` |
| 24 | `S19_components` | 实验结果对比 | IDEC 组件消融与重构权重 λ | `results/figures/components_lambda.png` |
| 25 | `S20_cost` | 实验结果对比 | 精度与训练成本 | `results/figures/cost_frontier.png` |
| 26 | `S21_stability` | 实验结果对比 | 没有标签时，“稳定”选不出“正确” | `results/figures/stability_vs_ari.png` |
| 27 | `S22_verdict` | 结论与选型 | 三个命题的回答 | 关键数值 |
| 28 | `S23_selection_guide` | 结论与选型 | 方法选择：先问表示从哪里来，再问数据和预算 | 选择流程 |
| 29 | `S24_limits_and_close` | 结论与选型 | 局限与展望 | 仓库链接与二维码 |

## 附录

| 页 | 页面 ID | 内容 |
| --- | --- | --- |
| A01 | `A01_benchmark_table` | 从零训练基准：三个数据集的 ACC、NMI、ARI |
| A02 | `A02_ablations` | IDEC 隐空间维度；λ 在早停与固定 50 轮下的对照 |
| A03 | `A03_sensitivity` | 簇数设错、训练集混入噪声图、类别不平衡 |
| A04 | `A04_encoder_clusterer_table` | 编码器 × 聚类器完整数值 |
| A05 | `A05_stability_table` | 无标签稳定性诊断完整记录 |
| A06 | `A06_references` | 参考文献 |

## 视觉规则

- 颜色只表示方法家族，全篇一致：灰色经典方法，蓝色自编码器系，紫色生成式，绿色对比学习，橙色外部预训练。文字一律中性色。
- 字号：标题 27 pt，正文 10–20 pt，图中文字不小于 11 pt；图表按幻灯片上的实际尺寸绘制。
- 所有数字在构建时从 `results/tables` 读取，讲稿中的 `{名称}` 占位符同样在构建时填入。
