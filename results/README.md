# 结果目录说明

先看 `tables/` 和 `figures/`。这些文件由 `scripts/analyze.py` 从 `raw/` 生成，是汇报和复核时最适合直接阅读的入口。

## 人类阅读入口

| 目录或文件 | 用途 |
| --- | --- |
| `tables/e01_summary.csv` | 三个数据集上的基础方法比较 |
| `tables/e02_ablation.csv` | 重构权重、隐空间和初始化组件消融 |
| `tables/e03_noise.csv` | 测试期噪声鲁棒性 |
| `tables/e03_sensitivity.csv` | 不平衡、样本量、聚类数和异常点敏感性 |
| `tables/e04_paired.csv` | 表征与聚类器解耦比较 |
| `tables/benchmark_cost_frontiers.csv` | 带测试标签的回顾性端到端成本基准 |
| `tables/selection.json` | 训练侧稳定性与坍缩风险诊断 |
| `tables/failures.csv` | 失败运行和保留的 traceback |
| `figures/` | 由上述表格自动生成的科研图 |

## 原始记录命名

`raw/` 中的 JSON 文件名按数据集、方法、seed、聚类数、隐空间维度和协议开关组成。它们服务于脚本批量读取和重复实验，阅读时使用上面的汇总表即可。原始文件会保留 `ok` 与 `failed` 状态，失败记录不被删除或改写成模拟数字。

`raw/E05_train/` 保存不读取测试标签的训练侧稳定性诊断。`raw/embeddings/` 保存少量用于离线重画嵌入图的测试嵌入。

`history/before-protocol-repair/` 是协议修复前的完整备份。`provenance/protocol-repair-2026-11-26/` 保存修复说明和运行日志。完整模型权重与缓存位于 Google Drive 归档，不进入 Git。

