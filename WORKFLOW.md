# 执行顺序

| 阶段 | 命令 | 通过条件 |
| --- | --- | --- |
| 0 | `python tests/test_core.py` | 指标、损失、选型规则的单元测试通过 |
| 1 | `python scripts/run_queue.py` | 每个作业写出 `ok` 或 `failed` JSON；失败保留 |
| 2 | `python scripts/eval_noise.py` | 只对图像模型的已有权重做测试期加噪 |
| 3 | `python scripts/analyze.py` | `results/tables` 与 `results/figures` 只含测到的格子 |
| 4 | `python scripts/build_pptx.py` | 结果页读取 CSV；没有的格子保持空白 |
| 5 | 打开 PowerPoint 后用 `scripts/ppt_com.py` 替换图片和数字 | 文件处于打开状态时不直接覆写 pptx |

队列大约 185 个作业，按优先级先跑 MNIST 与 Fashion-MNIST 的经典方法和 DEC/IDEC，再跑对比学习、CIFAR 和 E04。E03 的全量、平衡、无异常、K=10 不重复训练，作图时用 E01 的对应行。

噪声实验不重新训练。不平衡实验会用训练集标签构造子集，评价仍在干净测试集上。

Colab CLI 装在 WSL 的 `~/.local/bin/colab`。本地 12 GB 显存放得下当前网络时不用它。只有某个作业因为显存失败，才把该作业单独送到 Colab，并在 JSON 里记下运行位置。
