# 执行顺序

| 阶段 | 命令 | 通过条件 |
| --- | --- | --- |
| 0 | `python tests/test_core.py` 和 `python tests/test_experiment_execution.py` | 指标、损失、选型、队列路径和标签隔离测试通过 |
| 1 | `python scripts/run_queue.py` | 每个作业写出 `ok` 或 `failed` JSON，失败保留 traceback |
| 2 | `python scripts/eval_noise.py` | 只对已有权重做测试期加噪，不重训模型 |
| 3 | `python scripts/eval_train_stability.py` | 仅用训练侧预测生成 E05 的 NPZ 诊断 |
| 4 | `python scripts/analyze.py` | 表和图只从 `results/raw` 与 E05 训练诊断生成 |
| 5 | `python scripts/build_pptx.py` | 结果页从 CSV 和图读取，不能手填结果数字 |
| 6 | 打开 PowerPoint 后运行 `scripts/ppt_com.py` 做导出检查 | 通过 COM 检查页数、图片和文字，避免直接覆盖打开中的文件 |

当前 JSON 队列共有 809 条成功记录。E01 使用 3 个 seed。E02 和 E03 的协议配置固定为 seed 0。E04 的同一特征空间聚类器比较按 seed 0 运行，预训练 DINOv2/CLIP 的基准稳定性另有 3-seed 表。E05 是在基准完成后增加的训练侧选择诊断，不能当作独立测试集验证。

E03 的全量、平衡、无异常、K=10 情况复用 E01 对应行，不重复训练。噪声实验把噪声加到测试图并截断回 `[0, 1]`，不在噪声上重新拟合。无标签随机子集的 epoch 预算保持不变，因此样本更少时会看到更多遍数据。类别不平衡子集使用训练集标签构造，评价仍在干净测试集上。

外部模型权重和完整运行归档用 `rclone copy` 追加到 Google Drive，禁止用删除式同步覆盖历史归档。GitHub 只保存代码、协议、表、图和轻量文档，`results/checkpoints/` 与外部模型归档留在 Drive。
