# 本轮完成记录

这份记录面向需要阅读、讲解和复现实验的人，按日期说明仓库做过的工作和验证结果。内部运行留痕单独保存在 `results/history/`、`results/provenance/` 和 `results/attempts/`，首页不展开内部校验字段。

## 2026-11-27

### 首页与文件命名

- 重写 GitHub 首页，先放深度聚类范式总览图，再用通俗语言说明研究问题、主要发现、实验规则和阅读顺序。
- 首页去掉提交编号、代码校验串和面向工具的过程描述，改为面向听讲者的导航。
- 汇报文件统一使用 `presentation/deep-clustering-evolution-selection-guide.pptx` 和同名 PDF，文件名直接说明内容。
- 历史备份和协议修复目录使用可读的日期与用途命名，便于人类查找。

### PPT 可编辑化

- LaTeX 公式从 PNG 改成 Cambria Math 原生文本框。公式中的变量、上下标、希腊字母和罗马字符可以直接编辑。
- K-means、DEC、IDEC、VaDE、KL 和帕累托公式改用原生文本，避免分数线、长横线和罗马字符在紧裁切时变短。
- 范式演进、四维分类、DEC/IDEC、VaDE、对比聚类和预训练解耦图改用 PowerPoint 原生文本框、矩形和连接线。图中文字、边框和连接线可以单独移动或修改。
- 实验数据图继续由 `scripts/analyze.py` 从结果表生成，作为完整数据图保留在 PPT 中。
- PPT 采用一页一目的的讲解顺序，逐页备注保存在 `presentation/speaker_notes.md`。

### 实验与文档

- 保留统一协议、标签隔离、早停规则和失败运行记录。
- 汇总表和科研图继续从 `results/raw/` 自动生成，避免在结果目录或 PPT 中手填数字。
- README、实验协议、文献综述、数学基础、分类框架和方法选型文档同步更新，补充实现边界和读者说明。
- Google Drive 继续保存模型权重、外部预训练缓存和完整运行归档，GitHub 保留代码、协议、表格、轻量嵌入、图表和汇报文件。

### 验证

已运行以下检查：

```text
python tests/test_core.py
python tests/test_experiment_execution.py
python scripts/analyze.py
python scripts/build_pptx.py
```

PPT 共 30 页，讲稿按 Slide ID 绑定。公式和主要概念图在导出预览中检查过，长公式和连接线没有再使用紧裁切的公式图片。

## 已知边界

- 实验数据图是由结果表生成的图片，适合保持整张图的坐标轴、图例和统计标记。需要修改数据时，应改原始结果或分析脚本后重新生成。
- SimCLR 和 CC 使用课程版小型 CNN，CC 省略原论文边际簇熵项。VaDE 使用每张图的平方误差，GMM 使用标准化后的对角协方差。
- E02、E03 和 E04 的部分比较使用单个 seed，预训练路线包含外部数据预算。结论只适用于当前实验协议。
