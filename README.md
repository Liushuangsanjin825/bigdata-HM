# bigdata-HM

大数据分析课程作业仓库，主题为探索性数据分析（EDA）。

本仓库已整理为可直接执行的基础版本：
- `EDA_Checkpoint.ipynb` 提供完整 EDA 骨架（含待填写区块）
- `task.md` 提供可提交的任务书模板（按实际执行结果填写）
- `README.md` 说明运行方式、交付要求与质量标准

## 1. 项目目标

- 建立一套可复现的 EDA 分析流程。
- 识别数据中的关键模式、异常与潜在风险。
- 输出结构化结论，为后续建模或业务决策提供依据。

## 2. 仓库结构

- `EDA_Checkpoint.ipynb`：EDA 主笔记本（按章节执行与补充）
- `task.md`：任务书与执行记录
- `requirements.txt`：Python 依赖清单
- `doc_images/`：图表导出目录（报告引用）

## 3. 环境与依赖

### 3.1 推荐环境

- Python 3.9+
- Jupyter Notebook 或 VS Code Notebook
- Windows / Linux / macOS

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 依赖说明

- pandas：数据处理
- numpy：数值计算
- matplotlib：基础绘图
- seaborn：统计可视化
- jupyter：交互式笔记本

## 4. 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动笔记本

```bash
jupyter notebook EDA_Checkpoint.ipynb
```

3. 在笔记本首段设置数据路径 `DATA_PATH`，按顺序运行所有章节。

4. 将关键图表导出到 `doc_images/`，并在 `task.md` 中补充结论。

## 5. EDA 执行章节（对应笔记本）

1. 环境准备与数据加载
2. 数据概览与字段理解
3. 数据质量检查（缺失、重复、类型、异常）
4. 单变量分析（分布、集中趋势、离散程度）
5. 双变量/多变量分析（相关性、分组对比）
6. 总结与后续建议

## 6. 交付要求

- `EDA_Checkpoint.ipynb`：包含完整代码、图表、文字结论
- `task.md`：填写完整任务书（目标、过程、结果、问题与改进）
- `doc_images/`：核心图表文件（建议编号命名）

## 7. 质量检查清单

- [ ] 数据源与字段说明完整
- [ ] 缺失值、重复值、异常值处理策略清晰
- [ ] 每个核心图表都有文字结论
- [ ] 关键结论可回溯到代码与图表
- [ ] `task.md` 与实际执行结果一致

## 8. 维护说明

- 提交前清理无关输出单元，保留关键结果。
- 图表统一放入 `doc_images/`，便于报告引用。
- 若新增依赖，同步更新 `requirements.txt`。

## 9. 许可

本项目用于课程学习与交流，不用于商业用途。
