# bigdata-HM

《大数据分析与计算》综合实战项目仓库（H&M 推荐赛题）。

本仓库当前阶段定位：
- 已完成并提交任务书中的第一阶段（期中考核，20%）
- 第二阶段（期末工程交付，35%）和第三阶段（答辩与贡献，15%）尚未在本仓库完成

## 1. 任务书评分结构对齐

根据任务书（Project Rubric）实践板块，总计 70%：
1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码、架构、可解释性
3. Kaggle 打榜（15%）：MAP@12 竞赛结果与证明材料
4. 答辩与贡献度（15%）：GitHub 提交记录与答辩表现

当前仓库仅覆盖第 1 项（期中考核）。

## 2. 当前已完成内容（第一阶段）

已完成并可复现：
1. 数据可视化探索与业务洞察（对应期中 10 分）
2. 数据清洗规则制定与特征规划（对应期中 10 分）

对应交付物：
- `EDA_Checkpoint.ipynb`
- `doc_images/` 中的图表文件
- `task.md` 中的执行映射与状态记录

## 3. 仓库结构

- `EDA_Checkpoint.ipynb`：期中 EDA 与清洗方案 notebook
- `doc_images/`：期中阶段关键图表
- `task.md`：任务书要求映射与当前完成状态
- `requirements.txt`：当前 notebook 运行依赖
- `大作业考核任务书(Project_Evaluation).pdf`：课程任务书原文

## 4. 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 打开 notebook

```bash
jupyter notebook EDA_Checkpoint.ipynb
```

3. 修改 `BASE_PATH` 指向本地数据目录后运行。

## 5. 依赖说明

- numpy
- polars
- plotly
- matplotlib
- jupyter

## 6. 阶段状态

- [x] 第一阶段：期中 EDA 与清洗规则
- [ ] 第二阶段：Final_Project 流水线、K-Fold、XAI、Submission
- [ ] 第三阶段：Kaggle 公开榜证明与答辩材料

## 7. 说明

本仓库当前内容用于体现“已完成考核任务（期中）”的代码与材料，不代表最终全量大作业已经完成。
