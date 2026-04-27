# bigdata-HM

《大数据分析与计算》综合实战项目仓库（H&M 推荐赛题）。

本仓库当前阶段定位：
- 已完成并提交任务书中的第一阶段（期中考核，20%）
- 第二阶段已推进到 baseline+，并完成“生成层/评估层”职责拆分：
  - `Final_Project.py` 仅负责生成提交文件 `outputs/submission.csv`
  - `Final_Project_Eval.py` 独立负责离线评估与调参
- 第三阶段（答辩与贡献，15%）尚未在本仓库完成

## 1. 任务书评分结构对齐

根据任务书（Project Rubric）实践板块，总计 70%：
1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码、架构、可解释性
3. Kaggle 打榜（15%）：MAP@12 竞赛结果与证明材料
4. 答辩与贡献度（15%）：GitHub 提交记录与答辩表现

当前仓库仅完整覆盖第 1 项（期中考核）。
当前第 2 项已完成 baseline+ 工程实现（可运行脚本、离线 MAP@12 评估、提交文件生成与格式校验），并完成生成层与评估层解耦。

## 2. H&M 赛题关键规则（官方信息提炼）

以下内容来自 Kaggle 官方介绍，仅保留本项目实施必须信息，不包含排名奖励与时间限制：

1. 任务目标
   - 基于历史交易数据（transactions）与客户、商品元数据（customers/articles）做个性化商品推荐。
   - 元数据可包含结构化字段，也可扩展到文本与图像信息；方法路线不限。

2. 评估指标
   - 竞赛采用 `MAP@12`（Mean Average Precision @ 12）。
   - 需要对每位客户给出最多 12 个推荐结果，按顺序计算命中质量。
   - 评分时只统计测试期内确实发生购买行为的客户，但提交时仍需覆盖全部给定客户。
   - 对于真实购买不足 12 件的客户，提交满 12 条预测不会产生额外惩罚。

3. 提交要求
   - 提交文件必须是 CSV，且包含表头：`customer_id,prediction`。
   - `prediction` 字段为以空格分隔的 `article_id` 序列（最多 12 个）。
   - 预测目标是“训练窗口之后下一个 7 天内”的潜在购买商品。
   - 需要对给定 `customer_id` 列表中的所有客户产出预测，不能只覆盖训练集中活跃客户。

4. 实现注意事项
   - `customer_id` 与 `article_id` 应按字符串处理，避免前导零丢失。
   - 输出前建议做格式校验：列名、分隔符、每行最多 12 个 `article_id`。

## 3. 当前已完成内容（第一阶段）

已完成并可复现：
1. 数据可视化探索与业务洞察（对应期中 10 分）
2. 数据清洗规则制定与特征规划（对应期中 10 分）

对应交付物：
- `EDA_Checkpoint.ipynb`
- `doc_images/` 中的图表文件
- `task.md` 中的执行映射与状态记录

## 4. 仓库结构

- `EDA_Checkpoint.ipynb`：期中 EDA 与清洗方案 notebook
- `Final_Project.py`：第二阶段生成层脚本，仅负责输出 `outputs/submission.csv`
- `Final_Project_Eval.py`：第二阶段评估层脚本，负责离线评估与权重调参
- `Final_Project.ipynb`：第二阶段展示型 notebook（已对齐“生成层/评估层”拆分）
- `doc_images/`：期中阶段关键图表
- `task.md`：任务书要求映射与当前完成状态
- `AI_Assistant_Memo.md`：给后续智能体的接手规范与工作流程
- `requirements.txt`：当前 notebook 运行依赖
- `大作业考核任务书(Project_Evaluation).pdf`：课程任务书原文

## 5. 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 打开 notebook

```bash
jupyter notebook EDA_Checkpoint.ipynb
```

第二阶段生成层（提交文件生成）：

```bash
python Final_Project.py
```

3. 第二阶段评估层（离线评估与调参，耗时较长）：

```bash
python Final_Project_Eval.py
```

4. 修改 `BASE_PATH` 指向本地数据目录后运行；`Final_Project.py` 默认输出到仓库内 `outputs/submission.csv`（文件名固定不变）。

5. 仓库内已包含课程任务书 PDF；如需在 notebook 或脚本里读取 PDF 文本，请先安装依赖并使用 `pypdf`。

## 6. 依赖说明

- numpy
- polars
- plotly
- matplotlib
- jupyter
- pypdf

## 7. 阶段状态

- [x] 第一阶段：期中 EDA 与清洗规则
- [ ] 第二阶段：Final_Project 流水线、K-Fold、XAI、Submission（已完成 baseline+：5 折时间窗口 MAP@12 评估 + 三路召回 + 提交格式校验）
- [ ] 第三阶段：Kaggle 公开榜证明与答辩材料

## 8. 说明

本仓库当前内容用于体现“已完成考核任务（期中）”的代码与材料，不代表最终全量大作业已经完成。
