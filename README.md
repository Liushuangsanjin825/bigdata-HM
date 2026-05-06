# bigdata-HM

《大数据分析与计算》综合实战项目仓库（H&M 推荐赛题）。

本仓库当前阶段定位：
- 已完成并提交任务书中的第一阶段（期中考核，20%）
- 第二阶段已推进到“可持续调参 + 可追踪实验”阶段，并完成“生成层/评估层”职责拆分：
  - `Final_Project.py` 负责生成提交文件 `outputs/submission.csv`
  - `Final_Project_Eval.py` 负责离线评估与调参
  - `experiments/` 统一管理每次调参与 Kaggle 成绩记录
- 第三阶段（答辩与贡献，15%）尚未在本仓库完成

## 1. 任务书评分结构对齐

根据任务书（Project Rubric）实践板块，总计 70%：
1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码、架构、可解释性
3. Kaggle 打榜（15%）：MAP@12 竞赛结果与证明材料
4. 答辩与贡献度（15%）：GitHub 提交记录与答辩表现

当前仓库仅完整覆盖第 1 项（期中考核）。
当前第 2 项已完成“多路融合推荐器 + 可持续调参与实验追踪”工程实现（可运行脚本、离线 MAP@12 评估、提交文件生成与格式校验），并完成生成层与评估层解耦。

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
- `Final_Project_LGBM.py`：增强版 LightGBM 排序/分类流水线，负责扩展候选、构造训练集、离线 MAP@12 验证并生成 `outputs/submission.csv`
- `LGBM_Ablation_SHAP_Analysis.ipynb`：LightGBM baseline 的 Top-5 特征消融实验与 SHAP 可解释性分析 notebook
- `LGBM_Ablation_SHAP_Analysis.py`：同上分析的脚本版入口，适合在 Kaggle 直接运行并查看日志
- `homework/HM_LGBM_Ablation_SHAP_Report.md`：消融与 SHAP 分析报告（Markdown 版）
- `homework/HM_LGBM_Ablation_SHAP_Report.pdf`：消融与 SHAP 分析报告（PDF 版）
- `homework/` 下其余文件：已归档的消融表与 SHAP 图（`lgbm_ablation_top5.csv`、`lgbm_gain_importance.csv`、`shap_mean_abs_importance.csv`、`shap_beeswarm.png`、`shap_waterfall_sample0.png`）
- `Final_Project.ipynb`：第二阶段展示型 notebook（已对齐“生成层/评估层”拆分）
- `doc_images/`：期中阶段关键图表
- `task.md`：任务书要求映射与当前完成状态
- `experiments/`：调参与线上分数追踪目录（离线 run 快照 + 总索引）
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

4. 增强版 LightGBM 流水线（推荐在 Kaggle Notebook 上运行，耗时较长）：

```bash
python Final_Project_LGBM.py
```

Kaggle Notebook 中建议先设置数据路径，并将最终提交文件复制到工作目录根部：

```python
import os

os.environ["HNM_DATA_PATH"] = "/kaggle/input/h-and-m-personalized-fashion-recommendations"

!python Final_Project_LGBM.py
!cp outputs/submission.csv /kaggle/working/submission.csv
```

若要快速试跑或控制 Kaggle 运行时间，可在运行前设置环境变量：

```python
%env LGBM_TRAIN_CUSTOMER_CAP=40000
%env LGBM_VALIDATION_CUSTOMER_CAP=30000
%env LGBM_MAX_CANDIDATES_PER_CUSTOMER=60
%env LGBM_N_ESTIMATORS=260
```

冲分时建议先使用“保留 baseline 强候选 + 追加稳健召回”的配置：

```python
%env LGBM_TRAIN_CUSTOMER_CAP=80000
%env LGBM_VALIDATION_CUSTOMER_CAP=60000
%env LGBM_BASELINE_RECALL_TOP=80
%env LGBM_MAX_CANDIDATES_PER_CUSTOMER=100
%env LGBM_N_ESTIMATORS=450
%env LGBM_ENABLE_COLOUR_RECALL=0
%env LGBM_ENABLE_SECTION_RECALL=0
%env LGBM_DROP_NOISY_FEATURES=0
```

若验证发现 `candidate_rank/user_avg_price/user_recency_days` 仍有负贡献，可对比去噪版本：

```python
%env LGBM_DROP_NOISY_FEATURES=1
```

做“一次只加一个变量”的新特征消融时，固定其它参数，只切换：

```python
%env LGBM_FEATURE_EXPERIMENT=base
```

```python
%env LGBM_FEATURE_EXPERIMENT=add_item_pop_ratio_7d_30d
```

```python
%env LGBM_FEATURE_EXPERIMENT=add_item_pop_ratio_30d_all
```

```python
%env LGBM_FEATURE_EXPERIMENT=add_price_diff_user_item
```

每次运行 `python Final_Project_LGBM.py` 后记录日志中的 `LGBM validation ... MAP@12=...`。

完成消融实验与 SHAP 分析时，在 Kaggle 上打开并运行：

```text
LGBM_Ablation_SHAP_Analysis.ipynb
```

也可以直接运行脚本版（通常比 `nbconvert` 执行 notebook 更轻便）：

```python
!cp /kaggle/input/hm-project-code/Final_Project.py /kaggle/working/
!cp /kaggle/input/hm-project-code/Final_Project_LGBM.py /kaggle/working/
!cp /kaggle/input/hm-project-code/LGBM_Ablation_SHAP_Analysis.py /kaggle/working/

%cd /kaggle/working

!pip install -q lightgbm shap
!python LGBM_Ablation_SHAP_Analysis.py
```

该 notebook 会输出：

- `outputs/lgbm_ablation_top5.csv`
- `outputs/shap_beeswarm.png`
- `outputs/shap_waterfall_sample0.png`
- `outputs/shap_mean_abs_importance.csv`

5. 数据目录配置（推荐环境变量）：
   - 推荐设置环境变量 `HNM_DATA_PATH` 指向你本机的数据目录。
   - 也可直接在脚本内调整 `BASE_PATH`。
   - `Final_Project.py` 默认输出到仓库内 `outputs/submission.csv`（文件名固定不变）。

6. 仓库内已包含课程任务书 PDF；如需在 notebook 或脚本里读取 PDF 文本，请先安装依赖并使用 `pypdf`。

7. 若进行新一轮调参，请按 `experiments/README.md` 记录 run 与 Kaggle 分数，保证实验可追踪。

## 6. 依赖说明

- numpy
- polars
- plotly
- matplotlib
- jupyter
- pypdf
- lightgbm
- shap

## 7. 阶段状态

- [x] 第一阶段：期中 EDA 与清洗规则
- [ ] 第二阶段：Final_Project 流水线、K-Fold、XAI、Submission（已完成可持续调参与实验追踪；规则融合 5 折离线 MAP@12 最佳均值 `0.020833`，LGBM 单窗口验证 MAP@12 `0.030605`；已登记官方分数 baseline Public `0.01841` / Private `0.01847`，LGBM Public `0.02679` / Private `0.02717`）
- [ ] 第三阶段：Kaggle 公开榜证明与答辩材料

## 8. 调参实验管理

实验追踪统一使用 `experiments/` 目录：

- `experiments/tuning_runs/`：每次 run 的快照目录。
- `experiments/runs_index.csv`：离线实验总索引（run_id、参数、离线分数、提交哈希）。
- `experiments/kaggle_scores.csv`：手动提交 Kaggle 后填写 Public/Private 分数。

当前最佳离线记录（截至 2026-04-29，最佳 run 产生于 2026-04-27）：

- `run_id`: `run_2026-04-27_001_hybrid_multisource`
- `best_config`: `local_u40_l7_c5_d5_a5_t5.5_g1.2`
- `offline_mean_map12`: `0.020833152816925597`
- `latest_replay_offline_mean_map12`（2026-04-29 复跑）: `0.0207343934751573`
- `lgbm_single_window_map12`（2026-04-29）: `0.030605012565942353`

当前已登记官方成绩（截至 2026-04-29）：

- `run_id`: `run_2026-04-27_001_hybrid_multisource`
- `latest_submission_variant`: `Final_Project_LGBM.py`（延续 task 第16条补充）
- `public_score`: `0.02679`
- `private_score`: `0.02717`
- `submission status`: `after deadline`（late submission）
- `historical baseline score`: Public `0.01841` / Private `0.01847`

## 9. 说明

本仓库当前内容用于体现“已完成考核任务（期中）”的代码与材料，不代表最终全量大作业已经完成。
