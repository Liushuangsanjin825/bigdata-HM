# bigdata-HM

《大数据分析与计算》综合实战项目仓库（H&M 推荐赛题）。

本仓库当前阶段定位：
- 已完成并提交任务书中的第一阶段（期中考核，20%），包括 EDA、数据清洗规则和业务洞察。
- 第二阶段已推进到“端到端推荐系统 + 可追踪实验”阶段，并完成“生成层/评估层/增强模型层”职责拆分：
  - `Final_Project.py` 负责生成提交文件 `outputs/submission.csv`
  - `Final_Project_Eval.py` 负责离线评估与调参
  - `Final_Project_LGBM.py` 负责 LightGBM 召回-排序增强流水线
  - `LGBM_Optuna_Tuning.py` 与 `LGBM_MultiFold_Eval.py` 分别负责自动调参和多折时间验证
  - `experiments/` 统一管理每次调参与 Kaggle 成绩记录
- Kaggle 冲分阶段已完成多轮提交验证，当前最佳线上成绩为 Private `0.02881` / Public `0.02875`。
- 答辩与个人简历材料已形成项目历程报告，可用于后续归档、答辩复盘和保研简历包装。

## 1. 任务书评分结构对齐

根据任务书（Project Rubric）实践板块，总计 70%：
1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码、架构、可解释性
3. Kaggle 打榜（15%）：MAP@12 竞赛结果与证明材料
4. 答辩与贡献度（15%）：GitHub 提交记录与答辩表现

当前仓库已覆盖第 1 项、第 2 项和 Kaggle 打榜核心材料。项目已从早期规则 Baseline 推进到“多源候选召回 + LightGBM Classifier/Ranker 重排 + 时间验证 + 消融解释 + 线上提交验证”的完整推荐系统流程。

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
- `LGBM_Optuna_Tuning.py`：LightGBM/召回参数的 Optuna 自动调参入口，输出 `outputs/optuna_trials.csv` 与 `outputs/optuna_best_params.json`
- `LGBM_MultiFold_Eval.py`：多折时间窗口验证入口，用于检查本地分数稳定性与线上泛化风险
- `LGBM_Ablation_SHAP_Analysis.ipynb`：LightGBM baseline 的 Top-5 特征消融实验与 SHAP 可解释性分析 notebook
- `LGBM_Ablation_SHAP_Analysis.py`：同上分析的脚本版入口，适合在 Kaggle 直接运行并查看日志
- `homework/HM_LGBM_Ablation_SHAP_Report.md`：消融与 SHAP 分析报告（Markdown 版）
- `homework/HM_LGBM_Ablation_SHAP_Report.pdf`：消融与 SHAP 分析报告（PDF 版）
- `homework/HM_Project_Timeline_Report.md`：项目推进历程报告，记录从 EDA、Baseline 到 LGBM 优化、内存工程与融合提交的完整过程
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

当前结构化优化版默认增加了轻量商品共现召回，并默认使用 `LGBM_FEATURE_EXPERIMENT=best`，即保留两个商品热度比例特征、去掉消融为负的 `price_diff_user_item`。`LGBM_ENABLE_ID_MAPPING=1` 默认开启，会把超长 `customer_id/article_id` 映射成 `Int32`，训练和候选 join 更省内存，最终提交前仍会还原为原始字符串 ID。Kaggle 上建议先跑下面这组，不使用图片特征，也不需要 GPU：

```python
%env LGBM_ENABLE_ID_MAPPING=1
%env LGBM_TRAIN_CUSTOMER_CAP=80000
%env LGBM_VALIDATION_CUSTOMER_CAP=60000
%env LGBM_MAX_CANDIDATES_PER_CUSTOMER=100
%env LGBM_BASELINE_RECALL_TOP=80
%env LGBM_ATTRIBUTE_RECALL_TOP=8
%env LGBM_RECENT_GLOBAL_RECALL_TOP=12
%env LGBM_ENABLE_COLOUR_RECALL=0
%env LGBM_ENABLE_SECTION_RECALL=0
%env LGBM_ENABLE_COOCCURRENCE_RECALL=1
%env LGBM_COOCCURRENCE_RECALL_TOP=12
%env LGBM_TRAIN_WINDOW_COUNT=1
%env LGBM_FEATURE_EXPERIMENT=best
%env LGBM_DROP_NOISY_FEATURES=0
%env LGBM_N_ESTIMATORS=450
%env LGBM_SUBMISSION_CUSTOMER_CHUNK=10000

!python Final_Project_LGBM.py
!cp outputs/submission.csv /kaggle/working/submission.csv
```

若单窗口版本验证分数提升且内存稳定，可尝试更稳但更慢的双窗口训练。双窗口会把 `LGBM_TRAIN_CUSTOMER_CAP` 自动分摊到两个窗口，避免训练集直接翻倍：

```python
%env LGBM_TRAIN_WINDOW_COUNT=2
```

150 候选原版基础上做冷启动增强时，建议使用下面这组。它会给无历史/弱历史用户追加年龄段、会员状态、新闻订阅频率组合下的近期热门商品；邮编召回默认关闭，避免高基数带来过拟合和内存压力：

```python
%env LGBM_ENABLE_ID_MAPPING=1
%env LGBM_TRAIN_CUSTOMER_CAP=80000
%env LGBM_VALIDATION_CUSTOMER_CAP=60000
%env LGBM_MAX_CANDIDATES_PER_CUSTOMER=150
%env LGBM_BASELINE_RECALL_TOP=100
%env LGBM_ATTRIBUTE_RECALL_TOP=8
%env LGBM_RECENT_GLOBAL_RECALL_TOP=12
%env LGBM_ENABLE_COLOUR_RECALL=0
%env LGBM_ENABLE_SECTION_RECALL=0
%env LGBM_ENABLE_COOCCURRENCE_RECALL=1
%env LGBM_COOCCURRENCE_RECALL_TOP=20
%env LGBM_ENABLE_COLD_START_RECALL=1
%env LGBM_COLD_START_HISTORY_MAX_ITEMS=2
%env LGBM_COLD_START_RECALL_TOP=18
%env LGBM_ENABLE_POSTAL_COLD_START_RECALL=0
%env LGBM_TRAIN_WINDOW_COUNT=1
%env LGBM_FEATURE_EXPERIMENT=best
%env LGBM_DROP_NOISY_FEATURES=0
%env LGBM_N_ESTIMATORS=450
%env LGBM_SUBMISSION_CUSTOMER_CHUNK=6000

!python Final_Project_LGBM.py
!cp outputs/submission.csv /kaggle/working/submission.csv
```

若要用 Optuna 自动找更好的召回组合和 LightGBM 参数，建议先用较小 cap 做 10-30 次试跑，找到稳定方向后再把最优 `%env` 参数回填到 `Final_Project_LGBM.py` 的正式提交运行：

```python
!pip install -q optuna

%env LGBM_ENABLE_ID_MAPPING=1
%env OPTUNA_N_TRIALS=20
%env OPTUNA_TRAIN_CUSTOMER_CAP=20000
%env OPTUNA_VALIDATION_CUSTOMER_CAP=20000
%env OPTUNA_TRAIN_WINDOW_COUNT=1

!python LGBM_Optuna_Tuning.py
```

调参结果会写入：
- `outputs/optuna_trials.csv`：每个 trial 的参数、MAP@12 和候选行数
- `outputs/optuna_best_params.json`：最佳 MAP@12 与可直接复制到 Kaggle Notebook 的 `%env LGBM_...=...` 参数

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
- optuna（可选，仅用于轻度超参优化）

## 7. 阶段状态

- [x] 第一阶段：期中 EDA 与清洗规则
- [x] 第二阶段：端到端推荐流水线、时间验证、LightGBM 排序模型、消融与 SHAP 分析
- [x] Kaggle 打榜：已完成多轮 after-deadline 提交验证，当前最佳线上成绩 Private `0.02881` / Public `0.02875`
- [x] 答辩与材料沉淀：已形成数据分析报告、答辩 PPT、消融解释报告和项目历程报告
- [ ] 后续提升方向：更高质量 embedding 召回、多模态图文特征、跨窗口 OOF 融合与更大算力下的候选召回扩展

## 8. 调参实验管理

实验追踪统一使用 `experiments/` 目录：

- `experiments/tuning_runs/`：每次 run 的快照目录。
- `experiments/runs_index.csv`：离线实验总索引（run_id、参数、离线分数、提交哈希）。
- `experiments/kaggle_scores.csv`：手动提交 Kaggle 后填写 Public/Private 分数。

当前阶段性最佳结果（截至 2026-06-17）：

- 最佳线上融合方案：`ranker_source_rank 0.535 + classifier_userattr90 0.465`
- Kaggle Private：`0.02881`
- Kaggle Public：`0.02875`
- 早期规则 Baseline：Public `0.01828` / Private `0.01841`
- 早期 LightGBM Baseline：Public 约 `0.02702` / Private 约 `0.02680`
- 当前有效主线：多源召回、冷启动召回、用户画像特征、召回源排名特征、LightGBM Ranker/Classifier 与提交融合

关键工程优化：

- ID 映射：将长字符串 `customer_id/article_id` 映射为轻量整数，降低 join 与训练内存压力
- 分块预测与流式提交：控制 Kaggle 30GB 内存环境下的峰值占用
- 可配置召回与特征开关：通过 `%env LGBM_...` 管理候选数、召回源、特征实验、模型类型和训练窗口
- 多折时间验证：使用 `LGBM_MultiFold_Eval.py` 检查单窗口本地分数与线上表现的偏差

## 9. 说明

本仓库当前内容已覆盖课程项目从数据分析、Baseline、增强模型到 Kaggle 提交验证的主要过程。当前最适合作为阶段性结论的结果是 Private `0.02881` / Public `0.02875`；若继续冲分，优先方向是更高质量候选召回、embedding/多模态特征和更严格的跨窗口 OOF 融合。
