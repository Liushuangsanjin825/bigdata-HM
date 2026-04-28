# 综合实战任务执行记录（Project Rubric 对齐）

## 1. 基本信息

- 项目名称：基于多模态大数据的商品购买转化预测与推荐系统实战（H&M）
- 仓库名称：bigdata-HM
- 当前阶段：第二阶段进行中（期末工程交付，35%）
- 对齐依据：大作业考核任务书（Project_Evaluation）

## 2. 任务书要求映射

### 2.1 总体结构（实践板块 70%）

1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码体系与可解释性
3. Kaggle 打榜（15%）：MAP@12 公开竞赛结果
4. 答辩与贡献度（15%）：GitHub 记录与线下答辩

### 2.2 当前仓库覆盖范围

- 已覆盖：期中考核（20%）
- 已启动：期末工程交付（第二阶段）baseline 流水线
- 未覆盖：期末报告、Kaggle 打榜、答辩材料

### 2.3 Kaggle 官方规则提炼（不含奖励与时间线）

1. 任务定义
	- 基于历史交易 + 客户/商品元数据进行个性化商品推荐。
	- 可使用结构化特征，也可扩展文本与图像特征。
2. 评分规则
	- 采用 `MAP@12` 作为核心指标。
	- 每位客户最多提交 12 个推荐商品，按排序位置计算命中质量。
	- 评分仅统计测试期内有真实购买的客户，但提交时必须覆盖给定客户列表中的全部 `customer_id`。
	- 对真实购买数小于 12 的客户，提交满 12 条预测无惩罚。
3. 提交格式
	- 文件格式为 CSV，包含表头：`customer_id,prediction`。
	- `prediction` 为以空格分隔的 `article_id` 列表（最多 12 个）。
	- 预测目标窗口为训练期之后的下一个 7 天。
4. 工程约束
	- `customer_id`、`article_id` 全流程按字符串处理，避免前导零丢失。
	- 输出前需校验列名、分隔符与每行推荐数量上限（12）。

## 3. 第一阶段完成情况（已完成）

### 3.1 数据可视化探索与商业规律提取（10 分）

- 完成情况：已完成
- 证据：`EDA_Checkpoint.ipynb` 中多维分析与 Business Insight 章节
- 图表输出：`doc_images/` 中 fig_4_x、fig_5_x 系列图

### 3.2 数据清洗构建与 AI 辅助运用（10 分）

- 完成情况：已完成
- 证据：`EDA_Checkpoint.ipynb` 中质量诊断、清洗规则（Rulebook）、清洗代码实现
- 说明：当前仓库保留了清洗策略与实现过程，作为期中阶段交付

## 4. 当前交付物清单

1. `EDA_Checkpoint.ipynb`：期中考核主交付 notebook
2. `Final_Project.py`：第二阶段生成层脚本（仅负责生成 `outputs/submission.csv`）
3. `Final_Project_Eval.py`：第二阶段评估层脚本（离线评估与调参）
4. `Final_Project.ipynb`：第二阶段同步展示 notebook（已对齐“生成层/评估层”拆分）
5. `doc_images/`：关键图表（趋势、渠道、品类、用户特征等）
6. `README.md`：任务书对齐说明与阶段状态
7. `requirements.txt`：当前 notebook 运行依赖

## 5. 未完成项（后续阶段）

1. 第二阶段模型能力补全：K-Fold 训练、候选召回增强、排序模型优化
2. 调参评估口径优化：将调参阶段客户样本比例从当前约 2.9% 提升到约 5%（建议 `TUNING_CUSTOMER_CAP≈68,000~70,000`），并对比离线均值与波动稳定性
3. XAI 可解释性补全：SHAP 或 LIME 分析与报告
4. 竞赛交付补全：Kaggle 提交与 MAP@12 证明材料
5. 答辩展示材料与个人贡献说明

## 6. 当前结论

本仓库内容与任务书现阶段目标一致：
- 仅声明并提交已完成的期中考核任务
- 不将期末工程与竞赛内容误标为“已完成”

## 7. 自检

- [x] 仓库文档已按任务书分阶段表述
- [x] 已完成内容均有文件证据
- [x] 未完成内容已明确列出
- [ ] 第二、三阶段交付物仍待后续补充

## 8. 本次任务进度记录（2026-04-26）

1. 目标：将第二阶段从骨架推进到可运行 baseline，保证脚本可直接生成提交格式结果。
2. 输入：`Final_Project.py`、`Final_Project.ipynb`、主数据集目录 `G:\h-and-m-personalized-fashion-recommendations`。
3. 动作：
	- 在 `Final_Project.py` 中实现 baseline 特征构建与推荐生成逻辑。
	- 修复时间差计算报错（Polars API 由 `dt.days()` 调整为 `dt.total_days()`）。
	- 端到端执行脚本并验证输出路径与结果预览。
4. 结果：
	- 脚本执行成功（exit code 0）。
	- 已产出 `submission.csv` 到 `G:\h-and-m-personalized-fashion-recommendations\outputs\submission.csv`。
	- 第二阶段状态更新为“baseline 可运行，后续补 K-Fold/XAI/竞赛证明材料”。

## 9. 文档补充记录（2026-04-26）

1. 目标：补充官方赛题中与工程落地直接相关的规则说明。
2. 输入：Kaggle 官方 Description / Evaluation / Submission File 文本。
3. 动作：
	- 从官方介绍中抽取任务定义、MAP@12 评分规则、提交格式与预测窗口约束。
	- 显式剔除时间线与奖励金额等非工程实现信息。
	- 同步更新 `README.md` 与 `task.md`，保证后续实现可直接对照规则执行。
4. 结果：
	- 工作区文档已补齐“提交方式与要求”的统一口径。
	- 后续代码改造可直接围绕“全量客户覆盖 + 每人最多12条 + 7天预测窗口 + MAP@12”展开。

## 10. 第二阶段推进记录（2026-04-26）

1. 目标：将第二阶段由“可运行 baseline”推进到“可评估 + 可提交 + 可校验”的 baseline+。
2. 输入：`Final_Project.py`、`sample_submission.csv`、`transactions_train.csv`、新补充的官方提交规范。
3. 动作：
	- 重构 `Final_Project.py` 为端到端流程：数据读取、时间窗口评估、提交生成、格式校验。
	- 新增 5 折时间窗口离线评估（每折验证窗口 7 天），输出 MAP@12。
	- 召回从“用户历史 + 全局热门”扩展到“三路召回”：用户历史 + 渠道热门 + 全局热门，并进行加权重排。
	- 增加提交文件校验：列名、客户覆盖、每行推荐数量上限（12）与重复项检查。
	- 输出路径调整为仓库内 `outputs/submission.csv`，避免外部目录写权限问题。
	- 在 `AI_Assistant_Memo.md` 新增“6.1 task.md 更新频率规则（必须遵守）”，并补充中断交接与用户即时要求两类例外场景。
	- 在提交前清单加入“无中途重复写入 `task.md`”检查项。
4. 结果：
	- 端到端脚本运行成功（`python -B Final_Project.py`，exit code 0）。
	- 离线 5 折 MAP@12：
		- Fold1: 0.013006
		- Fold2: 0.012306
		- Fold3: 0.013601
		- Fold4: 0.014458
		- Fold5: 0.014540
		- Mean: 0.013582
	- 已生成提交文件：`G:\Users\caoruijie\bigdata-HM\outputs\submission.csv`。
	- 已明确“按完整对话收尾统一更新 task”的执行口径，可用于后续协作交接。

## 11. 任务收尾记录（2026-04-27）

1. 目标：在不改变提交文件名（`submission.csv`）的前提下，完成第二阶段脚本职责拆分、乱码修复与仓库收尾。
2. 输入：`Final_Project.py`、`Final_Project.ipynb`、`README.md`、既有 `outputs/ranker_tuning_metrics.csv`。
3. 动作：
	- 将 `Final_Project.py` 重构为“仅生成层”脚本：读取数据、加载/回退 ranker、生成并校验 `outputs/submission.csv`。
	- 新增 `Final_Project_Eval.py` 作为独立“评估层”脚本：离线 MAP@12 评估与权重调参，写出评估结果文件。
	- 覆盖更新 `Final_Project.ipynb`，同步到“双层架构”并修复中文乱码，统一 UTF-8 保存。
	- 在 `Final_Project.py`、`Final_Project_Eval.py` 增加 UTF-8 编码声明，降低 Windows 环境下中文乱码风险。
	- 更新 `README.md` 与 `task.md`，明确第二阶段执行入口与职责边界。
4. 结果：
	- 生成层与评估层已解耦，运行与排障路径更清晰。
	- `submission.csv` 输出文件名与路径保持不变：`outputs/submission.csv`。
	- Notebook 与脚本中文显示恢复正常，且语法校验通过。

## 12. 调参续跑记录（2026-04-27）

1. 目标：继续第二阶段评估层调参，扩大权重搜索空间并产出更合理的离线指标与提交结果。
2. 输入：`Final_Project_Eval.py`、`Final_Project.py`、既有 `outputs/ranker_tuning_metrics.csv` 与 `outputs/offline_fold_metrics.csv`。
3. 动作：
	- 在 `Final_Project_Eval.py` 中将调参流程升级为“两阶段”：先评估基线候选，再围绕最优候选进行邻域微调。
	- 新增调参上下文复用逻辑（按折预先构建 artifacts 与验证集），避免重复拟合导致的额外耗时。
	- 增加本地候选权重组（偏历史强化、渠道/部门轻微扰动、全局权重微调），并统一汇总排序输出。
	- 执行 `python -B Final_Project_Eval.py` 重新生成评估产物；随后执行 `python -B Final_Project.py` 用最新最优权重重生成提交文件。
4. 结果：
	- 选出新权重：`local_u40_c6_d6_g2`（`user=40.0, channel=6.0, department=6.0, global=2.0`）。
	- 调参榜首（3 折、每折 40000 客户）`mean_map12=0.014707`。
	- 5 折离线评估（全量客户）MAP@12：
		- Fold1: 0.013286
		- Fold2: 0.012523
		- Fold3: 0.014020
		- Fold4: 0.015023
		- Fold5: 0.014916
		- Mean: 0.013954
	- 已刷新输出文件：`outputs/ranker_tuning_metrics.csv`、`outputs/offline_fold_metrics.csv`、`outputs/selected_ranker.csv`、`outputs/submission.csv`。

## 13. 达标优化记录（2026-04-27）

1. 目标：将第二阶段离线 MAP@12 提升到任务书“0分基准线（0.018）”以上，并同步生成新提交文件。
2. 输入：`Final_Project.py`、`Final_Project_Eval.py`、`outputs/recommendation_explanations_sample.csv`、既有离线评估结果文件。
3. 动作：
	- 重构推荐策略为“近期历史 + 长期历史 + 渠道热门 + 部门热门 + 年龄段热门 + 趋势热门 + 全局热门”多路融合。
	- 在生成层新增年龄分桶映射与趋势热门构建逻辑，改造 cold-start 与打分融合方式。
	- 扩展 `RankerConfig` 权重维度（`long_history_weight / age_weight / trend_weight`），并同步缓存读取逻辑。
	- 在评估层同步升级调参与落盘逻辑，支持新权重字段并保留“两阶段调参（基线 + 邻域搜索）”。
	- 执行小规模 smoke test（2折、降采样客户）验证新逻辑后，完整执行 `python -B Final_Project_Eval.py` 与 `python -B Final_Project.py`。
4. 结果：
	- 最优权重更新为：`local_u40_l7_c5_d5_a5_t5.5_g1.2`（`user=40.0, long=7.0, channel=5.0, department=5.0, age=5.0, trend=5.5, global=1.2`）。
	- 5 折离线评估（全量客户）MAP@12：
		- Fold1: 0.019489
		- Fold2: 0.018212
		- Fold3: 0.021136
		- Fold4: 0.022912
		- Fold5: 0.022418
		- Mean: 0.020833
	- 离线均值已高于任务书 0 分基准线 `0.018`。
	- 已刷新输出文件：`outputs/ranker_tuning_metrics.csv`、`outputs/offline_fold_metrics.csv`、`outputs/selected_ranker.csv`、`outputs/submission.csv`。

## 14. 实验追踪体系落地（2026-04-27）

1. 目标：为“冲满分”阶段建立可持续调参台账，统一管理离线实验与 Kaggle 官方分数，避免实验过程不可复现。
2. 输入：`README.md`、当前 `outputs/*.csv`（调参与离线评估结果）、用户提出的“建立调参目录并持续记录”的需求。
3. 动作：
	- 新建 `experiments/` 目录体系，包含：
		- `experiments/tuning_runs/`（每次 run 快照）
		- `experiments/runs_index.csv`（离线实验总索引）
		- `experiments/kaggle_scores.csv`（线上成绩记录）
		- `experiments/README.md`（记录规范）
	- 归档当前最优 run：`run_2026-04-27_001_hybrid_multisource`，写入 `params.json`、`notes.md`、`submission_sha256.txt`，并复制本次调参产物文件。
	- 在 `runs_index.csv` 中登记本次 run 的离线 5 折结果与 `submission.csv` 哈希；在 `kaggle_scores.csv` 增加待填写行。
	- 更新 `README.md`：新增实验追踪说明、当前最佳离线记录与记录流程要求。
4. 结果：
	- 调参与线上分数已具备统一台账入口，可按 `run_id` 关联离线/线上表现。
	- 当前最佳 run（截至 2026-04-27）已完成结构化归档，便于后续持续优化与答辩举证。

## 15. 官方分数登记（2026-04-27）

1. 目标：将本轮 Kaggle 官方返回分数写入实验台账，完成离线-线上闭环记录。
2. 输入：用户提供的 Kaggle 结果截图（`submission.csv`；Public `0.01841`，Private `0.01847`，after deadline）。
3. 动作：
	- 更新 `experiments/kaggle_scores.csv` 对应 `run_id=run_2026-04-27_001_hybrid_multisource` 的提交记录。
	- 在 `experiments/runs_index.csv` 备注中补充该 run 的官方 Public/Private 分数。
	- 更新 `README.md` 中“调参实验管理”章节，新增“当前已登记官方成绩”信息。
4. 结果：
	- 当前 run 的离线与线上分数已可通过 `run_id` 直接关联，后续可持续跟踪“离线提升是否转化为线上提升”。

## 16. LightGBM 模型优化流水线补充（2026-04-28）

1. 目标：在现有规则融合 baseline 分数偏低的基础上，按“扩展候选召回 → 构造候选训练集 → 训练 LightGBM → MAP@12 时间窗口验证 → 生成 submission.csv”的流程升级模型。
2. 输入：`Final_Project.py` 中的数据读取、规则推荐器与提交校验函数；Kaggle H&M 官方数据集路径；用户提出的五步优化流程。
3. 动作：
	- 新增 `Final_Project_LGBM.py`，作为独立增强版流水线，不破坏原有 `Final_Project.py` 稳定 baseline。
	- 复用规则推荐器生成候选商品，并补充用户特征、商品热度特征、用户-商品交叉特征与商品属性特征。
	- 构造带标签的候选训练集：使用历史窗口作为特征期、未来 7 天真实购买作为标签。
	- 使用 LightGBM 二分类模型对候选商品进行排序，按预测概率为每个用户取前 12 个商品。
	- 新增单窗口 MAP@12 离线验证，并输出 `outputs/lgbm_validation_metrics.csv` 与 `outputs/lgbm_feature_importance.csv`。
	- 更新 `requirements.txt`，增加 `lightgbm` 依赖；更新 `README.md`，补充 Kaggle Notebook 运行入口。
4. 结果：
	- 新增增强版脚本：`Final_Project_LGBM.py`。
	- 已完成本地语法校验：`python -m py_compile Final_Project.py Final_Project_Eval.py Final_Project_LGBM.py`。
	- 由于完整数据集位于 Kaggle，本地未执行全量训练；完整验证与 `submission.csv` 生成需在 Kaggle Notebook 数据环境中运行。
