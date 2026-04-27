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
2. XAI 可解释性补全：SHAP 或 LIME 分析与报告
3. 竞赛交付补全：Kaggle 提交与 MAP@12 证明材料
4. 答辩展示材料与个人贡献说明

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
