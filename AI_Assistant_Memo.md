# AI 辅助分析备忘录

这份备忘录给后续接手本仓库的智能体使用，目标是让每次工作都能快速上手，并且严格对齐课程任务书的阶段要求。

## 1. 先读什么

接手时优先阅读以下文件：

1. [README.md](README.md)
2. [task.md](task.md)
3. [experiments/README.md](experiments/README.md)
4. [experiments/runs_index.csv](experiments/runs_index.csv)
5. [experiments/kaggle_scores.csv](experiments/kaggle_scores.csv)
6. [requirements.txt](requirements.txt)
7. [大作业考核任务书(Project_Evaluation).pdf](大作业考核任务书(Project_Evaluation).pdf)
8. [EDA_Checkpoint.ipynb](EDA_Checkpoint.ipynb)

先确认当前阶段，只做已经对齐的部分，不把未完成阶段写成已完成。

## 2. 任务书约束

课程任务书的实践板块分为四部分：

1. 期中考核（20%）：EDA 报告与特征初探
2. 期末报告（20%）：端到端模型代码、架构与可解释性
3. Kaggle 打榜（15%）：MAP@12 竞赛结果与证明材料
4. 答辩与贡献度（15%）：GitHub 提交记录与答辩表现

当前仓库已进入第二阶段（期末工程交付）持续优化阶段，包含离线调参与 Kaggle 成绩追踪；第三阶段（答辩）仍未完成，必须显式标注未完成状态。

## 3. 当前仓库规范

### 3.1 范围控制

- 只修改和当前任务相关的文件。
- 只在能被任务书或现有内容支持的范围内下结论。
- 如果是期中阶段，就只补 EDA、清洗规则、图表结论和执行记录。
- 如果是第二阶段调参与提交，必须同步维护 `experiments/` 台账（离线 run + 官方分数）。

### 3.2 依赖与环境

- notebook 运行依赖写在 [requirements.txt](requirements.txt)。
- 如果新增 PDF 读取、文本提取或其它能力，先补依赖，再更新说明。
- `.venv` 不进入仓库，属于本地环境，不要提交。

### 3.3 版本控制

- 修改前先看 `git status`。
- 每次只提交一组主题一致的变更。
- 提交信息尽量说明“改了什么 + 为什么改”。
- 远程默认仓库名以当前 GitHub 地址为准。

## 4. 每次工作的标准流程

### 4.1 接手检查

1. 读 README、task、requirements 和任务书 PDF。
2. 看 notebook 当前内容是否和文档一致。
3. 看 `git status`，确认有没有未提交改动。
4. 判断当前任务属于哪一阶段：期中、期末、Kaggle、答辩。

### 4.2 先做局部判断

每次编辑前先给出一个本地假设：

- 这段内容应该体现什么任务书要求？
- 哪个文件最能控制这个行为？
- 最便宜的验证是什么？

如果能用一个很小的修改验证假设，就先做那个修改，再看结果。

### 4.3 编辑原则

- 优先补最接近事实的说明，不编造不存在的结果。
- 文字和代码保持一致，文档里的状态要和 notebook、图表、依赖一致。
- 只补和当前阶段有关的内容，不提前把后续阶段写满。

### 4.4 验证原则

- 先做可以快速失败的检查，再扩大范围。
- notebook、README、task、requirements 改完后至少做一次文件级校验。
- 如果改了可执行内容，优先跑最小验证；如果只改文档，至少检查 diff 和内容一致性。

## 5. 任务书驱动的写作规范

### 5.1 期中考核写法

期中阶段文档和 notebook 需要覆盖：

- 数据可视化探索
- 商业规律提取
- 数据清洗规则
- AI 辅助运用过程
- 每张关键图表对应的业务洞察

### 5.2 期末与竞赛写法

如果后续开始做期末与 Kaggle 部分，必须额外补充：

- 防数据泄漏的说明
- Pipeline / ColumnTransformer 的实现
- K-Fold 稳定性与 random_state=610
- SHAP 或 LIME 可解释性
- Submission.csv 和成绩截图

### 5.3 答辩写法

答辩相关内容要能说明：

- 个人做了什么
- 为什么这么做
- 哪些问题已经解决，哪些还存在
- 贡献记录如何在 GitHub 中体现

## 6. AI 辅助记录规范

### 6.1 `task.md` 更新频率规则（必须遵守）

- 以“**一次完整对话**”为单位更新 `task.md`：从用户提出当前任务，到用户确认结束/切换任务为止。
- 在同一轮完整对话中，`task.md` **只在收尾阶段统一更新一次**，中途不重复写入。
- 中途进展通过对话消息同步，不写入 `task.md`，避免重复记录和版本噪音。
- 收尾更新时应合并本轮全部有效动作，至少包含：目标、输入、动作、结果、关键产物路径（如 `submission.csv`）与核心指标（如 MAP@12）。
- 只有两类例外可提前更新：
  1. 用户明确要求“现在就更新 `task.md`”；
  2. 会话需要提前中断并交接，需写入一条临时进度并标注“临时记录”。

每次使用 AI 辅助分析时，建议记录以下四项：

1. 目标：这次要解决什么问题
2. 输入：看了哪些文件、哪些数据或哪些图
3. 动作：做了什么修改、什么验证
4. 结果：最终产出是什么，是否能复现

如果是联合查询或生成代码，最好把“AI 负责什么、人负责什么”写清楚。

### 6.2 `experiments/` 更新规则（第二阶段必须遵守）

- 每次新调参 run 都要在 `experiments/tuning_runs/` 新建快照目录，并保留：
  - `ranker_tuning_metrics.csv`
  - `offline_fold_metrics.csv`
  - `selected_ranker.csv`
  - `submission_sha256.txt`
- 每次 run 必须追加更新 `experiments/runs_index.csv`，确保 `run_id` 唯一且可追溯。
- 每次手动提交 Kaggle 后，必须更新 `experiments/kaggle_scores.csv` 的 `public_score`、`private_score` 与提交时间。
- `README.md` 中“当前最佳离线记录 / 当前已登记官方成绩”应与 `experiments/` 台账保持一致。

## 7. 提交前清单

- [ ] README 和 task 与实际状态一致
- [ ] `task.md` 已按“每轮对话收尾一次更新”执行（无中途重复写入）
- [ ] requirements 与 notebook 导入一致
- [ ] PDF 读取、图表导出等能力已写进说明
- [ ] 任务书要求没有被误写成已完成
- [ ] `experiments/runs_index.csv` 与 `experiments/kaggle_scores.csv` 已同步更新
- [ ] `.venv` 没有被提交
- [ ] 已完成必要的 git 提交

## 8. 给下一位智能体的简短提示

如果你接手这个仓库，优先做两件事：

1. 对照任务书判断当前阶段，只做对应部分。
2. 每次改动都让文档、notebook、依赖、提交记录四者保持一致。
3. 第二阶段优化必须执行“先离线 run 归档，再登记官方分数”的闭环。
