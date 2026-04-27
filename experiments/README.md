# Experiment Tracking

本目录用于统一管理每次调参实验与 Kaggle 提交结果，避免“分数有了但过程不可复现”。

## Directory Layout

- `tuning_runs/`：每次离线调参的快照目录（一个 run 一个子目录）。
- `runs_index.csv`：离线实验总索引（参数、离线分数、提交哈希、备注）。
- `kaggle_scores.csv`：线上提交成绩表（手动补充 Public/Private 分）。

## Run Naming

建议格式：`run_YYYY-MM-DD_NNN_short_tag`

示例：`run_2026-04-27_001_hybrid_multisource`

## Minimal Checklist For Each New Run

1. 在 `tuning_runs/` 下创建新的 run 目录。
2. 复制本次输出文件：
   - `ranker_tuning_metrics.csv`
   - `offline_fold_metrics.csv`
   - `selected_ranker.csv`
3. 记录 `submission.csv` 的 `SHA256` 到 `submission_sha256.txt`。
4. 更新 `runs_index.csv` 一行。
5. 手动提交 Kaggle 后，更新 `kaggle_scores.csv`（关联同一个 `run_id`）。

## Notes

- `run_id` 是离线与线上记录的唯一关联键。
- `submission_sha256` 用于确认“线上提交文件”和“离线 run”一一对应。
