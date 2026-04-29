# run_2026-04-27_001_hybrid_multisource

## Summary

- 目标：把离线 MAP@12 从早期 `0.01x` 区间提升到 `0.018+` 基线以上。
- 方法：将召回/重排从旧的少量信号扩展为“近期历史 + 长期历史 + 渠道 + 部门 + 年龄段 + 趋势 + 全局”多路融合。
- 最优配置：`local_u40_l7_c5_d5_a5_t5.5_g1.2`。

## Offline Results

- Fold1: `0.019488854665846964`
- Fold2: `0.018211545328798175`
- Fold3: `0.02113577373349563`
- Fold4: `0.022911868498113806`
- Fold5: `0.022417721858373412`
- Mean: `0.020833152816925597`

## Submission Hash

- SHA256: `8CF3E85699884B3349591E8974AF9F2E17670C218B0C682EDAFBDD6596682677`

## Kaggle

- Public: `0.01841`
- Private: `0.01847`
- 提交状态：`after deadline`（late submission）
- LGBM 增强复跑（2026-04-29）：Public `0.02679` / Private `0.02717`（after deadline）
- 明细台账见 `experiments/kaggle_scores.csv`。
