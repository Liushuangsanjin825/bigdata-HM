# H&M LightGBM Baseline 消融实验与 SHAP 分析报告

## 1. 实验目标

本次分析基于 H&M 推荐任务中的 LightGBM baseline，完成两项模型解释工作：

1. 消融实验：按照 LightGBM gain 排名选取 Top-5 特征，逐个移除后重新训练模型，并记录 MAP@12 的变化。
2. SHAP 分析：使用 `shap.TreeExplainer` 计算 SHAP 值，生成全局蜂群图和单样本瀑布图，解释模型学到的推荐规律。

## 2. 消融实验

### 2.1 Top-5 特征选择

根据 `lgbm_gain_importance.csv`，当前 LightGBM baseline 的 gain 排名前五特征为：

| Gain 排名 | 特征 | Gain | Split |
|---:|---|---:|---:|
| 1 | `candidate_rank` | 4714382.9331 | 649 |
| 2 | `source_baseline` | 304661.0000 | 1 |
| 3 | `candidate_score` | 244729.3684 | 116 |
| 4 | `user_avg_price` | 91168.2414 | 805 |
| 5 | `user_recency_days` | 78244.2913 | 680 |

其中：

- `candidate_rank`：规则召回阶段中候选商品的排序位置。
- `source_baseline`：候选是否来自 baseline 召回来源。
- `candidate_score`：候选在规则召回阶段的基础分数。
- `user_avg_price`：用户历史购买均价。
- `user_recency_days`：用户最近一次购买距验证参考日期的天数。

### 2.2 边际贡献计算方式

边际贡献定义为：

```text
marginal_contribution = baseline_map12 - map12_after_removal
```

解释规则：

- `marginal_contribution > 0`：移除该特征后 MAP@12 下降，说明该特征对最终推荐指标有正贡献。
- `marginal_contribution = 0`：移除后指标基本不变，说明该特征对当前验证指标没有实际贡献。
- `marginal_contribution < 0`：移除后指标反而上升，说明该特征可能带来噪声或过拟合。

### 2.3 消融实验结果

Baseline MAP@12 为 `0.028581`。

| 移除特征 | Gain 排名 | Baseline MAP@12 | 移除后 MAP@12 | 边际贡献 | 是否伪重要特征 |
|---|---:|---:|---:|---:|---|
| `source_baseline` | 2 | 0.028581 | 0.028162 | 0.000419 | 否 |
| `candidate_score` | 3 | 0.028581 | 0.028457 | 0.000124 | 否 |
| `user_avg_price` | 4 | 0.028581 | 0.028812 | -0.000231 | 是 |
| `user_recency_days` | 5 | 0.028581 | 0.029147 | -0.000566 | 是 |
| `candidate_rank` | 1 | 0.028581 | 0.029423 | -0.000843 | 是 |
### 2.4 消融结论
从消融结果看，`source_baseline` 和 `candidate_score` 移除后 MAP@12 分别下降 `0.000419` 和 `0.000124`，说明它们对最终排序有正向贡献。
但 `candidate_rank`、`user_avg_price`、`user_recency_days` 虽然位于 gain 排名前五，移除后 MAP@12 反而上升，因此在当前验证窗口下属于“Gain 排名高但实际贡献为零或负贡献”的伪重要特征。尤其是 `candidate_rank` 的 gain 排名第 1，但移除后 MAP@12 从 `0.028581` 提升到 `0.029423`，说明该特征可能让模型过度依赖规则召回顺序，抑制了 LightGBM 对商品热度、用户行为和交互特征的重新排序能力。

## 3. SHAP 分析
**SHAP 蜂群图**
![SHAP 蜂群图](shap_beeswarm.png)

根据 `shap_mean_abs_importance.csv`，平均绝对 SHAP 值排名靠前的特征为：

| SHAP 排名 | 特征 | mean_abs_shap |
|---:|---|---:|
| 1 | `candidate_rank` | 0.315029 |
| 2 | `item_buyers_7d` | 0.268571 |
| 3 | `item_pop_7d` | 0.173789 |
| 4 | `item_recency_days` | 0.170581 |
| 5 | `ua_cnt` | 0.131962 |
| 6 | `ua_recency_days` | 0.107038 |
| 7 | `user_avg_price` | 0.094995 |
| 8 | `item_avg_price` | 0.082897 |
| 9 | `customer_age_bin_id` | 0.066112 |
| 10 | `user_recency_days` | 0.065922 |

SHAP 全局结果显示，模型最关注三类信号：

1. 候选召回阶段的排序信号，例如 `candidate_rank`。
2. 商品近期热度信号，例如 `item_buyers_7d`、`item_pop_7d`、`item_recency_days`。
3. 用户-商品历史交互信号，例如 `ua_cnt` 和 `ua_recency_days`。

这说明模型并不是单纯推荐全局热门商品，而是在 baseline 候选排序的基础上，进一步利用近期商品热度和用户历史交互强度进行重排。

**SHAP 瀑布图**

![SHAP 瀑布图](shap_waterfall_sample0.png)

瀑布图展示了单个候选商品样本的预测分数如何从基准值逐步被各个特征推高或压低。对于该样本，贡献为正的特征会提高模型判断其被购买的概率，贡献为负的特征会降低该概率；因此可以用它解释“为什么这个商品被排到当前用户推荐列表中的相应位置”。

## 4. 模型学到的规律

综合 SHAP 蜂群图和特征重要性表，LightGBM baseline 主要学到了三类推荐规律：第一，规则召回阶段给出的候选排序仍然是强信号，但消融实验表明它可能存在过拟合风险；第二，最近 7 天的商品购买人数、销量和商品新近程度对推荐分数有明显影响，说明服装推荐强依赖短期流行趋势；第三，用户与商品的历史交互次数和最近交互时间能够帮助模型识别用户延续性偏好。

同时，消融实验提醒我们，gain 高并不等价于对 MAP@12 有真实贡献。`candidate_rank`、`user_avg_price` 和 `user_recency_days` 在树分裂中很活跃，但移除后验证 MAP@12 反而提升，因此后续优化时应考虑降低这些特征的依赖，或通过更多时间窗口验证它们是否稳定有效。

## 5. 最终结论

本次分析完成了 Top-5 gain 特征的逐一消融，并发现当前模型中存在伪重要特征：`candidate_rank`、`user_avg_price` 和 `user_recency_days`。SHAP 分析表明，模型主要依赖候选排序、近期商品热度和用户-商品历史交互来判断推荐优先级；其中近期热度和交互特征更符合 H&M 服装推荐的业务规律，后续可以围绕这些稳定信号继续增强候选召回与排序特征。
