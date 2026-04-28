# -*- coding: utf-8 -*-
"""LightGBM ranking pipeline for the H&M recommendation task.

This script upgrades the rule-based baseline into:
1. Candidate recall expansion.
2. Candidate training set construction.
3. LightGBM binary ranking model training.
4. MAP@12 time-window validation.
5. `outputs/submission.csv` generation.

The original `Final_Project.py` remains the stable submission-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from Final_Project import (
    BASE_PATH,
    MAX_K,
    OUTPUT_DIR,
    DEFAULT_RANKER,
    RankerConfig,
    build_submission_frame_from_lists,
    describe_ranker,
    fit_recommender,
    load_ranker_from_cache,
    load_tables,
    predict_lists_for_customers,
    prepare_article_department,
    prepare_customer_age_bin,
    prepare_output_dir,
    prepare_transactions,
    resolve_base_path,
    summarize_inputs,
    validate_submission_format,
)

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - depends on runtime image.
    raise RuntimeError(
        "LightGBM is required for Final_Project_LGBM.py. "
        "On Kaggle, run: !pip install lightgbm"
    ) from exc


RANDOM_STATE = 610
LABEL_WINDOW_DAYS = 7
VALIDATION_CUSTOMER_CAP = 60000
TRAIN_CUSTOMER_CAP = 80000
NEGATIVE_SAMPLE_RATIO = 20
MAX_CANDIDATES_PER_CUSTOMER = 80
SUBMISSION_CUSTOMER_CHUNK = 50000

AGE_BIN_TO_ID = {
    "u18": 0,
    "18_24": 1,
    "25_34": 2,
    "35_44": 3,
    "45_54": 4,
    "55_64": 5,
    "65_plus": 6,
}

FEATURE_COLUMNS = [
    "candidate_rank",
    "candidate_score",
    "source_baseline",
    "user_txn_all",
    "user_txn_30d",
    "user_unique_items",
    "user_recency_days",
    "user_avg_price",
    "user_pref_channel",
    "user_pref_department",
    "customer_age_bin_id",
    "item_pop_7d",
    "item_pop_30d",
    "item_pop_all",
    "item_buyers_7d",
    "item_buyers_30d",
    "item_buyers_all",
    "item_recency_days",
    "item_avg_price",
    "ua_cnt",
    "ua_recency_days",
    "same_department_as_pref",
    "article_department_no",
    "article_product_type_no",
    "article_product_group_no",
    "article_colour_group_code",
    "article_index_group_no",
    "article_section_no",
    "article_garment_group_no",
]


@dataclass(frozen=True)
class WindowSpec:
    history_end: date
    label_start: date
    label_end: date


def apk(actual: list[str], predicted: list[str], k: int = MAX_K) -> float:
    if not actual:
        return 0.0
    score = 0.0
    hits = 0.0
    used: set[str] = set()
    for idx, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in used:
            hits += 1.0
            score += hits / idx
            used.add(item)
    return score / min(len(actual), k)


def mapk12(actual_list: list[list[str]], predicted_list: list[list[str]], k: int = MAX_K) -> float:
    if not actual_list:
        return 0.0
    return float(np.mean([apk(a, p, k=k) for a, p in zip(actual_list, predicted_list)]))


def prepare_article_model_features(articles_lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        articles_lf.select(
            pl.col("article_id").cast(pl.Utf8),
            pl.col("department_no").cast(pl.Float64, strict=False).alias("article_department_no"),
            pl.col("product_type_no").cast(pl.Float64, strict=False).alias("article_product_type_no"),
            pl.col("product_group_name").cast(pl.Categorical).to_physical().cast(pl.Float64).alias(
                "article_product_group_no"
            ),
            pl.col("colour_group_code").cast(pl.Float64, strict=False).alias("article_colour_group_code"),
            pl.col("index_group_no").cast(pl.Float64, strict=False).alias("article_index_group_no"),
            pl.col("section_no").cast(pl.Float64, strict=False).alias("article_section_no"),
            pl.col("garment_group_no").cast(pl.Float64, strict=False).alias("article_garment_group_no"),
        )
        .unique(subset=["article_id"], keep="first")
        .collect(engine="streaming")
    )


def _dict_frame(mapping: dict[Any, Any], key_name: str, value_name: str, value_dtype: pl.DataType) -> pl.DataFrame:
    if not mapping:
        return pl.DataFrame({key_name: [], value_name: []}, schema={key_name: pl.Utf8, value_name: value_dtype})
    return pl.DataFrame(
        {
            key_name: [str(key) for key in mapping.keys()],
            value_name: list(mapping.values()),
        },
        schema={key_name: pl.Utf8, value_name: value_dtype},
    )


def _collect_actual_items(
    label_tx: pl.DataFrame,
    customer_cap: int | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    actual_df = (
        label_tx.sort(["customer_id", "t_dat"])
        .group_by("customer_id")
        .agg(pl.col("article_id").unique(maintain_order=True).alias("actual_items"))
        .sort("customer_id")
    )
    if customer_cap is not None and customer_cap > 0 and actual_df.height > customer_cap:
        actual_df = actual_df.head(customer_cap)

    customers = actual_df.get_column("customer_id").to_list()
    actual_map = {row["customer_id"]: row["actual_items"] or [] for row in actual_df.iter_rows(named=True)}
    return customers, actual_map


def build_candidate_frame(
    customer_ids: list[str],
    artifacts: Any,
    ranker_config: RankerConfig,
    max_candidates: int = MAX_CANDIDATES_PER_CUSTOMER,
    actual_items_by_customer: dict[str, list[str]] | None = None,
    include_actual_items: bool = False,
) -> pl.DataFrame:
    baseline_lists = predict_lists_for_customers(
        customer_ids=customer_ids,
        artifacts=artifacts,
        ranker_config=ranker_config,
        k=max_candidates,
    )

    rows: list[dict[str, Any]] = []
    actual_items_by_customer = actual_items_by_customer or {}
    for customer_id, baseline_items in zip(customer_ids, baseline_lists):
        seen: set[str] = set()
        for rank, article_id in enumerate(baseline_items, start=1):
            if article_id in seen:
                continue
            rows.append(
                {
                    "customer_id": customer_id,
                    "article_id": article_id,
                    "candidate_rank": rank,
                    "candidate_score": 1.0 / rank,
                    "source_baseline": 1.0,
                    "source_actual_in_train": 0.0,
                }
            )
            seen.add(article_id)

        if include_actual_items:
            for article_id in actual_items_by_customer.get(customer_id, []):
                if article_id in seen:
                    continue
                rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": article_id,
                        "candidate_rank": max_candidates + 1,
                        "candidate_score": 0.0,
                        "source_baseline": 0.0,
                        "source_actual_in_train": 1.0,
                    }
                )
                seen.add(article_id)

    if not rows:
        return pl.DataFrame(
            {
                "customer_id": [],
                "article_id": [],
                "candidate_rank": [],
                "candidate_score": [],
                "source_baseline": [],
                "source_actual_in_train": [],
            }
        )
    return pl.DataFrame(rows).unique(subset=["customer_id", "article_id"], keep="first")


def build_user_feature_frame(
    history_tx: pl.DataFrame,
    artifacts: Any,
    customer_age_bin: dict[str, str],
    reference_date: date,
) -> pl.DataFrame:
    cutoff_30d = reference_date - timedelta(days=30)
    user_all = (
        history_tx.group_by("customer_id")
        .agg(
            pl.len().alias("user_txn_all"),
            pl.n_unique("article_id").alias("user_unique_items"),
            pl.mean("price").alias("user_avg_price"),
            pl.max("t_dat").alias("user_last_date"),
        )
        .with_columns(
            (pl.lit(reference_date) - pl.col("user_last_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("user_recency_days")
        )
        .drop("user_last_date")
    )
    user_30d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_30d))
        .group_by("customer_id")
        .agg(pl.len().alias("user_txn_30d"))
    )
    channel_df = _dict_frame(
        artifacts.customer_pref_channel,
        "customer_id",
        "user_pref_channel",
        pl.Float64,
    )
    department_df = _dict_frame(
        artifacts.customer_pref_department,
        "customer_id",
        "user_pref_department",
        pl.Float64,
    )
    age_df = pl.DataFrame(
        {
            "customer_id": list(customer_age_bin.keys()),
            "customer_age_bin_id": [AGE_BIN_TO_ID.get(value, -1) for value in customer_age_bin.values()],
        },
        schema={"customer_id": pl.Utf8, "customer_age_bin_id": pl.Float64},
    )
    return (
        user_all.join(user_30d, on="customer_id", how="left")
        .join(channel_df, on="customer_id", how="left")
        .join(department_df, on="customer_id", how="left")
        .join(age_df, on="customer_id", how="left")
    )


def build_item_feature_frame(history_tx: pl.DataFrame, reference_date: date) -> pl.DataFrame:
    cutoff_7d = reference_date - timedelta(days=7)
    cutoff_30d = reference_date - timedelta(days=30)
    item_all = (
        history_tx.group_by("article_id")
        .agg(
            pl.len().alias("item_pop_all"),
            pl.n_unique("customer_id").alias("item_buyers_all"),
            pl.mean("price").alias("item_avg_price"),
            pl.max("t_dat").alias("item_last_date"),
        )
        .with_columns(
            (pl.lit(reference_date) - pl.col("item_last_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("item_recency_days")
        )
        .drop("item_last_date")
    )
    item_7d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_7d))
        .group_by("article_id")
        .agg(
            pl.len().alias("item_pop_7d"),
            pl.n_unique("customer_id").alias("item_buyers_7d"),
        )
    )
    item_30d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_30d))
        .group_by("article_id")
        .agg(
            pl.len().alias("item_pop_30d"),
            pl.n_unique("customer_id").alias("item_buyers_30d"),
        )
    )
    return item_all.join(item_7d, on="article_id", how="left").join(item_30d, on="article_id", how="left")


def build_user_article_feature_frame(history_tx: pl.DataFrame, reference_date: date) -> pl.DataFrame:
    return (
        history_tx.group_by(["customer_id", "article_id"])
        .agg(
            pl.len().alias("ua_cnt"),
            pl.max("t_dat").alias("ua_last_date"),
        )
        .with_columns(
            (pl.lit(reference_date) - pl.col("ua_last_date"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("ua_recency_days")
        )
        .drop("ua_last_date")
    )


def build_feature_frame(
    candidates: pl.DataFrame,
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    artifacts: Any,
    reference_date: date,
) -> pl.DataFrame:
    if candidates.is_empty():
        return candidates

    user_features = build_user_feature_frame(
        history_tx=history_tx,
        artifacts=artifacts,
        customer_age_bin=customer_age_bin,
        reference_date=reference_date,
    )
    item_features = build_item_feature_frame(history_tx=history_tx, reference_date=reference_date)
    user_article_features = build_user_article_feature_frame(history_tx=history_tx, reference_date=reference_date)

    feature_df = (
        candidates.join(user_features, on="customer_id", how="left")
        .join(item_features, on="article_id", how="left")
        .join(user_article_features, on=["customer_id", "article_id"], how="left")
        .join(article_features, on="article_id", how="left")
        .with_columns(
            (
                pl.col("user_pref_department").fill_null(-1)
                == pl.col("article_department_no").fill_null(-2)
            )
            .cast(pl.Float64)
            .alias("same_department_as_pref")
        )
    )

    return feature_df.with_columns(
        [pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col) for col in FEATURE_COLUMNS]
    )


def label_candidates(candidates: pl.DataFrame, actual_items_by_customer: dict[str, list[str]]) -> pl.DataFrame:
    rows = [
        {"customer_id": customer_id, "article_id": article_id, "label": 1}
        for customer_id, items in actual_items_by_customer.items()
        for article_id in items
    ]
    if not rows:
        return candidates.with_columns(pl.lit(0).alias("label"))

    positives = pl.DataFrame(rows).unique(subset=["customer_id", "article_id"], keep="first")
    return (
        candidates.join(positives, on=["customer_id", "article_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )


def downsample_training_rows(train_df: pl.DataFrame) -> pl.DataFrame:
    positives = train_df.filter(pl.col("label") == 1)
    negatives = train_df.filter(pl.col("label") == 0)
    if positives.is_empty() or negatives.is_empty():
        return train_df

    negative_n = min(negatives.height, max(positives.height * NEGATIVE_SAMPLE_RATIO, 100000))
    sampled_negatives = negatives.sample(n=negative_n, seed=RANDOM_STATE, shuffle=True)
    return pl.concat([positives, sampled_negatives], how="vertical").sample(
        fraction=1.0,
        seed=RANDOM_STATE,
        shuffle=True,
    )


def make_training_matrix(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = train_df.select(FEATURE_COLUMNS).to_numpy()
    y = train_df.get_column("label").to_numpy()
    return x, y


def train_lgbm_model(train_df: pl.DataFrame) -> Any:
    train_df = downsample_training_rows(train_df)
    x_train, y_train = make_training_matrix(train_df)
    positive_count = float(np.sum(y_train))
    negative_count = float(len(y_train) - positive_count)
    scale_pos_weight = max(1.0, negative_count / max(positive_count, 1.0))

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        feature_name=FEATURE_COLUMNS,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"LightGBM trained: rows={len(y_train)} positives={int(positive_count)}")
    return model


def score_feature_frame(model: Any, feature_df: pl.DataFrame) -> pl.DataFrame:
    if feature_df.is_empty():
        return feature_df.with_columns(pl.lit(0.0).alias("score"))
    scores = model.predict_proba(feature_df.select(FEATURE_COLUMNS).to_numpy())[:, 1]
    return feature_df.with_columns(pl.Series("score", scores))


def scored_frame_to_prediction_map(scored_df: pl.DataFrame, k: int = MAX_K) -> dict[str, list[str]]:
    pred_df = (
        scored_df.sort(
            ["customer_id", "score", "candidate_rank", "article_id"],
            descending=[False, True, False, False],
        )
        .group_by("customer_id")
        .agg(pl.col("article_id").head(k).alias("prediction_items"))
    )
    return {row["customer_id"]: row["prediction_items"] or [] for row in pred_df.iter_rows(named=True)}


def build_labeled_window_dataset(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    window: WindowSpec,
    customer_cap: int,
) -> pl.DataFrame:
    history_tx = transactions.filter(pl.col("t_dat") <= pl.lit(window.history_end))
    label_tx = transactions.filter(
        (pl.col("t_dat") >= pl.lit(window.label_start)) & (pl.col("t_dat") <= pl.lit(window.label_end))
    )
    customers, actual_map = _collect_actual_items(label_tx, customer_cap=customer_cap)
    artifacts = fit_recommender(
        history_tx,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
    )
    candidates = build_candidate_frame(
        customer_ids=customers,
        artifacts=artifacts,
        ranker_config=ranker_config,
        max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
        actual_items_by_customer=actual_map,
        include_actual_items=True,
    )
    labeled = label_candidates(candidates, actual_map)
    features = build_feature_frame(
        candidates=labeled,
        history_tx=history_tx,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        artifacts=artifacts,
        reference_date=window.history_end,
    )
    print(
        f"window dataset: history_end={window.history_end} "
        f"labels={window.label_start}~{window.label_end} "
        f"customers={len(customers)} rows={features.height}"
    )
    return features


def run_single_window_validation(
    model: Any,
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    window: WindowSpec,
    customer_cap: int = VALIDATION_CUSTOMER_CAP,
) -> dict[str, Any]:
    history_tx = transactions.filter(pl.col("t_dat") <= pl.lit(window.history_end))
    label_tx = transactions.filter(
        (pl.col("t_dat") >= pl.lit(window.label_start)) & (pl.col("t_dat") <= pl.lit(window.label_end))
    )
    customers, actual_map = _collect_actual_items(label_tx, customer_cap=customer_cap)
    artifacts = fit_recommender(
        history_tx,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
    )
    candidates = build_candidate_frame(
        customer_ids=customers,
        artifacts=artifacts,
        ranker_config=ranker_config,
        max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
    )
    features = build_feature_frame(
        candidates=candidates,
        history_tx=history_tx,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        artifacts=artifacts,
        reference_date=window.history_end,
    )
    scored = score_feature_frame(model, features)
    pred_map = scored_frame_to_prediction_map(scored, k=MAX_K)
    predicted = [pred_map.get(customer_id, []) for customer_id in customers]
    actual = [actual_map.get(customer_id, []) for customer_id in customers]
    score = mapk12(actual, predicted, k=MAX_K)
    print(
        f"LGBM validation: history_end={window.history_end} "
        f"labels={window.label_start}~{window.label_end} "
        f"customers={len(customers)} MAP@12={score:.6f}"
    )
    return {
        "history_end": window.history_end,
        "label_start": window.label_start,
        "label_end": window.label_end,
        "customer_count": len(customers),
        "candidate_rows": candidates.height,
        "map12": score,
    }


def build_train_and_validation_windows(transactions: pl.DataFrame) -> tuple[WindowSpec, WindowSpec]:
    max_date = transactions.get_column("t_dat").max()
    validation_label_end = max_date
    validation_label_start = validation_label_end - timedelta(days=LABEL_WINDOW_DAYS - 1)
    validation_history_end = validation_label_start - timedelta(days=1)

    train_label_end = validation_history_end
    train_label_start = train_label_end - timedelta(days=LABEL_WINDOW_DAYS - 1)
    train_history_end = train_label_start - timedelta(days=1)
    return (
        WindowSpec(
            history_end=train_history_end,
            label_start=train_label_start,
            label_end=train_label_end,
        ),
        WindowSpec(
            history_end=validation_history_end,
            label_start=validation_label_start,
            label_end=validation_label_end,
        ),
    )


def build_final_training_window(transactions: pl.DataFrame) -> WindowSpec:
    max_date = transactions.get_column("t_dat").max()
    label_end = max_date
    label_start = label_end - timedelta(days=LABEL_WINDOW_DAYS - 1)
    history_end = label_start - timedelta(days=1)
    return WindowSpec(history_end=history_end, label_start=label_start, label_end=label_end)


def save_feature_importance(model: Any, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = prepare_output_dir(output_dir)
    importance_df = pl.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_.tolist(),
        }
    ).sort("importance", descending=True)
    path = output_dir / "lgbm_feature_importance.csv"
    importance_df.write_csv(path)
    print(f"feature importance saved: {path}")


def generate_lgbm_submission(
    model: Any,
    tables: dict[str, pl.LazyFrame],
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    output_dir: Path = OUTPUT_DIR,
) -> pl.DataFrame:
    customer_ids = (
        tables["submission"]
        .select(pl.col("customer_id").cast(pl.Utf8))
        .collect(engine="streaming")
        .get_column("customer_id")
        .to_list()
    )
    artifacts = fit_recommender(
        transactions,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
    )
    reference_date = transactions.get_column("t_dat").max()

    all_rows: list[dict[str, str]] = []
    total = len(customer_ids)
    for start in range(0, total, SUBMISSION_CUSTOMER_CHUNK):
        end = min(start + SUBMISSION_CUSTOMER_CHUNK, total)
        chunk_customers = customer_ids[start:end]
        candidates = build_candidate_frame(
            customer_ids=chunk_customers,
            artifacts=artifacts,
            ranker_config=ranker_config,
            max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
        )
        features = build_feature_frame(
            candidates=candidates,
            history_tx=transactions,
            article_features=article_features,
            customer_age_bin=customer_age_bin,
            artifacts=artifacts,
            reference_date=reference_date,
        )
        scored = score_feature_frame(model, features)
        pred_map = scored_frame_to_prediction_map(scored, k=MAX_K)
        fallback_lists = predict_lists_for_customers(
            customer_ids=chunk_customers,
            artifacts=artifacts,
            ranker_config=ranker_config,
            k=MAX_K,
        )
        for customer_id, fallback_items in zip(chunk_customers, fallback_lists):
            items = pred_map.get(customer_id, [])
            if len(items) < MAX_K:
                seen = set(items)
                for item in fallback_items:
                    if item not in seen:
                        items.append(item)
                        seen.add(item)
                    if len(items) >= MAX_K:
                        break
            all_rows.append({"customer_id": customer_id, "prediction": " ".join(items[:MAX_K])})
        print(f"[LGBM submission] {end}/{total} customers done")

    submission = pl.DataFrame(all_rows)
    validate_submission_format(submission, customer_ids, k=MAX_K)
    output_dir = prepare_output_dir(output_dir)
    output_path = output_dir / "submission.csv"
    submission.write_csv(output_path)
    print(f"LGBM submission saved: {output_path}")
    return submission


def run_lgbm_pipeline(base_path: Path = BASE_PATH) -> dict[str, Any]:
    base_path = resolve_base_path(base_path)
    tables = load_tables(base_path)
    summary = summarize_inputs(tables)
    transactions = prepare_transactions(tables["transactions"])
    article_department = prepare_article_department(tables["articles"])
    article_features = prepare_article_model_features(tables["articles"])
    customer_age_bin = prepare_customer_age_bin(tables["customers"])
    ranker_config = load_ranker_from_cache(output_dir=prepare_output_dir(OUTPUT_DIR), fallback=DEFAULT_RANKER)
    print(f"candidate ranker: {describe_ranker(ranker_config)}")

    train_window, validation_window = build_train_and_validation_windows(transactions)
    validation_train_df = build_labeled_window_dataset(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        window=train_window,
        customer_cap=TRAIN_CUSTOMER_CAP,
    )
    validation_model = train_lgbm_model(validation_train_df)
    validation_metrics = run_single_window_validation(
        model=validation_model,
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        window=validation_window,
        customer_cap=VALIDATION_CUSTOMER_CAP,
    )
    validation_path = prepare_output_dir(OUTPUT_DIR) / "lgbm_validation_metrics.csv"
    pl.DataFrame([validation_metrics]).write_csv(validation_path)
    print(f"LGBM validation metrics saved: {validation_path}")

    final_window = build_final_training_window(transactions)
    final_train_df = build_labeled_window_dataset(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        window=final_window,
        customer_cap=TRAIN_CUSTOMER_CAP,
    )
    final_model = train_lgbm_model(final_train_df)
    save_feature_importance(final_model, output_dir=OUTPUT_DIR)
    submission = generate_lgbm_submission(
        model=final_model,
        tables=tables,
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        output_dir=OUTPUT_DIR,
    )
    return {
        "summary": summary,
        "ranker_config": ranker_config,
        "validation_metrics": validation_metrics,
        "submission": submission,
    }


def main() -> dict[str, Any]:
    state = run_lgbm_pipeline(BASE_PATH)
    print("\nInput summary:")
    print(state["summary"])
    print("\nCandidate ranker:")
    print(describe_ranker(state["ranker_config"]))
    print("\nLGBM validation metrics:")
    print(state["validation_metrics"])
    print("\nSubmission preview:")
    print(state["submission"].head(5))
    return state


if __name__ == "__main__":
    main()
