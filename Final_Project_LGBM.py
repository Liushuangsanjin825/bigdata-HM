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

import gc
import os
from collections import Counter, defaultdict
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


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


RANDOM_STATE = 610
LABEL_WINDOW_DAYS = _env_int("LGBM_LABEL_WINDOW_DAYS", 7)
VALIDATION_CUSTOMER_CAP = _env_int("LGBM_VALIDATION_CUSTOMER_CAP", 60000)
TRAIN_CUSTOMER_CAP = _env_int("LGBM_TRAIN_CUSTOMER_CAP", 80000)
NEGATIVE_SAMPLE_RATIO = _env_int("LGBM_NEGATIVE_SAMPLE_RATIO", 20)
MAX_CANDIDATES_PER_CUSTOMER = _env_int("LGBM_MAX_CANDIDATES_PER_CUSTOMER", 100)
BASELINE_RECALL_TOP = _env_int("LGBM_BASELINE_RECALL_TOP", min(80, MAX_CANDIDATES_PER_CUSTOMER))
# Keep predictions unchanged while lowering per-chunk peak memory.
SUBMISSION_CUSTOMER_CHUNK = _env_int("LGBM_SUBMISSION_CUSTOMER_CHUNK", 20000)
ATTRIBUTE_RECALL_TOP = _env_int("LGBM_ATTRIBUTE_RECALL_TOP", 12)
RECENT_GLOBAL_RECALL_TOP = _env_int("LGBM_RECENT_GLOBAL_RECALL_TOP", 18)
LGBM_N_ESTIMATORS = _env_int("LGBM_N_ESTIMATORS", 450)
ENABLE_COLOUR_RECALL = _env_bool("LGBM_ENABLE_COLOUR_RECALL", False)
ENABLE_SECTION_RECALL = _env_bool("LGBM_ENABLE_SECTION_RECALL", False)
ENABLE_COOCCURRENCE_RECALL = _env_bool("LGBM_ENABLE_COOCCURRENCE_RECALL", True)
COOCCURRENCE_DAYS = _env_int("LGBM_COOCCURRENCE_DAYS", 30)
COOCCURRENCE_TOP_PER_ITEM = _env_int("LGBM_COOCCURRENCE_TOP_PER_ITEM", 16)
COOCCURRENCE_HISTORY_TOP = _env_int("LGBM_COOCCURRENCE_HISTORY_TOP", 4)
COOCCURRENCE_RECALL_TOP = _env_int("LGBM_COOCCURRENCE_RECALL_TOP", 12)
COOCCURRENCE_MAX_BASKET_SIZE = _env_int("LGBM_COOCCURRENCE_MAX_BASKET_SIZE", 20)
ENABLE_COLD_START_RECALL = _env_bool("LGBM_ENABLE_COLD_START_RECALL", True)
COLD_START_HISTORY_MAX_ITEMS = _env_int("LGBM_COLD_START_HISTORY_MAX_ITEMS", 2)
COLD_START_DAYS = _env_int("LGBM_COLD_START_DAYS", 30)
COLD_START_RECALL_TOP = _env_int("LGBM_COLD_START_RECALL_TOP", 18)
COLD_START_SEGMENT_TOP = _env_int("LGBM_COLD_START_SEGMENT_TOP", 32)
COLD_START_MIN_SEGMENT_BUYERS = _env_int("LGBM_COLD_START_MIN_SEGMENT_BUYERS", 3)
ENABLE_POSTAL_COLD_START_RECALL = _env_bool("LGBM_ENABLE_POSTAL_COLD_START_RECALL", False)
POSTAL_COLD_START_MIN_BUYERS = _env_int("LGBM_POSTAL_COLD_START_MIN_BUYERS", 8)
TRAIN_WINDOW_COUNT = _env_int("LGBM_TRAIN_WINDOW_COUNT", 1)
DROP_NOISY_FEATURES = _env_bool("LGBM_DROP_NOISY_FEATURES", False)
FEATURE_EXPERIMENT = os.getenv("LGBM_FEATURE_EXPERIMENT", "best").strip().lower()

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
    "source_product_type",
    "source_garment_group",
    "source_colour",
    "source_section",
    "source_recent_global",
    "source_cooccurrence",
    "cooccurrence_score",
    "source_cold_start",
    "cold_start_score",
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
    "item_pop_ratio_7d_30d",
    "item_pop_ratio_30d_all",
    "ua_cnt",
    "ua_recency_days",
    "price_diff_user_item",
    "same_department_as_pref",
    "article_department_no",
    "article_product_type_no",
    "article_product_group_no",
    "article_colour_group_code",
    "article_index_group_no",
    "article_section_no",
    "article_garment_group_no",
]

NOISY_FEATURE_COLUMNS = {"candidate_rank", "user_avg_price", "user_recency_days"}
NEW_EXPERIMENT_FEATURE_COLUMNS = {
    "item_pop_ratio_7d_30d",
    "item_pop_ratio_30d_all",
    "price_diff_user_item",
}


def _resolve_model_feature_columns() -> list[str]:
    feature_columns = list(FEATURE_COLUMNS)
    if DROP_NOISY_FEATURES:
        feature_columns = [feature for feature in feature_columns if feature not in NOISY_FEATURE_COLUMNS]

    experiment = FEATURE_EXPERIMENT
    if experiment in {"", "all"}:
        return feature_columns
    if experiment == "best":
        return [feature for feature in feature_columns if feature != "price_diff_user_item"]
    if experiment == "base":
        return [feature for feature in feature_columns if feature not in NEW_EXPERIMENT_FEATURE_COLUMNS]

    prefix = "add_"
    if experiment.startswith(prefix):
        added_feature = experiment[len(prefix) :]
        if added_feature not in NEW_EXPERIMENT_FEATURE_COLUMNS:
            raise ValueError(
                f"Unknown LGBM_FEATURE_EXPERIMENT={FEATURE_EXPERIMENT!r}. "
                f"Expected base, all, or add_ + one of {sorted(NEW_EXPERIMENT_FEATURE_COLUMNS)}"
            )
        base_features = [feature for feature in feature_columns if feature not in NEW_EXPERIMENT_FEATURE_COLUMNS]
        return base_features + [added_feature]

    raise ValueError(
        f"Unknown LGBM_FEATURE_EXPERIMENT={FEATURE_EXPERIMENT!r}. "
        "Expected best, base, all, add_item_pop_ratio_7d_30d, "
        "add_item_pop_ratio_30d_all, or add_price_diff_user_item."
    )


MODEL_FEATURE_COLUMNS = _resolve_model_feature_columns()


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


def _age_bin_expr(age_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(age_expr < 18)
        .then(pl.lit("u18"))
        .when(age_expr < 25)
        .then(pl.lit("18_24"))
        .when(age_expr < 35)
        .then(pl.lit("25_34"))
        .when(age_expr < 45)
        .then(pl.lit("35_44"))
        .when(age_expr < 55)
        .then(pl.lit("45_54"))
        .when(age_expr < 65)
        .then(pl.lit("55_64"))
        .otherwise(pl.lit("65_plus"))
    )


def _normalized_text_expr(column_name: str, fallback: str = "unknown") -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .replace({"": fallback, "none": "none", "nan": fallback, "null": fallback})
        .fill_null(fallback)
    )


def prepare_customer_model_features(customers_lf: pl.LazyFrame) -> pl.DataFrame:
    age_median = (
        customers_lf.select(pl.col("age").cast(pl.Float64, strict=False).drop_nulls().median().alias("age_median"))
        .collect(engine="streaming")
        .get_column("age_median")[0]
    )
    return (
        customers_lf.select(
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("age").cast(pl.Float64, strict=False).fill_null(age_median).alias("age_filled"),
            _normalized_text_expr("club_member_status").alias("club_member_status_norm"),
            _normalized_text_expr("fashion_news_frequency").alias("fashion_news_frequency_norm"),
            _normalized_text_expr("postal_code").alias("postal_code_norm"),
        )
        .with_columns(_age_bin_expr(pl.col("age_filled")).alias("cold_age_bin"))
        .with_columns(
            pl.concat_str(
                ["cold_age_bin", "club_member_status_norm", "fashion_news_frequency_norm"],
                separator="|",
            ).alias("cold_profile_key"),
            pl.concat_str(
                ["club_member_status_norm", "fashion_news_frequency_norm"],
                separator="|",
            ).alias("cold_club_news_key"),
        )
        .select(
            "customer_id",
            "cold_age_bin",
            "club_member_status_norm",
            "fashion_news_frequency_norm",
            "cold_profile_key",
            "cold_club_news_key",
            "postal_code_norm",
        )
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


def _build_article_attribute_maps(article_features: pl.DataFrame) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    for attr_col in [
        "article_product_type_no",
        "article_garment_group_no",
        "article_colour_group_code",
        "article_section_no",
    ]:
        maps[attr_col] = {
            row["article_id"]: float(row[attr_col])
            for row in article_features.select("article_id", attr_col).drop_nulls().iter_rows(named=True)
        }
    return maps


def _build_customer_preferred_attribute(
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    attr_col: str,
) -> dict[str, float]:
    pref_df = (
        history_tx.join(article_features.select("article_id", attr_col), on="article_id", how="left")
        .drop_nulls([attr_col])
        .group_by(["customer_id", attr_col])
        .agg(
            pl.len().alias("txn_cnt"),
            pl.max("t_dat").alias("last_date"),
        )
        .sort(["customer_id", "txn_cnt", "last_date"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col(attr_col).first().alias(attr_col))
    )
    return {row["customer_id"]: float(row[attr_col]) for row in pref_df.iter_rows(named=True)}


def _build_attribute_top_items(
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    attr_col: str,
    reference_date: date,
    top_n: int = ATTRIBUTE_RECALL_TOP,
) -> dict[float, list[str]]:
    cutoff = reference_date - timedelta(days=30)
    top_df = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .join(article_features.select("article_id", attr_col), on="article_id", how="left")
        .drop_nulls([attr_col])
        .group_by([attr_col, "article_id"])
        .agg(
            pl.len().alias("pop_30d"),
            pl.n_unique("customer_id").alias("buyers_30d"),
        )
        .sort([attr_col, "pop_30d", "buyers_30d", "article_id"], descending=[False, True, True, False])
        .group_by(attr_col)
        .agg(pl.col("article_id").head(top_n).alias("top_items"))
    )
    return {float(row[attr_col]): row["top_items"] or [] for row in top_df.iter_rows(named=True)}


def _build_recent_global_items(
    history_tx: pl.DataFrame,
    reference_date: date,
    top_n: int = RECENT_GLOBAL_RECALL_TOP,
) -> list[str]:
    cutoff = reference_date - timedelta(days=7)
    return (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by("article_id")
        .agg(
            pl.len().alias("pop_7d"),
            pl.n_unique("customer_id").alias("buyers_7d"),
        )
        .sort(["pop_7d", "buyers_7d", "article_id"], descending=[True, True, False])
        .select(pl.col("article_id").head(top_n))
        .get_column("article_id")
        .to_list()
    )


def _build_cooccurrence_items(
    history_tx: pl.DataFrame,
    reference_date: date,
) -> dict[str, list[tuple[str, float]]]:
    if not ENABLE_COOCCURRENCE_RECALL or COOCCURRENCE_RECALL_TOP <= 0:
        return {}

    cutoff = reference_date - timedelta(days=COOCCURRENCE_DAYS)
    baskets = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by(["customer_id", "t_dat"])
        .agg(pl.col("article_id").unique(maintain_order=True).alias("items"))
        .with_columns(pl.col("items").list.len().alias("basket_size"))
        .filter((pl.col("basket_size") > 1) & (pl.col("basket_size") <= COOCCURRENCE_MAX_BASKET_SIZE))
        .select("items")
    )

    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for items in baskets.get_column("items").to_list():
        basket_items = [str(item) for item in items if item is not None]
        for item in basket_items:
            counter = pair_counts[item]
            for other_item in basket_items:
                if other_item != item:
                    counter[other_item] += 1

    cooccurrence_items: dict[str, list[tuple[str, float]]] = {}
    for item, counter in pair_counts.items():
        cooccurrence_items[item] = [
            (other_item, float(count))
            for other_item, count in counter.most_common(COOCCURRENCE_TOP_PER_ITEM)
        ]
    print(
        "cooccurrence recall built: "
        f"items={len(cooccurrence_items)} days={COOCCURRENCE_DAYS} "
        f"top_per_item={COOCCURRENCE_TOP_PER_ITEM}"
    )
    return cooccurrence_items


def _build_customer_cold_start_segments(customer_features: pl.DataFrame) -> dict[str, tuple[str, str, str, str]]:
    if not ENABLE_COLD_START_RECALL:
        return {}
    result: dict[str, tuple[str, str, str, str]] = {}
    for row in customer_features.select(
        "customer_id",
        "cold_profile_key",
        "cold_age_bin",
        "cold_club_news_key",
        "postal_code_norm",
    ).iter_rows(named=True):
        result[row["customer_id"]] = (
            str(row["cold_profile_key"]),
            str(row["cold_age_bin"]),
            str(row["cold_club_news_key"]),
            str(row["postal_code_norm"]),
        )
    return result


def _build_segment_top_items(
    history_tx: pl.DataFrame,
    customer_features: pl.DataFrame,
    segment_col: str,
    reference_date: date,
    top_n: int,
    min_buyers: int,
) -> dict[str, list[tuple[str, float]]]:
    cutoff = reference_date - timedelta(days=COLD_START_DAYS)
    top_df = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .join(customer_features.select("customer_id", segment_col), on="customer_id", how="left")
        .drop_nulls([segment_col])
        .group_by([segment_col, "article_id"])
        .agg(
            pl.len().alias("pop_segment"),
            pl.n_unique("customer_id").alias("buyers_segment"),
        )
        .filter(pl.col("buyers_segment") >= min_buyers)
        .with_columns(
            (pl.col("pop_segment").cast(pl.Float64) + pl.col("buyers_segment").cast(pl.Float64) * 0.5).alias(
                "segment_score"
            )
        )
        .sort([segment_col, "segment_score", "buyers_segment", "article_id"], descending=[False, True, True, False])
        .group_by(segment_col)
        .agg(
            pl.struct("article_id", "segment_score")
            .head(top_n)
            .alias("segment_items")
        )
    )
    result: dict[str, list[tuple[str, float]]] = {}
    for row in top_df.iter_rows(named=True):
        result[str(row[segment_col])] = [
            (str(item["article_id"]), float(item["segment_score"]))
            for item in (row["segment_items"] or [])
        ]
    return result


def _build_cold_start_top_items(
    history_tx: pl.DataFrame,
    customer_features: pl.DataFrame,
    reference_date: date,
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    if not ENABLE_COLD_START_RECALL or COLD_START_RECALL_TOP <= 0:
        return {}

    segment_top_items = {
        "cold_profile_key": _build_segment_top_items(
            history_tx,
            customer_features,
            "cold_profile_key",
            reference_date,
            top_n=COLD_START_SEGMENT_TOP,
            min_buyers=COLD_START_MIN_SEGMENT_BUYERS,
        ),
        "cold_age_bin": _build_segment_top_items(
            history_tx,
            customer_features,
            "cold_age_bin",
            reference_date,
            top_n=COLD_START_SEGMENT_TOP,
            min_buyers=COLD_START_MIN_SEGMENT_BUYERS,
        ),
        "cold_club_news_key": _build_segment_top_items(
            history_tx,
            customer_features,
            "cold_club_news_key",
            reference_date,
            top_n=COLD_START_SEGMENT_TOP,
            min_buyers=COLD_START_MIN_SEGMENT_BUYERS,
        ),
    }
    if ENABLE_POSTAL_COLD_START_RECALL:
        segment_top_items["postal_code_norm"] = _build_segment_top_items(
            history_tx,
            customer_features,
            "postal_code_norm",
            reference_date,
            top_n=COLD_START_SEGMENT_TOP,
            min_buyers=POSTAL_COLD_START_MIN_BUYERS,
        )

    print(
        "cold-start recall built: "
        + ", ".join(f"{name}={len(items)}" for name, items in segment_top_items.items())
    )
    return segment_top_items


def build_recall_context(
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_features: pl.DataFrame,
    reference_date: date,
) -> dict[str, Any]:
    attr_cols = {
        "product_type": "article_product_type_no",
        "garment_group": "article_garment_group_no",
    }
    if ENABLE_COLOUR_RECALL:
        attr_cols["colour"] = "article_colour_group_code"
    if ENABLE_SECTION_RECALL:
        attr_cols["section"] = "article_section_no"
    preferred_attributes = {
        name: _build_customer_preferred_attribute(history_tx, article_features, attr_col)
        for name, attr_col in attr_cols.items()
    }
    attribute_top_items = {
        name: _build_attribute_top_items(history_tx, article_features, attr_col, reference_date)
        for name, attr_col in attr_cols.items()
    }
    return {
        "preferred_attributes": preferred_attributes,
        "attribute_top_items": attribute_top_items,
        "recent_global_items": _build_recent_global_items(history_tx, reference_date),
        "cooccurrence_items": _build_cooccurrence_items(history_tx, reference_date),
        "customer_cold_segments": _build_customer_cold_start_segments(customer_features),
        "cold_start_top_items": _build_cold_start_top_items(history_tx, customer_features, reference_date),
    }


def build_candidate_frame(
    customer_ids: list[str],
    artifacts: Any,
    ranker_config: RankerConfig,
    max_candidates: int = MAX_CANDIDATES_PER_CUSTOMER,
    actual_items_by_customer: dict[str, list[str]] | None = None,
    include_actual_items: bool = False,
    recall_context: dict[str, Any] | None = None,
) -> pl.DataFrame:
    baseline_recall_k = min(max_candidates, max(BASELINE_RECALL_TOP, MAX_K))
    baseline_lists = predict_lists_for_customers(
        customer_ids=customer_ids,
        artifacts=artifacts,
        ranker_config=ranker_config,
        k=baseline_recall_k,
    )

    rows: list[dict[str, Any]] = []
    actual_items_by_customer = actual_items_by_customer or {}
    recall_context = recall_context or {}
    preferred_attributes: dict[str, dict[str, float]] = recall_context.get("preferred_attributes", {})
    attribute_top_items: dict[str, dict[float, list[str]]] = recall_context.get("attribute_top_items", {})
    recent_global_items: list[str] = recall_context.get("recent_global_items", [])
    cooccurrence_items: dict[str, list[tuple[str, float]]] = recall_context.get("cooccurrence_items", {})
    customer_cold_segments: dict[str, tuple[str, str, str, str]] = recall_context.get("customer_cold_segments", {})
    cold_start_top_items: dict[str, dict[str, list[tuple[str, float]]]] = recall_context.get(
        "cold_start_top_items", {}
    )

    def add_candidate(
        customer_rows: list[dict[str, Any]],
        seen: set[str],
        customer_id: str,
        article_id: str,
        rank: int,
        score: float,
        source_col: str,
        extra_values: dict[str, float] | None = None,
    ) -> None:
        extra_values = extra_values or {}
        if article_id in seen:
            for row in reversed(customer_rows):
                if row["article_id"] == article_id:
                    row[source_col] = 1.0
                    row["candidate_score"] = max(float(row["candidate_score"]), score)
                    row["candidate_rank"] = min(int(row["candidate_rank"]), rank)
                    for key, value in extra_values.items():
                        row[key] = max(float(row.get(key, 0.0)), float(value))
                    return
        customer_rows.append(
            {
                "customer_id": customer_id,
                "article_id": article_id,
                "candidate_rank": rank,
                "candidate_score": score,
                "source_baseline": 0.0,
                "source_product_type": 0.0,
                "source_garment_group": 0.0,
                "source_colour": 0.0,
                "source_section": 0.0,
                "source_recent_global": 0.0,
                "source_cooccurrence": 0.0,
                "cooccurrence_score": 0.0,
                "source_cold_start": 0.0,
                "cold_start_score": 0.0,
                "source_actual_in_train": 0.0,
                source_col: 1.0,
                **extra_values,
            }
        )
        seen.add(article_id)

    for customer_id, baseline_items in zip(customer_ids, baseline_lists):
        seen: set[str] = set()
        customer_rows: list[dict[str, Any]] = []
        for rank, article_id in enumerate(baseline_items, start=1):
            add_candidate(
                customer_rows,
                seen,
                customer_id,
                article_id,
                rank,
                1.0 / rank,
                "source_baseline",
            )

        next_rank = len(customer_rows) + 1
        if len(customer_rows) < max_candidates and cooccurrence_items:
            co_scores: dict[str, float] = {}
            for history_pos, history_item in enumerate(
                artifacts.user_recent_history.get(customer_id, [])[:COOCCURRENCE_HISTORY_TOP],
                start=1,
            ):
                for co_item, co_score in cooccurrence_items.get(history_item, []):
                    if co_item in seen:
                        continue
                    co_scores[co_item] = co_scores.get(co_item, 0.0) + co_score / history_pos

            for co_rank, (article_id, co_score) in enumerate(
                sorted(co_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:COOCCURRENCE_RECALL_TOP],
                start=1,
            ):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.12 / max(next_rank, 1),
                    "source_cooccurrence",
                    {"cooccurrence_score": co_score / co_rank},
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break

        for source_name, source_col in [
            ("product_type", "source_product_type"),
            ("garment_group", "source_garment_group"),
            ("colour", "source_colour"),
            ("section", "source_section"),
        ]:
            if source_name == "colour" and not ENABLE_COLOUR_RECALL:
                continue
            if source_name == "section" and not ENABLE_SECTION_RECALL:
                continue
            pref_value = preferred_attributes.get(source_name, {}).get(customer_id)
            if pref_value is None:
                continue
            for article_id in attribute_top_items.get(source_name, {}).get(pref_value, []):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.15 / max(next_rank, 1),
                    source_col,
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break
                if len(customer_rows) >= max_candidates:
                    break

        history_item_count = len(artifacts.user_recent_history.get(customer_id, [])) + len(
            artifacts.user_long_history.get(customer_id, [])
        )
        if (
            len(customer_rows) < max_candidates
            and cold_start_top_items
            and history_item_count <= COLD_START_HISTORY_MAX_ITEMS
        ):
            segment_values = customer_cold_segments.get(customer_id)
            segment_plan: list[tuple[str, str, float]] = []
            if segment_values is not None:
                profile_key, age_bin, club_news_key, postal_code = segment_values
                segment_plan.extend(
                    [
                        ("cold_profile_key", profile_key, 1.0),
                        ("cold_age_bin", age_bin, 0.7),
                        ("cold_club_news_key", club_news_key, 0.5),
                    ]
                )
                if ENABLE_POSTAL_COLD_START_RECALL:
                    segment_plan.append(("postal_code_norm", postal_code, 0.9))

            cold_scores: dict[str, float] = {}
            for segment_name, segment_value, segment_weight in segment_plan:
                segment_items = cold_start_top_items.get(segment_name, {}).get(segment_value, [])
                for pos, (article_id, segment_score) in enumerate(segment_items, start=1):
                    cold_scores[article_id] = cold_scores.get(article_id, 0.0) + (
                        segment_weight * np.log1p(segment_score) / pos
                    )

            for cold_rank, (article_id, cold_score) in enumerate(
                sorted(cold_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:COLD_START_RECALL_TOP],
                start=1,
            ):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.10 / max(next_rank, 1),
                    "source_cold_start",
                    {"cold_start_score": cold_score / cold_rank},
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break

        if len(customer_rows) < max_candidates:
            for article_id in recent_global_items:
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.08 / max(next_rank, 1),
                    "source_recent_global",
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break

        if include_actual_items:
            for article_id in actual_items_by_customer.get(customer_id, []):
                if article_id in seen:
                    continue
                customer_rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": article_id,
                        "candidate_rank": max_candidates + 1,
                        "candidate_score": 0.0,
                        "source_baseline": 0.0,
                        "source_product_type": 0.0,
                        "source_garment_group": 0.0,
                        "source_colour": 0.0,
                        "source_section": 0.0,
                        "source_recent_global": 0.0,
                        "source_cooccurrence": 0.0,
                        "cooccurrence_score": 0.0,
                        "source_cold_start": 0.0,
                        "cold_start_score": 0.0,
                        "source_actual_in_train": 1.0,
                    }
                )
                seen.add(article_id)
        rows.extend(customer_rows[:max_candidates] if not include_actual_items else customer_rows)

    if not rows:
        return pl.DataFrame(
            {
                "customer_id": [],
                "article_id": [],
                "candidate_rank": [],
                "candidate_score": [],
                "source_baseline": [],
                "source_product_type": [],
                "source_garment_group": [],
                "source_colour": [],
                "source_section": [],
                "source_recent_global": [],
                "source_cooccurrence": [],
                "cooccurrence_score": [],
                "source_cold_start": [],
                "cold_start_score": [],
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
                pl.col("item_pop_7d").fill_null(0.0)
                / (pl.col("item_pop_30d").fill_null(0.0) + 1.0)
            ).alias("item_pop_ratio_7d_30d"),
            (
                pl.col("item_pop_30d").fill_null(0.0)
                / (pl.col("item_pop_all").fill_null(0.0) + 1.0)
            ).alias("item_pop_ratio_30d_all"),
            (pl.col("user_avg_price").fill_null(0.0) - pl.col("item_avg_price").fill_null(0.0))
            .abs()
            .alias("price_diff_user_item"),
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
    x = train_df.select(MODEL_FEATURE_COLUMNS).to_numpy()
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
        n_estimators=LGBM_N_ESTIMATORS,
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
        feature_name=MODEL_FEATURE_COLUMNS,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    print(f"LightGBM trained: rows={len(y_train)} positives={int(positive_count)}")
    return model


def score_feature_frame(model: Any, feature_df: pl.DataFrame) -> pl.DataFrame:
    if feature_df.is_empty():
        return feature_df.with_columns(pl.lit(0.0).alias("score"))
    scores = model.predict_proba(feature_df.select(MODEL_FEATURE_COLUMNS).to_numpy())[:, 1]
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
    customer_features: pl.DataFrame,
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
    recall_context = build_recall_context(
        history_tx=history_tx,
        article_features=article_features,
        customer_features=customer_features,
        reference_date=window.history_end,
    )
    candidates = build_candidate_frame(
        customer_ids=customers,
        artifacts=artifacts,
        ranker_config=ranker_config,
        max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
        actual_items_by_customer=actual_map,
        include_actual_items=True,
        recall_context=recall_context,
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


def build_labeled_windows_dataset(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    windows: list[WindowSpec],
    customer_cap: int,
) -> pl.DataFrame:
    if not windows:
        return pl.DataFrame()
    if len(windows) == 1:
        return build_labeled_window_dataset(
            transactions=transactions,
            article_department=article_department,
            article_features=article_features,
            customer_features=customer_features,
            customer_age_bin=customer_age_bin,
            ranker_config=ranker_config,
            window=windows[0],
            customer_cap=customer_cap,
        )

    per_window_cap = max(1000, customer_cap // len(windows))
    frames: list[pl.DataFrame] = []
    for idx, window in enumerate(windows, start=1):
        print(
            f"training window {idx}/{len(windows)}: "
            f"history_end={window.history_end} labels={window.label_start}~{window.label_end} "
            f"customer_cap={per_window_cap}"
        )
        frame = build_labeled_window_dataset(
            transactions=transactions,
            article_department=article_department,
            article_features=article_features,
            customer_features=customer_features,
            customer_age_bin=customer_age_bin,
            ranker_config=ranker_config,
            window=window,
            customer_cap=per_window_cap,
        )
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def run_single_window_validation(
    model: Any,
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_features: pl.DataFrame,
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
    recall_context = build_recall_context(
        history_tx=history_tx,
        article_features=article_features,
        customer_features=customer_features,
        reference_date=window.history_end,
    )
    candidates = build_candidate_frame(
        customer_ids=customers,
        artifacts=artifacts,
        ranker_config=ranker_config,
        max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
        recall_context=recall_context,
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


def build_rolling_training_windows(
    transactions: pl.DataFrame,
    latest_label_end: date,
    window_count: int = TRAIN_WINDOW_COUNT,
) -> list[WindowSpec]:
    min_date = transactions.get_column("t_dat").min()
    windows: list[WindowSpec] = []
    for offset in range(max(1, window_count)):
        label_end = latest_label_end - timedelta(days=LABEL_WINDOW_DAYS * offset)
        label_start = label_end - timedelta(days=LABEL_WINDOW_DAYS - 1)
        history_end = label_start - timedelta(days=1)
        if history_end <= min_date:
            continue
        windows.append(WindowSpec(history_end=history_end, label_start=label_start, label_end=label_end))
    return list(reversed(windows))


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
            "importance": [
                float(model.feature_importances_[MODEL_FEATURE_COLUMNS.index(feature)])
                if feature in MODEL_FEATURE_COLUMNS
                else 0.0
                for feature in FEATURE_COLUMNS
            ],
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
    customer_features: pl.DataFrame,
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
    recall_context = build_recall_context(
        history_tx=transactions,
        article_features=article_features,
        customer_features=customer_features,
        reference_date=reference_date,
    )

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
            recall_context=recall_context,
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
        # Release per-chunk tables aggressively to reduce memory pressure.
        del candidates, features, scored, pred_map, fallback_lists
        gc.collect()

    submission = pl.DataFrame(all_rows)
    validate_submission_format(submission, customer_ids, k=MAX_K)
    output_dir = prepare_output_dir(output_dir)
    output_path = output_dir / "submission.csv"
    submission.write_csv(output_path)
    print(f"LGBM submission saved: {output_path}")
    return submission


def run_lgbm_pipeline(base_path: Path = BASE_PATH) -> dict[str, Any]:
    base_path = resolve_base_path(base_path)
    print(
        "LGBM params: "
        f"train_cap={TRAIN_CUSTOMER_CAP}, valid_cap={VALIDATION_CUSTOMER_CAP}, "
        f"max_candidates={MAX_CANDIDATES_PER_CUSTOMER}, baseline_recall_top={BASELINE_RECALL_TOP}, "
        f"attribute_top={ATTRIBUTE_RECALL_TOP}, recent_global_top={RECENT_GLOBAL_RECALL_TOP}, "
        f"n_estimators={LGBM_N_ESTIMATORS}, "
        f"colour_recall={ENABLE_COLOUR_RECALL}, section_recall={ENABLE_SECTION_RECALL}, "
        f"cooccurrence_recall={ENABLE_COOCCURRENCE_RECALL}, cooccurrence_top={COOCCURRENCE_RECALL_TOP}, "
        f"cold_start_recall={ENABLE_COLD_START_RECALL}, cold_start_top={COLD_START_RECALL_TOP}, "
        f"cold_history_max={COLD_START_HISTORY_MAX_ITEMS}, postal_cold={ENABLE_POSTAL_COLD_START_RECALL}, "
        f"train_window_count={TRAIN_WINDOW_COUNT}, "
        f"drop_noisy_features={DROP_NOISY_FEATURES}, feature_experiment={FEATURE_EXPERIMENT}"
    )
    print(f"model feature count: {len(MODEL_FEATURE_COLUMNS)} / raw feature count: {len(FEATURE_COLUMNS)}")
    tables = load_tables(base_path)
    summary = summarize_inputs(tables)
    transactions = prepare_transactions(tables["transactions"])
    article_department = prepare_article_department(tables["articles"])
    article_features = prepare_article_model_features(tables["articles"])
    customer_features = prepare_customer_model_features(tables["customers"])
    customer_age_bin = prepare_customer_age_bin(tables["customers"])
    ranker_config = load_ranker_from_cache(output_dir=prepare_output_dir(OUTPUT_DIR), fallback=DEFAULT_RANKER)
    print(f"candidate ranker: {describe_ranker(ranker_config)}")

    train_window, validation_window = build_train_and_validation_windows(transactions)
    validation_train_windows = build_rolling_training_windows(
        transactions,
        latest_label_end=train_window.label_end,
        window_count=TRAIN_WINDOW_COUNT,
    )
    validation_train_df = build_labeled_windows_dataset(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_features=customer_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        windows=validation_train_windows,
        customer_cap=TRAIN_CUSTOMER_CAP,
    )
    validation_model = train_lgbm_model(validation_train_df)
    validation_metrics = run_single_window_validation(
        model=validation_model,
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_features=customer_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        window=validation_window,
        customer_cap=VALIDATION_CUSTOMER_CAP,
    )
    validation_path = prepare_output_dir(OUTPUT_DIR) / "lgbm_validation_metrics.csv"
    pl.DataFrame([validation_metrics]).write_csv(validation_path)
    print(f"LGBM validation metrics saved: {validation_path}")

    final_window = build_final_training_window(transactions)
    final_train_windows = build_rolling_training_windows(
        transactions,
        latest_label_end=final_window.label_end,
        window_count=TRAIN_WINDOW_COUNT,
    )
    final_train_df = build_labeled_windows_dataset(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_features=customer_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        windows=final_train_windows,
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
        customer_features=customer_features,
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
