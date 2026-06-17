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

import csv
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


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return float(raw_value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


RANDOM_STATE = 610
ID_DTYPE = pl.Int32
ENABLE_ID_MAPPING = _env_bool("LGBM_ENABLE_ID_MAPPING", True)
LABEL_WINDOW_DAYS = _env_int("LGBM_LABEL_WINDOW_DAYS", 7)
VALIDATION_CUSTOMER_CAP = _env_int("LGBM_VALIDATION_CUSTOMER_CAP", 60000)
TRAIN_CUSTOMER_CAP = _env_int("LGBM_TRAIN_CUSTOMER_CAP", 80000)
NEGATIVE_SAMPLE_RATIO = _env_int("LGBM_NEGATIVE_SAMPLE_RATIO", 20)
NEGATIVE_SAMPLE_MODE = os.getenv("LGBM_NEGATIVE_SAMPLE_MODE", "random").strip().lower()
HARD_NEGATIVE_FRACTION = _env_float("LGBM_HARD_NEGATIVE_FRACTION", 0.6)
HARD_NEGATIVE_TOP_RANK = _env_int("LGBM_HARD_NEGATIVE_TOP_RANK", 40)
ENABLE_SEGMENT_RERANK = _env_bool("LGBM_ENABLE_SEGMENT_RERANK", False)
ENABLE_SEGMENT_CANDIDATES = _env_bool("LGBM_ENABLE_SEGMENT_CANDIDATES", False)
SEGMENT_ACTIVE_DAYS = _env_int("LGBM_SEGMENT_ACTIVE_DAYS", 30)
SEGMENT_STALE_DAYS = _env_int("LGBM_SEGMENT_STALE_DAYS", 90)
SEGMENT_COLD_TXN_MAX = _env_int("LGBM_SEGMENT_COLD_TXN_MAX", 2)
SEGMENT_ACTIVE_RECENT_MIN = _env_int("LGBM_SEGMENT_ACTIVE_RECENT_MIN", 2)
SEGMENT_ACTIVE_COOCCURRENCE_BONUS = _env_float("LGBM_SEGMENT_ACTIVE_COOCCURRENCE_BONUS", 0.0)
SEGMENT_ACTIVE_ATTR_BONUS = _env_float("LGBM_SEGMENT_ACTIVE_ATTR_BONUS", 0.0)
SEGMENT_COLD_START_BONUS = _env_float("LGBM_SEGMENT_COLD_START_BONUS", 0.0)
SEGMENT_COLD_RECENT_BONUS = _env_float("LGBM_SEGMENT_COLD_RECENT_BONUS", 0.0)
SEGMENT_STALE_RECENT_BONUS = _env_float("LGBM_SEGMENT_STALE_RECENT_BONUS", 0.0)
SEGMENT_TOP_RANK_BONUS = _env_float("LGBM_SEGMENT_TOP_RANK_BONUS", 0.0)
SEGMENT_TOP_RANK_CUTOFF = _env_int("LGBM_SEGMENT_TOP_RANK_CUTOFF", 24)
MAX_CANDIDATES_PER_CUSTOMER = _env_int("LGBM_MAX_CANDIDATES_PER_CUSTOMER", 100)
BASELINE_RECALL_TOP = _env_int("LGBM_BASELINE_RECALL_TOP", min(80, MAX_CANDIDATES_PER_CUSTOMER))
# Keep predictions unchanged while lowering per-chunk peak memory.
SUBMISSION_CUSTOMER_CHUNK = _env_int("LGBM_SUBMISSION_CUSTOMER_CHUNK", 20000)
ATTRIBUTE_RECALL_TOP = _env_int("LGBM_ATTRIBUTE_RECALL_TOP", 12)
ENABLE_GROUP_RECALL = _env_bool("LGBM_ENABLE_GROUP_RECALL", False)
GROUP_RECALL_TOP = _env_int("LGBM_GROUP_RECALL_TOP", 6)
GROUP_RECALL_DAYS = _env_int("LGBM_GROUP_RECALL_DAYS", 30)
RECENT_GLOBAL_RECALL_TOP = _env_int("LGBM_RECENT_GLOBAL_RECALL_TOP", 18)
LGBM_N_ESTIMATORS = _env_int("LGBM_N_ESTIMATORS", 450)
LGBM_LEARNING_RATE = _env_float("LGBM_LEARNING_RATE", 0.045)
LGBM_NUM_LEAVES = _env_int("LGBM_NUM_LEAVES", 96)
LGBM_MAX_DEPTH = _env_int("LGBM_MAX_DEPTH", -1)
LGBM_MIN_CHILD_SAMPLES = _env_int("LGBM_MIN_CHILD_SAMPLES", 80)
LGBM_SUBSAMPLE = _env_float("LGBM_SUBSAMPLE", 0.85)
LGBM_COLSAMPLE_BYTREE = _env_float("LGBM_COLSAMPLE_BYTREE", 0.85)
LGBM_REG_ALPHA = _env_float("LGBM_REG_ALPHA", 0.1)
LGBM_REG_LAMBDA = _env_float("LGBM_REG_LAMBDA", 1.0)
LGBM_N_JOBS = _env_int("LGBM_N_JOBS", -1)
LGBM_FORCE_COL_WISE = _env_bool("LGBM_FORCE_COL_WISE", False)
SAVE_FEATURE_IMPORTANCE = _env_bool("LGBM_SAVE_FEATURE_IMPORTANCE", True)
LGBM_MODEL_TYPE = os.getenv("LGBM_MODEL_TYPE", "classifier").strip().lower()
SKIP_SUBMISSION = _env_bool("LGBM_SKIP_SUBMISSION", False)
SKIP_VALIDATION = _env_bool("LGBM_SKIP_VALIDATION", False)
STREAM_SUBMISSION = _env_bool("LGBM_STREAM_SUBMISSION", True)
ENABLE_COLOUR_RECALL = _env_bool("LGBM_ENABLE_COLOUR_RECALL", False)
ENABLE_SECTION_RECALL = _env_bool("LGBM_ENABLE_SECTION_RECALL", False)
ENABLE_COOCCURRENCE_RECALL = _env_bool("LGBM_ENABLE_COOCCURRENCE_RECALL", True)
COOCCURRENCE_DAYS = _env_int("LGBM_COOCCURRENCE_DAYS", 30)
COOCCURRENCE_TOP_PER_ITEM = _env_int("LGBM_COOCCURRENCE_TOP_PER_ITEM", 16)
COOCCURRENCE_HISTORY_TOP = _env_int("LGBM_COOCCURRENCE_HISTORY_TOP", 4)
COOCCURRENCE_RECALL_TOP = _env_int("LGBM_COOCCURRENCE_RECALL_TOP", 12)
COOCCURRENCE_MAX_BASKET_SIZE = _env_int("LGBM_COOCCURRENCE_MAX_BASKET_SIZE", 20)
ENABLE_ITEMCF_RECALL = _env_bool("LGBM_ENABLE_ITEMCF_RECALL", False)
ITEMCF_DAYS = _env_int("LGBM_ITEMCF_DAYS", 45)
ITEMCF_TOP_PER_ITEM = _env_int("LGBM_ITEMCF_TOP_PER_ITEM", 24)
ITEMCF_HISTORY_TOP = _env_int("LGBM_ITEMCF_HISTORY_TOP", 5)
ITEMCF_RECALL_TOP = _env_int("LGBM_ITEMCF_RECALL_TOP", 16)
ITEMCF_MAX_BASKET_SIZE = _env_int("LGBM_ITEMCF_MAX_BASKET_SIZE", 20)
ENABLE_PRODUCT_CODE_RECALL = _env_bool("LGBM_ENABLE_PRODUCT_CODE_RECALL", False)
PRODUCT_CODE_DAYS = _env_int("LGBM_PRODUCT_CODE_DAYS", 60)
PRODUCT_CODE_HISTORY_TOP = _env_int("LGBM_PRODUCT_CODE_HISTORY_TOP", 4)
PRODUCT_CODE_RECALL_TOP = _env_int("LGBM_PRODUCT_CODE_RECALL_TOP", 8)
ENABLE_PRICE_BAND_RECALL = _env_bool("LGBM_ENABLE_PRICE_BAND_RECALL", False)
PRICE_BAND_DAYS = _env_int("LGBM_PRICE_BAND_DAYS", 30)
PRICE_BAND_RECALL_TOP = _env_int("LGBM_PRICE_BAND_RECALL_TOP", 12)
PRICE_BAND_COUNT = _env_int("LGBM_PRICE_BAND_COUNT", 5)
ENABLE_MULTI_DAY_TREND_RECALL = _env_bool("LGBM_ENABLE_MULTI_DAY_TREND_RECALL", False)
MULTI_DAY_TREND_RECALL_TOP = _env_int("LGBM_MULTI_DAY_TREND_RECALL_TOP", 12)
ENABLE_SELLABLE_FILTER = _env_bool("LGBM_ENABLE_SELLABLE_FILTER", False)
SELLABLE_DAYS = _env_int("LGBM_SELLABLE_DAYS", 42)
SELLABLE_ALLOW_HISTORY = _env_bool("LGBM_SELLABLE_ALLOW_HISTORY", True)
ENABLE_DYNAMIC_ITEM_FEATURES = _env_bool("LGBM_ENABLE_DYNAMIC_ITEM_FEATURES", False)
ENABLE_COLD_START_RECALL = _env_bool("LGBM_ENABLE_COLD_START_RECALL", True)
COLD_START_HISTORY_MAX_ITEMS = _env_int("LGBM_COLD_START_HISTORY_MAX_ITEMS", 2)
COLD_START_DAYS = _env_int("LGBM_COLD_START_DAYS", 30)
COLD_START_RECALL_TOP = _env_int("LGBM_COLD_START_RECALL_TOP", 18)
COLD_START_SEGMENT_TOP = _env_int("LGBM_COLD_START_SEGMENT_TOP", 32)
COLD_START_MIN_SEGMENT_BUYERS = _env_int("LGBM_COLD_START_MIN_SEGMENT_BUYERS", 3)
SEGMENT_ACTIVE_BASELINE_TOP = _env_int("LGBM_SEGMENT_ACTIVE_BASELINE_TOP", BASELINE_RECALL_TOP)
SEGMENT_WARM_BASELINE_TOP = _env_int("LGBM_SEGMENT_WARM_BASELINE_TOP", BASELINE_RECALL_TOP)
SEGMENT_COLD_BASELINE_TOP = _env_int("LGBM_SEGMENT_COLD_BASELINE_TOP", min(BASELINE_RECALL_TOP, 70))
SEGMENT_COLD_START_RECALL_TOP = _env_int(
    "LGBM_SEGMENT_COLD_START_RECALL_TOP",
    max(COLD_START_RECALL_TOP, 24),
)
ENABLE_POSTAL_COLD_START_RECALL = _env_bool("LGBM_ENABLE_POSTAL_COLD_START_RECALL", False)
POSTAL_COLD_START_MIN_BUYERS = _env_int("LGBM_POSTAL_COLD_START_MIN_BUYERS", 8)
TRAIN_WINDOW_COUNT = _env_int("LGBM_TRAIN_WINDOW_COUNT", 1)
DROP_NOISY_FEATURES = _env_bool("LGBM_DROP_NOISY_FEATURES", False)
FEATURE_EXPERIMENT = os.getenv("LGBM_FEATURE_EXPERIMENT", "best").strip().lower()
ENABLE_USER_ATTR_FEATURES = _env_bool("LGBM_ENABLE_USER_ATTR_FEATURES", False)
USER_ATTR_FEATURE_DAYS = _env_int("LGBM_USER_ATTR_FEATURE_DAYS", 180)
ENABLE_EXTENDED_USER_ATTR_FEATURES = _env_bool("LGBM_ENABLE_EXTENDED_USER_ATTR_FEATURES", False)
ENABLE_SOURCE_RANK_FEATURES = _env_bool("LGBM_ENABLE_SOURCE_RANK_FEATURES", False)

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
    "source_count",
    "source_baseline",
    "baseline_rr",
    "source_product_type",
    "product_type_rr",
    "source_garment_group",
    "garment_group_rr",
    "source_colour",
    "colour_rr",
    "source_section",
    "section_rr",
    "source_department_group",
    "department_group_rr",
    "source_index_group",
    "index_group_rr",
    "source_product_group",
    "product_group_rr",
    "source_recent_global",
    "recent_global_rr",
    "source_cooccurrence",
    "cooccurrence_rr",
    "cooccurrence_score",
    "source_itemcf",
    "itemcf_rr",
    "itemcf_score",
    "source_product_code",
    "product_code_rr",
    "product_code_score",
    "source_price_band",
    "price_band_rr",
    "price_band_score",
    "source_trend",
    "trend_rr",
    "trend_score",
    "source_cold_start",
    "cold_start_rr",
    "cold_start_score",
    "user_txn_all",
    "user_txn_30d",
    "user_unique_items",
    "user_recency_days",
    "user_avg_price",
    "user_pref_channel",
    "user_pref_department",
    "customer_age_bin_id",
    "item_pop_1d",
    "item_buyers_1d",
    "item_rank_1d",
    "item_pop_3d",
    "item_buyers_3d",
    "item_rank_3d",
    "item_pop_7d",
    "item_pop_30d",
    "item_pop_14d",
    "item_buyers_14d",
    "item_rank_14d",
    "item_pop_all",
    "item_buyers_7d",
    "item_buyers_30d",
    "item_buyers_all",
    "item_recency_days",
    "item_avg_price",
    "item_pop_ratio_1d_7d",
    "item_pop_ratio_3d_14d",
    "item_pop_ratio_7d_30d",
    "item_pop_ratio_30d_all",
    "ua_cnt",
    "ua_recency_days",
    "price_diff_user_item",
    "same_department_as_pref",
    "user_product_type_cnt",
    "user_product_type_share",
    "user_product_type_recency_days",
    "user_garment_group_cnt",
    "user_garment_group_share",
    "user_garment_group_recency_days",
    "user_colour_cnt",
    "user_colour_share",
    "user_colour_recency_days",
    "user_section_cnt",
    "user_section_share",
    "user_section_recency_days",
    "user_department_cnt",
    "user_department_share",
    "user_department_recency_days",
    "user_index_group_cnt",
    "user_index_group_share",
    "user_index_group_recency_days",
    "user_product_group_cnt",
    "user_product_group_share",
    "user_product_group_recency_days",
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
SOURCE_RANK_FEATURE_COLUMNS = {
    "source_count",
    "baseline_rr",
    "product_type_rr",
    "garment_group_rr",
    "colour_rr",
    "section_rr",
    "department_group_rr",
    "index_group_rr",
    "product_group_rr",
    "recent_global_rr",
    "cooccurrence_rr",
    "itemcf_rr",
    "product_code_rr",
    "price_band_rr",
    "trend_rr",
    "cold_start_rr",
}
USER_ATTR_FEATURE_COLUMNS = {
    "user_product_type_cnt",
    "user_product_type_share",
    "user_product_type_recency_days",
    "user_garment_group_cnt",
    "user_garment_group_share",
    "user_garment_group_recency_days",
    "user_colour_cnt",
    "user_colour_share",
    "user_colour_recency_days",
    "user_section_cnt",
    "user_section_share",
    "user_section_recency_days",
}
EXTENDED_USER_ATTR_FEATURE_COLUMNS = {
    "user_department_cnt",
    "user_department_share",
    "user_department_recency_days",
    "user_index_group_cnt",
    "user_index_group_share",
    "user_index_group_recency_days",
    "user_product_group_cnt",
    "user_product_group_share",
    "user_product_group_recency_days",
}
BASE_USER_ATTR_PREF_SPECS = [
    ("product_type", "article_product_type_no"),
    ("garment_group", "article_garment_group_no"),
    ("colour", "article_colour_group_code"),
    ("section", "article_section_no"),
]
EXTENDED_USER_ATTR_PREF_SPECS = [
    ("department", "article_department_no"),
    ("index_group", "article_index_group_no"),
    ("product_group", "article_product_group_no"),
]
USER_ATTR_PREF_SPECS = (
    BASE_USER_ATTR_PREF_SPECS + EXTENDED_USER_ATTR_PREF_SPECS
    if ENABLE_EXTENDED_USER_ATTR_FEATURES
    else BASE_USER_ATTR_PREF_SPECS
)
DYNAMIC_ITEM_FEATURE_COLUMNS = {
    "item_pop_1d",
    "item_buyers_1d",
    "item_rank_1d",
    "item_pop_3d",
    "item_buyers_3d",
    "item_rank_3d",
    "item_pop_14d",
    "item_buyers_14d",
    "item_rank_14d",
    "item_pop_ratio_1d_7d",
    "item_pop_ratio_3d_14d",
}


def _resolve_model_feature_columns() -> list[str]:
    feature_columns = list(FEATURE_COLUMNS)
    if DROP_NOISY_FEATURES:
        feature_columns = [feature for feature in feature_columns if feature not in NOISY_FEATURE_COLUMNS]
    if not ENABLE_SOURCE_RANK_FEATURES:
        feature_columns = [feature for feature in feature_columns if feature not in SOURCE_RANK_FEATURE_COLUMNS]
    if not ENABLE_USER_ATTR_FEATURES:
        disabled_user_attr_features = USER_ATTR_FEATURE_COLUMNS | EXTENDED_USER_ATTR_FEATURE_COLUMNS
        feature_columns = [feature for feature in feature_columns if feature not in disabled_user_attr_features]
    elif not ENABLE_EXTENDED_USER_ATTR_FEATURES:
        feature_columns = [
            feature for feature in feature_columns if feature not in EXTENDED_USER_ATTR_FEATURE_COLUMNS
        ]
    if not ENABLE_DYNAMIC_ITEM_FEATURES:
        feature_columns = [feature for feature in feature_columns if feature not in DYNAMIC_ITEM_FEATURE_COLUMNS]

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


@dataclass(frozen=True)
class IdMapping:
    customer_to_idx: dict[str, int]
    article_to_idx: dict[str, int]
    idx_to_customer: list[str]
    idx_to_article: list[str]

    def encode_customer_ids(self, customer_ids: list[str]) -> list[int]:
        return [self.customer_to_idx[str(customer_id)] for customer_id in customer_ids]

    def decode_article_ids(self, article_ids: list[Any]) -> list[str]:
        return [self.idx_to_article[int(article_id)] for article_id in article_ids]

    def encode_customer_age_bin(self, customer_age_bin: dict[str, str]) -> dict[int, str]:
        encoded: dict[int, str] = {}
        for customer_id, age_bin in customer_age_bin.items():
            customer_key = str(customer_id)
            if customer_key in self.customer_to_idx:
                encoded[self.customer_to_idx[customer_key]] = age_bin
        return encoded


def _encode_id_column(
    frame: pl.DataFrame,
    column_name: str,
    mapping: dict[str, int],
    encoded_name: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.col(column_name).cast(ID_DTYPE, strict=False))

    map_df = pl.DataFrame(
        {
            column_name: list(mapping.keys()),
            encoded_name: list(mapping.values()),
        },
        schema={column_name: pl.Utf8, encoded_name: ID_DTYPE},
    )
    original_columns = frame.columns
    encoded = frame.join(map_df, on=column_name, how="left")
    missing_count = encoded.filter(pl.col(encoded_name).is_null()).height
    if missing_count:
        raise ValueError(f"ID mapping failed for {missing_count} rows in column {column_name!r}")
    return encoded.drop(column_name).rename({encoded_name: column_name}).select(original_columns)


def build_id_mapping(tables: dict[str, pl.LazyFrame], transactions: pl.DataFrame) -> IdMapping:
    customer_ids = (
        tables["submission"]
        .select(pl.col("customer_id").cast(pl.Utf8))
        .collect(engine="streaming")
        .get_column("customer_id")
        .to_list()
    )
    seen_customers = set(customer_ids)
    for customer_id in transactions.select("customer_id").unique().get_column("customer_id").to_list():
        customer_key = str(customer_id)
        if customer_key not in seen_customers:
            customer_ids.append(customer_key)
            seen_customers.add(customer_key)

    article_ids = (
        tables["articles"]
        .select(pl.col("article_id").cast(pl.Utf8))
        .collect(engine="streaming")
        .get_column("article_id")
        .to_list()
    )
    seen_articles = set(article_ids)
    for article_id in transactions.select("article_id").unique().get_column("article_id").to_list():
        article_key = str(article_id)
        if article_key not in seen_articles:
            article_ids.append(article_key)
            seen_articles.add(article_key)

    mapping = IdMapping(
        customer_to_idx={customer_id: idx for idx, customer_id in enumerate(customer_ids)},
        article_to_idx={article_id: idx for idx, article_id in enumerate(article_ids)},
        idx_to_customer=customer_ids,
        idx_to_article=article_ids,
    )
    print(
        "ID mapping built: "
        f"customers={len(mapping.idx_to_customer)} articles={len(mapping.idx_to_article)} dtype={ID_DTYPE}"
    )
    return mapping


def apply_id_mapping(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    id_mapping: IdMapping,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[int, str]]:
    transactions = _encode_id_column(transactions, "customer_id", id_mapping.customer_to_idx, "customer_idx")
    transactions = _encode_id_column(transactions, "article_id", id_mapping.article_to_idx, "article_idx")
    article_department = _encode_id_column(
        article_department,
        "article_id",
        id_mapping.article_to_idx,
        "article_idx",
    )
    article_features = _encode_id_column(article_features, "article_id", id_mapping.article_to_idx, "article_idx")
    customer_features = _encode_id_column(
        customer_features,
        "customer_id",
        id_mapping.customer_to_idx,
        "customer_idx",
    )
    return (
        transactions,
        article_department,
        article_features,
        customer_features,
        id_mapping.encode_customer_age_bin(customer_age_bin),
    )


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
            pl.col("product_code").cast(pl.Utf8, strict=False),
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


def _dict_frame(
    mapping: dict[Any, Any],
    key_name: str,
    value_name: str,
    value_dtype: pl.DataType,
    key_dtype: pl.DataType = pl.Utf8,
) -> pl.DataFrame:
    if not mapping:
        return pl.DataFrame({key_name: [], value_name: []}, schema={key_name: key_dtype, value_name: value_dtype})
    return pl.DataFrame(
        {
            key_name: list(mapping.keys()),
            value_name: list(mapping.values()),
        },
        schema={key_name: key_dtype, value_name: value_dtype},
    )


def _infer_id_dtype(values: list[Any], default: pl.DataType = pl.Utf8) -> pl.DataType:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, np.integer)):
            return ID_DTYPE
        return pl.Utf8
    return default


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
    days: int = 30,
) -> dict[float, list[str]]:
    cutoff = reference_date - timedelta(days=days)
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

    pair_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
    for items in baskets.get_column("items").to_list():
        basket_items = [item for item in items if item is not None]
        for item in basket_items:
            counter = pair_counts[item]
            for other_item in basket_items:
                if other_item != item:
                    counter[other_item] += 1

    cooccurrence_items: dict[Any, list[tuple[Any, float]]] = {}
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


def _build_itemcf_items(
    history_tx: pl.DataFrame,
    reference_date: date,
) -> dict[Any, list[tuple[Any, float]]]:
    if not ENABLE_ITEMCF_RECALL or ITEMCF_RECALL_TOP <= 0:
        return {}

    cutoff = reference_date - timedelta(days=ITEMCF_DAYS)
    baskets = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by(["customer_id", "t_dat"])
        .agg(pl.col("article_id").unique(maintain_order=True).alias("items"))
        .with_columns(pl.col("items").list.len().alias("basket_size"))
        .filter((pl.col("basket_size") > 1) & (pl.col("basket_size") <= ITEMCF_MAX_BASKET_SIZE))
        .select("items")
    )

    item_counts: Counter[Any] = Counter()
    pair_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
    for items in baskets.get_column("items").to_list():
        basket_items = [item for item in items if item is not None]
        for item in basket_items:
            item_counts[item] += 1
        for item in basket_items:
            counter = pair_counts[item]
            for other_item in basket_items:
                if other_item != item:
                    counter[other_item] += 1

    itemcf_items: dict[Any, list[tuple[Any, float]]] = {}
    for item, counter in pair_counts.items():
        item_count = max(float(item_counts[item]), 1.0)
        scored_items = []
        for other_item, pair_count in counter.items():
            other_count = max(float(item_counts[other_item]), 1.0)
            score = float(pair_count) / np.sqrt(item_count * other_count)
            scored_items.append((other_item, score))
        scored_items.sort(key=lambda pair: (-pair[1], pair[0]))
        itemcf_items[item] = scored_items[:ITEMCF_TOP_PER_ITEM]

    print(
        "item-CF recall built: "
        f"items={len(itemcf_items)} days={ITEMCF_DAYS} top_per_item={ITEMCF_TOP_PER_ITEM}"
    )
    return itemcf_items


def _build_product_code_variant_items(
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    reference_date: date,
) -> dict[Any, list[tuple[Any, float]]]:
    if not ENABLE_PRODUCT_CODE_RECALL or PRODUCT_CODE_RECALL_TOP <= 0:
        return {}
    if "product_code" not in article_features.columns:
        return {}

    article_product_code = article_features.select("article_id", "product_code").drop_nulls(["product_code"])
    item_to_product_code = {
        row["article_id"]: str(row["product_code"])
        for row in article_product_code.iter_rows(named=True)
    }
    if not item_to_product_code:
        return {}

    cutoff = reference_date - timedelta(days=PRODUCT_CODE_DAYS)
    product_code_items = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by("article_id")
        .agg(
            pl.len().alias("product_code_pop"),
            pl.n_unique("customer_id").alias("product_code_buyers"),
        )
        .join(article_product_code, on="article_id", how="left")
        .drop_nulls(["product_code"])
        .with_columns(
            (
                pl.col("product_code_pop").cast(pl.Float64)
                + pl.col("product_code_buyers").cast(pl.Float64) * 0.5
            ).alias("product_code_score")
        )
        .sort(
            ["product_code", "product_code_score", "product_code_buyers", "article_id"],
            descending=[False, True, True, False],
        )
        .group_by("product_code")
        .agg(
            pl.struct("article_id", "product_code_score")
            .head(max(PRODUCT_CODE_RECALL_TOP * 2, PRODUCT_CODE_RECALL_TOP))
            .alias("variant_items")
        )
    )
    variants_by_code: dict[str, list[tuple[Any, float]]] = {}
    for row in product_code_items.iter_rows(named=True):
        variants_by_code[str(row["product_code"])] = [
            (item["article_id"], float(item["product_code_score"]))
            for item in (row["variant_items"] or [])
        ]

    variant_items: dict[Any, list[tuple[Any, float]]] = {}
    for article_id, product_code in item_to_product_code.items():
        candidates = [
            (variant_id, score)
            for variant_id, score in variants_by_code.get(product_code, [])
            if variant_id != article_id
        ]
        if candidates:
            variant_items[article_id] = candidates[:PRODUCT_CODE_RECALL_TOP]

    print(
        "product-code variant recall built: "
        f"items={len(variant_items)} days={PRODUCT_CODE_DAYS} top={PRODUCT_CODE_RECALL_TOP}"
    )
    return variant_items


def _price_band_expr(price_expr: pl.Expr, cutoffs: list[float]) -> pl.Expr:
    expr = pl.lit(max(len(cutoffs), 0))
    for idx, cutoff in reversed(list(enumerate(cutoffs))):
        expr = pl.when(price_expr <= cutoff).then(pl.lit(idx)).otherwise(expr)
    return expr.cast(pl.Int16)


def _build_price_band_context(
    history_tx: pl.DataFrame,
    reference_date: date,
) -> tuple[dict[Any, int], dict[int, list[tuple[Any, float]]]]:
    if not ENABLE_PRICE_BAND_RECALL or PRICE_BAND_RECALL_TOP <= 0:
        return {}, {}

    price_values = history_tx.get_column("price").drop_nulls()
    if len(price_values) == 0:
        return {}, {}

    band_count = max(2, PRICE_BAND_COUNT)
    quantiles = [idx / band_count for idx in range(1, band_count)]
    cutoffs = sorted(
        {
            float(value)
            for value in (price_values.quantile(q) for q in quantiles)
            if value is not None
        }
    )
    if not cutoffs:
        return {}, {}

    with_band = history_tx.with_columns(_price_band_expr(pl.col("price"), cutoffs).alias("price_band_id"))
    customer_price_band = (
        with_band.group_by(["customer_id", "price_band_id"])
        .agg(
            pl.len().alias("band_txn"),
            pl.max("t_dat").alias("band_last_date"),
        )
        .sort(["customer_id", "band_txn", "band_last_date"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("price_band_id").first().alias("price_band_id"))
    )
    customer_band_map = {
        row["customer_id"]: int(row["price_band_id"])
        for row in customer_price_band.iter_rows(named=True)
    }

    cutoff = reference_date - timedelta(days=PRICE_BAND_DAYS)
    price_band_top = (
        with_band.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by(["price_band_id", "article_id"])
        .agg(
            pl.len().alias("price_band_pop"),
            pl.n_unique("customer_id").alias("price_band_buyers"),
        )
        .with_columns(
            (
                pl.col("price_band_pop").cast(pl.Float64)
                + pl.col("price_band_buyers").cast(pl.Float64) * 0.5
            ).alias("price_band_score")
        )
        .sort(
            ["price_band_id", "price_band_score", "price_band_buyers", "article_id"],
            descending=[False, True, True, False],
        )
        .group_by("price_band_id")
        .agg(
            pl.struct("article_id", "price_band_score")
            .head(PRICE_BAND_RECALL_TOP)
            .alias("price_band_items")
        )
    )
    price_band_top_items: dict[int, list[tuple[Any, float]]] = {}
    for row in price_band_top.iter_rows(named=True):
        price_band_top_items[int(row["price_band_id"])] = [
            (item["article_id"], float(item["price_band_score"]))
            for item in (row["price_band_items"] or [])
        ]

    print(
        "price-band recall built: "
        f"customers={len(customer_band_map)} bands={len(price_band_top_items)} days={PRICE_BAND_DAYS}"
    )
    return customer_band_map, price_band_top_items


def _build_multi_day_trend_items(
    history_tx: pl.DataFrame,
    reference_date: date,
) -> list[tuple[Any, float]]:
    if not ENABLE_MULTI_DAY_TREND_RECALL or MULTI_DAY_TREND_RECALL_TOP <= 0:
        return []

    cutoff_1d = reference_date - timedelta(days=1)
    cutoff_3d = reference_date - timedelta(days=3)
    cutoff_7d = reference_date - timedelta(days=7)
    pop_1d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_1d))
        .group_by("article_id")
        .agg(pl.len().alias("pop_1d"))
    )
    pop_3d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_3d))
        .group_by("article_id")
        .agg(pl.len().alias("pop_3d"))
    )
    pop_7d = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff_7d))
        .group_by("article_id")
        .agg(
            pl.len().alias("pop_7d"),
            pl.n_unique("customer_id").alias("buyers_7d"),
        )
    )
    trend_df = (
        pop_7d.join(pop_3d, on="article_id", how="left")
        .join(pop_1d, on="article_id", how="left")
        .with_columns(
            pl.col("pop_1d").fill_null(0.0).cast(pl.Float64),
            pl.col("pop_3d").fill_null(0.0).cast(pl.Float64),
            pl.col("pop_7d").fill_null(0.0).cast(pl.Float64),
            pl.col("buyers_7d").fill_null(0.0).cast(pl.Float64),
        )
        .with_columns(
            (
                pl.col("pop_1d") * 2.0
                + pl.col("pop_3d") * 1.0
                + pl.col("pop_7d") * 0.35
                + pl.col("buyers_7d") * 0.25
            ).alias("trend_score")
        )
        .sort(["trend_score", "buyers_7d", "article_id"], descending=[True, True, False])
        .select("article_id", "trend_score")
        .head(MULTI_DAY_TREND_RECALL_TOP)
    )
    trend_items = [
        (row["article_id"], float(row["trend_score"]))
        for row in trend_df.iter_rows(named=True)
    ]
    print(f"multi-day trend recall built: items={len(trend_items)}")
    return trend_items


def _build_sellable_items(history_tx: pl.DataFrame, reference_date: date) -> set[Any]:
    if not ENABLE_SELLABLE_FILTER:
        return set()
    cutoff = reference_date - timedelta(days=SELLABLE_DAYS)
    sellable_items = set(
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .select("article_id")
        .unique()
        .get_column("article_id")
        .to_list()
    )
    print(f"sellable item filter built: items={len(sellable_items)} days={SELLABLE_DAYS}")
    return sellable_items


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
) -> dict[str, list[tuple[Any, float]]]:
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
    result: dict[str, list[tuple[Any, float]]] = {}
    for row in top_df.iter_rows(named=True):
        result[str(row[segment_col])] = [
            (item["article_id"], float(item["segment_score"]))
            for item in (row["segment_items"] or [])
        ]
    return result


def _build_cold_start_top_items(
    history_tx: pl.DataFrame,
    customer_features: pl.DataFrame,
    reference_date: date,
) -> dict[str, dict[str, list[tuple[Any, float]]]]:
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
    if ENABLE_GROUP_RECALL:
        attr_cols["department_group"] = "article_department_no"
        attr_cols["index_group"] = "article_index_group_no"
        attr_cols["product_group"] = "article_product_group_no"
    preferred_attributes = {
        name: _build_customer_preferred_attribute(history_tx, article_features, attr_col)
        for name, attr_col in attr_cols.items()
    }
    attribute_top_items = {
        name: _build_attribute_top_items(
            history_tx,
            article_features,
            attr_col,
            reference_date,
            top_n=GROUP_RECALL_TOP if name in {"department_group", "index_group", "product_group"} else ATTRIBUTE_RECALL_TOP,
            days=GROUP_RECALL_DAYS if name in {"department_group", "index_group", "product_group"} else 30,
        )
        for name, attr_col in attr_cols.items()
    }
    customer_price_band, price_band_top_items = _build_price_band_context(history_tx, reference_date)
    return {
        "preferred_attributes": preferred_attributes,
        "attribute_top_items": attribute_top_items,
        "recent_global_items": _build_recent_global_items(history_tx, reference_date),
        "cooccurrence_items": _build_cooccurrence_items(history_tx, reference_date),
        "itemcf_items": _build_itemcf_items(history_tx, reference_date),
        "product_code_variant_items": _build_product_code_variant_items(
            history_tx,
            article_features,
            reference_date,
        ),
        "customer_price_band": customer_price_band,
        "price_band_top_items": price_band_top_items,
        "multi_day_trend_items": _build_multi_day_trend_items(history_tx, reference_date),
        "sellable_items": _build_sellable_items(history_tx, reference_date),
        "customer_cold_segments": _build_customer_cold_start_segments(customer_features),
        "cold_start_top_items": _build_cold_start_top_items(history_tx, customer_features, reference_date),
    }


def build_candidate_frame(
    customer_ids: list[Any],
    artifacts: Any,
    ranker_config: RankerConfig,
    max_candidates: int = MAX_CANDIDATES_PER_CUSTOMER,
    actual_items_by_customer: dict[Any, list[Any]] | None = None,
    include_actual_items: bool = False,
    recall_context: dict[str, Any] | None = None,
) -> pl.DataFrame:
    segment_baseline_tops = [
        SEGMENT_ACTIVE_BASELINE_TOP,
        SEGMENT_WARM_BASELINE_TOP,
        SEGMENT_COLD_BASELINE_TOP,
    ] if ENABLE_SEGMENT_CANDIDATES else [BASELINE_RECALL_TOP]
    baseline_recall_k = min(max_candidates, max(max(segment_baseline_tops), MAX_K))
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
    cooccurrence_items: dict[Any, list[tuple[Any, float]]] = recall_context.get("cooccurrence_items", {})
    itemcf_items: dict[Any, list[tuple[Any, float]]] = recall_context.get("itemcf_items", {})
    product_code_variant_items: dict[Any, list[tuple[Any, float]]] = recall_context.get(
        "product_code_variant_items", {}
    )
    customer_price_band: dict[Any, int] = recall_context.get("customer_price_band", {})
    price_band_top_items: dict[int, list[tuple[Any, float]]] = recall_context.get("price_band_top_items", {})
    multi_day_trend_items: list[tuple[Any, float]] = recall_context.get("multi_day_trend_items", [])
    sellable_items: set[Any] = recall_context.get("sellable_items", set())
    has_sellable_filter = ENABLE_SELLABLE_FILTER and bool(sellable_items)
    customer_cold_segments: dict[str, tuple[str, str, str, str]] = recall_context.get("customer_cold_segments", {})
    cold_start_top_items: dict[str, dict[str, list[tuple[Any, float]]]] = recall_context.get(
        "cold_start_top_items", {}
    )
    source_rr_cols = {
        "source_baseline": "baseline_rr",
        "source_product_type": "product_type_rr",
        "source_garment_group": "garment_group_rr",
        "source_colour": "colour_rr",
        "source_section": "section_rr",
        "source_department_group": "department_group_rr",
        "source_index_group": "index_group_rr",
        "source_product_group": "product_group_rr",
        "source_recent_global": "recent_global_rr",
        "source_cooccurrence": "cooccurrence_rr",
        "source_itemcf": "itemcf_rr",
        "source_product_code": "product_code_rr",
        "source_price_band": "price_band_rr",
        "source_trend": "trend_rr",
        "source_cold_start": "cold_start_rr",
    }

    def add_candidate(
        customer_rows: list[dict[str, Any]],
        seen: set[Any],
        customer_id: Any,
        article_id: Any,
        rank: int,
        score: float,
        source_col: str,
        extra_values: dict[str, float] | None = None,
        allow_unsellable: bool = False,
    ) -> None:
        extra_values = extra_values or {}
        if has_sellable_filter and article_id not in sellable_items and not allow_unsellable:
            return
        rr_col = source_rr_cols.get(source_col)
        source_rr = 1.0 / max(rank, 1)
        if article_id in seen:
            for row in reversed(customer_rows):
                if row["article_id"] == article_id:
                    previous_source_value = float(row.get(source_col, 0.0))
                    row[source_col] = 1.0
                    if previous_source_value <= 0.0:
                        row["source_count"] = float(row.get("source_count", 0.0)) + 1.0
                    if rr_col is not None:
                        row[rr_col] = max(float(row.get(rr_col, 0.0)), source_rr)
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
                "source_count": 1.0,
                "source_baseline": 0.0,
                "baseline_rr": 0.0,
                "source_product_type": 0.0,
                "product_type_rr": 0.0,
                "source_garment_group": 0.0,
                "garment_group_rr": 0.0,
                "source_colour": 0.0,
                "colour_rr": 0.0,
                "source_section": 0.0,
                "section_rr": 0.0,
                "source_department_group": 0.0,
                "department_group_rr": 0.0,
                "source_index_group": 0.0,
                "index_group_rr": 0.0,
                "source_product_group": 0.0,
                "product_group_rr": 0.0,
                "source_recent_global": 0.0,
                "recent_global_rr": 0.0,
                "source_cooccurrence": 0.0,
                "cooccurrence_rr": 0.0,
                "cooccurrence_score": 0.0,
                "source_itemcf": 0.0,
                "itemcf_rr": 0.0,
                "itemcf_score": 0.0,
                "source_product_code": 0.0,
                "product_code_rr": 0.0,
                "product_code_score": 0.0,
                "source_price_band": 0.0,
                "price_band_rr": 0.0,
                "price_band_score": 0.0,
                "source_trend": 0.0,
                "trend_rr": 0.0,
                "trend_score": 0.0,
                "source_cold_start": 0.0,
                "cold_start_rr": 0.0,
                "cold_start_score": 0.0,
                "source_actual_in_train": 0.0,
                source_col: 1.0,
                **({rr_col: source_rr} if rr_col is not None else {}),
                **extra_values,
            }
        )
        seen.add(article_id)

    def user_segment(customer_id: Any) -> str:
        recent_count = len(artifacts.user_recent_history.get(customer_id, []))
        long_count = len(artifacts.user_long_history.get(customer_id, []))
        history_count = recent_count + long_count
        if history_count <= SEGMENT_COLD_TXN_MAX:
            return "cold"
        if recent_count >= SEGMENT_ACTIVE_RECENT_MIN:
            return "active"
        return "warm"

    def segment_baseline_top(segment_name: str) -> int:
        if not ENABLE_SEGMENT_CANDIDATES:
            return BASELINE_RECALL_TOP
        if segment_name == "cold":
            return SEGMENT_COLD_BASELINE_TOP
        if segment_name == "active":
            return SEGMENT_ACTIVE_BASELINE_TOP
        return SEGMENT_WARM_BASELINE_TOP

    def add_cold_start_candidates(
        customer_rows: list[dict[str, Any]],
        seen: set[Any],
        customer_id: Any,
        next_rank: int,
        top_n: int,
    ) -> int:
        if not cold_start_top_items or top_n <= 0:
            return next_rank
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

        cold_scores: dict[Any, float] = {}
        for segment_name, segment_value, segment_weight in segment_plan:
            segment_items = cold_start_top_items.get(segment_name, {}).get(segment_value, [])
            for pos, (article_id, segment_score) in enumerate(segment_items, start=1):
                if article_id in seen:
                    continue
                cold_scores[article_id] = cold_scores.get(article_id, 0.0) + (
                    segment_weight * np.log1p(segment_score) / pos
                )

        for cold_rank, (article_id, cold_score) in enumerate(
            sorted(cold_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n],
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
        return next_rank

    for customer_id, baseline_items in zip(customer_ids, baseline_lists):
        seen: set[Any] = set()
        customer_rows: list[dict[str, Any]] = []
        segment_name = user_segment(customer_id)
        baseline_top = min(max_candidates, max(MAX_K, segment_baseline_top(segment_name)))
        user_history_items = set(artifacts.user_recent_history.get(customer_id, [])) | set(
            artifacts.user_long_history.get(customer_id, [])
        )
        for rank, article_id in enumerate(baseline_items[:baseline_top], start=1):
            add_candidate(
                customer_rows,
                seen,
                customer_id,
                article_id,
                rank,
                1.0 / rank,
                "source_baseline",
                allow_unsellable=SELLABLE_ALLOW_HISTORY and article_id in user_history_items,
            )

        next_rank = len(customer_rows) + 1
        if (
            ENABLE_SEGMENT_CANDIDATES
            and segment_name == "cold"
            and len(customer_rows) < max_candidates
        ):
            next_rank = add_cold_start_candidates(
                customer_rows,
                seen,
                customer_id,
                next_rank,
                SEGMENT_COLD_START_RECALL_TOP,
            )

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

        if len(customer_rows) < max_candidates and itemcf_items:
            itemcf_scores: dict[Any, float] = {}
            for history_pos, history_item in enumerate(
                artifacts.user_recent_history.get(customer_id, [])[:ITEMCF_HISTORY_TOP],
                start=1,
            ):
                for cf_item, cf_score in itemcf_items.get(history_item, []):
                    if cf_item in seen:
                        continue
                    itemcf_scores[cf_item] = itemcf_scores.get(cf_item, 0.0) + cf_score / history_pos

            for cf_rank, (article_id, cf_score) in enumerate(
                sorted(itemcf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:ITEMCF_RECALL_TOP],
                start=1,
            ):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.13 / max(next_rank, 1),
                    "source_itemcf",
                    {"itemcf_score": cf_score / cf_rank},
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break

        if len(customer_rows) < max_candidates and product_code_variant_items:
            product_code_scores: dict[Any, float] = {}
            for history_pos, history_item in enumerate(
                artifacts.user_recent_history.get(customer_id, [])[:PRODUCT_CODE_HISTORY_TOP],
                start=1,
            ):
                for variant_item, variant_score in product_code_variant_items.get(history_item, []):
                    if variant_item in seen:
                        continue
                    product_code_scores[variant_item] = product_code_scores.get(variant_item, 0.0) + (
                        variant_score / history_pos
                    )

            for variant_rank, (article_id, variant_score) in enumerate(
                sorted(product_code_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:PRODUCT_CODE_RECALL_TOP],
                start=1,
            ):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.11 / max(next_rank, 1),
                    "source_product_code",
                    {"product_code_score": variant_score / variant_rank},
                )
                next_rank += 1
                if len(customer_rows) >= max_candidates:
                    break

        for source_name, source_col in [
            ("product_type", "source_product_type"),
            ("garment_group", "source_garment_group"),
            ("colour", "source_colour"),
            ("section", "source_section"),
            ("department_group", "source_department_group"),
            ("index_group", "source_index_group"),
            ("product_group", "source_product_group"),
        ]:
            if source_name == "colour" and not ENABLE_COLOUR_RECALL:
                continue
            if source_name == "section" and not ENABLE_SECTION_RECALL:
                continue
            if source_name in {"department_group", "index_group", "product_group"} and not ENABLE_GROUP_RECALL:
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

        if len(customer_rows) < max_candidates and price_band_top_items:
            price_band_id = customer_price_band.get(customer_id)
            if price_band_id is not None:
                for band_rank, (article_id, band_score) in enumerate(
                    price_band_top_items.get(price_band_id, [])[:PRICE_BAND_RECALL_TOP],
                    start=1,
                ):
                    add_candidate(
                        customer_rows,
                        seen,
                        customer_id,
                        article_id,
                        next_rank,
                        0.09 / max(next_rank, 1),
                        "source_price_band",
                        {"price_band_score": band_score / band_rank},
                    )
                    next_rank += 1
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

        if len(customer_rows) < max_candidates and multi_day_trend_items:
            for trend_rank, (article_id, trend_score) in enumerate(multi_day_trend_items, start=1):
                add_candidate(
                    customer_rows,
                    seen,
                    customer_id,
                    article_id,
                    next_rank,
                    0.085 / max(next_rank, 1),
                    "source_trend",
                    {"trend_score": trend_score / trend_rank},
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
                        "source_count": 0.0,
                        "source_baseline": 0.0,
                        "baseline_rr": 0.0,
                        "source_product_type": 0.0,
                        "product_type_rr": 0.0,
                        "source_garment_group": 0.0,
                        "garment_group_rr": 0.0,
                        "source_colour": 0.0,
                        "colour_rr": 0.0,
                        "source_section": 0.0,
                        "section_rr": 0.0,
                        "source_department_group": 0.0,
                        "department_group_rr": 0.0,
                        "source_index_group": 0.0,
                        "index_group_rr": 0.0,
                        "source_product_group": 0.0,
                        "product_group_rr": 0.0,
                        "source_recent_global": 0.0,
                        "recent_global_rr": 0.0,
                        "source_cooccurrence": 0.0,
                        "cooccurrence_rr": 0.0,
                        "cooccurrence_score": 0.0,
                        "source_itemcf": 0.0,
                        "itemcf_rr": 0.0,
                        "itemcf_score": 0.0,
                        "source_product_code": 0.0,
                        "product_code_rr": 0.0,
                        "product_code_score": 0.0,
                        "source_price_band": 0.0,
                        "price_band_rr": 0.0,
                        "price_band_score": 0.0,
                        "source_trend": 0.0,
                        "trend_rr": 0.0,
                        "trend_score": 0.0,
                        "source_cold_start": 0.0,
                        "cold_start_rr": 0.0,
                        "cold_start_score": 0.0,
                        "source_actual_in_train": 1.0,
                    }
                )
                seen.add(article_id)
        rows.extend(customer_rows[:max_candidates] if not include_actual_items else customer_rows)

    if not rows:
        customer_dtype = _infer_id_dtype(customer_ids)
        article_dtype = ID_DTYPE if customer_dtype == ID_DTYPE else pl.Utf8
        return pl.DataFrame(
            {
                "customer_id": [],
                "article_id": [],
                "candidate_rank": [],
                "candidate_score": [],
                "source_count": [],
                "source_baseline": [],
                "baseline_rr": [],
                "source_product_type": [],
                "product_type_rr": [],
                "source_garment_group": [],
                "garment_group_rr": [],
                "source_colour": [],
                "colour_rr": [],
                "source_section": [],
                "section_rr": [],
                "source_department_group": [],
                "department_group_rr": [],
                "source_index_group": [],
                "index_group_rr": [],
                "source_product_group": [],
                "product_group_rr": [],
                "source_recent_global": [],
                "recent_global_rr": [],
                "source_cooccurrence": [],
                "cooccurrence_rr": [],
                "cooccurrence_score": [],
                "source_itemcf": [],
                "itemcf_rr": [],
                "itemcf_score": [],
                "source_product_code": [],
                "product_code_rr": [],
                "product_code_score": [],
                "source_price_band": [],
                "price_band_rr": [],
                "price_band_score": [],
                "source_trend": [],
                "trend_rr": [],
                "trend_score": [],
                "source_cold_start": [],
                "cold_start_rr": [],
                "cold_start_score": [],
                "source_actual_in_train": [],
            },
            schema={
                "customer_id": customer_dtype,
                "article_id": article_dtype,
                "candidate_rank": pl.Int64,
                "candidate_score": pl.Float64,
                "source_count": pl.Float64,
                "source_baseline": pl.Float64,
                "baseline_rr": pl.Float64,
                "source_product_type": pl.Float64,
                "product_type_rr": pl.Float64,
                "source_garment_group": pl.Float64,
                "garment_group_rr": pl.Float64,
                "source_colour": pl.Float64,
                "colour_rr": pl.Float64,
                "source_section": pl.Float64,
                "section_rr": pl.Float64,
                "source_department_group": pl.Float64,
                "department_group_rr": pl.Float64,
                "source_index_group": pl.Float64,
                "index_group_rr": pl.Float64,
                "source_product_group": pl.Float64,
                "product_group_rr": pl.Float64,
                "source_recent_global": pl.Float64,
                "recent_global_rr": pl.Float64,
                "source_cooccurrence": pl.Float64,
                "cooccurrence_rr": pl.Float64,
                "cooccurrence_score": pl.Float64,
                "source_itemcf": pl.Float64,
                "itemcf_rr": pl.Float64,
                "itemcf_score": pl.Float64,
                "source_product_code": pl.Float64,
                "product_code_rr": pl.Float64,
                "product_code_score": pl.Float64,
                "source_price_band": pl.Float64,
                "price_band_rr": pl.Float64,
                "price_band_score": pl.Float64,
                "source_trend": pl.Float64,
                "trend_rr": pl.Float64,
                "trend_score": pl.Float64,
                "source_cold_start": pl.Float64,
                "cold_start_rr": pl.Float64,
                "cold_start_score": pl.Float64,
                "source_actual_in_train": pl.Float64,
            },
        )
    customer_dtype = _infer_id_dtype(customer_ids)
    article_dtype = _infer_id_dtype([row["article_id"] for row in rows])
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("customer_id").cast(customer_dtype, strict=False),
            pl.col("article_id").cast(article_dtype, strict=False),
        )
        .unique(subset=["customer_id", "article_id"], keep="first")
    )


def build_user_feature_frame(
    history_tx: pl.DataFrame,
    artifacts: Any,
    customer_age_bin: dict[str, str],
    reference_date: date,
) -> pl.DataFrame:
    cutoff_30d = reference_date - timedelta(days=30)
    customer_id_dtype = history_tx.schema.get("customer_id", pl.Utf8)
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
        key_dtype=customer_id_dtype,
    )
    department_df = _dict_frame(
        artifacts.customer_pref_department,
        "customer_id",
        "user_pref_department",
        pl.Float64,
        key_dtype=customer_id_dtype,
    )
    age_df = pl.DataFrame(
        {
            "customer_id": list(customer_age_bin.keys()),
            "customer_age_bin_id": [AGE_BIN_TO_ID.get(value, -1) for value in customer_age_bin.values()],
        },
        schema={"customer_id": customer_id_dtype, "customer_age_bin_id": pl.Float64},
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

    def window_item_stats(days: int, suffix: str, with_rank: bool = False) -> pl.DataFrame:
        cutoff = reference_date - timedelta(days=days)
        pop_col = f"item_pop_{suffix}"
        buyers_col = f"item_buyers_{suffix}"
        stats = (
            history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
            .group_by("article_id")
            .agg(
                pl.len().alias(pop_col),
                pl.n_unique("customer_id").alias(buyers_col),
            )
        )
        if with_rank:
            rank_col = f"item_rank_{suffix}"
            stats = (
                stats.sort([pop_col, buyers_col, "article_id"], descending=[True, True, False])
                .with_row_index(rank_col, offset=1)
                .with_columns(pl.col(rank_col).cast(pl.Float64))
            )
        return stats

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
    item_7d = window_item_stats(days=7, suffix="7d")
    item_30d = window_item_stats(days=30, suffix="30d")
    feature_df = item_all.join(item_7d, on="article_id", how="left").join(item_30d, on="article_id", how="left")
    if ENABLE_DYNAMIC_ITEM_FEATURES:
        item_1d = window_item_stats(days=1, suffix="1d", with_rank=True)
        item_3d = window_item_stats(days=3, suffix="3d", with_rank=True)
        item_14d = window_item_stats(days=14, suffix="14d", with_rank=True)
        feature_df = (
            feature_df.join(item_1d, on="article_id", how="left")
            .join(item_3d, on="article_id", how="left")
            .join(item_14d, on="article_id", how="left")
        )
    return feature_df


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


def build_user_attribute_preference_frames(
    history_tx: pl.DataFrame,
    article_features: pl.DataFrame,
    reference_date: date,
) -> dict[str, pl.DataFrame]:
    if not ENABLE_USER_ATTR_FEATURES:
        return {}

    attr_cols = [attr_col for _, attr_col in USER_ATTR_PREF_SPECS]
    cutoff = reference_date - timedelta(days=USER_ATTR_FEATURE_DAYS)
    history_with_attrs = (
        history_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .join(article_features.select(["article_id", *attr_cols]), on="article_id", how="left")
    )
    if history_with_attrs.is_empty():
        return {}

    user_window_counts = history_with_attrs.group_by("customer_id").agg(
        pl.len().alias("user_attr_window_txn")
    )
    frames: dict[str, pl.DataFrame] = {}
    for name, attr_col in USER_ATTR_PREF_SPECS:
        cnt_col = f"user_{name}_cnt"
        share_col = f"user_{name}_share"
        recency_col = f"user_{name}_recency_days"
        frame = (
            history_with_attrs.drop_nulls([attr_col])
            .group_by(["customer_id", attr_col])
            .agg(
                pl.len().alias(cnt_col),
                pl.max("t_dat").alias("_user_attr_last_date"),
            )
            .join(user_window_counts, on="customer_id", how="left")
            .with_columns(
                (
                    pl.col(cnt_col).cast(pl.Float64)
                    / pl.col("user_attr_window_txn").cast(pl.Float64).clip(lower_bound=1.0)
                ).alias(share_col),
                (pl.lit(reference_date) - pl.col("_user_attr_last_date"))
                .dt.total_days()
                .cast(pl.Float64)
                .alias(recency_col),
            )
            .drop("_user_attr_last_date", "user_attr_window_txn")
        )
        frames[name] = frame

    print(
        "user-attribute preference features built: "
        + ", ".join(f"{name}={frame.height}" for name, frame in frames.items())
        + f" days={USER_ATTR_FEATURE_DAYS}"
    )
    return frames


def _ensure_feature_columns(feature_df: pl.DataFrame) -> pl.DataFrame:
    missing_exprs = [
        pl.lit(0.0).alias(col)
        for col in FEATURE_COLUMNS
        if col not in feature_df.columns
    ]
    if missing_exprs:
        feature_df = feature_df.with_columns(missing_exprs)
    return feature_df


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
    user_attr_frames = build_user_attribute_preference_frames(
        history_tx=history_tx,
        article_features=article_features,
        reference_date=reference_date,
    )

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
    if ENABLE_DYNAMIC_ITEM_FEATURES:
        feature_df = feature_df.with_columns(
            (
                pl.col("item_pop_1d").fill_null(0.0)
                / (pl.col("item_pop_7d").fill_null(0.0) + 1.0)
            ).alias("item_pop_ratio_1d_7d"),
            (
                pl.col("item_pop_3d").fill_null(0.0)
                / (pl.col("item_pop_14d").fill_null(0.0) + 1.0)
            ).alias("item_pop_ratio_3d_14d"),
        )
    for name, attr_col in USER_ATTR_PREF_SPECS:
        attr_frame = user_attr_frames.get(name)
        if attr_frame is not None and not attr_frame.is_empty():
            feature_df = feature_df.join(attr_frame, on=["customer_id", attr_col], how="left")

    feature_df = _ensure_feature_columns(feature_df)
    return feature_df.with_columns(
        [pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col) for col in FEATURE_COLUMNS]
    )


def label_candidates(candidates: pl.DataFrame, actual_items_by_customer: dict[Any, list[Any]]) -> pl.DataFrame:
    rows = [
        {"customer_id": customer_id, "article_id": article_id, "label": 1}
        for customer_id, items in actual_items_by_customer.items()
        for article_id in items
    ]
    if not rows:
        return candidates.with_columns(pl.lit(0).alias("label"))

    positives = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("customer_id").cast(candidates.schema.get("customer_id", pl.Utf8), strict=False),
            pl.col("article_id").cast(candidates.schema.get("article_id", pl.Utf8), strict=False),
            pl.col("label").cast(pl.Int8),
        )
        .unique(subset=["customer_id", "article_id"], keep="first")
    )
    return (
        candidates.join(positives, on=["customer_id", "article_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )


def summarize_candidate_recall(
    candidates: pl.DataFrame,
    actual_items_by_customer: dict[Any, list[Any]],
) -> dict[str, Any]:
    rows = [
        {"customer_id": customer_id, "article_id": article_id}
        for customer_id, items in actual_items_by_customer.items()
        for article_id in items
    ]
    if not rows:
        return {
            "candidate_recall": 0.0,
            "candidate_hit_count": 0,
            "actual_item_count": 0,
            "candidate_customer_hit_rate": 0.0,
        }
    positives = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("customer_id").cast(candidates.schema.get("customer_id", pl.Utf8), strict=False),
            pl.col("article_id").cast(candidates.schema.get("article_id", pl.Utf8), strict=False),
        )
        .unique(subset=["customer_id", "article_id"], keep="first")
    )
    source_cols = [
        col
        for col in candidates.columns
        if col.startswith("source_") and col not in {"source_actual_in_train", "source_count"}
    ]
    hit_df = positives.join(candidates, on=["customer_id", "article_id"], how="inner")
    actual_item_count = positives.height
    hit_count = hit_df.height
    customer_hit_count = hit_df.select("customer_id").n_unique()
    summary: dict[str, Any] = {
        "candidate_recall": float(hit_count / actual_item_count) if actual_item_count else 0.0,
        "candidate_hit_count": int(hit_count),
        "actual_item_count": int(actual_item_count),
        "candidate_customer_hit_rate": float(customer_hit_count / len(actual_items_by_customer))
        if actual_items_by_customer
        else 0.0,
    }
    for source_col in source_cols:
        source_hit_count = hit_df.filter(pl.col(source_col) > 0).height if source_col in hit_df.columns else 0
        short_name = source_col.removeprefix("source_")
        summary[f"candidate_hit_{short_name}"] = int(source_hit_count)
        summary[f"candidate_recall_{short_name}"] = (
            float(source_hit_count / actual_item_count) if actual_item_count else 0.0
        )

    source_bits = []
    for col in source_cols:
        short_name = col.removeprefix("source_")
        hit_key = f"candidate_hit_{short_name}"
        source_bits.append(f"{short_name}={summary.get(hit_key, 0)}")
    print(
        "candidate recall: "
        f"hits={hit_count}/{actual_item_count} recall={summary['candidate_recall']:.4f} "
        f"customer_hit_rate={summary['candidate_customer_hit_rate']:.4f} "
        f"sources: {', '.join(source_bits)}"
    )
    return summary


def _sample_frame(frame: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    if n <= 0 or frame.is_empty():
        return frame.head(0)
    return frame.sample(n=min(n, frame.height), seed=seed, shuffle=True)


def _hard_negative_mask(frame: pl.DataFrame) -> pl.Expr | None:
    mask_parts: list[pl.Expr] = []
    if "candidate_rank" in frame.columns:
        mask_parts.append(pl.col("candidate_rank") <= HARD_NEGATIVE_TOP_RANK)
    for col in ("source_cooccurrence", "source_recent_global", "source_product_type", "source_garment_group"):
        if col in frame.columns:
            mask_parts.append(pl.col(col) > 0)
    if not mask_parts:
        return None

    mask = mask_parts[0]
    for expr in mask_parts[1:]:
        mask = mask | expr
    return mask


def downsample_training_rows(train_df: pl.DataFrame) -> pl.DataFrame:
    positives = train_df.filter(pl.col("label") == 1)
    negatives = train_df.filter(pl.col("label") == 0)
    if positives.is_empty() or negatives.is_empty():
        return train_df

    negative_n = min(negatives.height, max(positives.height * NEGATIVE_SAMPLE_RATIO, 100000))
    mode = NEGATIVE_SAMPLE_MODE if NEGATIVE_SAMPLE_MODE in {"random", "hard", "mixed"} else "random"

    if mode == "random":
        sampled_negatives = _sample_frame(negatives, negative_n, RANDOM_STATE)
        hard_count = 0
        easy_count = sampled_negatives.height
    else:
        hard_mask = _hard_negative_mask(negatives)
        if hard_mask is None:
            sampled_negatives = _sample_frame(negatives, negative_n, RANDOM_STATE)
            hard_count = 0
            easy_count = sampled_negatives.height
        else:
            hard_pool = negatives.filter(hard_mask)
            easy_pool = negatives.filter(~hard_mask)
            hard_fraction = min(max(HARD_NEGATIVE_FRACTION, 0.0), 1.0)
            hard_target = negative_n if mode == "hard" else int(negative_n * hard_fraction)
            sampled_hard = _sample_frame(hard_pool, hard_target, RANDOM_STATE)
            sampled_easy = _sample_frame(
                easy_pool,
                negative_n - sampled_hard.height,
                RANDOM_STATE + 1,
            )
            sampled_negatives = pl.concat([sampled_hard, sampled_easy], how="vertical")
            hard_count = sampled_hard.height
            easy_count = sampled_easy.height

    print(
        "negative sampling: "
        f"mode={mode}, ratio={NEGATIVE_SAMPLE_RATIO}, target={negative_n}, "
        f"sampled={sampled_negatives.height}, hard={hard_count}, easy={easy_count}, "
        f"hard_top_rank={HARD_NEGATIVE_TOP_RANK}, hard_fraction={HARD_NEGATIVE_FRACTION}"
    )
    return pl.concat([positives, sampled_negatives], how="vertical").sample(
        fraction=1.0,
        seed=RANDOM_STATE,
        shuffle=True,
    )


def make_training_matrix(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = train_df.select(MODEL_FEATURE_COLUMNS).to_numpy()
    y = train_df.get_column("label").to_numpy()
    return x, y


def make_ranking_matrix(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sorted_df = train_df.sort(
        ["customer_id", "label", "candidate_rank", "article_id"],
        descending=[False, True, False, False],
    )
    group_sizes = (
        sorted_df.group_by("customer_id", maintain_order=True)
        .agg(pl.len().alias("group_size"))
        .get_column("group_size")
        .to_list()
    )
    x_train, y_train = make_training_matrix(sorted_df)
    return x_train, y_train, [int(size) for size in group_sizes]


def train_lgbm_model(train_df: pl.DataFrame, model_params: dict[str, Any] | None = None) -> Any:
    train_df = downsample_training_rows(train_df)
    if LGBM_MODEL_TYPE in {"ranker", "lambdarank"}:
        x_train, y_train, group_train = make_ranking_matrix(train_df)
    else:
        x_train, y_train = make_training_matrix(train_df)
        group_train = []
    positive_count = float(np.sum(y_train))
    negative_count = float(len(y_train) - positive_count)
    scale_pos_weight = max(1.0, negative_count / max(positive_count, 1.0))

    params: dict[str, Any] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": LGBM_N_ESTIMATORS,
        "learning_rate": LGBM_LEARNING_RATE,
        "num_leaves": LGBM_NUM_LEAVES,
        "max_depth": LGBM_MAX_DEPTH,
        "min_child_samples": LGBM_MIN_CHILD_SAMPLES,
        "subsample": LGBM_SUBSAMPLE,
        "subsample_freq": 1,
        "colsample_bytree": LGBM_COLSAMPLE_BYTREE,
        "reg_alpha": LGBM_REG_ALPHA,
        "reg_lambda": LGBM_REG_LAMBDA,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_jobs": LGBM_N_JOBS,
    }
    if LGBM_FORCE_COL_WISE:
        params["force_col_wise"] = True
    if model_params:
        params.update(model_params)
        params["scale_pos_weight"] = scale_pos_weight
        params["random_state"] = RANDOM_STATE

    if LGBM_MODEL_TYPE in {"ranker", "lambdarank"}:
        params.pop("scale_pos_weight", None)
        params["objective"] = "lambdarank"
        params["metric"] = "ndcg"
        model = lgb.LGBMRanker(**params)
        model.fit(
            x_train,
            y_train,
            group=group_train,
            feature_name=MODEL_FEATURE_COLUMNS,
            callbacks=[lgb.log_evaluation(period=50)],
        )
        print(
            f"LightGBM ranker trained: rows={len(y_train)} positives={int(positive_count)} "
            f"groups={len(group_train)}"
        )
    else:
        model = lgb.LGBMClassifier(
            **params,
        )
        model.fit(
            x_train,
            y_train,
            feature_name=MODEL_FEATURE_COLUMNS,
            callbacks=[lgb.log_evaluation(period=50)],
        )
        print(f"LightGBM classifier trained: rows={len(y_train)} positives={int(positive_count)}")
    return model


def score_feature_frame(model: Any, feature_df: pl.DataFrame) -> pl.DataFrame:
    if feature_df.is_empty():
        return feature_df.with_columns(pl.lit(0.0).alias("score"))
    x_score = feature_df.select(MODEL_FEATURE_COLUMNS).to_numpy()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_score)[:, 1]
    else:
        scores = model.predict(x_score)
    scored = feature_df.with_columns(pl.Series("score", scores))
    return apply_segment_rerank(scored)


def apply_segment_rerank(scored_df: pl.DataFrame) -> pl.DataFrame:
    if not ENABLE_SEGMENT_RERANK or scored_df.is_empty():
        return scored_df

    active_user = (pl.col("user_recency_days").fill_null(9999.0) <= float(SEGMENT_ACTIVE_DAYS)) | (
        pl.col("user_txn_30d").fill_null(0.0) > 0.0
    )
    cold_user = pl.col("user_txn_all").fill_null(0.0) <= float(SEGMENT_COLD_TXN_MAX)
    stale_user = pl.col("user_recency_days").fill_null(9999.0) >= float(SEGMENT_STALE_DAYS)
    attr_strength = (
        pl.col("user_product_type_share").fill_null(0.0)
        + pl.col("user_garment_group_share").fill_null(0.0)
        + pl.col("user_section_share").fill_null(0.0)
    ) / 3.0
    top_rank = pl.col("candidate_rank").fill_null(9999) <= SEGMENT_TOP_RANK_CUTOFF

    rerank_bonus = (
        pl.when(active_user)
        .then(
            SEGMENT_ACTIVE_COOCCURRENCE_BONUS * pl.col("source_cooccurrence").fill_null(0.0)
            + SEGMENT_ACTIVE_ATTR_BONUS * attr_strength
        )
        .otherwise(0.0)
        + pl.when(cold_user)
        .then(
            SEGMENT_COLD_START_BONUS * pl.col("source_cold_start").fill_null(0.0)
            + SEGMENT_COLD_RECENT_BONUS * pl.col("source_recent_global").fill_null(0.0)
        )
        .otherwise(0.0)
        + pl.when(stale_user)
        .then(SEGMENT_STALE_RECENT_BONUS * pl.col("source_recent_global").fill_null(0.0))
        .otherwise(0.0)
        + pl.when(top_rank).then(SEGMENT_TOP_RANK_BONUS).otherwise(0.0)
    )
    return scored_df.with_columns((pl.col("score") + rerank_bonus).alias("score"))


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
    candidate_recall_metrics = summarize_candidate_recall(candidates, actual_map)
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
        **candidate_recall_metrics,
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
    if not SAVE_FEATURE_IMPORTANCE:
        print("LGBM_SAVE_FEATURE_IMPORTANCE=0: skip feature importance export.")
        return
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
    id_mapping: IdMapping | None = None,
) -> pl.DataFrame:
    original_customer_ids = (
        tables["submission"]
        .select(pl.col("customer_id").cast(pl.Utf8))
        .collect(engine="streaming")
        .get_column("customer_id")
        .to_list()
    )
    customer_ids: list[Any]
    if id_mapping is not None:
        customer_ids = id_mapping.encode_customer_ids(original_customer_ids)
    else:
        customer_ids = original_customer_ids

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

    output_dir = prepare_output_dir(output_dir)
    output_path = output_dir / "submission.csv"
    all_rows: list[dict[str, str]] = []
    preview_rows: list[dict[str, str]] = []
    stream_file = None
    writer: csv.DictWriter | None = None
    if STREAM_SUBMISSION:
        stream_file = output_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream_file, fieldnames=["customer_id", "prediction"])
        writer.writeheader()

    written_count = 0
    total = len(customer_ids)
    try:
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
                output_customer_id = (
                    id_mapping.idx_to_customer[int(customer_id)] if id_mapping is not None else str(customer_id)
                )
                output_items = (
                    id_mapping.decode_article_ids(items[:MAX_K]) if id_mapping is not None else items[:MAX_K]
                )
                if len(output_items) != MAX_K:
                    raise ValueError(
                        f"Prediction for customer {output_customer_id!r} has {len(output_items)} items; "
                        f"expected {MAX_K}."
                    )
                row = {
                    "customer_id": output_customer_id,
                    "prediction": " ".join(str(item) for item in output_items),
                }
                if writer is not None:
                    writer.writerow(row)
                    if len(preview_rows) < 5:
                        preview_rows.append(row)
                else:
                    all_rows.append(row)
                written_count += 1
            print(f"[LGBM submission] {end}/{total} customers done")
            # Release per-chunk tables aggressively to reduce memory pressure.
            del candidates, features, scored, pred_map, fallback_lists
            gc.collect()
    finally:
        if stream_file is not None:
            stream_file.close()

    if written_count != len(original_customer_ids):
        raise ValueError(f"Submission row count mismatch: got {written_count}, expected {len(original_customer_ids)}")

    if STREAM_SUBMISSION:
        submission_preview = pl.DataFrame(preview_rows)
        print(f"LGBM submission saved: {output_path}")
        return submission_preview

    submission = pl.DataFrame(all_rows)
    validate_submission_format(submission, original_customer_ids, k=MAX_K)
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
        f"group_recall={ENABLE_GROUP_RECALL}, group_top={GROUP_RECALL_TOP}, group_days={GROUP_RECALL_DAYS}, "
        f"n_estimators={LGBM_N_ESTIMATORS}, "
        f"colour_recall={ENABLE_COLOUR_RECALL}, section_recall={ENABLE_SECTION_RECALL}, "
        f"cooccurrence_recall={ENABLE_COOCCURRENCE_RECALL}, cooccurrence_top={COOCCURRENCE_RECALL_TOP}, "
        f"itemcf_recall={ENABLE_ITEMCF_RECALL}, itemcf_top={ITEMCF_RECALL_TOP}, "
        f"product_code_recall={ENABLE_PRODUCT_CODE_RECALL}, product_code_top={PRODUCT_CODE_RECALL_TOP}, "
        f"price_band_recall={ENABLE_PRICE_BAND_RECALL}, price_band_top={PRICE_BAND_RECALL_TOP}, "
        f"multi_day_trend_recall={ENABLE_MULTI_DAY_TREND_RECALL}, trend_top={MULTI_DAY_TREND_RECALL_TOP}, "
        f"sellable_filter={ENABLE_SELLABLE_FILTER}, sellable_days={SELLABLE_DAYS}, "
        f"cold_start_recall={ENABLE_COLD_START_RECALL}, cold_start_top={COLD_START_RECALL_TOP}, "
        f"cold_history_max={COLD_START_HISTORY_MAX_ITEMS}, postal_cold={ENABLE_POSTAL_COLD_START_RECALL}, "
        f"train_window_count={TRAIN_WINDOW_COUNT}, "
        f"drop_noisy_features={DROP_NOISY_FEATURES}, feature_experiment={FEATURE_EXPERIMENT}, "
        f"negative_sample_mode={NEGATIVE_SAMPLE_MODE}, negative_ratio={NEGATIVE_SAMPLE_RATIO}, "
        f"hard_negative_fraction={HARD_NEGATIVE_FRACTION}, hard_negative_top_rank={HARD_NEGATIVE_TOP_RANK}, "
        f"segment_candidates={ENABLE_SEGMENT_CANDIDATES}, "
        f"segment_baseline_top=active:{SEGMENT_ACTIVE_BASELINE_TOP}/warm:{SEGMENT_WARM_BASELINE_TOP}/cold:{SEGMENT_COLD_BASELINE_TOP}, "
        f"segment_cold_start_top={SEGMENT_COLD_START_RECALL_TOP}, "
        f"segment_rerank={ENABLE_SEGMENT_RERANK}, active_days={SEGMENT_ACTIVE_DAYS}, "
        f"stale_days={SEGMENT_STALE_DAYS}, cold_txn_max={SEGMENT_COLD_TXN_MAX}, "
        f"user_attr_features={ENABLE_USER_ATTR_FEATURES}, user_attr_days={USER_ATTR_FEATURE_DAYS}, "
        f"extended_user_attr_features={ENABLE_EXTENDED_USER_ATTR_FEATURES}, "
        f"source_rank_features={ENABLE_SOURCE_RANK_FEATURES}, "
        f"dynamic_item_features={ENABLE_DYNAMIC_ITEM_FEATURES}, "
        f"id_mapping={ENABLE_ID_MAPPING}, skip_validation={SKIP_VALIDATION}, "
        f"stream_submission={STREAM_SUBMISSION}, skip_submission={SKIP_SUBMISSION}"
    )
    print(
        "LGBM model params: "
        f"model_type={LGBM_MODEL_TYPE}, "
        f"learning_rate={LGBM_LEARNING_RATE}, num_leaves={LGBM_NUM_LEAVES}, "
        f"min_child_samples={LGBM_MIN_CHILD_SAMPLES}, subsample={LGBM_SUBSAMPLE}, "
        f"colsample_bytree={LGBM_COLSAMPLE_BYTREE}, reg_alpha={LGBM_REG_ALPHA}, "
        f"reg_lambda={LGBM_REG_LAMBDA}, n_jobs={LGBM_N_JOBS}, "
        f"force_col_wise={LGBM_FORCE_COL_WISE}, save_feature_importance={SAVE_FEATURE_IMPORTANCE}"
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
    id_mapping: IdMapping | None = None
    if ENABLE_ID_MAPPING:
        id_mapping = build_id_mapping(tables, transactions)
        (
            transactions,
            article_department,
            article_features,
            customer_features,
            customer_age_bin,
        ) = apply_id_mapping(
            transactions=transactions,
            article_department=article_department,
            article_features=article_features,
            customer_features=customer_features,
            customer_age_bin=customer_age_bin,
            id_mapping=id_mapping,
        )
        gc.collect()

    validation_metrics: dict[str, Any] | None = None
    validation_model: Any | None = None
    run_validation = SKIP_SUBMISSION or not SKIP_VALIDATION
    if run_validation:
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
        del validation_train_df
        gc.collect()
    else:
        print("LGBM_SKIP_VALIDATION=1: skip offline validation for submission run.")
    if SKIP_SUBMISSION:
        if validation_model is not None:
            save_feature_importance(validation_model, output_dir=OUTPUT_DIR)
        print("LGBM_SKIP_SUBMISSION=1: stop after local validation.")
        return {
            "summary": summary,
            "ranker_config": ranker_config,
            "validation_metrics": validation_metrics,
            "submission": None,
        }

    if validation_model is not None:
        del validation_model
        gc.collect()

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
    del final_train_df
    gc.collect()
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
        id_mapping=id_mapping,
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
    if state["validation_metrics"] is not None:
        print("\nLGBM validation metrics:")
        print(state["validation_metrics"])
    else:
        print("\nLGBM validation metrics: skipped")
    if state["submission"] is not None:
        print("\nSubmission preview:")
        print(state["submission"].head(5))
    return state


if __name__ == "__main__":
    main()
