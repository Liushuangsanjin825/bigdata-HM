# -*- coding: utf-8 -*-
"""Submission-only pipeline for the H&M recommendation task.

This script is responsible only for:
1. Loading data.
2. Building recommendation artifacts.
3. Generating and validating `outputs/submission.csv`.

Offline evaluation and ranker tuning are moved to `Final_Project_Eval.py`.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

BASE_PATH = Path(r"G:\h-and-m-personalized-fashion-recommendations")
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_K = 12
USER_HISTORY_TOP = 24
CHANNEL_TOP = 24
DEPARTMENT_TOP = 24
GLOBAL_TOP = 120


@dataclass(frozen=True)
class EnvironmentInfo:
    python: str
    platform: str
    polars: str


@dataclass(frozen=True)
class RankerConfig:
    name: str
    user_weight: float
    channel_weight: float
    department_weight: float
    global_weight: float


@dataclass(frozen=True)
class RecommenderArtifacts:
    reference_date: date
    user_history: dict[str, list[str]]
    customer_pref_channel: dict[str, int]
    customer_pref_department: dict[str, int]
    channel_top_items: dict[int, list[str]]
    department_top_items: dict[int, list[str]]
    global_top_items: list[str]
    global_rank: dict[str, int]


@dataclass(frozen=True)
class PredictionContext:
    base_global_scores: dict[str, float]
    cold_start_prediction: list[str]


RANKER_CANDIDATES: tuple[RankerConfig, ...] = (
    RankerConfig("baseline_plus", user_weight=30.0, channel_weight=8.0, department_weight=6.0, global_weight=2.0),
    RankerConfig("history_heavy", user_weight=36.0, channel_weight=8.0, department_weight=6.0, global_weight=2.0),
    RankerConfig("channel_heavy", user_weight=30.0, channel_weight=12.0, department_weight=6.0, global_weight=2.0),
    RankerConfig("department_heavy", user_weight=30.0, channel_weight=8.0, department_weight=10.0, global_weight=2.0),
    RankerConfig("global_smooth", user_weight=26.0, channel_weight=8.0, department_weight=6.0, global_weight=3.5),
    RankerConfig("balanced_fresh", user_weight=28.0, channel_weight=10.0, department_weight=8.0, global_weight=2.5),
)
DEFAULT_RANKER = RANKER_CANDIDATES[0]
RANKER_BY_NAME = {cfg.name: cfg for cfg in RANKER_CANDIDATES}


def get_environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        polars=pl.__version__,
    )


def describe_ranker(config: RankerConfig) -> str:
    return (
        f"{config.name} "
        f"(user={config.user_weight}, channel={config.channel_weight}, "
        f"department={config.department_weight}, global={config.global_weight})"
    )


def print_environment_info() -> None:
    info = get_environment_info()
    print(f"Python: {info.python}")
    print(f"Platform: {info.platform}")
    print(f"polars: {info.polars}")


def print_data_file_status(base_path: Path = BASE_PATH) -> dict[str, Path]:
    resolved_files = {
        "articles": base_path / "articles.csv",
        "customers": base_path / "customers.csv",
        "transactions": base_path / "transactions_train.csv",
        "submission": base_path / "sample_submission.csv",
    }
    for name, path in resolved_files.items():
        print(f"{name:12s} exists={path.exists()} path={path}")
    return resolved_files


def load_tables(base_path: Path) -> dict[str, pl.LazyFrame]:
    return {
        "articles": pl.scan_csv(
            base_path / "articles.csv",
            schema_overrides={"article_id": pl.Utf8, "product_code": pl.Utf8},
        ),
        "customers": pl.scan_csv(
            base_path / "customers.csv",
            schema_overrides={"customer_id": pl.Utf8, "postal_code": pl.Utf8},
        ),
        "transactions": pl.scan_csv(
            base_path / "transactions_train.csv",
            schema_overrides={
                "t_dat": pl.Utf8,
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
            },
        ),
        "submission": pl.scan_csv(
            base_path / "sample_submission.csv",
            schema_overrides={"customer_id": pl.Utf8},
        ),
    }


def summarize_inputs(tables: dict[str, pl.LazyFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, lf in tables.items():
        schema = lf.collect_schema()
        row_count = lf.select(pl.len().alias("rows")).collect(engine="streaming")["rows"][0]
        rows.append(
            {
                "table": name,
                "rows": int(row_count),
                "columns": len(schema.names()),
                "column_list": ", ".join(schema.names()),
            }
        )
    return pl.DataFrame(rows)


def prepare_transactions(transactions_lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        transactions_lf.select(
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("article_id").cast(pl.Utf8),
            pl.col("t_dat").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("t_dat"),
            pl.col("price").cast(pl.Float64),
            pl.col("sales_channel_id").cast(pl.Int64),
        )
        .drop_nulls(["customer_id", "article_id", "t_dat"])
        .collect(engine="streaming")
    )


def prepare_article_department(articles_lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        articles_lf.select(
            pl.col("article_id").cast(pl.Utf8),
            pl.col("department_no").cast(pl.Int64, strict=False),
        )
        .drop_nulls(["article_id", "department_no"])
        .unique(subset=["article_id"], keep="first")
        .collect(engine="streaming")
    )


def _build_user_history(train_tx: pl.DataFrame) -> dict[str, list[str]]:
    history = (
        train_tx.group_by(["customer_id", "article_id"])
        .agg(
            pl.len().alias("ua_cnt"),
            pl.max("t_dat").alias("ua_last_date"),
        )
        .sort(["customer_id", "ua_cnt", "ua_last_date"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("article_id").head(USER_HISTORY_TOP).alias("hist_items"))
    )
    result: dict[str, list[str]] = {}
    for row in history.iter_rows(named=True):
        result[row["customer_id"]] = row["hist_items"] or []
    return result


def _build_customer_pref_channel(train_tx: pl.DataFrame) -> dict[str, int]:
    pref = (
        train_tx.group_by(["customer_id", "sales_channel_id"])
        .agg(
            pl.len().alias("txn_cnt"),
            pl.max("t_dat").alias("last_date"),
        )
        .sort(["customer_id", "txn_cnt", "last_date"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("sales_channel_id").first().alias("pref_channel"))
    )
    result: dict[str, int] = {}
    for row in pref.iter_rows(named=True):
        result[row["customer_id"]] = int(row["pref_channel"])
    return result


def _build_customer_pref_department(tx_with_dept: pl.DataFrame) -> dict[str, int]:
    pref = (
        tx_with_dept.drop_nulls(["department_no"])
        .group_by(["customer_id", "department_no"])
        .agg(
            pl.len().alias("txn_cnt"),
            pl.max("t_dat").alias("last_date"),
        )
        .sort(["customer_id", "txn_cnt", "last_date"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("department_no").first().alias("pref_department"))
    )
    result: dict[str, int] = {}
    for row in pref.iter_rows(named=True):
        result[row["customer_id"]] = int(row["pref_department"])
    return result


def _build_channel_top(train_tx: pl.DataFrame, ref_date: date) -> dict[int, list[str]]:
    cutoff = ref_date - timedelta(days=30)
    channel_top = (
        train_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by(["sales_channel_id", "article_id"])
        .agg(pl.len().alias("pop_30d"))
        .sort(["sales_channel_id", "pop_30d", "article_id"], descending=[False, True, False])
        .group_by("sales_channel_id")
        .agg(pl.col("article_id").head(CHANNEL_TOP).alias("channel_items"))
    )
    result: dict[int, list[str]] = {}
    for row in channel_top.iter_rows(named=True):
        result[int(row["sales_channel_id"])] = row["channel_items"] or []
    return result


def _build_department_top(tx_with_dept: pl.DataFrame, ref_date: date) -> dict[int, list[str]]:
    cutoff = ref_date - timedelta(days=30)
    department_top = (
        tx_with_dept.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .drop_nulls(["department_no"])
        .group_by(["department_no", "article_id"])
        .agg(pl.len().alias("pop_30d"))
        .sort(["department_no", "pop_30d", "article_id"], descending=[False, True, False])
        .group_by("department_no")
        .agg(pl.col("article_id").head(DEPARTMENT_TOP).alias("department_items"))
    )
    result: dict[int, list[str]] = {}
    for row in department_top.iter_rows(named=True):
        result[int(row["department_no"])] = row["department_items"] or []
    return result


def _build_global_top(train_tx: pl.DataFrame, ref_date: date) -> list[str]:
    cutoff = ref_date - timedelta(days=30)
    top_recent = (
        train_tx.filter(pl.col("t_dat") >= pl.lit(cutoff))
        .group_by("article_id")
        .agg(
            pl.len().alias("pop_30d"),
            pl.n_unique("customer_id").alias("buyer_cnt_30d"),
        )
        .sort(["pop_30d", "buyer_cnt_30d", "article_id"], descending=[True, True, False])
        .select(pl.col("article_id").head(GLOBAL_TOP))
        .get_column("article_id")
        .to_list()
    )
    top_all = (
        train_tx.group_by("article_id")
        .agg(
            pl.len().alias("pop_all"),
            pl.n_unique("customer_id").alias("buyer_cnt_all"),
        )
        .sort(["pop_all", "buyer_cnt_all", "article_id"], descending=[True, True, False])
        .select(pl.col("article_id").head(GLOBAL_TOP))
        .get_column("article_id")
        .to_list()
    )
    merged: list[str] = []
    seen: set[str] = set()
    for item in top_recent + top_all:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged[:GLOBAL_TOP]


def fit_recommender(train_tx: pl.DataFrame, article_department: pl.DataFrame) -> RecommenderArtifacts:
    reference_date = train_tx.get_column("t_dat").max()
    tx_with_dept = train_tx.join(article_department, on="article_id", how="left")
    user_history = _build_user_history(train_tx)
    customer_pref_channel = _build_customer_pref_channel(train_tx)
    customer_pref_department = _build_customer_pref_department(tx_with_dept)
    channel_top_items = _build_channel_top(train_tx, reference_date)
    department_top_items = _build_department_top(tx_with_dept, reference_date)
    global_top_items = _build_global_top(train_tx, reference_date)
    global_rank = {item: idx for idx, item in enumerate(global_top_items)}
    return RecommenderArtifacts(
        reference_date=reference_date,
        user_history=user_history,
        customer_pref_channel=customer_pref_channel,
        customer_pref_department=customer_pref_department,
        channel_top_items=channel_top_items,
        department_top_items=department_top_items,
        global_top_items=global_top_items,
        global_rank=global_rank,
    )


def _build_prediction_context(artifacts: RecommenderArtifacts, ranker_config: RankerConfig) -> PredictionContext:
    base_global_scores: dict[str, float] = {}
    if ranker_config.global_weight > 0:
        for idx, item in enumerate(artifacts.global_top_items):
            base_global_scores[item] = ranker_config.global_weight / (idx + 1)
    return PredictionContext(
        base_global_scores=base_global_scores,
        cold_start_prediction=artifacts.global_top_items[:MAX_K],
    )


def predict_lists_for_customers(
    customer_ids: list[str],
    artifacts: RecommenderArtifacts,
    ranker_config: RankerConfig,
    k: int = MAX_K,
    show_progress: bool = False,
    progress_label: str = "predict",
) -> list[list[str]]:
    context = _build_prediction_context(artifacts, ranker_config)
    global_items = artifacts.global_top_items
    global_rank = artifacts.global_rank
    user_history = artifacts.user_history
    pref_channel_map = artifacts.customer_pref_channel
    pref_department_map = artifacts.customer_pref_department
    channel_top_items = artifacts.channel_top_items
    department_top_items = artifacts.department_top_items
    log_every = 200000

    results: list[list[str]] = []
    total = len(customer_ids)
    for idx, customer_id in enumerate(customer_ids, start=1):
        user_items = user_history.get(customer_id, [])
        pref_channel = pref_channel_map.get(customer_id)
        pref_department = pref_department_map.get(customer_id)
        channel_items = channel_top_items.get(pref_channel, []) if pref_channel is not None else []
        department_items = department_top_items.get(pref_department, []) if pref_department is not None else []

        if not user_items and pref_channel is None and pref_department is None:
            ranked_items = context.cold_start_prediction[:k]
        else:
            scores = context.base_global_scores.copy()
            for pos, item in enumerate(user_items):
                scores[item] = scores.get(item, 0.0) + ranker_config.user_weight / (pos + 1)
            for pos, item in enumerate(channel_items):
                scores[item] = scores.get(item, 0.0) + ranker_config.channel_weight / (pos + 1)
            for pos, item in enumerate(department_items):
                scores[item] = scores.get(item, 0.0) + ranker_config.department_weight / (pos + 1)

            ranked_pairs = sorted(
                scores.items(),
                key=lambda kv: (-kv[1], global_rank.get(kv[0], 10**9), kv[0]),
            )
            ranked_items = [item for item, _ in ranked_pairs[:k]]
            if len(ranked_items) < k:
                seen = set(ranked_items)
                for item in global_items:
                    if item not in seen:
                        ranked_items.append(item)
                        seen.add(item)
                    if len(ranked_items) >= k:
                        break
            ranked_items = ranked_items[:k]
        results.append(ranked_items)

        if show_progress and total >= log_every and (idx % log_every == 0 or idx == total):
            print(f"[{progress_label}] {idx}/{total} customers done")
    return results


def build_submission_frame_from_lists(customer_ids: list[str], prediction_lists: list[list[str]]) -> pl.DataFrame:
    prediction_text = [" ".join(pred_items) for pred_items in prediction_lists]
    return pl.DataFrame({"customer_id": customer_ids, "prediction": prediction_text})


def validate_submission_format(
    submission: pl.DataFrame,
    expected_customer_ids: list[str],
    k: int = MAX_K,
) -> None:
    required_cols = {"customer_id", "prediction"}
    if set(submission.columns) != required_cols:
        raise ValueError(f"提交列名错误：期望 {required_cols}，实际 {set(submission.columns)}")

    if submission.height != len(expected_customer_ids):
        raise ValueError(f"提交行数错误：期望 {len(expected_customer_ids)}，实际 {submission.height}")

    expected_set = set(expected_customer_ids)
    actual_ids = submission.get_column("customer_id").to_list()
    actual_set = set(actual_ids)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if missing or extra:
        raise ValueError(f"customer_id 覆盖错误：missing={len(missing)} extra={len(extra)}")

    token_stats = (
        submission.with_columns(pl.col("prediction").str.split(" ").alias("pred_tokens"))
        .with_columns(
            pl.col("pred_tokens")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .alias("pred_tokens")
        )
        .with_columns(
            pl.col("pred_tokens").list.len().alias("pred_len"),
            pl.col("pred_tokens").list.n_unique().alias("pred_unique_len"),
        )
    )
    over_k = token_stats.filter(pl.col("pred_len") > k).height
    empty_row = token_stats.filter(pl.col("pred_len") == 0).height
    has_dup = token_stats.filter(pl.col("pred_len") != pl.col("pred_unique_len")).height
    if over_k > 0 or empty_row > 0 or has_dup > 0:
        raise ValueError(
            "提交格式校验失败："
            f"over_k={over_k}, empty_row={empty_row}, duplicate_items_rows={has_dup}"
        )


def prepare_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_ranker_from_cache(
    output_dir: Path = OUTPUT_DIR,
    fallback: RankerConfig = DEFAULT_RANKER,
) -> RankerConfig:
    cache_path = output_dir / "ranker_tuning_metrics.csv"
    if not cache_path.exists():
        print(f"ranker cache not found, use fallback: {describe_ranker(fallback)}")
        return fallback
    try:
        metrics = pl.read_csv(cache_path)
    except Exception as exc:  # pragma: no cover
        print(f"failed to read ranker cache ({exc}), use fallback: {describe_ranker(fallback)}")
        return fallback
    if metrics.is_empty():
        print(f"ranker cache is empty, use fallback: {describe_ranker(fallback)}")
        return fallback

    sort_cols = [col for col in ["mean_map12", "std_map12"] if col in metrics.columns]
    if sort_cols:
        descending = [True if col == "mean_map12" else False for col in sort_cols]
        metrics = metrics.sort(sort_cols, descending=descending)
    row = metrics.row(0, named=True)

    required_weight_cols = {"user_weight", "channel_weight", "department_weight", "global_weight"}
    if required_weight_cols.issubset(metrics.columns):
        cfg = RankerConfig(
            name=str(row.get("config", "cached_ranker")),
            user_weight=float(row["user_weight"]),
            channel_weight=float(row["channel_weight"]),
            department_weight=float(row["department_weight"]),
            global_weight=float(row["global_weight"]),
        )
        print(f"use cached ranker weights: {describe_ranker(cfg)}")
        return cfg

    cache_name = str(row.get("config", ""))
    if cache_name in RANKER_BY_NAME:
        cfg = RANKER_BY_NAME[cache_name]
        print(f"use cached ranker by name: {describe_ranker(cfg)}")
        return cfg

    print(f"ranker cache format mismatch, use fallback: {describe_ranker(fallback)}")
    return fallback


def generate_submission(
    tables: dict[str, pl.LazyFrame],
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
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
    artifacts = fit_recommender(transactions, article_department=article_department)
    prediction_lists = predict_lists_for_customers(
        customer_ids=customer_ids,
        artifacts=artifacts,
        ranker_config=ranker_config,
        k=MAX_K,
        show_progress=True,
        progress_label="submission",
    )
    submission = build_submission_frame_from_lists(customer_ids, prediction_lists)
    validate_submission_format(submission, customer_ids, k=MAX_K)

    prepare_output_dir(output_dir)
    output_path = output_dir / "submission.csv"
    submission.write_csv(output_path)
    print(f"submission saved: {output_path}")
    return submission


def run_pipeline(base_path: Path = BASE_PATH) -> dict[str, Any]:
    tables = load_tables(base_path)
    summary = summarize_inputs(tables)
    transactions = prepare_transactions(tables["transactions"])
    article_department = prepare_article_department(tables["articles"])
    ranker_config = load_ranker_from_cache(output_dir=prepare_output_dir(OUTPUT_DIR))
    submission = generate_submission(
        tables=tables,
        transactions=transactions,
        article_department=article_department,
        ranker_config=ranker_config,
        output_dir=OUTPUT_DIR,
    )
    return {
        "summary": summary,
        "ranker_config": ranker_config,
        "submission": submission,
    }


def main() -> dict[str, Any]:
    print_environment_info()
    print_data_file_status(BASE_PATH)
    state = run_pipeline(BASE_PATH)

    print("\nInput summary:")
    print(state["summary"])
    print("\nSelected ranker:")
    print(describe_ranker(state["ranker_config"]))
    print("\nSubmission preview:")
    print(state["submission"].head(5))
    return state


if __name__ == "__main__":
    main()
