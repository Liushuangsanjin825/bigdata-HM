"""Stage-2 baseline+ pipeline for the H&M recommendation task.

This script provides:
1. Time-window offline evaluation with MAP@12.
2. Multi-source candidate recall (user history + channel hot + global hot).
3. Competition-format submission generation and validation.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import plotly.express as px

BASE_PATH = Path(r"G:\h-and-m-personalized-fashion-recommendations")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "articles": BASE_PATH / "articles.csv",
    "customers": BASE_PATH / "customers.csv",
    "transactions": BASE_PATH / "transactions_train.csv",
    "submission": BASE_PATH / "sample_submission.csv",
}
RANDOM_STATE = 610
N_SPLITS = 5
MAX_K = 12
EVAL_WINDOW_DAYS = 7
OUTPUT_DIR = PROJECT_ROOT / "outputs"

USER_HISTORY_TOP = 24
CHANNEL_TOP = 24
GLOBAL_TOP = 120


@dataclass(frozen=True)
class EnvironmentInfo:
    python: str
    platform: str
    polars: str
    numpy: str
    matplotlib: str
    plotly: str


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_end: date
    valid_start: date
    valid_end: date
    customer_count: int
    map12: float


@dataclass(frozen=True)
class RecommenderArtifacts:
    reference_date: date
    user_history: dict[str, list[str]]
    customer_pref_channel: dict[str, int]
    channel_top_items: dict[int, list[str]]
    global_top_items: list[str]
    global_rank: dict[str, int]


def get_environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        polars=pl.__version__,
        numpy=np.__version__,
        matplotlib=plt.matplotlib.__version__,
        plotly=getattr(px, "__version__", "unknown"),
    )


def print_environment_info() -> None:
    info = get_environment_info()
    print(f"Python: {info.python}")
    print(f"Platform: {info.platform}")
    print(f"polars: {info.polars}")
    print(f"numpy: {info.numpy}")
    print(f"matplotlib: {info.matplotlib}")
    print(f"plotly: {info.plotly}")


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
    scores = [apk(actual, predicted, k=k) for actual, predicted in zip(actual_list, predicted_list)]
    return float(np.mean(scores))


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


def fit_recommender(train_tx: pl.DataFrame) -> RecommenderArtifacts:
    reference_date = train_tx.get_column("t_dat").max()
    user_history = _build_user_history(train_tx)
    customer_pref_channel = _build_customer_pref_channel(train_tx)
    channel_top_items = _build_channel_top(train_tx, reference_date)
    global_top_items = _build_global_top(train_tx, reference_date)
    global_rank = {item: idx for idx, item in enumerate(global_top_items)}
    return RecommenderArtifacts(
        reference_date=reference_date,
        user_history=user_history,
        customer_pref_channel=customer_pref_channel,
        channel_top_items=channel_top_items,
        global_top_items=global_top_items,
        global_rank=global_rank,
    )


def _rank_candidates(
    user_items: list[str],
    channel_items: list[str],
    global_items: list[str],
    global_rank: dict[str, int],
    k: int = MAX_K,
) -> list[str]:
    scores: dict[str, float] = {}
    for idx, item in enumerate(user_items):
        scores[item] = scores.get(item, 0.0) + 30.0 / (idx + 1)
    for idx, item in enumerate(channel_items):
        scores[item] = scores.get(item, 0.0) + 8.0 / (idx + 1)
    for idx, item in enumerate(global_items):
        scores[item] = scores.get(item, 0.0) + 2.0 / (idx + 1)

    ranked = sorted(
        scores.keys(),
        key=lambda item: (
            -scores[item],
            global_rank.get(item, 10**9),
            item,
        ),
    )
    return ranked[:k]


def predict_list_for_customer(
    customer_id: str,
    artifacts: RecommenderArtifacts,
    k: int = MAX_K,
) -> list[str]:
    user_items = artifacts.user_history.get(customer_id, [])
    pref_channel = artifacts.customer_pref_channel.get(customer_id)
    channel_items = artifacts.channel_top_items.get(pref_channel, []) if pref_channel is not None else []
    global_items = artifacts.global_top_items
    ranked_items = _rank_candidates(user_items, channel_items, global_items, artifacts.global_rank, k=k)

    # Safety fallback: ensure we always return up to k items.
    if len(ranked_items) < k:
        seen = set(ranked_items)
        for item in global_items:
            if item not in seen:
                ranked_items.append(item)
                seen.add(item)
            if len(ranked_items) >= k:
                break
    return ranked_items[:k]


def predict_for_customers(
    customer_ids: list[str],
    artifacts: RecommenderArtifacts,
    k: int = MAX_K,
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for customer_id in customer_ids:
        results[customer_id] = predict_list_for_customer(customer_id, artifacts, k=k)
    return results


def build_submission_frame(customer_ids: list[str], prediction_map: dict[str, list[str]]) -> pl.DataFrame:
    rows = []
    for customer_id in customer_ids:
        pred_items = prediction_map.get(customer_id, [])
        rows.append({"customer_id": customer_id, "prediction": " ".join(pred_items)})
    return pl.DataFrame(rows)


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
        submission.with_columns(
            pl.col("prediction").str.split(" ").alias("pred_tokens"),
        )
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


def _build_eval_folds(
    transactions: pl.DataFrame,
    n_splits: int = N_SPLITS,
    window_days: int = EVAL_WINDOW_DAYS,
) -> list[tuple[date, date, date]]:
    if transactions.is_empty():
        return []
    min_date = transactions.get_column("t_dat").min()
    max_date = transactions.get_column("t_dat").max()
    folds: list[tuple[date, date, date]] = []
    for fold_idx in range(n_splits):
        offset = (n_splits - 1 - fold_idx) * window_days
        valid_end = max_date - timedelta(days=offset)
        valid_start = valid_end - timedelta(days=window_days - 1)
        train_end = valid_start - timedelta(days=1)
        if train_end <= min_date:
            continue
        folds.append((train_end, valid_start, valid_end))
    return folds


def evaluate_fold(
    transactions: pl.DataFrame,
    fold_id: int,
    train_end: date,
    valid_start: date,
    valid_end: date,
) -> FoldResult:
    train_tx = transactions.filter(pl.col("t_dat") <= pl.lit(train_end))
    valid_tx = transactions.filter(
        (pl.col("t_dat") >= pl.lit(valid_start)) & (pl.col("t_dat") <= pl.lit(valid_end))
    )
    if train_tx.is_empty() or valid_tx.is_empty():
        return FoldResult(
            fold_id=fold_id,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            customer_count=0,
            map12=0.0,
        )

    artifacts = fit_recommender(train_tx)
    actual_df = (
        valid_tx.sort(["customer_id", "t_dat"])
        .group_by("customer_id")
        .agg(pl.col("article_id").unique(maintain_order=True).alias("actual_items"))
    )
    customers = actual_df.get_column("customer_id").to_list()
    prediction_map = predict_for_customers(customers, artifacts, k=MAX_K)

    actual_list: list[list[str]] = []
    predicted_list: list[list[str]] = []
    for row in actual_df.iter_rows(named=True):
        customer_id = row["customer_id"]
        actual_items = row["actual_items"] or []
        actual_list.append(actual_items)
        predicted_list.append(prediction_map.get(customer_id, []))

    score = mapk12(actual_list, predicted_list, k=MAX_K)
    return FoldResult(
        fold_id=fold_id,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        customer_count=len(customers),
        map12=score,
    )


def run_time_window_evaluation(
    transactions: pl.DataFrame,
    n_splits: int = N_SPLITS,
    window_days: int = EVAL_WINDOW_DAYS,
) -> pl.DataFrame:
    folds = _build_eval_folds(transactions, n_splits=n_splits, window_days=window_days)
    if not folds:
        return pl.DataFrame(
            {
                "fold_id": [],
                "train_end": [],
                "valid_start": [],
                "valid_end": [],
                "customer_count": [],
                "map12": [],
            }
        )

    results: list[FoldResult] = []
    for fold_idx, (train_end, valid_start, valid_end) in enumerate(folds, start=1):
        result = evaluate_fold(transactions, fold_idx, train_end, valid_start, valid_end)
        results.append(result)
        print(
            f"[Fold {fold_idx}] train_end={train_end} valid={valid_start}~{valid_end} "
            f"customers={result.customer_count} MAP@12={result.map12:.6f}"
        )

    return pl.DataFrame(
        {
            "fold_id": [r.fold_id for r in results],
            "train_end": [r.train_end for r in results],
            "valid_start": [r.valid_start for r in results],
            "valid_end": [r.valid_end for r in results],
            "customer_count": [r.customer_count for r in results],
            "map12": [r.map12 for r in results],
        }
    )


def prepare_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_submission(
    tables: dict[str, pl.LazyFrame],
    transactions: pl.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pl.DataFrame:
    customer_ids = (
        tables["submission"]
        .select(pl.col("customer_id").cast(pl.Utf8))
        .collect(engine="streaming")
        .get_column("customer_id")
        .to_list()
    )
    artifacts = fit_recommender(transactions)
    prediction_map = predict_for_customers(customer_ids, artifacts, k=MAX_K)
    submission = build_submission_frame(customer_ids, prediction_map)
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
    fold_metrics = run_time_window_evaluation(transactions, n_splits=N_SPLITS, window_days=EVAL_WINDOW_DAYS)
    submission = generate_submission(tables, transactions, output_dir=OUTPUT_DIR)
    return {
        "tables": tables,
        "summary": summary,
        "transactions": transactions,
        "fold_metrics": fold_metrics,
        "submission": submission,
    }


def main() -> dict[str, Any]:
    print_environment_info()
    print_data_file_status(BASE_PATH)
    state = run_pipeline(BASE_PATH)
    print("\nInput summary:")
    print(state["summary"])
    print("\nOffline MAP@12 summary:")
    fold_metrics = state["fold_metrics"]
    print(fold_metrics)
    if fold_metrics.height > 0:
        print(f"mean MAP@12: {fold_metrics.get_column('map12').mean():.6f}")
    print("\nSubmission preview:")
    print(state["submission"].head(5))
    return state


if __name__ == "__main__":
    main()
