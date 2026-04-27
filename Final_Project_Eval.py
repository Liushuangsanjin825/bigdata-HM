# -*- coding: utf-8 -*-
"""Offline evaluation and ranker tuning layer for H&M recommendation.

This script is separated from `Final_Project.py` on purpose:
- `Final_Project.py` only generates submission data.
- `Final_Project_Eval.py` performs tuning/evaluation and saves metrics.
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
    RANKER_CANDIDATES,
    DEFAULT_RANKER,
    RankerConfig,
    describe_ranker,
    fit_recommender,
    load_tables,
    predict_lists_for_customers,
    prepare_article_department,
    prepare_customer_age_bin,
    prepare_output_dir,
    prepare_transactions,
    summarize_inputs,
)

N_SPLITS = 5
EVAL_WINDOW_DAYS = 7
TUNING_FOLDS = 3
TUNING_CUSTOMER_CAP = 40000
OFFLINE_EVAL_CUSTOMER_CAP: int | None = None


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_end: date
    valid_start: date
    valid_end: date
    customer_count: int
    map12: float


@dataclass
class TuningFoldContext:
    fold_id: int
    train_end: date
    valid_start: date
    valid_end: date
    customer_count: int
    artifacts: Any
    customers: list[str]
    actual_list: list[list[str]]


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


def _empty_fold_metrics() -> pl.DataFrame:
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


def _empty_tuning_metrics() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "config": [],
            "user_weight": [],
            "long_history_weight": [],
            "channel_weight": [],
            "department_weight": [],
            "age_weight": [],
            "trend_weight": [],
            "global_weight": [],
            "fold_count": [],
            "mean_map12": [],
            "std_map12": [],
        }
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


def _collect_actual_df(valid_tx: pl.DataFrame, customer_cap: int | None = None) -> pl.DataFrame:
    actual_df = (
        valid_tx.sort(["customer_id", "t_dat"])
        .group_by("customer_id")
        .agg(pl.col("article_id").unique(maintain_order=True).alias("actual_items"))
        .sort("customer_id")
    )
    if customer_cap is not None and customer_cap > 0 and actual_df.height > customer_cap:
        actual_df = actual_df.head(customer_cap)
    return actual_df


def evaluate_fold(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    fold_id: int,
    train_end: date,
    valid_start: date,
    valid_end: date,
    customer_cap: int | None = None,
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

    artifacts = fit_recommender(
        train_tx,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
    )
    actual_df = _collect_actual_df(valid_tx, customer_cap=customer_cap)
    if actual_df.is_empty():
        return FoldResult(
            fold_id=fold_id,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            customer_count=0,
            map12=0.0,
        )

    customers = actual_df.get_column("customer_id").to_list()
    predicted_list = predict_lists_for_customers(
        customer_ids=customers,
        artifacts=artifacts,
        ranker_config=ranker_config,
        k=MAX_K,
    )
    actual_list = [(row["actual_items"] or []) for row in actual_df.iter_rows(named=True)]
    score = mapk12(actual_list, predicted_list, k=MAX_K)

    return FoldResult(
        fold_id=fold_id,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        customer_count=len(customers),
        map12=score,
    )


def _ranker_weight_key(config: RankerConfig) -> tuple[float, float, float, float, float, float, float]:
    return (
        round(config.user_weight, 6),
        round(config.long_history_weight, 6),
        round(config.channel_weight, 6),
        round(config.department_weight, 6),
        round(config.age_weight, 6),
        round(config.trend_weight, 6),
        round(config.global_weight, 6),
    )


def _dedupe_ranker_configs(configs: list[RankerConfig]) -> list[RankerConfig]:
    seen: set[tuple[float, ...]] = set()
    deduped: list[RankerConfig] = []
    for cfg in configs:
        key = _ranker_weight_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def _build_local_ranker_candidates(anchor: RankerConfig) -> list[RankerConfig]:
    # Local neighborhood search around the current best ranker.
    raw_weights = [
        (
            anchor.user_weight + 4.0,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight + 2.0,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight - 2.0,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight + 3.0,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight - 2.0,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight + 2.0,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight - 2.0,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight + 2.0,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight + 2.0,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight - 1.5,
            anchor.trend_weight,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight + 2.0,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight - 1.0,
            anchor.global_weight,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight + 0.6,
        ),
        (
            anchor.user_weight,
            anchor.long_history_weight,
            anchor.channel_weight,
            anchor.department_weight,
            anchor.age_weight,
            anchor.trend_weight,
            anchor.global_weight - 0.4,
        ),
        (
            anchor.user_weight + 2.0,
            anchor.long_history_weight + 2.0,
            anchor.channel_weight - 1.0,
            anchor.department_weight,
            anchor.age_weight + 1.0,
            anchor.trend_weight + 1.0,
            anchor.global_weight - 0.2,
        ),
        (
            anchor.user_weight + 4.0,
            anchor.long_history_weight - 1.0,
            anchor.channel_weight - 1.0,
            anchor.department_weight - 1.0,
            anchor.age_weight + 1.0,
            anchor.trend_weight + 1.5,
            anchor.global_weight,
        ),
    ]
    candidates: list[RankerConfig] = []
    for user_w, long_w, channel_w, department_w, age_w, trend_w, global_w in raw_weights:
        user_w = max(1.0, float(user_w))
        long_w = max(0.5, float(long_w))
        channel_w = max(0.5, float(channel_w))
        department_w = max(0.5, float(department_w))
        age_w = max(0.5, float(age_w))
        trend_w = max(0.5, float(trend_w))
        global_w = max(0.5, float(global_w))
        name = (
            f"local_u{user_w:g}_l{long_w:g}_c{channel_w:g}_d{department_w:g}_"
            f"a{age_w:g}_t{trend_w:g}_g{global_w:g}"
        )
        candidates.append(
            RankerConfig(
                name=name,
                user_weight=user_w,
                long_history_weight=long_w,
                channel_weight=channel_w,
                department_weight=department_w,
                age_weight=age_w,
                trend_weight=trend_w,
                global_weight=global_w,
            )
        )
    return _dedupe_ranker_configs(candidates)


def _prepare_tuning_contexts(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    customer_age_bin: dict[str, str],
    selected_folds: list[tuple[date, date, date]],
    customer_cap: int,
) -> list[TuningFoldContext]:
    contexts: list[TuningFoldContext] = []
    for tune_fold_idx, (train_end, valid_start, valid_end) in enumerate(selected_folds, start=1):
        train_tx = transactions.filter(pl.col("t_dat") <= pl.lit(train_end))
        valid_tx = transactions.filter(
            (pl.col("t_dat") >= pl.lit(valid_start)) & (pl.col("t_dat") <= pl.lit(valid_end))
        )
        if train_tx.is_empty() or valid_tx.is_empty():
            continue

        artifacts = fit_recommender(
            train_tx,
            article_department=article_department,
            customer_age_bin=customer_age_bin,
        )
        actual_df = _collect_actual_df(valid_tx, customer_cap=customer_cap)
        if actual_df.is_empty():
            continue

        customers = actual_df.get_column("customer_id").to_list()
        actual_list = [(row["actual_items"] or []) for row in actual_df.iter_rows(named=True)]
        contexts.append(
            TuningFoldContext(
                fold_id=tune_fold_idx,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                customer_count=len(customers),
                artifacts=artifacts,
                customers=customers,
                actual_list=actual_list,
            )
        )
        print(
            f"[Tune Setup Fold {tune_fold_idx}] train_end={train_end} "
            f"valid={valid_start}~{valid_end} customers={len(customers)}"
        )
    return contexts


def _score_rankers_on_contexts(
    contexts: list[TuningFoldContext],
    candidate_configs: list[RankerConfig],
    stage_name: str,
) -> dict[str, list[float]]:
    score_board: dict[str, list[float]] = {cfg.name: [] for cfg in candidate_configs}
    for context in contexts:
        for cfg in candidate_configs:
            predicted_list = predict_lists_for_customers(
                customer_ids=context.customers,
                artifacts=context.artifacts,
                ranker_config=cfg,
                k=MAX_K,
            )
            score = mapk12(context.actual_list, predicted_list, k=MAX_K)
            score_board[cfg.name].append(score)
            print(
                f"[Tune {stage_name} Fold {context.fold_id}] config={cfg.name} "
                f"customers={context.customer_count} MAP@12={score:.6f}"
            )
    return score_board


def _build_tuning_metrics(
    candidate_configs: list[RankerConfig],
    score_board: dict[str, list[float]],
) -> pl.DataFrame:
    if not candidate_configs:
        return _empty_tuning_metrics()
    rows: list[dict[str, Any]] = []
    for cfg in candidate_configs:
        scores = score_board.get(cfg.name, [])
        rows.append(
            {
                "config": cfg.name,
                "user_weight": cfg.user_weight,
                "long_history_weight": cfg.long_history_weight,
                "channel_weight": cfg.channel_weight,
                "department_weight": cfg.department_weight,
                "age_weight": cfg.age_weight,
                "trend_weight": cfg.trend_weight,
                "global_weight": cfg.global_weight,
                "fold_count": len(scores),
                "mean_map12": float(np.mean(scores)) if scores else 0.0,
                "std_map12": float(np.std(scores)) if scores else 0.0,
            }
        )
    return pl.DataFrame(rows).sort(["mean_map12", "std_map12"], descending=[True, False])


def tune_ranker_config(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    customer_age_bin: dict[str, str],
    n_splits: int = N_SPLITS,
    window_days: int = EVAL_WINDOW_DAYS,
    tuning_folds: int = TUNING_FOLDS,
    customer_cap: int = TUNING_CUSTOMER_CAP,
) -> tuple[RankerConfig, pl.DataFrame]:
    folds = _build_eval_folds(transactions, n_splits=n_splits, window_days=window_days)
    if not folds:
        return DEFAULT_RANKER, _empty_tuning_metrics()

    use_fold_count = min(max(tuning_folds, 1), len(folds))
    selected_folds = folds[-use_fold_count:]
    contexts = _prepare_tuning_contexts(
        transactions=transactions,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
        selected_folds=selected_folds,
        customer_cap=customer_cap,
    )
    if not contexts:
        return DEFAULT_RANKER, _empty_tuning_metrics()

    base_candidates = _dedupe_ranker_configs(list(RANKER_CANDIDATES))
    base_score_board = _score_rankers_on_contexts(contexts, base_candidates, stage_name="base")
    base_metrics = _build_tuning_metrics(base_candidates, base_score_board)
    if base_metrics.is_empty():
        return DEFAULT_RANKER, base_metrics

    best_base_row = base_metrics.row(0, named=True)
    best_base_ranker = RankerConfig(
        name=str(best_base_row["config"]),
        user_weight=float(best_base_row["user_weight"]),
        long_history_weight=float(best_base_row["long_history_weight"]),
        channel_weight=float(best_base_row["channel_weight"]),
        department_weight=float(best_base_row["department_weight"]),
        age_weight=float(best_base_row["age_weight"]),
        trend_weight=float(best_base_row["trend_weight"]),
        global_weight=float(best_base_row["global_weight"]),
    )
    print(f"base best ranker: {describe_ranker(best_base_ranker)}")

    local_candidates = _build_local_ranker_candidates(best_base_ranker)
    base_keys = {_ranker_weight_key(cfg) for cfg in base_candidates}
    local_candidates = [cfg for cfg in local_candidates if _ranker_weight_key(cfg) not in base_keys]
    local_score_board: dict[str, list[float]] = {}
    if local_candidates:
        print(f"local search candidate count: {len(local_candidates)}")
        local_score_board = _score_rankers_on_contexts(contexts, local_candidates, stage_name="local")

    all_candidates = base_candidates + local_candidates
    merged_score_board: dict[str, list[float]] = {}
    for cfg in all_candidates:
        merged_score_board[cfg.name] = base_score_board.get(cfg.name, []) + local_score_board.get(cfg.name, [])

    tuning_metrics = _build_tuning_metrics(all_candidates, merged_score_board)
    if tuning_metrics.is_empty():
        return DEFAULT_RANKER, _empty_tuning_metrics()

    best_row = tuning_metrics.row(0, named=True)
    best_ranker = RankerConfig(
        name=str(best_row["config"]),
        user_weight=float(best_row["user_weight"]),
        long_history_weight=float(best_row["long_history_weight"]),
        channel_weight=float(best_row["channel_weight"]),
        department_weight=float(best_row["department_weight"]),
        age_weight=float(best_row["age_weight"]),
        trend_weight=float(best_row["trend_weight"]),
        global_weight=float(best_row["global_weight"]),
    )
    print(f"selected ranker: {describe_ranker(best_ranker)}")
    return best_ranker, tuning_metrics


def run_time_window_evaluation(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config: RankerConfig,
    n_splits: int = N_SPLITS,
    window_days: int = EVAL_WINDOW_DAYS,
    customer_cap: int | None = OFFLINE_EVAL_CUSTOMER_CAP,
) -> pl.DataFrame:
    folds = _build_eval_folds(transactions, n_splits=n_splits, window_days=window_days)
    if not folds:
        return _empty_fold_metrics()

    cap_text = "all_customers" if customer_cap is None else f"customer_cap={customer_cap}"
    print(f"offline evaluation ranker: {describe_ranker(ranker_config)} ({cap_text})")

    results: list[FoldResult] = []
    for fold_idx, (train_end, valid_start, valid_end) in enumerate(folds, start=1):
        result = evaluate_fold(
            transactions=transactions,
            article_department=article_department,
            customer_age_bin=customer_age_bin,
            ranker_config=ranker_config,
            fold_id=fold_idx,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            customer_cap=customer_cap,
        )
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


def save_eval_outputs(
    tuning_metrics: pl.DataFrame,
    fold_metrics: pl.DataFrame,
    best_ranker: RankerConfig,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = prepare_output_dir(output_dir)
    tuning_path = output_dir / "ranker_tuning_metrics.csv"
    fold_path = output_dir / "offline_fold_metrics.csv"
    ranker_path = output_dir / "selected_ranker.csv"

    tuning_metrics.write_csv(tuning_path)
    fold_metrics.write_csv(fold_path)
    pl.DataFrame(
        [
            {
                "config": best_ranker.name,
                "user_weight": best_ranker.user_weight,
                "long_history_weight": best_ranker.long_history_weight,
                "channel_weight": best_ranker.channel_weight,
                "department_weight": best_ranker.department_weight,
                "age_weight": best_ranker.age_weight,
                "trend_weight": best_ranker.trend_weight,
                "global_weight": best_ranker.global_weight,
            }
        ]
    ).write_csv(ranker_path)

    print(f"ranker tuning metrics saved: {tuning_path}")
    print(f"offline fold metrics saved: {fold_path}")
    print(f"selected ranker saved: {ranker_path}")


def run_eval_pipeline(base_path: Path = BASE_PATH) -> dict[str, Any]:
    tables = load_tables(base_path)
    summary = summarize_inputs(tables)
    transactions = prepare_transactions(tables["transactions"])
    article_department = prepare_article_department(tables["articles"])
    customer_age_bin = prepare_customer_age_bin(tables["customers"])

    best_ranker, tuning_metrics = tune_ranker_config(
        transactions=transactions,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
        n_splits=N_SPLITS,
        window_days=EVAL_WINDOW_DAYS,
        tuning_folds=TUNING_FOLDS,
        customer_cap=TUNING_CUSTOMER_CAP,
    )
    fold_metrics = run_time_window_evaluation(
        transactions=transactions,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
        ranker_config=best_ranker,
        n_splits=N_SPLITS,
        window_days=EVAL_WINDOW_DAYS,
        customer_cap=OFFLINE_EVAL_CUSTOMER_CAP,
    )
    save_eval_outputs(
        tuning_metrics=tuning_metrics,
        fold_metrics=fold_metrics,
        best_ranker=best_ranker,
        output_dir=OUTPUT_DIR,
    )
    return {
        "summary": summary,
        "best_ranker": best_ranker,
        "tuning_metrics": tuning_metrics,
        "fold_metrics": fold_metrics,
    }


def main() -> dict[str, Any]:
    state = run_eval_pipeline(BASE_PATH)
    print("\nInput summary:")
    print(state["summary"])
    print("\nSelected ranker:")
    print(describe_ranker(state["best_ranker"]))
    print("\nTuning summary:")
    print(state["tuning_metrics"])
    print("\nOffline MAP@12 summary:")
    print(state["fold_metrics"])
    if state["fold_metrics"].height > 0:
        print(f"mean MAP@12: {state['fold_metrics'].get_column('map12').mean():.6f}")
    return state


if __name__ == "__main__":
    main()
