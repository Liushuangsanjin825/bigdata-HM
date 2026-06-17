"""Multi-fold offline validation for the H&M LightGBM pipeline.

Run this on Kaggle after setting the same LGBM_* environment variables used by
Final_Project_LGBM.py. It trains/evaluates rolling weekly folds sequentially and
writes outputs/multifold_metrics.csv.
"""

from __future__ import annotations

import gc
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

import Final_Project_LGBM as fp


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


MULTIFOLD_COUNT = _env_int("LGBM_MULTIFOLD_COUNT", 3)
MULTIFOLD_TRAIN_CUSTOMER_CAP = _env_int("LGBM_MULTIFOLD_TRAIN_CUSTOMER_CAP", fp.TRAIN_CUSTOMER_CAP)
MULTIFOLD_VALIDATION_CUSTOMER_CAP = _env_int(
    "LGBM_MULTIFOLD_VALIDATION_CUSTOMER_CAP",
    fp.VALIDATION_CUSTOMER_CAP,
)


def build_validation_folds(transactions: pl.DataFrame, fold_count: int) -> list[fp.WindowSpec]:
    max_date = transactions.get_column("t_dat").max()
    min_date = transactions.get_column("t_dat").min()
    folds: list[fp.WindowSpec] = []
    for offset in range(max(1, fold_count)):
        label_end = max_date - timedelta(days=fp.LABEL_WINDOW_DAYS * offset)
        label_start = label_end - timedelta(days=fp.LABEL_WINDOW_DAYS - 1)
        history_end = label_start - timedelta(days=1)
        if history_end <= min_date:
            continue
        folds.append(fp.WindowSpec(history_end=history_end, label_start=label_start, label_end=label_end))
    return list(reversed(folds))


def _prepare_pipeline_inputs(base_path: Path) -> dict[str, Any]:
    tables = fp.load_tables(fp.resolve_base_path(base_path))
    fp.summarize_inputs(tables)
    transactions = fp.prepare_transactions(tables["transactions"])
    article_department = fp.prepare_article_department(tables["articles"])
    article_features = fp.prepare_article_model_features(tables["articles"])
    customer_features = fp.prepare_customer_model_features(tables["customers"])
    customer_age_bin = fp.prepare_customer_age_bin(tables["customers"])
    ranker_config = fp.load_ranker_from_cache(
        output_dir=fp.prepare_output_dir(fp.OUTPUT_DIR),
        fallback=fp.DEFAULT_RANKER,
    )
    print(f"candidate ranker: {fp.describe_ranker(ranker_config)}")

    id_mapping = None
    if fp.ENABLE_ID_MAPPING:
        id_mapping = fp.build_id_mapping(tables, transactions)
        (
            transactions,
            article_department,
            article_features,
            customer_features,
            customer_age_bin,
        ) = fp.apply_id_mapping(
            transactions=transactions,
            article_department=article_department,
            article_features=article_features,
            customer_features=customer_features,
            customer_age_bin=customer_age_bin,
            id_mapping=id_mapping,
        )
        gc.collect()

    return {
        "tables": tables,
        "transactions": transactions,
        "article_department": article_department,
        "article_features": article_features,
        "customer_features": customer_features,
        "customer_age_bin": customer_age_bin,
        "ranker_config": ranker_config,
        "id_mapping": id_mapping,
    }


def run_multifold_eval(base_path: Path = fp.BASE_PATH) -> pl.DataFrame:
    print(
        "LGBM multifold params: "
        f"fold_count={MULTIFOLD_COUNT}, train_cap={MULTIFOLD_TRAIN_CUSTOMER_CAP}, "
        f"valid_cap={MULTIFOLD_VALIDATION_CUSTOMER_CAP}, train_window_count={fp.TRAIN_WINDOW_COUNT}, "
        f"max_candidates={fp.MAX_CANDIDATES_PER_CUSTOMER}, user_attr={fp.ENABLE_USER_ATTR_FEATURES}, "
        f"user_attr_days={fp.USER_ATTR_FEATURE_DAYS}, model_type={fp.LGBM_MODEL_TYPE}"
    )
    inputs = _prepare_pipeline_inputs(base_path)
    transactions: pl.DataFrame = inputs["transactions"]
    folds = build_validation_folds(transactions, MULTIFOLD_COUNT)
    rows: list[dict[str, Any]] = []

    for fold_idx, validation_window in enumerate(folds, start=1):
        print(
            "\n"
            + "=" * 80
            + f"\nFOLD {fold_idx}/{len(folds)}: "
            + f"history_end={validation_window.history_end} "
            + f"labels={validation_window.label_start}~{validation_window.label_end}\n"
            + "=" * 80
        )
        train_windows = fp.build_rolling_training_windows(
            transactions,
            latest_label_end=validation_window.history_end,
            window_count=fp.TRAIN_WINDOW_COUNT,
        )
        train_df = fp.build_labeled_windows_dataset(
            transactions=transactions,
            article_department=inputs["article_department"],
            article_features=inputs["article_features"],
            customer_features=inputs["customer_features"],
            customer_age_bin=inputs["customer_age_bin"],
            ranker_config=inputs["ranker_config"],
            windows=train_windows,
            customer_cap=MULTIFOLD_TRAIN_CUSTOMER_CAP,
        )
        model = fp.train_lgbm_model(train_df)
        metrics = fp.run_single_window_validation(
            model=model,
            transactions=transactions,
            article_department=inputs["article_department"],
            article_features=inputs["article_features"],
            customer_features=inputs["customer_features"],
            customer_age_bin=inputs["customer_age_bin"],
            ranker_config=inputs["ranker_config"],
            window=validation_window,
            customer_cap=MULTIFOLD_VALIDATION_CUSTOMER_CAP,
        )
        metrics["fold"] = fold_idx
        rows.append(metrics)

        out_df = pl.DataFrame(rows)
        out_path = fp.prepare_output_dir(fp.OUTPUT_DIR) / "multifold_metrics.csv"
        out_df.write_csv(out_path)
        print(f"multifold metrics saved: {out_path}")

        del train_df, model
        gc.collect()

    metrics_df = pl.DataFrame(rows)
    if not metrics_df.is_empty():
        mean_map = metrics_df.get_column("map12").mean()
        min_map = metrics_df.get_column("map12").min()
        mean_recall = metrics_df.get_column("candidate_recall").mean()
        print(
            "\nMultifold summary: "
            f"folds={metrics_df.height}, mean_map12={mean_map:.6f}, "
            f"min_map12={min_map:.6f}, mean_candidate_recall={mean_recall:.6f}"
        )
    return metrics_df


def main() -> None:
    run_multifold_eval(fp.BASE_PATH)


if __name__ == "__main__":
    main()
