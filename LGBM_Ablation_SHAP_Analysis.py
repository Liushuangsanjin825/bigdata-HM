# -*- coding: utf-8 -*-
"""Ablation and SHAP analysis for the H&M LightGBM baseline.

Run on Kaggle after attaching:
- H&M competition dataset.
- A code dataset containing `Final_Project.py` and `Final_Project_LGBM.py`.

Outputs:
- outputs/lgbm_ablation_top5.csv
- outputs/shap_beeswarm.png
- outputs/shap_waterfall_sample0.png
- outputs/shap_mean_abs_importance.csv
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def prepare_kaggle_files() -> None:
    """Copy project scripts from /kaggle/input to /kaggle/working when needed."""
    input_dir = Path("/kaggle/input")
    work_dir = Path("/kaggle/working")
    if not input_dir.exists() or not work_dir.exists():
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["Final_Project.py", "Final_Project_LGBM.py"]:
        target = work_dir / filename
        if target.exists():
            continue
        matches = list(input_dir.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Cannot find {filename}. Attach the code dataset containing this file.")
        shutil.copy(matches[0], target)
        print(f"copied {matches[0]} -> {target}")

    data_candidates = [
        Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
        Path("/kaggle/input/competitions/h-and-m-personalized-fashion-recommendations"),
    ]
    required_files = ["articles.csv", "customers.csv", "transactions_train.csv", "sample_submission.csv"]
    if not os.getenv("HNM_DATA_PATH"):
        for candidate in data_candidates:
            if all((candidate / name).exists() for name in required_files):
                os.environ["HNM_DATA_PATH"] = str(candidate)
                break

    os.chdir(work_dir)
    print(f"cwd = {Path.cwd()}")
    print(f"HNM_DATA_PATH = {os.getenv('HNM_DATA_PATH')}")


prepare_kaggle_files()

import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import shap  # noqa: E402

from Final_Project import (  # noqa: E402
    BASE_PATH,
    DEFAULT_RANKER,
    MAX_K,
    OUTPUT_DIR,
    fit_recommender,
    load_ranker_from_cache,
    load_tables,
    prepare_article_department,
    prepare_customer_age_bin,
    prepare_output_dir,
    prepare_transactions,
    resolve_base_path,
)
from Final_Project_LGBM import (  # noqa: E402
    FEATURE_COLUMNS,
    MAX_CANDIDATES_PER_CUSTOMER,
    RANDOM_STATE,
    _collect_actual_items,
    build_candidate_frame,
    build_feature_frame,
    build_labeled_window_dataset,
    build_train_and_validation_windows,
    downsample_training_rows,
    mapk12,
    prepare_article_model_features,
    scored_frame_to_prediction_map,
)


ANALYSIS_TRAIN_CUSTOMER_CAP = _env_int("ANALYSIS_TRAIN_CUSTOMER_CAP", 10000)
ANALYSIS_VALID_CUSTOMER_CAP = _env_int("ANALYSIS_VALID_CUSTOMER_CAP", 10000)
SHAP_SAMPLE_SIZE = _env_int("SHAP_SAMPLE_SIZE", 1000)
N_ESTIMATORS = _env_int("N_ESTIMATORS", 120)


def train_model(train_frame: pl.DataFrame, feature_cols: list[str], n_estimators: int = N_ESTIMATORS) -> lgb.LGBMClassifier:
    x_train = train_frame.select(feature_cols).to_pandas()
    y_train = train_frame.get_column("label").to_numpy()
    positive_count = float(np.sum(y_train))
    negative_count = float(len(y_train) - positive_count)
    scale_pos_weight = max(1.0, negative_count / max(positive_count, 1.0))

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=96,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y_train, feature_name=feature_cols)
    return model


def evaluate_map12(
    model: lgb.LGBMClassifier,
    feature_frame: pl.DataFrame,
    feature_cols: list[str],
    valid_customers: list[str],
    actual_list: list[list[str]],
) -> float:
    x_valid = feature_frame.select(feature_cols).to_pandas()
    scores = model.predict_proba(x_valid)[:, 1]
    scored = feature_frame.with_columns(pl.Series("score", scores))
    pred_map = scored_frame_to_prediction_map(scored, k=MAX_K)
    predicted_list = [pred_map.get(customer_id, []) for customer_id in valid_customers]
    return mapk12(actual_list, predicted_list, k=MAX_K)


def build_validation_features(
    transactions: pl.DataFrame,
    article_department: pl.DataFrame,
    article_features: pl.DataFrame,
    customer_age_bin: dict[str, str],
    ranker_config,
    validation_window,
) -> tuple[pl.DataFrame, list[str], list[list[str]]]:
    history_tx = transactions.filter(pl.col("t_dat") <= pl.lit(validation_window.history_end))
    label_tx = transactions.filter(
        (pl.col("t_dat") >= pl.lit(validation_window.label_start))
        & (pl.col("t_dat") <= pl.lit(validation_window.label_end))
    )
    valid_customers, actual_map = _collect_actual_items(label_tx, customer_cap=ANALYSIS_VALID_CUSTOMER_CAP)
    valid_artifacts = fit_recommender(
        history_tx,
        article_department=article_department,
        customer_age_bin=customer_age_bin,
    )
    valid_candidates = build_candidate_frame(
        customer_ids=valid_customers,
        artifacts=valid_artifacts,
        ranker_config=ranker_config,
        max_candidates=MAX_CANDIDATES_PER_CUSTOMER,
    )
    valid_features = build_feature_frame(
        candidates=valid_candidates,
        history_tx=history_tx,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        artifacts=valid_artifacts,
        reference_date=validation_window.history_end,
    )
    actual_list = [actual_map.get(customer_id, []) for customer_id in valid_customers]
    return valid_features, valid_customers, actual_list


def run_analysis() -> dict[str, object]:
    output_dir = prepare_output_dir(OUTPUT_DIR)
    base_path = resolve_base_path(BASE_PATH)
    print(f"analysis params: train_cap={ANALYSIS_TRAIN_CUSTOMER_CAP}, valid_cap={ANALYSIS_VALID_CUSTOMER_CAP}, "
          f"shap_sample={SHAP_SAMPLE_SIZE}, n_estimators={N_ESTIMATORS}")

    tables = load_tables(base_path)
    transactions = prepare_transactions(tables["transactions"])
    article_department = prepare_article_department(tables["articles"])
    article_features = prepare_article_model_features(tables["articles"])
    customer_age_bin = prepare_customer_age_bin(tables["customers"])
    ranker_config = load_ranker_from_cache(output_dir=output_dir, fallback=DEFAULT_RANKER)

    train_window, validation_window = build_train_and_validation_windows(transactions)
    print(f"train_window = {train_window}")
    print(f"validation_window = {validation_window}")

    train_df_raw = build_labeled_window_dataset(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        window=train_window,
        customer_cap=ANALYSIS_TRAIN_CUSTOMER_CAP,
    )
    train_df = downsample_training_rows(train_df_raw)
    print(f"train rows after downsample: {train_df.height}")

    valid_features, valid_customers, actual_list = build_validation_features(
        transactions=transactions,
        article_department=article_department,
        article_features=article_features,
        customer_age_bin=customer_age_bin,
        ranker_config=ranker_config,
        validation_window=validation_window,
    )
    print(f"validation customers: {len(valid_customers)}")
    print(f"validation candidate rows: {valid_features.height}")

    baseline_model = train_model(train_df, FEATURE_COLUMNS)
    baseline_map12 = evaluate_map12(
        baseline_model,
        valid_features,
        FEATURE_COLUMNS,
        valid_customers,
        actual_list,
    )
    print(f"Baseline MAP@12 = {baseline_map12:.6f}")

    gain_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": baseline_model.booster_.feature_importance(importance_type="gain"),
            "split": baseline_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    gain_df["gain_rank"] = np.arange(1, len(gain_df) + 1)
    top5_features = gain_df.head(5)["feature"].tolist()
    gain_df.to_csv(output_dir / "lgbm_gain_importance.csv", index=False)
    print("Top-5 gain features:")
    print(gain_df.head(5).to_string(index=False))

    ablation_rows: list[dict[str, object]] = []
    for removed_feature in top5_features:
        ablated_features = [col for col in FEATURE_COLUMNS if col != removed_feature]
        model = train_model(train_df, ablated_features)
        removed_map12 = evaluate_map12(
            model,
            valid_features,
            ablated_features,
            valid_customers,
            actual_list,
        )
        marginal_contribution = baseline_map12 - removed_map12
        gain_row = gain_df[gain_df["feature"] == removed_feature].iloc[0]
        ablation_rows.append(
            {
                "removed_feature": removed_feature,
                "gain_rank": int(gain_row["gain_rank"]),
                "gain": float(gain_row["gain"]),
                "baseline_map12": baseline_map12,
                "map12_after_removal": removed_map12,
                "marginal_contribution": marginal_contribution,
                "pseudo_important": marginal_contribution <= 0.0,
            }
        )
        print(
            f"removed={removed_feature:28s} MAP@12={removed_map12:.6f} "
            f"contribution={marginal_contribution:.6f}"
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("marginal_contribution", ascending=False)
    ablation_path = output_dir / "lgbm_ablation_top5.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"ablation saved: {ablation_path}")
    print(ablation_df.to_string(index=False))

    shap_n = min(SHAP_SAMPLE_SIZE, valid_features.height)
    shap_frame = valid_features.sample(n=shap_n, seed=RANDOM_STATE, shuffle=True)
    x_shap = shap_frame.select(FEATURE_COLUMNS).to_pandas()
    explainer = shap.TreeExplainer(baseline_model)
    shap_explanation = explainer(x_shap)

    if len(shap_explanation.values.shape) == 3:
        shap_class1 = shap.Explanation(
            values=shap_explanation.values[:, :, 1],
            base_values=shap_explanation.base_values[:, 1],
            data=x_shap,
            feature_names=FEATURE_COLUMNS,
        )
    else:
        shap_class1 = shap_explanation

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_class1, max_display=20, show=False)
    plt.tight_layout()
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"SHAP beeswarm saved: {beeswarm_path}")

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_class1[0], max_display=15, show=False)
    plt.tight_layout()
    waterfall_path = output_dir / "shap_waterfall_sample0.png"
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall saved: {waterfall_path}")

    mean_abs_shap = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": np.abs(shap_class1.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    mean_abs_path = output_dir / "shap_mean_abs_importance.csv"
    mean_abs_shap.to_csv(mean_abs_path, index=False)
    print(f"SHAP mean absolute importance saved: {mean_abs_path}")
    print("Top SHAP features:")
    print(mean_abs_shap.head(10).to_string(index=False))

    pseudo_features = ablation_df.loc[ablation_df["pseudo_important"], "removed_feature"].tolist()
    if pseudo_features:
        print(f"Potential pseudo-important high-gain features: {pseudo_features}")
    else:
        print("No high-gain feature has zero/negative marginal contribution in this validation run.")

    return {
        "baseline_map12": baseline_map12,
        "top5_features": top5_features,
        "ablation": ablation_df,
        "gain": gain_df,
        "mean_abs_shap": mean_abs_shap,
    }


if __name__ == "__main__":
    run_analysis()
