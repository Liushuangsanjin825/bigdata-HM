# -*- coding: utf-8 -*-
"""Optuna tuning entrypoint for the H&M LightGBM pipeline.

The tuner reuses the production candidate generation and MAP@12 validation
code in Final_Project_LGBM.py, then writes the best Kaggle `%env` lines to
`outputs/optuna_best_params.json`.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import polars as pl

import Final_Project_LGBM as pipe

try:
    import optuna
except ImportError as exc:  # pragma: no cover - depends on runtime image.
    raise RuntimeError("Optuna is required. On Kaggle, run: !pip install -q optuna") from exc


PIPE_GLOBAL_BY_ENV = {
    "LGBM_MAX_CANDIDATES_PER_CUSTOMER": "MAX_CANDIDATES_PER_CUSTOMER",
    "LGBM_BASELINE_RECALL_TOP": "BASELINE_RECALL_TOP",
    "LGBM_ATTRIBUTE_RECALL_TOP": "ATTRIBUTE_RECALL_TOP",
    "LGBM_RECENT_GLOBAL_RECALL_TOP": "RECENT_GLOBAL_RECALL_TOP",
    "LGBM_ENABLE_COOCCURRENCE_RECALL": "ENABLE_COOCCURRENCE_RECALL",
    "LGBM_COOCCURRENCE_RECALL_TOP": "COOCCURRENCE_RECALL_TOP",
    "LGBM_ENABLE_ITEMCF_RECALL": "ENABLE_ITEMCF_RECALL",
    "LGBM_ITEMCF_RECALL_TOP": "ITEMCF_RECALL_TOP",
    "LGBM_ITEMCF_HISTORY_TOP": "ITEMCF_HISTORY_TOP",
    "LGBM_ENABLE_PRODUCT_CODE_RECALL": "ENABLE_PRODUCT_CODE_RECALL",
    "LGBM_PRODUCT_CODE_RECALL_TOP": "PRODUCT_CODE_RECALL_TOP",
    "LGBM_PRODUCT_CODE_HISTORY_TOP": "PRODUCT_CODE_HISTORY_TOP",
    "LGBM_ENABLE_PRICE_BAND_RECALL": "ENABLE_PRICE_BAND_RECALL",
    "LGBM_PRICE_BAND_RECALL_TOP": "PRICE_BAND_RECALL_TOP",
    "LGBM_ENABLE_MULTI_DAY_TREND_RECALL": "ENABLE_MULTI_DAY_TREND_RECALL",
    "LGBM_MULTI_DAY_TREND_RECALL_TOP": "MULTI_DAY_TREND_RECALL_TOP",
    "LGBM_ENABLE_SELLABLE_FILTER": "ENABLE_SELLABLE_FILTER",
    "LGBM_SELLABLE_DAYS": "SELLABLE_DAYS",
    "LGBM_ENABLE_COLD_START_RECALL": "ENABLE_COLD_START_RECALL",
    "LGBM_COLD_START_HISTORY_MAX_ITEMS": "COLD_START_HISTORY_MAX_ITEMS",
    "LGBM_COLD_START_RECALL_TOP": "COLD_START_RECALL_TOP",
    "LGBM_NEGATIVE_SAMPLE_RATIO": "NEGATIVE_SAMPLE_RATIO",
    "LGBM_FEATURE_EXPERIMENT": "FEATURE_EXPERIMENT",
    "LGBM_DROP_NOISY_FEATURES": "DROP_NOISY_FEATURES",
    "LGBM_ENABLE_USER_ATTR_FEATURES": "ENABLE_USER_ATTR_FEATURES",
    "LGBM_USER_ATTR_FEATURE_DAYS": "USER_ATTR_FEATURE_DAYS",
    "LGBM_ENABLE_DYNAMIC_ITEM_FEATURES": "ENABLE_DYNAMIC_ITEM_FEATURES",
}

MODEL_PARAM_BY_ENV = {
    "LGBM_N_ESTIMATORS": "n_estimators",
    "LGBM_LEARNING_RATE": "learning_rate",
    "LGBM_NUM_LEAVES": "num_leaves",
    "LGBM_MAX_DEPTH": "max_depth",
    "LGBM_MIN_CHILD_SAMPLES": "min_child_samples",
    "LGBM_SUBSAMPLE": "subsample",
    "LGBM_COLSAMPLE_BYTREE": "colsample_bytree",
    "LGBM_REG_ALPHA": "reg_alpha",
    "LGBM_REG_LAMBDA": "reg_lambda",
}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _env_float_or_none(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return float(raw_value)


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _apply_pipeline_env(env_values: dict[str, Any]) -> None:
    for env_name, global_name in PIPE_GLOBAL_BY_ENV.items():
        if env_name in env_values:
            setattr(pipe, global_name, env_values[env_name])
    pipe.MODEL_FEATURE_COLUMNS = pipe._resolve_model_feature_columns()


def _suggest_params(trial: optuna.Trial) -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline_env = {
        "LGBM_MAX_CANDIDATES_PER_CUSTOMER": trial.suggest_categorical(
            "LGBM_MAX_CANDIDATES_PER_CUSTOMER", [100, 120, 150, 180]
        ),
        "LGBM_BASELINE_RECALL_TOP": trial.suggest_categorical("LGBM_BASELINE_RECALL_TOP", [80, 100, 120]),
        "LGBM_ATTRIBUTE_RECALL_TOP": trial.suggest_categorical("LGBM_ATTRIBUTE_RECALL_TOP", [6, 8, 10, 12]),
        "LGBM_RECENT_GLOBAL_RECALL_TOP": trial.suggest_categorical(
            "LGBM_RECENT_GLOBAL_RECALL_TOP", [8, 12, 16, 20]
        ),
        "LGBM_ENABLE_COOCCURRENCE_RECALL": trial.suggest_categorical(
            "LGBM_ENABLE_COOCCURRENCE_RECALL", [True, False]
        ),
        "LGBM_COOCCURRENCE_RECALL_TOP": trial.suggest_categorical(
            "LGBM_COOCCURRENCE_RECALL_TOP", [12, 16, 20, 24]
        ),
        "LGBM_ENABLE_ITEMCF_RECALL": trial.suggest_categorical("LGBM_ENABLE_ITEMCF_RECALL", [True, False]),
        "LGBM_ITEMCF_RECALL_TOP": trial.suggest_categorical("LGBM_ITEMCF_RECALL_TOP", [8, 12, 16]),
        "LGBM_ITEMCF_HISTORY_TOP": trial.suggest_categorical("LGBM_ITEMCF_HISTORY_TOP", [3, 5, 7]),
        "LGBM_ENABLE_PRODUCT_CODE_RECALL": trial.suggest_categorical(
            "LGBM_ENABLE_PRODUCT_CODE_RECALL", [True, False]
        ),
        "LGBM_PRODUCT_CODE_RECALL_TOP": trial.suggest_categorical("LGBM_PRODUCT_CODE_RECALL_TOP", [4, 8, 12]),
        "LGBM_PRODUCT_CODE_HISTORY_TOP": trial.suggest_categorical("LGBM_PRODUCT_CODE_HISTORY_TOP", [2, 4, 6]),
        "LGBM_ENABLE_PRICE_BAND_RECALL": trial.suggest_categorical(
            "LGBM_ENABLE_PRICE_BAND_RECALL", [True, False]
        ),
        "LGBM_PRICE_BAND_RECALL_TOP": trial.suggest_categorical("LGBM_PRICE_BAND_RECALL_TOP", [6, 10, 12]),
        "LGBM_ENABLE_MULTI_DAY_TREND_RECALL": trial.suggest_categorical(
            "LGBM_ENABLE_MULTI_DAY_TREND_RECALL", [True, False]
        ),
        "LGBM_MULTI_DAY_TREND_RECALL_TOP": trial.suggest_categorical(
            "LGBM_MULTI_DAY_TREND_RECALL_TOP", [6, 10, 12]
        ),
        "LGBM_ENABLE_SELLABLE_FILTER": trial.suggest_categorical(
            "LGBM_ENABLE_SELLABLE_FILTER", [True, False]
        ),
        "LGBM_SELLABLE_DAYS": trial.suggest_categorical("LGBM_SELLABLE_DAYS", [35, 42, 56]),
        "LGBM_ENABLE_COLD_START_RECALL": trial.suggest_categorical(
            "LGBM_ENABLE_COLD_START_RECALL", [True, False]
        ),
        "LGBM_COLD_START_HISTORY_MAX_ITEMS": trial.suggest_categorical(
            "LGBM_COLD_START_HISTORY_MAX_ITEMS", [1, 2, 3]
        ),
        "LGBM_COLD_START_RECALL_TOP": trial.suggest_categorical("LGBM_COLD_START_RECALL_TOP", [12, 18, 24]),
        "LGBM_NEGATIVE_SAMPLE_RATIO": trial.suggest_categorical("LGBM_NEGATIVE_SAMPLE_RATIO", [10, 15, 20, 30]),
        "LGBM_FEATURE_EXPERIMENT": trial.suggest_categorical("LGBM_FEATURE_EXPERIMENT", ["best", "base", "all"]),
        "LGBM_DROP_NOISY_FEATURES": trial.suggest_categorical("LGBM_DROP_NOISY_FEATURES", [False, True]),
        "LGBM_ENABLE_USER_ATTR_FEATURES": trial.suggest_categorical(
            "LGBM_ENABLE_USER_ATTR_FEATURES", [True, False]
        ),
        "LGBM_USER_ATTR_FEATURE_DAYS": trial.suggest_categorical("LGBM_USER_ATTR_FEATURE_DAYS", [90, 180, 365]),
        "LGBM_ENABLE_DYNAMIC_ITEM_FEATURES": trial.suggest_categorical(
            "LGBM_ENABLE_DYNAMIC_ITEM_FEATURES", [True, False]
        ),
    }
    model_env = {
        "LGBM_N_ESTIMATORS": trial.suggest_int("LGBM_N_ESTIMATORS", 300, 600, step=50),
        "LGBM_LEARNING_RATE": trial.suggest_float("LGBM_LEARNING_RATE", 0.025, 0.075),
        "LGBM_NUM_LEAVES": trial.suggest_categorical("LGBM_NUM_LEAVES", [64, 96, 128, 160]),
        "LGBM_MAX_DEPTH": trial.suggest_categorical("LGBM_MAX_DEPTH", [-1, 8, 10, 12]),
        "LGBM_MIN_CHILD_SAMPLES": trial.suggest_categorical("LGBM_MIN_CHILD_SAMPLES", [50, 80, 120, 160]),
        "LGBM_SUBSAMPLE": trial.suggest_float("LGBM_SUBSAMPLE", 0.75, 0.95),
        "LGBM_COLSAMPLE_BYTREE": trial.suggest_float("LGBM_COLSAMPLE_BYTREE", 0.75, 0.95),
        "LGBM_REG_ALPHA": trial.suggest_float("LGBM_REG_ALPHA", 1e-3, 2.0, log=True),
        "LGBM_REG_LAMBDA": trial.suggest_float("LGBM_REG_LAMBDA", 0.1, 6.0, log=True),
    }
    model_params = {MODEL_PARAM_BY_ENV[env_name]: value for env_name, value in model_env.items()}
    return pipeline_env | model_env, model_params


def _prepare_data(base_path: Path) -> dict[str, Any]:
    tables = pipe.load_tables(base_path)
    transactions = pipe.prepare_transactions(tables["transactions"])
    article_department = pipe.prepare_article_department(tables["articles"])
    article_features = pipe.prepare_article_model_features(tables["articles"])
    customer_features = pipe.prepare_customer_model_features(tables["customers"])
    customer_age_bin = pipe.prepare_customer_age_bin(tables["customers"])
    ranker_config = pipe.load_ranker_from_cache(
        output_dir=pipe.prepare_output_dir(pipe.OUTPUT_DIR),
        fallback=pipe.DEFAULT_RANKER,
    )

    id_mapping = None
    if pipe.ENABLE_ID_MAPPING:
        id_mapping = pipe.build_id_mapping(tables, transactions)
        (
            transactions,
            article_department,
            article_features,
            customer_features,
            customer_age_bin,
        ) = pipe.apply_id_mapping(
            transactions=transactions,
            article_department=article_department,
            article_features=article_features,
            customer_features=customer_features,
            customer_age_bin=customer_age_bin,
            id_mapping=id_mapping,
        )
        gc.collect()

    return {
        "transactions": transactions,
        "article_department": article_department,
        "article_features": article_features,
        "customer_features": customer_features,
        "customer_age_bin": customer_age_bin,
        "ranker_config": ranker_config,
        "id_mapping": id_mapping,
    }


def _trial_records(study: optuna.Study) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"attr_{key}": value for key, value in trial.user_attrs.items()})
        records.append(row)
    return records


def _write_outputs(study: optuna.Study, output_dir: Path) -> None:
    output_dir = pipe.prepare_output_dir(output_dir)
    records = _trial_records(study)
    if records:
        pl.DataFrame(records).write_csv(output_dir / "optuna_trials.csv")

    best_params = dict(study.best_params)
    recommended_env = {"LGBM_ENABLE_ID_MAPPING": pipe.ENABLE_ID_MAPPING}
    recommended_env.update(best_params)
    payload = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": best_params,
        "recommended_env": {key: _format_env_value(value) for key, value in recommended_env.items()},
    }
    with (output_dir / "optuna_best_params.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("Best trial:", study.best_trial.number)
    print(f"Best MAP@12: {study.best_value:.6f}")
    print("Recommended Kaggle env lines:")
    for key, value in payload["recommended_env"].items():
        print(f"%env {key}={value}")


def main() -> dict[str, Any]:
    n_trials = _env_int("OPTUNA_N_TRIALS", 20)
    train_customer_cap = _env_int("OPTUNA_TRAIN_CUSTOMER_CAP", 20000)
    validation_customer_cap = _env_int("OPTUNA_VALIDATION_CUSTOMER_CAP", 20000)
    train_window_count = _env_int("OPTUNA_TRAIN_WINDOW_COUNT", 1)
    timeout = _env_float_or_none("OPTUNA_TIMEOUT")
    study_name = os.getenv("OPTUNA_STUDY_NAME", "hm_lgbm_tuning")
    storage = os.getenv("OPTUNA_STORAGE") or None
    output_dir = Path(os.getenv("OPTUNA_OUTPUT_DIR", str(pipe.OUTPUT_DIR))).expanduser()
    base_path = pipe.resolve_base_path()

    print(
        "Optuna tuning config: "
        f"trials={n_trials}, train_cap={train_customer_cap}, validation_cap={validation_customer_cap}, "
        f"train_windows={train_window_count}, id_mapping={pipe.ENABLE_ID_MAPPING}"
    )
    print(f"data path: {base_path}")

    data = _prepare_data(base_path)
    train_window, validation_window = pipe.build_train_and_validation_windows(data["transactions"])
    training_windows = pipe.build_rolling_training_windows(
        data["transactions"],
        latest_label_end=train_window.label_end,
        window_count=train_window_count,
    )
    if not training_windows:
        raise RuntimeError("No training windows available for Optuna tuning.")

    sampler = optuna.samplers.TPESampler(seed=pipe.RANDOM_STATE, multivariate=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        train_df = None
        model = None
        summary = None
        env_values, model_params = _suggest_params(trial)
        _apply_pipeline_env(env_values)
        trial.set_user_attr("feature_count", len(pipe.MODEL_FEATURE_COLUMNS))

        try:
            train_df = pipe.build_labeled_windows_dataset(
                transactions=data["transactions"],
                article_department=data["article_department"],
                article_features=data["article_features"],
                customer_features=data["customer_features"],
                customer_age_bin=data["customer_age_bin"],
                ranker_config=data["ranker_config"],
                windows=training_windows,
                customer_cap=train_customer_cap,
            )
            if train_df.is_empty():
                raise optuna.TrialPruned("empty training matrix")

            trial.set_user_attr("train_rows", train_df.height)
            model = pipe.train_lgbm_model(train_df, model_params=model_params)
            summary = pipe.run_single_window_validation(
                model=model,
                transactions=data["transactions"],
                article_department=data["article_department"],
                article_features=data["article_features"],
                customer_features=data["customer_features"],
                customer_age_bin=data["customer_age_bin"],
                ranker_config=data["ranker_config"],
                window=validation_window,
                customer_cap=validation_customer_cap,
            )
            trial.set_user_attr("candidate_rows", summary["candidate_rows"])
            trial.set_user_attr("validation_customers", summary["customer_count"])
            return float(summary["map12"])
        finally:
            del train_df, model, summary
            gc.collect()

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    _write_outputs(study, output_dir)
    return {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": dict(study.best_params),
    }


if __name__ == "__main__":
    main()
