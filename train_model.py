import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from model_estimators import (
    CatBoostAdapter,
    MarketResidualModel,
    TemporalEnsemble,
    make_gradient_model,
    make_spline_logistic,
)
from model_pipeline import (
    CATEGORICAL_FEATURES,
    DIFFERENCE_FEATURES,
    FEATURE_COLUMNS,
    FULL_FEATURE_COLUMNS,
    FULL_NUMERIC_FEATURES,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    build_history,
    clipped_logit,
    sigmoid,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
MODEL_PATH = ROOT / "betting_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
RESIDUAL_FEATURES = (
    "age_diff",
    "glicko_logit",
    "glicko_uncertainty_mean",
    "layoff_diff",
    "str_against_division_z_diff",
    "td_for_division_z_diff",
)


def metrics(y_true, probability):
    return {
        "accuracy": float(accuracy_score(y_true, probability >= 0.5)),
        "auc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def temporal_partitions(frame):
    dates = np.array(sorted(frame["event_date"].unique()))
    calibration_start = dates[int(len(dates) * 0.70)]
    test_start = dates[int(len(dates) * 0.85)]
    development = frame[frame["event_date"] < calibration_start].copy()
    calibration = frame[
        (frame["event_date"] >= calibration_start) & (frame["event_date"] < test_start)
    ].copy()
    test = frame[frame["event_date"] >= test_start].copy()
    return development, calibration, test


def expanding_folds(frame, initial_fraction=0.40, fold_count=4):
    dates = np.array(sorted(frame["event_date"].unique()))
    first = max(1, int(len(dates) * initial_fraction))
    remaining = len(dates) - first
    fold_size = max(1, math_ceil(remaining / fold_count))
    for index in range(fold_count):
        validation_start = first + index * fold_size
        if validation_start >= len(dates):
            break
        validation_end = min(len(dates), validation_start + fold_size)
        train_dates = set(dates[:validation_start])
        validation_dates = set(dates[validation_start:validation_end])
        train = frame[frame["event_date"].isin(train_dates)]
        validation = frame[frame["event_date"].isin(validation_dates)]
        if not train.empty and not validation.empty:
            yield train, validation


def math_ceil(value):
    return int(np.ceil(value))


def model_specs(numeric_columns, feature_columns):
    return {
        "catboost": (
            CatBoostAdapter(categorical_columns=tuple(CATEGORICAL_FEATURES)),
            list(feature_columns),
        ),
        "gradient": (make_gradient_model(list(numeric_columns), depth=3), list(numeric_columns)),
        "spline_logistic": (
            make_spline_logistic(list(numeric_columns), CATEGORICAL_FEATURES, c_value=0.1),
            list(feature_columns),
        ),
    }


def fit_models(frame, numeric_columns, feature_columns):
    fitted, columns = {}, {}
    for name, (model, selected_columns) in model_specs(numeric_columns, feature_columns).items():
        fitted[name] = clone(model).fit(frame[selected_columns], frame["target"])
        columns[name] = selected_columns
    return fitted, columns


def base_predictions(models, columns, frame):
    return {
        name: model.predict_proba(frame[columns[name]])[:, 1]
        for name, model in models.items()
    }


def meta_matrix(predictions, frame):
    ordered = [predictions[name] for name in sorted(predictions)]
    ordered.append(sigmoid(frame["glicko_logit"].to_numpy()))
    return np.column_stack(ordered)


def walk_forward_ablation_score(frame, numeric_columns):
    scores = []
    for train, validation in expanding_folds(frame, initial_fraction=0.55, fold_count=3):
        model = make_gradient_model(list(numeric_columns), depth=3)
        model.fit(train[list(numeric_columns)], train["target"])
        probability = model.predict_proba(validation[list(numeric_columns)])[:, 1]
        scores.append(log_loss(validation["target"], probability, labels=[0, 1]))
    return float(np.mean(scores))


def build_oof_predictions(frame, numeric_columns, feature_columns):
    names = sorted(model_specs(numeric_columns, feature_columns))
    oof = pd.DataFrame(index=frame.index, columns=names, dtype=float)
    for train, validation in expanding_folds(frame):
        models, columns = fit_models(train, numeric_columns, feature_columns)
        predictions = base_predictions(models, columns, validation)
        for name in names:
            oof.loc[validation.index, name] = predictions[name]
    oof["glicko"] = sigmoid(frame["glicko_logit"].to_numpy())
    return oof.dropna(subset=names)


def fit_platt(probability, target):
    calibrator = LogisticRegression(C=1000.0, max_iter=1000)
    calibrator.fit(clipped_logit(probability).reshape(-1, 1), target)
    return calibrator


def apply_platt(calibrator, probability):
    return calibrator.predict_proba(clipped_logit(probability).reshape(-1, 1))[:, 1]


def symmetric_residual_predict(residual, model_probability, market_probability, frame):
    direct = residual.predict_probability(model_probability, market_probability, frame)
    mirrored = frame.copy()
    for column in DIFFERENCE_FEATURES:
        mirrored[column] = -mirrored[column]
    reverse = residual.predict_probability(
        1.0 - np.asarray(model_probability),
        1.0 - np.asarray(market_probability),
        mirrored,
    )
    return (direct + 1.0 - reverse) / 2.0


def crossfit_meta_for_residual(oof, frame):
    usable = frame.loc[oof.index].sort_values("event_date")
    oof = oof.loc[usable.index]
    dates = np.array(sorted(usable["event_date"].unique()))
    split_points = (0.55, 0.75, 1.0)
    output = pd.Series(np.nan, index=usable.index, dtype=float)
    for start, end in zip(split_points[:-1], split_points[1:]):
        train_end = dates[int(len(dates) * start)]
        validation_end = dates[min(len(dates) - 1, int(len(dates) * end) - 1)]
        train_mask = usable["event_date"] < train_end
        validation_mask = (
            (usable["event_date"] >= train_end) & (usable["event_date"] <= validation_end)
        )
        meta = LogisticRegression(C=0.1, max_iter=1500)
        meta.fit(oof.loc[train_mask], usable.loc[train_mask, "target"])
        output.loc[validation_mask] = meta.predict_proba(oof.loc[validation_mask])[:, 1]
    return output.dropna()


def select_residual(
    development,
    oof,
    calibration,
    calibration_probability,
):
    crossfit_probability = crossfit_meta_for_residual(oof, development)
    residual_train = development.loc[crossfit_probability.index]
    train_mask = residual_train["market_probability"].notna()
    residual_train = residual_train.loc[train_mask]
    train_probability = crossfit_probability.loc[residual_train.index].to_numpy()
    market_probability = residual_train["market_probability"].to_numpy()

    calibration_mask = calibration["market_probability"].notna().to_numpy()
    calibration_market = calibration.loc[calibration_mask, "market_probability"].to_numpy()
    calibration_model = calibration_probability[calibration_mask]
    calibration_frame = calibration.loc[calibration_mask]
    leaderboard = {}
    fitted = {}
    for l2 in (1.0, 10.0, 50.0, 200.0):
        residual = MarketResidualModel(RESIDUAL_FEATURES, l2=l2)
        residual.fit(
            train_probability,
            market_probability,
            residual_train,
            residual_train["target"],
        )
        probability = symmetric_residual_predict(
            residual,
            calibration_model,
            calibration_market,
            calibration_frame,
        )
        leaderboard[str(l2)] = float(log_loss(
            calibration_frame["target"], probability, labels=[0, 1]
        ))
        fitted[str(l2)] = residual
    winner = min(leaderboard, key=leaderboard.get)
    return fitted[winner], winner, leaderboard, int(len(residual_train))


def expected_calibration_error(y_true, probability, bins=10):
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y_true[mask].mean())
    return float(error)


def train(data_dir=DATA_DIR, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    frame, _, _, _ = build_history(data_dir)
    development, calibration, test = temporal_partitions(frame)

    ablation = {
        "core": walk_forward_ablation_score(development, NUMERIC_FEATURES),
        "full": walk_forward_ablation_score(development, FULL_NUMERIC_FEATURES),
    }
    use_full = ablation["full"] + 0.001 < ablation["core"]
    numeric_columns = FULL_NUMERIC_FEATURES if use_full else NUMERIC_FEATURES
    feature_columns = FULL_FEATURE_COLUMNS if use_full else FEATURE_COLUMNS
    feature_set_name = "full" if use_full else "ablated_core"

    oof = build_oof_predictions(development, numeric_columns, feature_columns)
    oof_frame = development.loc[oof.index]
    meta_model = LogisticRegression(C=0.1, max_iter=1500)
    meta_model.fit(oof.to_numpy(), oof_frame["target"])

    development_models, model_columns = fit_models(
        development, numeric_columns, feature_columns
    )
    calibration_base = base_predictions(development_models, model_columns, calibration)
    test_base = base_predictions(development_models, model_columns, test)
    calibration_raw = meta_model.predict_proba(meta_matrix(calibration_base, calibration))[:, 1]
    test_raw = meta_model.predict_proba(meta_matrix(test_base, test))[:, 1]
    calibrator = fit_platt(calibration_raw, calibration["target"])
    calibration_probability = apply_platt(calibrator, calibration_raw)
    test_probability = apply_platt(calibrator, test_raw)

    residual, residual_l2, residual_leaderboard, residual_samples = select_residual(
        development,
        oof,
        calibration,
        calibration_probability,
    )
    market_test_mask = test["market_probability"].notna().to_numpy()
    market_test = test.loc[market_test_mask]
    market_probability = market_test["market_probability"].to_numpy()
    residual_probability = symmetric_residual_predict(
        residual,
        test_probability[market_test_mask],
        market_probability,
        market_test,
    )
    market_metrics = metrics(market_test["target"], market_probability)
    residual_metrics = metrics(market_test["target"], residual_probability)
    backtest_gate_passed = bool(
        residual_metrics["log_loss"] + 0.001 < market_metrics["log_loss"]
        and residual_metrics["brier"] < market_metrics["brier"]
    )

    base_test_metrics = {
        name: metrics(test["target"], probability)
        for name, probability in test_base.items()
    }
    base_test_metrics["dynamic_glicko"] = metrics(
        test["target"], sigmoid(test["glicko_logit"].to_numpy())
    )
    calibration_error = expected_calibration_error(calibration["target"], calibration_probability)
    uncertainty_margin = max(0.03, min(0.15, calibration_error * 1.5))
    report = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": "temporal_oof_ensemble",
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "sample_count": int(len(frame)),
        "event_count": int(frame["event_name"].nunique()),
        "splits": {
            "development": int(len(development)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
            "ensemble_oof": int(len(oof)),
            "residual_oof": residual_samples,
        },
        "feature_ablation_log_loss": ablation,
        "base_test": base_test_metrics,
        "calibrated_test": metrics(test["target"], test_probability),
        "ensemble_weights": {
            name: float(weight)
            for name, weight in zip(oof.columns, meta_model.coef_[0])
        },
        "market_test": market_metrics,
        "market_residual_test": residual_metrics,
        "market_residual_l2": float(residual_l2),
        "market_residual_calibration_log_loss": residual_leaderboard,
        "calibration_error": calibration_error,
        "uncertainty_margin": uncertainty_margin,
        "backtest_gate_passed": backtest_gate_passed,
        "betting_enabled": False,
        "gate_reason": (
            "Backtest passed; forward evidence is still required."
            if backtest_gate_passed
            else "Market residual did not beat the no-vig market out of sample."
        ),
    }

    production_models, production_columns = fit_models(frame, numeric_columns, feature_columns)
    ensemble = TemporalEnsemble(
        production_models,
        production_columns,
        meta_model,
        calibrator,
        "glicko_logit",
    )
    artifact = {
        "version": MODEL_VERSION,
        "ensemble": ensemble,
        "market_residual": residual,
        "feature_columns": list(feature_columns),
        "numeric_columns": list(numeric_columns),
        "metadata": report,
    }
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and validate the Better probability ensemble.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    report = train(args.data_dir, args.model_path, args.metrics_path)
    print(json.dumps({
        "selected_model": report["selected_model"],
        "feature_set": report["feature_set"],
        "test_log_loss": report["calibrated_test"]["log_loss"],
        "market_log_loss": report["market_test"]["log_loss"],
        "residual_log_loss": report["market_residual_test"]["log_loss"],
        "backtest_gate_passed": report["backtest_gate_passed"],
        "betting_enabled": report["betting_enabled"],
    }, indent=2))


if __name__ == "__main__":
    main()
