"""Chronological UFC fight-method model.

The target is the way a bout ends, independent of the winning corner.  All
features are constructed before the event and made order-invariant, so swapping
the fighters leaves a prediction unchanged.
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from model_pipeline import (
    CATEGORICAL_FEATURES,
    DIFFERENCE_FEATURES,
    FULL_NUMERIC_FEATURES,
    build_history,
    current_features,
    load_source_data,
    update_states,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
MODEL_PATH = ROOT / "fight_method_model.joblib"
REPORT_PATH = ROOT / "fight_method_metrics.json"
MODEL_VERSION = "1.0.0"
CLASSES = ("ko_tko", "submission", "decision", "nc")

# Pair features are absolute differences plus pair-level features. This makes
# the response independent of the order supplied by the client.
METHOD_NUMERIC_FEATURES = [f"abs_{name}" for name in FULL_NUMERIC_FEATURES if name not in {"glicko_logit"}]
METHOD_CATEGORICAL_FEATURES = list(CATEGORICAL_FEATURES) + ["scheduled_rounds"]
METHOD_FEATURES = METHOD_NUMERIC_FEATURES + METHOD_CATEGORICAL_FEATURES


def classify_method(row):
    """Map UFCStats values to the four published labels, or None if excluded."""
    if row.result == "nc":
        return "nc"
    text = str(row.method or "").strip().casefold()
    if text.startswith("ko/tko"):
        return "ko_tko"
    if text.startswith("sub"):
        return "submission"
    if "dec" in text:
        return "decision"
    # DQs, overturned results, and empty/other methods are not one of the four
    # requested methods. They still update the historical fighter state.
    return None


def method_features(pair_frame, scheduled_rounds=3):
    row = pair_frame.iloc[0]
    values = {f"abs_{name}": abs(float(row[name])) if pd.notna(row[name]) else np.nan
              for name in FULL_NUMERIC_FEATURES if name != "glicko_logit"}
    values.update({name: row[name] for name in CATEGORICAL_FEATURES})
    values["scheduled_rounds"] = str(int(scheduled_rounds))
    return values


def build_method_history(data_dir=DATA_DIR):
    """Return one pre-fight row per completed eligible bout, in date order."""
    # Re-run the same state update logic event by event so the target never leaks
    # into a feature row from the current event.
    fights, fighters, _ = load_source_data(data_dir)
    from collections import defaultdict
    from model_pipeline import FighterState, RunningMoments, build_profiles

    profiles = build_profiles(fighters)
    states = defaultdict(FighterState)
    division_states = defaultdict(lambda: defaultdict(RunningMoments))
    completed = fights[fights["result"].isin(["win", "draw", "nc"]) & fights["event_date_dt"].notna()].copy()
    completed = completed.sort_values("event_date_dt")
    records = []

    for event_date, event_rows in completed.groupby("event_date_dt", sort=True):
        event_day = pd.Timestamp(event_date).date()
        for row in event_rows.itertuples(index=False):
            target = classify_method(row)
            if target is None:
                continue
            pair = current_features(
                row.fighter_red, row.fighter_blue, row.weight_class, event_day,
                states, profiles, division_states,
            )
            records.append({
                "fight_id": row.fight_link,
                "event_name": row.event_name,
                "event_date": event_day.isoformat(),
                "fighter_red": row.fighter_red,
                "fighter_blue": row.fighter_blue,
                "weight_class": row.weight_class,
                "target": target,
                **method_features(pair),
            })
        for row in event_rows.itertuples(index=False):
            update_states(row, states, division_states, event_day)
    return pd.DataFrame(records), states, profiles, division_states


def temporal_partitions(frame):
    dates = np.array(sorted(frame.event_date.unique()))
    calibration_start = dates[int(len(dates) * 0.70)]
    test_start = dates[int(len(dates) * 0.85)]
    return (
        frame[frame.event_date < calibration_start].copy(),
        frame[(frame.event_date >= calibration_start) & (frame.event_date < test_start)].copy(),
        frame[frame.event_date >= test_start].copy(),
    )


def make_logistic():
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), METHOD_NUMERIC_FEATURES),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), METHOD_CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=2500, C=0.3))])


def make_gradient():
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), METHOD_NUMERIC_FEATURES),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), METHOD_CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", GradientBoostingClassifier(
        learning_rate=0.04, max_depth=2, n_estimators=180, min_samples_leaf=18, random_state=42
    ))])


def aligned_probabilities(model, frame):
    raw = model.predict_proba(frame[METHOD_FEATURES])
    output = np.full((len(frame), len(CLASSES)), 1e-9)
    for index, label in enumerate(model.classes_):
        output[:, CLASSES.index(label)] = raw[:, index]
    return output / output.sum(axis=1, keepdims=True)


def fit_candidate(name, model, frame):
    return model.fit(frame[METHOD_FEATURES], frame.target)


def frequency_probability(train, count):
    # Dirichlet smoothing prevents an unjustified zero probability for NC.
    counts = train.target.value_counts()
    values = np.array([counts.get(label, 0) + 2.0 for label in CLASSES], dtype=float)
    return np.tile(values / values.sum(), (count, 1))


def multiclass_brier(y_true, probability):
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(len(y_true)), [CLASSES.index(value) for value in y_true]] = 1.0
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def ece(y_true, probability, bins=10):
    encoded = np.array([CLASSES.index(value) for value in y_true])
    output = {}
    for class_index, label in enumerate(CLASSES):
        values, actual = probability[:, class_index], (encoded == class_index).astype(float)
        total = 0.0
        for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
            mask = (values >= lo) & (values < hi if hi < 1 else values <= hi)
            if mask.any(): total += mask.mean() * abs(values[mask].mean() - actual[mask].mean())
        output[label] = float(total)
    return output


def evaluate(frame, probability):
    encoded = np.array([CLASSES.index(value) for value in frame.target])
    predicted = probability.argmax(axis=1)
    precision, recall, _, support = precision_recall_fscore_support(encoded, predicted, labels=range(len(CLASSES)), zero_division=0)
    return {
        "sample_count": int(len(frame)),
        "class_counts": {label: int((frame.target == label).sum()) for label in CLASSES},
        "log_loss": float(log_loss(encoded, probability, labels=range(len(CLASSES)))),
        "brier": multiclass_brier(frame.target.to_numpy(), probability),
        "calibration_ece": ece(frame.target.to_numpy(), probability),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])} for i, label in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(encoded, predicted, labels=range(len(CLASSES))).tolist(),
    }


def segment_evaluations(frame, probability):
    out = {}
    for field, values in (("weight_class", frame.weight_class), ("scheduled_rounds", pd.Series("3", index=frame.index)), ("limited_history", frame.abs_experience_total < np.log1p(3))):
        groups = pd.Series(values, index=frame.index).fillna("Unknown")
        out[field] = {}
        for value, indexes in groups.groupby(groups).groups.items():
            if len(indexes) >= 20:
                out[field][str(value)] = evaluate(frame.loc[indexes], probability[frame.index.get_indexer(indexes)])
    return out


def train(data_dir=DATA_DIR, model_path=MODEL_PATH, report_path=REPORT_PATH):
    frame, states, profiles, division_states = build_method_history(data_dir)
    development, calibration, test = temporal_partitions(frame)
    # Select the estimator on the last chronological quarter of development.
    dates = np.array(sorted(development.event_date.unique()))
    boundary = dates[int(len(dates) * .75)]
    select_train, select_validation = development[development.event_date < boundary], development[development.event_date >= boundary]
    candidates = {"multinomial_logistic": make_logistic(), "gradient_boosting": make_gradient()}
    leaderboard = {}
    for name, candidate in candidates.items():
        fit_candidate(name, candidate, select_train)
        leaderboard[name] = evaluate(select_validation, aligned_probabilities(candidate, select_validation))["log_loss"]
    selected_name = min(leaderboard, key=leaderboard.get)
    model = fit_candidate(selected_name, candidates[selected_name], development)

    # The calibration period is reserved for a conservative blend with the
    # smoothed class-frequency baseline. This avoids overfitting rare NC cases.
    raw_calibration = aligned_probabilities(model, calibration)
    baseline_calibration = frequency_probability(development, len(calibration))
    blend_scores = {}
    for blend in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        blend_scores[str(blend)] = evaluate(calibration, (1 - blend) * raw_calibration + blend * baseline_calibration)["log_loss"]
    blend = float(min(blend_scores, key=blend_scores.get))
    test_probability = (1 - blend) * aligned_probabilities(model, test) + blend * frequency_probability(development, len(test))
    baseline_test = frequency_probability(development, len(test))
    report = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(frame.event_date.max()),
        "target_definition": {"ko_tko": "Methods beginning KO/TKO", "submission": "Methods beginning SUB", "decision": "Methods containing DEC, including draw decisions", "nc": "Official result NC"},
        "excluded_methods": "DQ, Overturned, Other, and blank methods are excluded from targets but update historical state.",
        "scheduled_rounds": "Source data has no reliable scheduled-round field; all training rows use the standard 3-round context. Predictions for title/main-event 5-round fights should be treated as limited-data estimates.",
        "feature_schema": METHOD_FEATURES,
        "split_rows": {"development": len(development), "calibration": len(calibration), "test": len(test)},
        "model_selection": {"selected": selected_name, "validation_log_loss": leaderboard},
        "baseline_test": evaluate(test, baseline_test),
        "test": evaluate(test, test_probability),
        "segment_test": segment_evaluations(test, test_probability),
        "calibration_blend": {"selected_baseline_weight": blend, "validation_log_loss": blend_scores},
        "dependencies": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "platform": platform.platform()},
        "limitations": "NC is extremely rare. Its probability is smoothed toward its historical rate and should not be read as a reliable event-level NC forecast.",
    }
    artifact = {"version": MODEL_VERSION, "classes": CLASSES, "model": model, "baseline_weight": blend, "baseline_counts": development.target.value_counts().to_dict(), "feature_schema": METHOD_FEATURES, "metadata": report}
    joblib.dump(artifact, model_path)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return artifact, report, states, profiles, division_states


def predict_method(artifact, pair_frame):
    features = method_features(pair_frame)
    frame = pd.DataFrame([features])[artifact["feature_schema"]]
    raw = aligned_probabilities(artifact["model"], frame)
    counts = artifact["baseline_counts"]
    base = np.array([counts.get(label, 0) + 2.0 for label in CLASSES], dtype=float)
    probability = (1 - artifact["baseline_weight"]) * raw[0] + artifact["baseline_weight"] * base / base.sum()
    probability = np.maximum(probability, 0)
    probability /= probability.sum()
    return {label: float(value) for label, value in zip(CLASSES, probability)}


def main():
    _, report, _, _, _ = train()
    print(f"Trained fight-method model v{MODEL_VERSION}; test log loss {report['test']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
