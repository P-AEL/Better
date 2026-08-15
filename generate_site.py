import json
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from model_pipeline import (
    build_history,
    current_features,
    residual_probability,
    symmetric_component_probabilities,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
SITE_DATA_PATH = ROOT / "site" / "data" / "site-data.json"
MODEL_PATH = ROOT / "betting_model.joblib"
FORWARD_RESULTS_PATH = ROOT / "site" / "data" / "forward-results.json"
BASE_ELO = 1500.0
K_FACTOR = 32.0


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_event(value):
    return re.sub(r"\bvs\.?\b", "vs", normalize_text(value))


def elo_probability(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def american_to_decimal(value):
    if pd.isna(value):
        return None
    odds = float(value)
    if odds == 0:
        return None
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def american_to_implied(value):
    decimal = american_to_decimal(value)
    return 1.0 / decimal if decimal else None


def display_american(value):
    if value is None or pd.isna(value):
        return None
    number = int(float(value))
    return f"+{number}" if number > 0 else str(number)


def new_fighter_state():
    return {
        "elo": BASE_ELO,
        "peak_elo": BASE_ELO,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "no_contests": 0,
        "fights": 0,
        "division": "Unknown",
        "last_fight": None,
        "form": [],
        "last_change": 0.0,
    }


def prepare_fights(data_dir=DATA_DIR):
    fights = pd.read_csv(Path(data_dir) / "ufc_fights.csv")
    events = pd.read_csv(Path(data_dir) / "ufc_events.csv")
    events["event_date_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    fights["source_order"] = range(len(fights))
    fights = fights.merge(
        events[["event_name", "event_date", "event_date_dt", "location"]],
        on="event_name",
        how="left",
    )
    return fights, events


def compute_elo_states(fights, cutoff=None):
    states = defaultdict(new_fighter_state)
    ordered = fights.sort_values(
        ["event_date_dt", "source_order"], ascending=[True, False], na_position="last"
    )

    for row in ordered.itertuples(index=False):
        if row.result == "scheduled" or pd.isna(row.event_date_dt):
            continue
        fight_date = row.event_date_dt.date()
        if cutoff and fight_date > cutoff:
            continue

        red = states[row.fighter_red]
        blue = states[row.fighter_blue]
        red["division"] = row.weight_class
        blue["division"] = row.weight_class
        red["last_fight"] = fight_date.isoformat()
        blue["last_fight"] = fight_date.isoformat()
        red["fights"] += 1
        blue["fights"] += 1

        if row.result == "nc":
            red["no_contests"] += 1
            blue["no_contests"] += 1
            red["form"].append("NC")
            blue["form"].append("NC")
            continue

        expected_red = elo_probability(red["elo"], blue["elo"])
        if row.result == "draw":
            score_red = 0.5
            red["draws"] += 1
            blue["draws"] += 1
            red["form"].append("D")
            blue["form"].append("D")
        else:
            score_red = 1.0 if row.winner == row.fighter_red else 0.0
            winner, loser = (red, blue) if score_red == 1.0 else (blue, red)
            winner["wins"] += 1
            loser["losses"] += 1
            red["form"].append("W" if score_red == 1.0 else "L")
            blue["form"].append("L" if score_red == 1.0 else "W")

        change = K_FACTOR * (score_red - expected_red)
        red["elo"] += change
        blue["elo"] -= change
        red["last_change"] = change
        blue["last_change"] = -change
        red["peak_elo"] = max(red["peak_elo"], red["elo"])
        blue["peak_elo"] = max(blue["peak_elo"], blue["elo"])

    return states


def choose_next_event(fights):
    scheduled = fights[fights["result"].eq("scheduled")].copy()
    if scheduled.empty:
        return None, scheduled

    today = date.today()
    dated = scheduled[scheduled["event_date_dt"].notna()].copy()
    upcoming = dated[dated["event_date_dt"].dt.date >= today]
    candidates = upcoming if not upcoming.empty else dated
    if candidates.empty:
        event_name = scheduled.iloc[0]["event_name"]
    else:
        event_name = candidates.sort_values("event_date_dt").iloc[0]["event_name"]
    return event_name, scheduled[scheduled["event_name"].eq(event_name)].copy()


def build_odds_lookup(data_dir=DATA_DIR):
    odds_path = Path(data_dir) / "ufc_fight_odds.csv"
    if not odds_path.exists():
        return {}

    odds = pd.read_csv(odds_path)
    lookup = {}
    status_rank = {"matched": 2, "partial": 1, "no_match": 0}
    for row in odds.itertuples(index=False):
        key = (
            normalize_event(row.event_name),
            frozenset((normalize_text(row.fighter_red), normalize_text(row.fighter_blue))),
        )
        current = lookup.get(key)
        if current is None or status_rank.get(row.match_status, -1) > status_rank.get(
            current.match_status, -1
        ):
            lookup[key] = row
    return lookup


def market_for_fight(event_name, red_name, blue_name, odds_lookup):
    key = (
        normalize_event(event_name),
        frozenset((normalize_text(red_name), normalize_text(blue_name))),
    )
    row = odds_lookup.get(key)
    if row is None or row.match_status != "matched":
        return None

    same_orientation = normalize_text(row.fighter_red) == normalize_text(red_name)
    red_open = row.red_open if same_orientation else row.blue_open
    blue_open = row.blue_open if same_orientation else row.red_open
    red_current = row.red_close_high if same_orientation else row.blue_close_high
    blue_current = row.blue_close_high if same_orientation else row.red_close_high
    red_price = red_current if pd.notna(red_current) else red_open
    blue_price = blue_current if pd.notna(blue_current) else blue_open
    if pd.isna(red_price) or pd.isna(blue_price):
        return None

    red_implied = american_to_implied(red_price)
    blue_implied = american_to_implied(blue_price)
    total = red_implied + blue_implied
    source = getattr(row, "odds_source", None)
    observed_at = getattr(row, "scraped_at_utc", None)
    return {
        "red_american": display_american(red_price),
        "blue_american": display_american(blue_price),
        "red_decimal": american_to_decimal(red_price),
        "blue_decimal": american_to_decimal(blue_price),
        "red_fair_probability": red_implied / total,
        "blue_fair_probability": blue_implied / total,
        "source": source if pd.notna(source) else "BestFightOdds",
        "observed_at": observed_at if pd.notna(observed_at) else None,
    }


def load_model_context(data_dir):
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    _, feature_states, profiles, division_states = build_history(data_dir)
    return artifact, feature_states, profiles, division_states


def load_forward_results():
    if not FORWARD_RESULTS_PATH.exists():
        return {"forward_gate_passed": False, "priced_settled_predictions": 0}
    return json.loads(FORWARD_RESULTS_PATH.read_text(encoding="utf-8"))


def predict_event(
    event_name,
    event_fights,
    states,
    odds_lookup,
    event_date=None,
    model_context=None,
    forward_results=None,
):
    predictions = []
    for row in event_fights.itertuples(index=False):
        red_state = states[row.fighter_red]
        blue_state = states[row.fighter_blue]
        raw_probability = elo_probability(red_state["elo"], blue_state["elo"])
        model_metadata = None
        artifact = None
        features = None
        model_probabilities = None
        if model_context and event_date:
            artifact, feature_states, profiles, division_states = model_context
            features = current_features(
                row.fighter_red,
                row.fighter_blue,
                row.weight_class,
                event_date,
                feature_states,
                profiles,
                division_states,
            )
            model_probabilities = symmetric_component_probabilities(artifact, features)
            raw_probability = model_probabilities["ensemble"]
            model_metadata = artifact["metadata"]

        red_probability = raw_probability
        blue_probability = 1.0 - red_probability
        market = market_for_fight(
            event_name, row.fighter_red, row.fighter_blue, odds_lookup
        )

        recommendation = None
        best_ev = None
        edge = None
        conservative_ev = None
        model_market_gap = None
        paper_candidate = False
        call = "Unpriced"
        if market:
            model_market_gap = (
                raw_probability - market["red_fair_probability"]
                if raw_probability >= 0.5
                else (1.0 - raw_probability) - market["blue_fair_probability"]
            )
            if model_metadata and artifact is not None:
                red_probability = residual_probability(
                    artifact,
                    raw_probability,
                    market["red_fair_probability"],
                    features,
                )
                blue_probability = 1.0 - red_probability
            red_ev = red_probability * market["red_decimal"] - 1.0
            blue_ev = blue_probability * market["blue_decimal"] - 1.0
            if red_ev >= blue_ev:
                recommendation = row.fighter_red
                best_ev = red_ev
                edge = red_probability - market["red_fair_probability"]
                recommended_odds = market["red_american"]
                recommended_probability = red_probability
                recommended_decimal = market["red_decimal"]
            else:
                recommendation = row.fighter_blue
                best_ev = blue_ev
                edge = blue_probability - market["blue_fair_probability"]
                recommended_odds = market["blue_american"]
                recommended_probability = blue_probability
                recommended_decimal = market["blue_decimal"]

            uncertainty = model_metadata["uncertainty_margin"] if model_metadata else 0.15
            conservative_probability = max(0.0, recommended_probability - uncertainty)
            conservative_ev = conservative_probability * recommended_decimal - 1.0
            backtest_validated = bool(
                model_metadata and model_metadata.get("backtest_gate_passed", False)
            )
            forward_validated = bool(
                forward_results and forward_results.get("forward_gate_passed", False)
            )
            validated = backtest_validated and forward_validated
            qualifies = edge >= 0.03 and best_ev >= 0.05 and conservative_ev > 0
            paper_candidate = bool(backtest_validated and qualifies)
            call = (
                "Signal" if validated and qualifies
                else "Paper" if paper_candidate
                else "Experimental" if best_ev > 0
                else "Pass"
            )
        else:
            recommended_odds = None

        predicted_winner = row.fighter_red if raw_probability >= 0.5 else row.fighter_blue
        predictions.append(
            {
                "fighter_red": row.fighter_red,
                "fighter_blue": row.fighter_blue,
                "weight_class": row.weight_class,
                "red_elo": round(red_state["elo"]),
                "blue_elo": round(blue_state["elo"]),
                "red_probability": round(red_probability, 4),
                "blue_probability": round(blue_probability, 4),
                "independent_red_probability": round(raw_probability, 4),
                "model_probabilities": {
                    name: round(probability, 4)
                    for name, probability in (model_probabilities or {"ensemble": raw_probability}).items()
                },
                "predicted_winner": predicted_winner,
                "confidence": round(max(raw_probability, 1.0 - raw_probability), 4),
                "market": market,
                "recommendation": recommendation,
                "recommended_odds": recommended_odds,
                "expected_value": round(best_ev, 4) if best_ev is not None else None,
                "conservative_expected_value": (
                    round(conservative_ev, 4) if conservative_ev is not None else None
                ),
                "edge": round(edge, 4) if edge is not None else None,
                "model_market_gap": (
                    round(model_market_gap, 4) if model_market_gap is not None else None
                ),
                "paper_candidate": paper_candidate,
                "call": call,
            }
        )

    predictions.sort(
        key=lambda item: (
            item["expected_value"] is not None,
            item["expected_value"] if item["expected_value"] is not None else -1,
            item["confidence"],
        ),
        reverse=True,
    )
    return predictions


def build_rankings(states):
    ranked = sorted(
        ((name, state) for name, state in states.items() if state["fights"]),
        key=lambda item: item[1]["elo"],
        reverse=True,
    )
    return [
        {
            "rank": index,
            "fighter": name,
            "elo": round(state["elo"]),
            "peak_elo": round(state["peak_elo"]),
            "change": round(state["last_change"]),
            "division": state["division"],
            "wins": state["wins"],
            "losses": state["losses"],
            "draws": state["draws"],
            "no_contests": state["no_contests"],
            "fights": state["fights"],
            "last_fight": state["last_fight"],
            "form": state["form"][-5:],
        }
        for index, (name, state) in enumerate(ranked, start=1)
    ]


def build_site_data(data_dir=DATA_DIR):
    fights, events = prepare_fights(data_dir)
    event_name, next_fights = choose_next_event(fights)
    all_states = compute_elo_states(fights)
    rankings = build_rankings(all_states)
    predictions = []
    event_payload = None
    model_context = load_model_context(data_dir)
    forward_results = load_forward_results()

    if event_name:
        event_row = events[events["event_name"].eq(event_name)].iloc[0]
        event_date = event_row["event_date_dt"].date()
        prediction_states = compute_elo_states(fights, cutoff=event_date)
        predictions = predict_event(
            event_name,
            next_fights,
            prediction_states,
            build_odds_lookup(data_dir),
            event_date=event_date,
            model_context=model_context,
            forward_results=forward_results,
        )
        event_payload = {
            "name": event_name,
            "date": event_date.isoformat(),
            "date_display": event_row["event_date"],
            "location": event_row["location"],
            "fight_count": len(predictions),
        }

    priced = [item for item in predictions if item["market"]]
    signals = [item for item in priced if item["call"] == "Signal"]
    paper_candidates = [item for item in priced if item["paper_candidate"]]
    watches = [
        item for item in priced
        if item["expected_value"] is not None and item["expected_value"] > 0
    ]
    best_bet = max(signals, key=lambda item: item["conservative_expected_value"]) if signals else None

    if model_context:
        metadata = model_context[0]["metadata"]
        backtest_gate = bool(metadata.get("backtest_gate_passed", False))
        forward_gate = bool(forward_results.get("forward_gate_passed", False))
        betting_enabled = backtest_gate and forward_gate
        model_payload = {
            "name": "Calibrated temporal ensemble",
            "version": model_context[0]["version"],
            "selected_model": metadata["selected_model"],
            "feature_set": metadata.get("feature_set"),
            "calibration_error": metadata["calibration_error"],
            "uncertainty_margin": metadata["uncertainty_margin"],
            "backtest_gate_passed": backtest_gate,
            "forward_gate_passed": forward_gate,
            "betting_enabled": betting_enabled,
            "gate_reason": (
                "Backtest and forward gates passed."
                if betting_enabled
                else metadata["gate_reason"]
                if not backtest_gate
                else "Backtest passed; collecting at least 100 forward-priced outcomes."
            ),
            "test_metrics": metadata["calibrated_test"],
            "market_metrics": metadata["market_test"],
            "residual_metrics": metadata.get("market_residual_test"),
            "prediction_models": [
                {"key": "ensemble", "label": "Calibrated ensemble"},
                {"key": "spline_logistic", "label": "Spline logistic"},
                {"key": "gradient", "label": "Gradient boosting"},
                {"key": "catboost", "label": "CatBoost"},
                {"key": "dynamic_glicko", "label": "Dynamic Glicko"},
            ],
            "performance_models": [
                {"key": "ensemble", "label": "Calibrated ensemble"},
                {"key": "spline_logistic", "label": "Spline logistic"},
                {"key": "gradient", "label": "Gradient boosting"},
                {"key": "catboost", "label": "CatBoost"},
                {"key": "dynamic_glicko", "label": "Dynamic Glicko"},
            ],
            "historical_performance": metadata.get("historical_performance", []),
            "forward_results": forward_results,
        }
    else:
        model_payload = {
            "name": "Elo 32 fallback",
            "version": "fallback",
            "betting_enabled": False,
            "gate_reason": "No trained artifact is available; betting signals are disabled.",
            "base_rating": BASE_ELO,
            "k_factor": K_FACTOR,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            **model_payload,
            "decisive_fights": int(fights["result"].eq("win").sum()),
            "draws": int(fights["result"].eq("draw").sum()),
        },
        "next_event": event_payload,
        "predictions": predictions,
        "best_bet": best_bet,
        "rankings": rankings,
        "summary": {
            "ranked_fighters": len(rankings),
            "priced_fights": len(priced),
            "experimental_edges": len(watches),
            "paper_candidates": len(paper_candidates),
            "validated_signals": len(signals),
        },
    }


def main():
    payload = build_site_data()
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Generated {SITE_DATA_PATH}: "
        f"{len(payload['predictions'])} predictions, {len(payload['rankings'])} rankings"
    )


if __name__ == "__main__":
    main()
