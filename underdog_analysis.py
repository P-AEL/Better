"""Chronological historical paper analysis for market underdogs.

This is deliberately a fixed-stake accounting rule, not a recommendation engine.
It uses archived opening prices, chooses a price cutoff on an earlier calibration
period, then reports the untouched shared 100-card test period.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation_split import load_or_build_manifest
from model_pipeline import american_to_decimal, normalize_event, normalize_text


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
CUTOFFS = (100, 125, 150, 175, 200, 250, 300, 400, 500)
CALIBRATION_FRACTION = 0.15
MIN_CALIBRATION_BETS = 40


def _pair_key(fighter_a, fighter_b):
    return tuple(sorted((normalize_text(fighter_a), normalize_text(fighter_b))))


def _event_key(event_name, event_date):
    return normalize_event(event_name), str(event_date)


def _summarize(rows):
    sample_count = len(rows)
    if not sample_count:
        return {
            "sample_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "net_units": 0.0,
            "roi": None,
        }
    profit = float(rows["profit_units"].sum())
    wins = int(rows["won"].sum())
    return {
        "sample_count": sample_count,
        "wins": wins,
        "losses": sample_count - wins,
        "win_rate": wins / sample_count,
        "net_units": profit,
        "roi": profit / sample_count,
    }


def _cutoff_summary(rows, cutoff):
    return {"cutoff": cutoff, **_summarize(rows[rows["underdog_american"] >= cutoff])}


def _choose_cutoff(calibration):
    candidates = [_cutoff_summary(calibration, cutoff) for cutoff in CUTOFFS]
    eligible = [item for item in candidates if item["sample_count"] >= MIN_CALIBRATION_BETS]
    if not eligible:
        eligible = [item for item in candidates if item["sample_count"]]
    if not eligible:
        raise ValueError("No priced underdog bets are available for cutoff selection")
    # The choice sees calibration results only. More observations break identical ROI ties.
    selected = max(eligible, key=lambda item: (item["roi"], item["sample_count"], -item["cutoff"]))
    return selected["cutoff"], candidates


def build_bets(data_dir=DATA_DIR):
    """Return settled opening-price underdog rows in chronological order."""
    data_dir = Path(data_dir)
    fights = pd.read_csv(data_dir / "ufc_fights.csv")
    events = pd.read_csv(data_dir / "ufc_events.csv")
    odds = pd.read_csv(data_dir / "ufc_fight_odds.csv")

    event_dates = {
        normalize_event(row.event_name): str(pd.to_datetime(row.event_date).date())
        for row in events.itertuples(index=False)
        if pd.notna(row.event_date)
    }
    fights = fights[fights["result"].eq("win")].copy()
    fights["event_date"] = fights["event_name"].map(lambda value: event_dates.get(normalize_event(value)))
    fights = fights[fights["event_date"].notna()].copy()

    odds_index = {}
    for row in odds.itertuples(index=False):
        date_value = str(pd.to_datetime(row.event_date).date()) if pd.notna(row.event_date) else None
        key = (normalize_event(row.event_name), date_value, _pair_key(row.fighter_red, row.fighter_blue))
        # The source should have a single current record. Preserve the last one if it does not.
        odds_index[key] = row

    rows = []
    for fight in fights.itertuples(index=False):
        key = (normalize_event(fight.event_name), str(fight.event_date), _pair_key(fight.fighter_red, fight.fighter_blue))
        odds_row = odds_index.get(key)
        if odds_row is None:
            continue
        red_open = getattr(odds_row, "red_open", None)
        blue_open = getattr(odds_row, "blue_open", None)
        if pd.isna(red_open) or pd.isna(blue_open):
            continue
        if normalize_text(odds_row.fighter_red) == normalize_text(fight.fighter_red):
            red_price, blue_price = float(red_open), float(blue_open)
        elif normalize_text(odds_row.fighter_red) == normalize_text(fight.fighter_blue):
            red_price, blue_price = float(blue_open), float(red_open)
        else:
            continue
        red_decimal = american_to_decimal(red_price)
        blue_decimal = american_to_decimal(blue_price)
        if not red_decimal or not blue_decimal or math.isclose(red_decimal, blue_decimal):
            continue
        if red_decimal > blue_decimal:
            fighter, price, decimal = fight.fighter_red, red_price, red_decimal
        else:
            fighter, price, decimal = fight.fighter_blue, blue_price, blue_decimal
        # A negative price is a marginal market underdog but not a conventional plus-money dog.
        if price <= 0:
            continue
        won = normalize_text(fight.winner) == normalize_text(fighter)
        rows.append({
            "event_name": fight.event_name,
            "event_date": str(fight.event_date),
            "fighter": fighter,
            "opponent": fight.fighter_blue if fighter == fight.fighter_red else fight.fighter_red,
            "underdog_american": int(price),
            "underdog_decimal": round(decimal, 4),
            "won": bool(won),
            "profit_units": round(decimal - 1 if won else -1.0, 4),
        })
    return pd.DataFrame(rows).sort_values(["event_date", "event_name", "fighter"]).reset_index(drop=True)


def _partition_bets(bets, data_dir):
    manifest = load_or_build_manifest(data_dir)
    test_keys = {_event_key(item["event_name"], item["event_date"]) for item in manifest["events"]}
    keys = bets.apply(lambda row: _event_key(row.event_name, row.event_date), axis=1)
    test = bets.loc[keys.isin(test_keys)].copy()
    earlier = bets.loc[~keys.isin(test_keys)].copy()
    event_order = (
        earlier[["event_name", "event_date"]]
        .drop_duplicates()
        .sort_values(["event_date", "event_name"])
        .reset_index(drop=True)
    )
    start = max(1, int(len(event_order) * (1 - CALIBRATION_FRACTION)))
    calibration_keys = {
        _event_key(row.event_name, row.event_date)
        for row in event_order.iloc[start:].itertuples(index=False)
    }
    calibration = earlier.loc[keys.loc[earlier.index].isin(calibration_keys)].copy()
    development = earlier.loc[~keys.loc[earlier.index].isin(calibration_keys)].copy()
    return development, calibration, test, manifest


def build_analysis(data_dir=DATA_DIR):
    bets = build_bets(data_dir)
    if bets.empty:
        raise ValueError("No matched historical opening prices are available")
    development, calibration, test, manifest = _partition_bets(bets, data_dir)
    selected_cutoff, calibration_candidates = _choose_cutoff(calibration)
    comparisons = []
    for calibration_item in calibration_candidates:
        cutoff = calibration_item["cutoff"]
        comparisons.append({
            "cutoff": cutoff,
            "calibration": calibration_item,
            "test": _cutoff_summary(test, cutoff),
        })
    selected_test = _cutoff_summary(test, selected_cutoff)
    selected_bets = test[test["underdog_american"] >= selected_cutoff].copy()
    selected_bets = selected_bets.sort_values(["event_date", "event_name", "fighter"], ascending=False)
    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "Underdog price-cutoff paper analysis",
        "definition": "One flat unit on the market underdog when its opening American price meets the cutoff. A win returns decimal odds minus one unit; a loss costs one unit.",
        "limitations": (
            "Historical paper analysis only. Opening prices are archived market records and their exact availability "
            "at a specific bookmaker or stake size cannot be verified from this dataset. A positive past ROI is not evidence of a future edge."
        ),
        "price_source": "BestFightOdds archived opening prices",
        "test_event_count": manifest["test_event_count"],
        "coverage": {
            "all_priced_underdogs": len(bets),
            "development": len(development),
            "calibration": len(calibration),
            "test": len(test),
        },
        "selection": {
            "cutoff": selected_cutoff,
            "minimum_calibration_bets": MIN_CALIBRATION_BETS,
            "calibration": _cutoff_summary(calibration, selected_cutoff),
            "test": selected_test,
            "selection_note": "The cutoff was selected using calibration cards before the fixed 100-card test period; the test results were not used to choose it.",
        },
        "cutoff_comparison": comparisons,
        "test_bets": selected_bets.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(json.dumps(build_analysis(), ensure_ascii=False, indent=2))
