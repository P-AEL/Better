import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scraped_data"
SITE_DATA_PATH = ROOT / "site" / "data" / "site-data.json"
JOURNAL_PATH = ROOT / "data" / "forward_predictions.csv"
RESULTS_PATH = ROOT / "site" / "data" / "forward-results.json"
JOURNAL_COLUMNS = [
    "recorded_at",
    "snapshot_id",
    "model_version",
    "event_name",
    "event_date",
    "fighter_red",
    "fighter_blue",
    "independent_red_probability",
    "residual_red_probability",
    "market_red_probability",
    "red_decimal",
    "blue_decimal",
    "recommendation",
    "recommended_decimal",
    "paper_candidate",
    "outcome_red",
    "settled_at",
]


def binary_log_loss(target, probability):
    probability = min(1 - 1e-6, max(1e-6, float(probability)))
    return -(target * math.log(probability) + (1 - target) * math.log(1 - probability))


def load_journal(path=JOURNAL_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    frame = pd.read_csv(path)
    for column in JOURNAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[JOURNAL_COLUMNS]


def record_current_predictions(journal, site_data):
    rows = []
    event = site_data.get("next_event")
    if not event:
        return journal
    for fight in site_data.get("predictions", []):
        market = fight.get("market")
        if not market:
            continue
        snapshot_id = market.get("observed_at") or (
            f"legacy:{event['name']}:"
            f"{market.get('red_american', market['red_decimal'])}:"
            f"{market.get('blue_american', market['blue_decimal'])}"
        )
        recommendation = fight.get("recommendation")
        if recommendation == fight["fighter_red"]:
            recommended_decimal = market["red_decimal"]
        elif recommendation == fight["fighter_blue"]:
            recommended_decimal = market["blue_decimal"]
        else:
            recommended_decimal = None
        rows.append({
            "recorded_at": site_data["generated_at"],
            "snapshot_id": snapshot_id,
            "model_version": site_data["model"]["version"],
            "event_name": event["name"],
            "event_date": event["date"],
            "fighter_red": fight["fighter_red"],
            "fighter_blue": fight["fighter_blue"],
            "independent_red_probability": fight["independent_red_probability"],
            "residual_red_probability": fight["red_probability"],
            "market_red_probability": market["red_fair_probability"],
            "red_decimal": market["red_decimal"],
            "blue_decimal": market["blue_decimal"],
            "recommendation": recommendation,
            "recommended_decimal": recommended_decimal,
            "paper_candidate": bool(fight.get("paper_candidate", False)),
            "outcome_red": np.nan,
            "settled_at": np.nan,
        })
    if not rows:
        return journal
    incoming = pd.DataFrame(rows, columns=JOURNAL_COLUMNS)
    combined = pd.concat([journal, incoming], ignore_index=True)
    unique = [
        "snapshot_id",
        "model_version",
        "event_name",
        "fighter_red",
        "fighter_blue",
    ]
    return combined.drop_duplicates(unique, keep="first")


def settle_predictions(journal, fights):
    completed = fights[fights["result"].eq("win")]
    winners = {
        (row.event_name, frozenset((row.fighter_red, row.fighter_blue))): row.winner
        for row in completed.itertuples(index=False)
    }
    settled_at = datetime.now(timezone.utc).isoformat()
    pending = journal["outcome_red"].isna()
    for index, row in journal.loc[pending].iterrows():
        winner = winners.get((row["event_name"], frozenset((row["fighter_red"], row["fighter_blue"]))))
        if winner:
            journal.at[index, "outcome_red"] = float(winner == row["fighter_red"])
            journal.at[index, "settled_at"] = settled_at
    return journal


def summarize(journal):
    settled = journal[journal["outcome_red"].notna()].copy()
    priced = settled[
        settled[[
            "independent_red_probability",
            "residual_red_probability",
            "market_red_probability",
        ]].notna().all(axis=1)
    ]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recorded_predictions": int(len(journal)),
        "settled_predictions": int(len(settled)),
        "priced_settled_predictions": int(len(priced)),
        "minimum_required": 100,
        "forward_gate_passed": False,
        "metrics": None,
        "paper": {"settled": 0, "net_units": 0.0, "roi": None},
    }
    if not priced.empty:
        target = priced["outcome_red"].astype(float).to_numpy()
        model = priced["independent_red_probability"].astype(float).to_numpy()
        residual = priced["residual_red_probability"].astype(float).to_numpy()
        market = priced["market_red_probability"].astype(float).to_numpy()
        summary["metrics"] = {
            "independent_log_loss": float(np.mean([
                binary_log_loss(y, p) for y, p in zip(target, model)
            ])),
            "residual_log_loss": float(np.mean([
                binary_log_loss(y, p) for y, p in zip(target, residual)
            ])),
            "market_log_loss": float(np.mean([
                binary_log_loss(y, p) for y, p in zip(target, market)
            ])),
            "independent_brier": float(np.mean((model - target) ** 2)),
            "residual_brier": float(np.mean((residual - target) ** 2)),
            "market_brier": float(np.mean((market - target) ** 2)),
        }
        summary["forward_gate_passed"] = bool(
            len(priced) >= summary["minimum_required"]
            and summary["metrics"]["residual_log_loss"] + 0.005
            < summary["metrics"]["market_log_loss"]
            and summary["metrics"]["residual_brier"] < summary["metrics"]["market_brier"]
        )

    paper = settled[settled["paper_candidate"].astype(str).str.casefold().eq("true")]
    pnl = []
    for row in paper.itertuples(index=False):
        recommended_red = row.recommendation == row.fighter_red
        won = bool(row.outcome_red) == recommended_red
        pnl.append(float(row.recommended_decimal) - 1.0 if won else -1.0)
    if pnl:
        summary["paper"] = {
            "settled": len(pnl),
            "net_units": round(sum(pnl), 4),
            "roi": round(sum(pnl) / len(pnl), 4),
        }
    return summary


def run(
    data_dir=DATA_DIR,
    site_data_path=SITE_DATA_PATH,
    journal_path=JOURNAL_PATH,
    results_path=RESULTS_PATH,
    settle_only=False,
):
    journal = load_journal(journal_path)
    fights = pd.read_csv(Path(data_dir) / "ufc_fights.csv")
    journal = settle_predictions(journal, fights)
    if not settle_only and Path(site_data_path).exists():
        site_data = json.loads(Path(site_data_path).read_text(encoding="utf-8"))
        journal = record_current_predictions(journal, site_data)
    summary = summarize(journal)
    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
    journal.to_csv(journal_path, index=False)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Maintain the append-only forward prediction journal.")
    parser.add_argument("--settle-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(settle_only=args.settle_only), indent=2))


if __name__ == "__main__":
    main()
