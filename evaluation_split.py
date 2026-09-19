"""Shared, chronological fixed evaluation split for all prediction models."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = ROOT / "data" / "evaluation_test_manifest.json"
TEST_EVENT_COUNT = 100


def completed_events(data_dir):
    data_dir = Path(data_dir)
    fights = pd.read_csv(data_dir / "ufc_fights.csv")
    events = pd.read_csv(data_dir / "ufc_events.csv")
    events["event_date_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    names = set(fights.loc[fights["result"].isin(["win", "draw", "nc"]), "event_name"])
    output = events[events["event_name"].isin(names) & events["event_date_dt"].notna()].copy()
    return output.sort_values(["event_date_dt", "event_name"])


def build_manifest(data_dir, manifest_path=DEFAULT_MANIFEST_PATH):
    events = completed_events(data_dir)
    if len(events) < TEST_EVENT_COUNT:
        raise ValueError(f"Need {TEST_EVENT_COUNT} completed events; found {len(events)}")
    test = events.tail(TEST_EVENT_COUNT)
    payload = {
        "purpose": "Fixed historical test set shared by winner and decision models.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_event_count": TEST_EVENT_COUNT,
        "events": [
            {"event_name": row.event_name, "event_date": row.event_date_dt.date().isoformat()}
            for row in test.itertuples(index=False)
        ],
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_or_build_manifest(data_dir, manifest_path=DEFAULT_MANIFEST_PATH):
    return build_manifest(data_dir, manifest_path)


def partition_frame(frame, data_dir, manifest_path=DEFAULT_MANIFEST_PATH, calibration_fraction=0.15):
    manifest = load_or_build_manifest(data_dir, manifest_path)
    test_keys = {(item["event_name"], item["event_date"]) for item in manifest["events"]}
    test_mask = pd.Series(
        [(name, str(event_date)) in test_keys for name, event_date in zip(frame["event_name"], frame["event_date"])],
        index=frame.index,
    )
    test = frame.loc[test_mask].copy()
    earlier = frame.loc[~test_mask].copy()
    if test["event_name"].nunique() != TEST_EVENT_COUNT:
        raise ValueError("Test manifest does not map to exactly 100 events in model frame")
    dates = sorted(earlier["event_date"].unique())
    calibration_start = dates[max(1, int(len(dates) * (1 - calibration_fraction)))]
    development = earlier[earlier["event_date"] < calibration_start].copy()
    calibration = earlier[earlier["event_date"] >= calibration_start].copy()
    return development, calibration, test, manifest
