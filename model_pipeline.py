import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


BASE_ELO = 1500.0
ELO_K = 32.0
GLICKO_INITIAL_RD = 350.0
GLICKO_MIN_RD = 50.0
EWMA_ALPHA = 0.35
MODEL_VERSION = "3.0.0"
STAT_NAMES = ("kd", "str", "td", "sub")

CORE_DIFFERENCE_FEATURES = [
    "elo_diff",
    "glicko_logit",
    "glicko_rd_diff",
    "win_rate_diff",
    "finish_rate_diff",
    "recent3_win_rate_diff",
    "recent5_win_rate_diff",
    "age_diff",
    "age_peak_distance_diff",
    "height_diff",
    "layoff_diff",
    "opponent_quality_diff",
    "str_trend_diff",
    "td_trend_diff",
    *[f"{stat}_{kind}_division_z_diff" for stat in STAT_NAMES for kind in ("for", "against")],
]
ABLATION_ONLY_FEATURES = [
    "experience_diff",
    "reach_diff",
    "avg_duration_diff",
    *[f"{stat}_{kind}_raw_diff" for stat in STAT_NAMES for kind in ("for", "against")],
]
DIFFERENCE_FEATURES = CORE_DIFFERENCE_FEATURES + ABLATION_ONLY_FEATURES
SYMMETRIC_NUMERIC_FEATURES = [
    "experience_total",
    "age_mean",
    "glicko_uncertainty_mean",
]
CATEGORICAL_FEATURES = ["weight_class", "stance_pair"]
SYMMETRIC_FEATURES = SYMMETRIC_NUMERIC_FEATURES + CATEGORICAL_FEATURES
NUMERIC_FEATURES = CORE_DIFFERENCE_FEATURES + SYMMETRIC_NUMERIC_FEATURES
FULL_NUMERIC_FEATURES = DIFFERENCE_FEATURES + SYMMETRIC_NUMERIC_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FULL_FEATURE_COLUMNS = FULL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_event(value):
    return re.sub(r"\bvs\.?\b", "vs", normalize_text(value))


def american_to_decimal(value):
    if value is None or pd.isna(value):
        return None
    odds = float(value)
    if odds == 0:
        return None
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def no_vig_probabilities(red_odds, blue_odds):
    red_decimal = american_to_decimal(red_odds)
    blue_decimal = american_to_decimal(blue_odds)
    if not red_decimal or not blue_decimal:
        return None
    red_raw, blue_raw = 1.0 / red_decimal, 1.0 / blue_decimal
    total = red_raw + blue_raw
    return red_raw / total, blue_raw / total


def elo_probability(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def clipped_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -30, 30)))


def safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_height(value):
    match = re.fullmatch(r"(\d+)'\s*(\d+)\"", str(value or "").strip())
    return float(int(match.group(1)) * 12 + int(match.group(2))) if match else np.nan


def parse_reach(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\"", str(value or "").strip())
    return float(match.group(1)) if match else np.nan


def fight_minutes(round_number, clock):
    try:
        minutes, seconds = str(clock).split(":", 1)
        return max(1 / 60, (float(round_number) - 1) * 5 + int(minutes) + int(seconds) / 60)
    except (TypeError, ValueError):
        return 5.0


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value):
        if value is None or not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def z_score(self, value):
        if value is None or not np.isfinite(value) or self.count < 20:
            return np.nan
        variance = self.m2 / max(1, self.count - 1)
        return (value - self.mean) / max(math.sqrt(variance), 1e-6)


@dataclass
class FighterState:
    elo: float = BASE_ELO
    glicko_rating: float = BASE_ELO
    glicko_rd: float = GLICKO_INITIAL_RD
    fights: int = 0
    wins: int = 0
    finishes: int = 0
    duration_total: float = 0.0
    opponent_rating_total: float = 0.0
    last_date: date | None = None
    ewma: dict = field(default_factory=dict)
    rate_history: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    results: deque = field(default_factory=lambda: deque(maxlen=5))

    def rate(self, key):
        return self.ewma.get(key, np.nan)

    def trend(self, key):
        values = list(self.rate_history.get(key, ()))
        if len(values) < 4:
            return np.nan
        return float(np.mean(values[-2:]) - np.mean(values[:-2]))

    def recent_win_rate(self, window):
        values = list(self.results)[-window:]
        return (sum(values) + 1.0) / (len(values) + 2.0)


def build_profiles(fighters):
    profiles = {}
    for row in fighters.itertuples(index=False):
        first = str(row.first_name).strip() if pd.notna(row.first_name) else ""
        last = str(row.last_name).strip() if pd.notna(row.last_name) else ""
        name = f"{first} {last}".strip()
        if not name:
            continue
        profiles[normalize_text(name)] = {
            "height": parse_height(row.height),
            "reach": parse_reach(row.reach),
            "stance": str(row.stance) if pd.notna(row.stance) else "Unknown",
            "dob": pd.to_datetime(row.dob, errors="coerce"),
        }
    return profiles


def profile_for(name, profiles):
    return profiles.get(
        normalize_text(name),
        {"height": np.nan, "reach": np.nan, "stance": "Unknown", "dob": pd.NaT},
    )


def age_on(profile, event_date):
    if pd.isna(profile["dob"]) or pd.isna(event_date):
        return np.nan
    return (pd.Timestamp(event_date) - profile["dob"]).days / 365.2425


def division_peak_age(weight_class):
    division = normalize_text(weight_class)
    if any(token in division for token in ("flyweight", "bantamweight", "featherweight", "women")):
        return 29.0
    if any(token in division for token in ("middleweight", "light heavyweight")):
        return 31.0
    if "heavyweight" in division:
        return 32.0
    return 30.0


def smoothed_rate(wins, fights):
    return (wins + 1.0) / (fights + 2.0)


def layoff_days(state, event_date):
    if state.last_date is None:
        return np.nan
    return min(1825.0, max(0.0, (event_date - state.last_date).days))


def effective_glicko_rd(state, event_date):
    if state.last_date is None:
        return state.glicko_rd
    years = max(0.0, (event_date - state.last_date).days / 365.2425)
    return min(GLICKO_INITIAL_RD, math.sqrt(state.glicko_rd ** 2 + years * 55.0 ** 2))


def glicko_g(rd):
    q = math.log(10) / 400.0
    return 1.0 / math.sqrt(1.0 + 3.0 * q * q * rd * rd / (math.pi ** 2))


def glicko_probability(red, blue, event_date):
    combined_rd = math.sqrt(
        effective_glicko_rd(red, event_date) ** 2 + effective_glicko_rd(blue, event_date) ** 2
    )
    scaled_difference = glicko_g(combined_rd) * (red.glicko_rating - blue.glicko_rating)
    return 1.0 / (1.0 + 10 ** (-scaled_difference / 400.0))


def stance_pair(red_profile, blue_profile):
    return " | ".join(sorted((str(red_profile["stance"]), str(blue_profile["stance"]))))


def pair_features(red_name, blue_name, weight_class, event_date, states, profiles, division_states):
    red, blue = states[red_name], states[blue_name]
    red_profile, blue_profile = profile_for(red_name, profiles), profile_for(blue_name, profiles)
    red_age, blue_age = age_on(red_profile, event_date), age_on(blue_profile, event_date)
    peak_age = division_peak_age(weight_class)
    division_state = division_states[normalize_text(weight_class)]
    red_rd, blue_rd = effective_glicko_rd(red, event_date), effective_glicko_rd(blue, event_date)
    glicko_p = glicko_probability(red, blue, event_date)

    values = {
        "elo_diff": (red.elo - blue.elo) / 400.0,
        "glicko_logit": float(clipped_logit(glicko_p)),
        "glicko_rd_diff": (red_rd - blue_rd) / GLICKO_INITIAL_RD,
        "glicko_uncertainty_mean": (red_rd + blue_rd) / (2 * GLICKO_INITIAL_RD),
        "experience_diff": math.log1p(red.fights) - math.log1p(blue.fights),
        "experience_total": math.log1p(red.fights + blue.fights),
        "win_rate_diff": smoothed_rate(red.wins, red.fights) - smoothed_rate(blue.wins, blue.fights),
        "finish_rate_diff": smoothed_rate(red.finishes, red.fights) - smoothed_rate(blue.finishes, blue.fights),
        "recent3_win_rate_diff": red.recent_win_rate(3) - blue.recent_win_rate(3),
        "recent5_win_rate_diff": red.recent_win_rate(5) - blue.recent_win_rate(5),
        "age_diff": red_age - blue_age,
        "age_mean": np.nanmean([red_age, blue_age]) if not (np.isnan(red_age) and np.isnan(blue_age)) else np.nan,
        "age_peak_distance_diff": abs(red_age - peak_age) - abs(blue_age - peak_age),
        "height_diff": red_profile["height"] - blue_profile["height"],
        "reach_diff": red_profile["reach"] - blue_profile["reach"],
        "layoff_diff": layoff_days(red, event_date) - layoff_days(blue, event_date),
        "avg_duration_diff": (
            red.duration_total / red.fights if red.fights else np.nan
        ) - (blue.duration_total / blue.fights if blue.fights else np.nan),
        "opponent_quality_diff": (
            red.opponent_rating_total / red.fights if red.fights else BASE_ELO
        ) - (blue.opponent_rating_total / blue.fights if blue.fights else BASE_ELO),
        "str_trend_diff": red.trend("str_for") - blue.trend("str_for"),
        "td_trend_diff": red.trend("td_for") - blue.trend("td_for"),
        "weight_class": str(weight_class or "Unknown"),
        "stance_pair": stance_pair(red_profile, blue_profile),
    }
    for stat in STAT_NAMES:
        for kind in ("for", "against"):
            key = f"{stat}_{kind}"
            values[f"{key}_raw_diff"] = red.rate(key) - blue.rate(key)
            values[f"{key}_division_z_diff"] = (
                division_state[key].z_score(red.rate(key))
                - division_state[key].z_score(blue.rate(key))
            )
    return values


def mirror_features(features):
    mirrored = dict(features)
    for column in DIFFERENCE_FEATURES:
        mirrored[column] = -features[column]
    return mirrored


def odds_lookup(odds):
    lookup = {}
    if odds is None:
        return lookup
    for row in odds.itertuples(index=False):
        if getattr(row, "match_status", None) != "matched":
            continue
        key = (
            normalize_event(row.event_name),
            frozenset((normalize_text(row.fighter_red), normalize_text(row.fighter_blue))),
        )
        lookup[key] = row
    return lookup


def market_for_pair(event_name, red_name, blue_name, lookup):
    key = (
        normalize_event(event_name),
        frozenset((normalize_text(red_name), normalize_text(blue_name))),
    )
    row = lookup.get(key)
    if row is None:
        return None
    same = normalize_text(row.fighter_red) == normalize_text(red_name)
    red_odds = row.red_close_high if same else row.blue_close_high
    blue_odds = row.blue_close_high if same else row.red_close_high
    if pd.isna(red_odds) or pd.isna(blue_odds):
        red_odds = row.red_open if same else row.blue_open
        blue_odds = row.blue_open if same else row.red_open
    return no_vig_probabilities(red_odds, blue_odds)


def update_ewma(state, key, value):
    previous = state.ewma.get(key)
    state.ewma[key] = value if previous is None else EWMA_ALPHA * value + (1 - EWMA_ALPHA) * previous
    state.rate_history[key].append(value)


def glicko_update(rating, rd, opponent_rating, opponent_rd, score):
    q = math.log(10) / 400.0
    g = glicko_g(opponent_rd)
    expected = 1.0 / (1.0 + 10 ** (-g * (rating - opponent_rating) / 400.0))
    variance = 1.0 / (q * q * g * g * expected * (1 - expected))
    denominator = 1.0 / (rd * rd) + 1.0 / variance
    new_rating = rating + q / denominator * g * (score - expected)
    new_rd = math.sqrt(1.0 / denominator)
    return new_rating, max(GLICKO_MIN_RD, min(GLICKO_INITIAL_RD, new_rd))


def update_states(row, states, division_states, event_date):
    red, blue = states[row.fighter_red], states[row.fighter_blue]
    if row.result == "nc":
        return

    red_elo, blue_elo = red.elo, blue.elo
    red_score = 0.5 if row.result == "draw" else 1.0
    change = ELO_K * (red_score - elo_probability(red_elo, blue_elo))
    red.elo += change
    blue.elo -= change

    red_rating, blue_rating = red.glicko_rating, blue.glicko_rating
    red_rd, blue_rd = effective_glicko_rd(red, event_date), effective_glicko_rd(blue, event_date)
    red.glicko_rating, red.glicko_rd = glicko_update(red_rating, red_rd, blue_rating, blue_rd, red_score)
    blue.glicko_rating, blue.glicko_rd = glicko_update(blue_rating, blue_rd, red_rating, red_rd, 1.0 - red_score)

    duration = fight_minutes(row.round, row.time)
    red.opponent_rating_total += blue_rating
    blue.opponent_rating_total += red_rating
    for state in (red, blue):
        state.fights += 1
        state.duration_total += duration
        state.last_date = event_date
    red.results.append(red_score)
    blue.results.append(1.0 - red_score)
    if row.result == "win":
        red.wins += 1
        if "decision" not in normalize_text(row.method):
            red.finishes += 1

    division_state = division_states[normalize_text(row.weight_class)]
    for stat in STAT_NAMES:
        red_value = safe_number(getattr(row, f"{stat}_red")) / duration
        blue_value = safe_number(getattr(row, f"{stat}_blue")) / duration
        for state, own, opponent in ((red, red_value, blue_value), (blue, blue_value, red_value)):
            update_ewma(state, f"{stat}_for", own)
            update_ewma(state, f"{stat}_against", opponent)
            division_state[f"{stat}_for"].update(own)
            division_state[f"{stat}_against"].update(opponent)


def load_source_data(data_dir):
    data_dir = Path(data_dir)
    fights = pd.read_csv(data_dir / "ufc_fights.csv")
    events = pd.read_csv(data_dir / "ufc_events.csv")
    fighters = pd.read_csv(data_dir / "ufc_fighters_basic_with_dob.csv")
    odds_path = data_dir / "ufc_fight_odds.csv"
    odds = pd.read_csv(odds_path) if odds_path.exists() else None
    events["event_date_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    fights = fights.merge(events[["event_name", "event_date_dt"]], on="event_name", how="left")
    return fights, fighters, odds


def build_history(data_dir):
    fights, fighters, odds = load_source_data(data_dir)
    profiles = build_profiles(fighters)
    lookup = odds_lookup(odds)
    states = defaultdict(FighterState)
    division_states = defaultdict(lambda: defaultdict(RunningMoments))
    records = []
    completed = fights[
        fights["result"].isin(["win", "draw", "nc"]) & fights["event_date_dt"].notna()
    ].sort_values("event_date_dt")

    for event_date, event_rows in completed.groupby("event_date_dt", sort=True):
        event_day = pd.Timestamp(event_date).date()
        for row in event_rows.itertuples(index=False):
            if row.result != "win":
                continue
            features = pair_features(
                row.fighter_red,
                row.fighter_blue,
                row.weight_class,
                event_day,
                states,
                profiles,
                division_states,
            )
            market = market_for_pair(row.event_name, row.fighter_red, row.fighter_blue, lookup)
            common = {
                "event_name": row.event_name,
                "event_date": event_day.isoformat(),
                "fight_id": row.fight_link,
                "fighter_red": row.fighter_red,
                "fighter_blue": row.fighter_blue,
                "target": 1,
                "market_probability": market[0] if market else np.nan,
            }
            records.append({**common, **features})
            records.append({
                **common,
                "fighter_red": row.fighter_blue,
                "fighter_blue": row.fighter_red,
                "target": 0,
                "market_probability": market[1] if market else np.nan,
                **mirror_features(features),
            })
        for row in event_rows.itertuples(index=False):
            update_states(row, states, division_states, event_day)

    return pd.DataFrame(records), states, profiles, division_states


def current_features(red_name, blue_name, weight_class, event_date, states, profiles, division_states):
    event_day = pd.Timestamp(event_date).date()
    return pd.DataFrame([
        pair_features(
            red_name,
            blue_name,
            weight_class,
            event_day,
            states,
            profiles,
            division_states,
        )
    ])[FULL_FEATURE_COLUMNS]


def calibrated_probabilities(artifact, frame):
    return artifact["ensemble"].predict_proba(frame)[:, 1]


def symmetric_probability(artifact, frame):
    direct = calibrated_probabilities(artifact, frame)[0]
    mirrored = frame.copy()
    for column in DIFFERENCE_FEATURES:
        mirrored[column] = -mirrored[column]
    reverse = calibrated_probabilities(artifact, mirrored)[0]
    return float(np.clip((direct + 1.0 - reverse) / 2.0, 0.01, 0.99))


def symmetric_component_probabilities(artifact, frame):
    ensemble = artifact["ensemble"]
    mirrored = frame.copy()
    for column in DIFFERENCE_FEATURES:
        mirrored[column] = -mirrored[column]

    probabilities = {"ensemble": symmetric_probability(artifact, frame)}
    for name in ensemble.model_names:
        model = ensemble.models[name]
        columns = ensemble.model_columns[name]
        direct = model.predict_proba(frame[columns])[:, 1][0]
        reverse = model.predict_proba(mirrored[columns])[:, 1][0]
        probabilities[name] = float(np.clip((direct + 1.0 - reverse) / 2.0, 0.01, 0.99))

    direct_glicko = sigmoid(frame[ensemble.glicko_column].to_numpy())[0]
    reverse_glicko = sigmoid(mirrored[ensemble.glicko_column].to_numpy())[0]
    probabilities["dynamic_glicko"] = float(
        np.clip((direct_glicko + 1.0 - reverse_glicko) / 2.0, 0.01, 0.99)
    )
    return probabilities


def residual_probability(artifact, model_probability, market_probability, frame):
    residual = artifact.get("market_residual")
    if residual is None:
        return float(market_probability)
    direct = residual.predict_probability(
        np.array([model_probability]),
        np.array([market_probability]),
        frame,
    )[0]
    mirrored = frame.copy()
    for column in DIFFERENCE_FEATURES:
        mirrored[column] = -mirrored[column]
    reverse = residual.predict_probability(
        np.array([1.0 - model_probability]),
        np.array([1.0 - market_probability]),
        mirrored,
    )[0]
    probability = (direct + 1.0 - reverse) / 2.0
    return float(np.clip(probability, 0.01, 0.99))


def market_anchored_probability(model_probability, market_probability, alpha):
    return float(sigmoid(clipped_logit(market_probability) + alpha * (
        clipped_logit(model_probability) - clipped_logit(market_probability)
    )))
