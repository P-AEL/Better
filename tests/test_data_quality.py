import unittest
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "scraped_data"

EVENT_COLUMNS = {
    "event_name", "event_date", "location", "link",
}
FIGHT_COLUMNS = {
    "event_name", "fighter_red", "fighter_blue", "result", "winner",
    "kd_red", "kd_blue", "str_red", "str_blue", "td_red", "td_blue",
    "sub_red", "sub_blue", "weight_class", "method", "round", "time",
    "fight_link",
}
FIGHTER_COLUMNS = {
    "first_name", "last_name", "nickname", "height", "weight", "reach",
    "stance", "wins", "losses", "draws", "belt", "profile_url", "dob",
}
ODDS_COLUMNS = {
    "event_name", "event_date", "fighter_red", "fighter_blue", "red_open",
    "blue_open", "red_close_low", "red_close_high", "blue_close_low",
    "blue_close_high", "red_open_decimal", "blue_open_decimal", "red_bfo_url",
    "blue_bfo_url", "bfo_event_date", "bfo_event_url", "match_status",
    "odds_source", "scraped_at_utc",
}


def nonblank(series):
    return series.notna() & series.astype(str).str.strip().ne("")


def unordered_fight_key(row):
    return row.event_name, *sorted((str(row.fighter_red), str(row.fighter_blue)))


class DataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = pd.read_csv(DATA_DIR / "ufc_events.csv")
        cls.fights = pd.read_csv(DATA_DIR / "ufc_fights.csv")
        cls.fighters = pd.read_csv(DATA_DIR / "ufc_fighters_basic_with_dob.csv")
        cls.odds = pd.read_csv(DATA_DIR / "ufc_fight_odds.csv")

    def test_expected_schemas(self):
        self.assertEqual(set(self.events.columns), EVENT_COLUMNS)
        self.assertEqual(set(self.fights.columns), FIGHT_COLUMNS)
        self.assertEqual(set(self.fighters.columns), FIGHTER_COLUMNS)
        self.assertEqual(set(self.odds.columns), ODDS_COLUMNS)

    def test_datasets_are_not_empty_or_truncated(self):
        self.assertGreater(len(self.events), 700)
        self.assertGreater(len(self.fights), 8_000)
        self.assertGreater(len(self.fighters), 4_000)
        self.assertGreater(len(self.odds), 7_000)

    def test_events_have_unique_valid_identity(self):
        required = self.events[["event_name", "event_date", "location", "link"]]
        self.assertTrue(required.apply(nonblank).all().all())
        self.assertFalse(self.events["event_name"].duplicated().any())
        self.assertFalse(self.events["link"].duplicated().any())
        self.assertTrue(pd.to_datetime(self.events["event_date"], errors="coerce").notna().all())
        self.assertTrue(
            self.events["link"].str.match(
                r"^https?://(?:www\.)?ufcstats\.com/event-details/[0-9a-f]+$"
            ).all()
        )

    def test_fights_reference_events_and_distinct_fighters(self):
        required = self.fights[["event_name", "fighter_red", "fighter_blue"]]
        self.assertTrue(required.apply(nonblank).all().all())
        self.assertTrue(self.fights["event_name"].isin(self.events["event_name"]).all())
        self.assertFalse(self.fights["fighter_red"].eq(self.fights["fighter_blue"]).any())

    def test_fight_outcomes_are_consistent(self):
        self.assertTrue(self.fights["result"].isin({"win", "draw", "nc", "scheduled"}).all())

        wins = self.fights["result"].eq("win")
        non_wins = ~wins
        self.assertTrue(self.fights.loc[wins, "winner"].eq(self.fights.loc[wins, "fighter_red"]).all())
        self.assertTrue(self.fights.loc[non_wins, "winner"].isna().all())
        self.assertTrue(
            self.fights.loc[self.fights["winner"].notna()].apply(
                lambda row: row.winner in {row.fighter_red, row.fighter_blue}, axis=1
            ).all()
        )

    def test_head_to_head_wins_and_losses_are_reciprocal(self):
        ledger = defaultdict(Counter)
        decisive = self.fights[self.fights["result"].eq("win")]

        for row in decisive.itertuples(index=False):
            loser = row.fighter_blue if row.winner == row.fighter_red else row.fighter_red
            ledger[(row.winner, loser)]["wins"] += 1
            ledger[(loser, row.winner)]["losses"] += 1

        for (fighter, opponent), record in ledger.items():
            reciprocal = ledger[(opponent, fighter)]
            self.assertEqual(
                record["wins"],
                reciprocal["losses"],
                f"{fighter} wins over {opponent} do not match reciprocal losses",
            )
            self.assertEqual(
                record["losses"],
                reciprocal["wins"],
                f"{fighter} losses to {opponent} do not match reciprocal wins",
            )

    def test_completed_fight_links_are_valid_and_unique(self):
        completed = self.fights[~self.fights["result"].eq("scheduled")]
        self.assertTrue(nonblank(completed["fight_link"]).all())
        self.assertFalse(completed["fight_link"].duplicated().any())
        self.assertTrue(
            completed["fight_link"].str.match(
                r"^https?://(?:www\.)?ufcstats\.com/fight-details/[0-9a-f]+$"
            ).all()
        )

    def test_completed_fight_stats_are_valid(self):
        completed = self.fights[~self.fights["result"].eq("scheduled")]
        stat_columns = [
            "kd_red", "kd_blue", "str_red", "str_blue", "td_red", "td_blue",
            "sub_red", "sub_blue",
        ]
        for column in stat_columns:
            raw = completed[column].astype(str).str.strip()
            valid = raw.eq("--") | raw.str.fullmatch(r"\d+")
            self.assertTrue(valid.all(), f"Invalid values in {column}: {raw[~valid].unique()[:5]}")

        rounds = pd.to_numeric(completed["round"], errors="coerce")
        self.assertTrue(rounds.notna().all())
        self.assertTrue(rounds.between(1, 5).all())

        times = completed["time"].astype(str).str.extract(r"^(\d+):(\d{2})$")
        self.assertTrue(times.notna().all().all())
        self.assertTrue(pd.to_numeric(times[1]).lt(60).all())
        self.assertTrue(nonblank(completed["method"]).all())
        self.assertTrue(nonblank(completed["weight_class"]).all())

    def test_scheduled_fights_do_not_contain_results(self):
        scheduled = self.fights[self.fights["result"].eq("scheduled")]
        self.assertTrue(scheduled["winner"].isna().all())
        self.assertTrue(scheduled["method"].isna().all())
        self.assertTrue(scheduled["fight_link"].isna().all())

    def test_fighter_profiles_and_records_are_valid(self):
        has_name = nonblank(self.fighters["first_name"]) | nonblank(self.fighters["last_name"])
        self.assertTrue(has_name.all())
        self.assertTrue(nonblank(self.fighters["profile_url"]).all())
        self.assertFalse(self.fighters["profile_url"].duplicated().any())
        self.assertTrue(
            self.fighters["profile_url"].str.match(
                r"^https?://(?:www\.)?ufcstats\.com/fighter-details/[0-9a-f]+$"
            ).all()
        )

        for column in ("wins", "losses", "draws"):
            values = pd.to_numeric(self.fighters[column], errors="coerce")
            self.assertTrue(values.notna().all())
            self.assertTrue(values.ge(0).all())
            self.assertTrue(values.mod(1).eq(0).all())

    def test_fighter_physical_data_and_dobs_have_valid_formats(self):
        self.assertTrue(self.fighters["height"].astype(str).str.fullmatch(r"--|\d+' \d+\"").all())
        self.assertTrue(self.fighters["weight"].astype(str).str.fullmatch(r"--|\d+ lbs\.").all())
        self.assertTrue(self.fighters["reach"].astype(str).str.fullmatch(r"--|\d+(?:\.\d+)?\"").all())
        self.assertTrue(
            self.fighters["stance"].dropna().isin(
                {"Orthodox", "Southpaw", "Switch", "Open Stance", "Sideways"}
            ).all()
        )

        dobs = pd.to_datetime(self.fighters["dob"], format="%b %d, %Y", errors="coerce")
        supplied = self.fighters["dob"].notna()
        self.assertTrue(dobs[supplied].notna().all())
        self.assertTrue(dobs[supplied].dt.year.between(1900, 2010).all())

    def test_fight_participants_are_well_covered_by_fighter_profiles(self):
        full_names = (
            self.fighters["first_name"].fillna("").str.strip()
            + " "
            + self.fighters["last_name"].fillna("").str.strip()
        ).str.strip()
        participants = set(
            pd.concat([self.fights["fighter_red"], self.fights["fighter_blue"]]).dropna()
        )
        coverage = len(participants.intersection(set(full_names))) / len(participants)
        self.assertGreaterEqual(coverage, 0.95)

    def test_odds_status_and_required_values_are_consistent(self):
        required = self.odds[["event_name", "fighter_red", "fighter_blue", "match_status"]]
        self.assertTrue(required.apply(nonblank).all().all())
        self.assertFalse(self.odds["fighter_red"].eq(self.odds["fighter_blue"]).any())
        self.assertTrue(self.odds["match_status"].isin({"matched", "partial", "no_match"}).all())
        supplied_sources = self.odds["odds_source"].dropna()
        self.assertTrue(supplied_sources.eq("BestFightOdds").all())
        supplied_times = self.odds["scraped_at_utc"].dropna()
        self.assertTrue(pd.to_datetime(supplied_times, utc=True, errors="coerce").notna().all())

        matched = self.odds[self.odds["match_status"].eq("matched")]
        odds_columns = [
            "red_open", "blue_open", "red_close_low", "red_close_high",
            "blue_close_low", "blue_close_high",
        ]
        self.assertTrue(matched[odds_columns].notna().all().all())

        no_match = self.odds[self.odds["match_status"].eq("no_match")]
        self.assertTrue(no_match[odds_columns].isna().all().all())

    def test_american_and_decimal_odds_agree(self):
        american_columns = [
            "red_open", "blue_open", "red_close_low", "red_close_high",
            "blue_close_low", "blue_close_high",
        ]
        for column in american_columns:
            values = pd.to_numeric(self.odds[column], errors="coerce")
            self.assertTrue(values[self.odds[column].notna()].notna().all())
            self.assertTrue(values.dropna().abs().ge(100).all())

        for side in ("red", "blue"):
            american = pd.to_numeric(self.odds[f"{side}_open"], errors="coerce")
            decimal = pd.to_numeric(self.odds[f"{side}_open_decimal"], errors="coerce")
            expected = (1 + american / 100).where(
                american > 0, 1 + 100 / american.abs()
            )
            comparable = american.notna() & decimal.notna()
            self.assertTrue((decimal[comparable] - expected[comparable]).abs().lt(1e-9).all())

    def test_odds_refer_to_known_fights_and_dates(self):
        fight_keys = {unordered_fight_key(row) for row in self.fights.itertuples(index=False)}
        odds_keys = [unordered_fight_key(row) for row in self.odds.itertuples(index=False)]
        coverage = sum(key in fight_keys for key in odds_keys) / len(odds_keys)
        self.assertGreaterEqual(coverage, 0.99)

        event_dates = pd.to_datetime(self.odds["event_date"], errors="coerce")
        bfo_dates = pd.to_datetime(self.odds["bfo_event_date"], errors="coerce")
        comparable = event_dates.notna() & bfo_dates.notna()
        self.assertTrue((event_dates[comparable] - bfo_dates[comparable]).abs().dt.days.le(2).all())


if __name__ == "__main__":
    unittest.main()
