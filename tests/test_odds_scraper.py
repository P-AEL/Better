import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import odds_scraper


class OddsScraperTests(unittest.TestCase):
    @patch("odds_scraper.pd.read_csv")
    def test_load_next_event_limits_fights_to_nearest_scheduled_card(self, read_csv):
        fights_data = pd.DataFrame(
            [
                {"event_name": "Later", "fighter_red": "C", "fighter_blue": "D", "result": "scheduled"},
                {"event_name": "Next", "fighter_red": "A", "fighter_blue": "B", "result": "scheduled"},
                {"event_name": "Next", "fighter_red": "E", "fighter_blue": "F", "result": "win"},
            ]
        )
        events_data = pd.DataFrame(
            [
                {"event_name": "Later", "event_date": "January 20, 2030"},
                {"event_name": "Next", "event_date": "January 10, 2030"},
            ]
        )
        read_csv.side_effect = [fights_data, events_data]

        event_name, fights = odds_scraper.load_next_event_fights(
            "unused", today=date(2030, 1, 1)
        )

        self.assertEqual(event_name, "Next")
        self.assertEqual(list(fights["fighter_red"]), ["A"])
        self.assertEqual(fights.iloc[0]["event_date"], date(2030, 1, 10))

    @patch("odds_scraper.Path.open")
    @patch("odds_scraper.Path.read_text")
    @patch("odds_scraper.Path.exists", return_value=True)
    def test_writer_preserves_untouched_historical_lines(self, _exists, read_text, open_file):
        header = ",".join(odds_scraper.ODDS_COLUMNS) + "\n"
        historical = "History,,Past A,Past B,,,,,,,,,,,,,no_match\n"
        replaced = "Next,,Old A,Old B,,,,,,,,,,,,,no_match\n"
        read_text.return_value = header + historical + replaced
        handle = open_file.return_value.__enter__.return_value
        current = pd.DataFrame(
            [self._row("Next", "New A", "New B")],
            columns=odds_scraper.ODDS_COLUMNS,
        )

        odds_scraper.write_event_odds(Path("odds.csv"), current, "Next")

        written = "".join(call.args[0] for call in handle.write.call_args_list)
        written += "".join(handle.writelines.call_args.args[0])
        self.assertIn(historical, written)
        self.assertNotIn(replaced, written)
        self.assertIn("Next,,New A,New B", written)

    @patch("odds_scraper.fetch_fighter_pairs")
    def test_build_event_odds_maps_pair_to_fight_orientation(self, fetch_pairs):
        pair = {
            "event_date": date(2030, 1, 10),
            "event_url": "https://example.test/event",
            "fighter_A": "Blue Fighter",
            "fighter_B": "Red Fighter",
            "fighter_A_url": "https://example.test/blue",
            "fighter_B_url": "https://example.test/red",
            "A_open": "+150", "A_close": "+140",
            "B_open": "-170", "B_close": "-160",
        }
        fetch_pairs.return_value = ("https://example.test/red", [pair])
        fights = pd.DataFrame(
            [{
                "event_name": "Next",
                "event_date": date(2030, 1, 10),
                "fighter_red": "Red Fighter",
                "fighter_blue": "Blue Fighter",
            }]
        )

        result = odds_scraper.build_event_odds(fights).iloc[0]

        self.assertEqual(result["match_status"], "matched")
        self.assertEqual(result["red_open"], "-170")
        self.assertEqual(result["blue_open"], "+150")
        self.assertEqual(result["red_close_high"], "-160")
        self.assertEqual(result["blue_close_high"], "+140")

    @staticmethod
    def _row(event_name, red, blue):
        row = {column: None for column in odds_scraper.ODDS_COLUMNS}
        row.update({
            "event_name": event_name,
            "fighter_red": red,
            "fighter_blue": blue,
            "match_status": "no_match",
        })
        return row


if __name__ == "__main__":
    unittest.main()
