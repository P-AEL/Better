import unittest

import pandas as pd

import underdog_analysis


class UnderdogAnalysisTests(unittest.TestCase):
    def test_flat_stake_summary_uses_decimal_profit_and_losses(self):
        rows = pd.DataFrame([
            {"won": True, "profit_units": 1.5},
            {"won": False, "profit_units": -1.0},
        ])

        summary = underdog_analysis._summarize(rows)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["net_units"], 0.5)
        self.assertAlmostEqual(summary["roi"], 0.25)

    def test_cutoff_selection_uses_calibration_rows_only(self):
        calibration = pd.DataFrame([
            {"underdog_american": 125, "won": True, "profit_units": 1.25}
            for _ in range(40)
        ] + [
            {"underdog_american": 100, "won": False, "profit_units": -1.0}
            for _ in range(40)
        ])

        cutoff, candidates = underdog_analysis._choose_cutoff(calibration)

        self.assertEqual(cutoff, 125)
        selected = next(item for item in candidates if item["cutoff"] == cutoff)
        self.assertEqual(selected["sample_count"], 40)
        self.assertGreater(selected["roi"], 0)

    def test_cutoff_summary_excludes_shorter_prices(self):
        rows = pd.DataFrame([
            {"underdog_american": 100, "won": True, "profit_units": 1.0},
            {"underdog_american": 150, "won": False, "profit_units": -1.0},
        ])

        summary = underdog_analysis._cutoff_summary(rows, 125)

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["net_units"], -1.0)


if __name__ == "__main__":
    unittest.main()
