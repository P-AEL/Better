import unittest

import pandas as pd

import forward_test


class ForwardTestTests(unittest.TestCase):
    def test_records_and_settles_prediction(self):
        site_data = {
            "generated_at": "2030-01-01T10:00:00+00:00",
            "model": {"version": "3.0.0"},
            "next_event": {"name": "Test Event", "date": "2030-01-02"},
            "predictions": [{
                "fighter_red": "Red Fighter",
                "fighter_blue": "Blue Fighter",
                "independent_red_probability": 0.6,
                "red_probability": 0.58,
                "recommendation": "Red Fighter",
                "paper_candidate": True,
                "market": {
                    "observed_at": "2030-01-01T09:00:00+00:00",
                    "red_fair_probability": 0.55,
                    "red_decimal": 1.9,
                    "blue_decimal": 2.0,
                },
            }],
        }
        journal = forward_test.record_current_predictions(
            pd.DataFrame(columns=forward_test.JOURNAL_COLUMNS), site_data
        )
        journal = forward_test.record_current_predictions(journal, site_data)
        fights = pd.DataFrame([{
            "event_name": "Test Event",
            "fighter_red": "Red Fighter",
            "fighter_blue": "Blue Fighter",
            "result": "win",
            "winner": "Red Fighter",
        }])
        settled = forward_test.settle_predictions(journal, fights)
        summary = forward_test.summarize(settled)

        self.assertEqual(settled.iloc[0]["outcome_red"], 1.0)
        self.assertEqual(len(settled), 1)
        self.assertEqual(summary["paper"]["settled"], 1)
        self.assertAlmostEqual(summary["paper"]["net_units"], 0.9)

    def test_forward_gate_requires_minimum_sample_and_market_improvement(self):
        rows = []
        for index in range(100):
            target = float(index % 2 == 0)
            rows.append({
                **{column: None for column in forward_test.JOURNAL_COLUMNS},
                "independent_red_probability": 0.8 if target else 0.2,
                "residual_red_probability": 0.8 if target else 0.2,
                "market_red_probability": 0.5,
                "outcome_red": target,
                "paper_candidate": False,
            })
        summary = forward_test.summarize(pd.DataFrame(rows))
        self.assertTrue(summary["forward_gate_passed"])


if __name__ == "__main__":
    unittest.main()
