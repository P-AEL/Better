import unittest
from datetime import date, timedelta

import pandas as pd

import model_pipeline
import train_model


class ModelPipelineTests(unittest.TestCase):
    def test_no_vig_probabilities_sum_to_one(self):
        red, blue = model_pipeline.no_vig_probabilities(-200, 170)
        self.assertAlmostEqual(red + blue, 1.0)
        self.assertGreater(red, blue)

    def test_mirror_negates_only_directional_features(self):
        original = {
            column: float(index + 1)
            for index, column in enumerate(model_pipeline.FULL_FEATURE_COLUMNS)
        }
        mirrored = model_pipeline.mirror_features(original)
        for column in model_pipeline.DIFFERENCE_FEATURES:
            self.assertEqual(mirrored[column], -original[column])
        for column in model_pipeline.SYMMETRIC_FEATURES:
            self.assertEqual(mirrored[column], original[column])
        for column in model_pipeline.CATEGORICAL_FEATURES:
            self.assertEqual(mirrored[column], original[column])

    def test_dynamic_glicko_probability_is_complementary(self):
        red = model_pipeline.FighterState(glicko_rating=1650, glicko_rd=90)
        blue = model_pipeline.FighterState(glicko_rating=1500, glicko_rd=120)
        event_date = date(2030, 1, 1)
        red_probability = model_pipeline.glicko_probability(red, blue, event_date)
        blue_probability = model_pipeline.glicko_probability(blue, red, event_date)
        self.assertAlmostEqual(red_probability + blue_probability, 1.0)
        self.assertGreater(red_probability, 0.5)

    def test_glicko_uncertainty_increases_during_inactivity(self):
        state = model_pipeline.FighterState(
            glicko_rd=80,
            last_date=date(2025, 1, 1),
        )
        active_rd = model_pipeline.effective_glicko_rd(state, date(2025, 2, 1))
        inactive_rd = model_pipeline.effective_glicko_rd(state, date(2030, 1, 1))
        self.assertGreater(inactive_rd, active_rd)
        self.assertLessEqual(inactive_rd, model_pipeline.GLICKO_INITIAL_RD)

    def test_running_division_normalization_uses_prior_observations(self):
        moments = model_pipeline.RunningMoments()
        for value in range(1, 31):
            moments.update(value)
        self.assertGreater(moments.z_score(30), 0)

    def test_market_anchor_with_zero_alpha_equals_market(self):
        probability = model_pipeline.market_anchored_probability(0.80, 0.35, 0.0)
        self.assertAlmostEqual(probability, 0.35)

    def test_temporal_partitions_never_train_on_future_events(self):
        frame = pd.DataFrame({
            "event_date": [f"2024-{month:02d}-01" for month in range(1, 11)],
            "target": [0, 1] * 5,
        })
        development, calibration, test = train_model.temporal_partitions(frame)
        self.assertLess(development["event_date"].max(), calibration["event_date"].min())
        self.assertLess(calibration["event_date"].max(), test["event_date"].min())

    def test_fight_probabilities_are_symmetrized(self):
        frame = pd.DataFrame({"fight_id": ["a", "a", "b"], "target": [1, 0, 1]})
        probability = train_model.symmetrize_fight_probabilities(
            frame, [0.7, 0.4, 0.8]
        )
        self.assertAlmostEqual(probability[0], 0.65)
        self.assertAlmostEqual(probability[1], 0.35)
        self.assertAlmostEqual(probability[2], 0.8)

    def test_event_performance_counts_each_fight_once(self):
        frame = pd.DataFrame({
            "event_name": ["Event A", "Event A", "Event A", "Event A"],
            "event_date": ["2025-01-01"] * 4,
            "fight_id": ["a", "a", "b", "b"],
            "target": [1, 0, 1, 0],
        })
        result = train_model.event_performance(
            frame, {"ensemble": [0.7, 0.3, 0.4, 0.6]}
        )
        self.assertEqual(result[0]["fight_count"], 2)
        self.assertAlmostEqual(result[0]["models"]["ensemble"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
