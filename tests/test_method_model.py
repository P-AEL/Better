import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

import method_model


class MethodModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Model artifacts are intentionally ignored by Git. Build an isolated
        # artifact here so this test remains valid in a clean CI checkout and
        # does not depend on a previous workflow step leaving a file behind.
        cls.temporary_dir = TemporaryDirectory()
        model_path = Path(cls.temporary_dir.name) / "fight_method_model.joblib"
        report_path = Path(cls.temporary_dir.name) / "fight_method_metrics.json"
        method_model.train(
            model_path=model_path,
            report_path=report_path,
        )
        cls.artifact = joblib.load(model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_dir.cleanup()

    def test_method_mapping_keeps_requested_classes_distinct(self):
        self.assertEqual(method_model.classify_method(pd.Series({"result": "win", "method": "KO/TKO Punches"})), "not_decision")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "win", "method": "SUB Armbar"})), "not_decision")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "draw", "method": "U-DEC"})), "decision")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "nc", "method": "Overturned"})), "not_decision")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "win", "method": "DQ"})), "not_decision")

    def test_method_features_are_fighter_order_invariant(self):
        values = {name: 2.0 for name in method_model.FULL_NUMERIC_FEATURES}
        values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        forward = method_model.method_features(pd.DataFrame([values]))
        reversed_values = {name: -2.0 for name in method_model.FULL_NUMERIC_FEATURES}
        reversed_values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        reverse = method_model.method_features(pd.DataFrame([reversed_values]))
        self.assertEqual(forward, reverse)

    def test_prediction_probabilities_are_normalized(self):
        artifact = self.artifact
        values = {name: 0.0 for name in method_model.FULL_NUMERIC_FEATURES}
        values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        result = method_model.predict_method(artifact, pd.DataFrame([values]))
        self.assertEqual(set(result), set(method_model.CLASSES))
        self.assertTrue(all(value >= 0 for value in result.values()))
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_historical_predictions_keep_outcome_and_probabilities(self):
        frame = pd.DataFrame([{
            "event_name": "Test Event", "event_date": "2026-01-01",
            "fighter_red": "Red", "fighter_blue": "Blue", "weight_class": "Lightweight",
            "target": "not_decision",
        }])
        rows = method_model.historical_predictions(frame, [[0.2, 0.8]])
        self.assertEqual(rows[0]["actual_method"], "not_decision")
        self.assertEqual(rows[0]["predicted_method"], "not_decision")
        self.assertAlmostEqual(sum(rows[0]["probabilities"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
