import unittest

import joblib
import pandas as pd

import method_model


class MethodModelTests(unittest.TestCase):
    def test_method_mapping_keeps_requested_classes_distinct(self):
        self.assertEqual(method_model.classify_method(pd.Series({"result": "win", "method": "KO/TKO Punches"})), "ko_tko")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "win", "method": "SUB Armbar"})), "submission")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "draw", "method": "U-DEC"})), "decision")
        self.assertEqual(method_model.classify_method(pd.Series({"result": "nc", "method": "Overturned"})), "nc")
        self.assertIsNone(method_model.classify_method(pd.Series({"result": "win", "method": "DQ"})))

    def test_method_features_are_fighter_order_invariant(self):
        values = {name: 2.0 for name in method_model.FULL_NUMERIC_FEATURES}
        values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        forward = method_model.method_features(pd.DataFrame([values]))
        reversed_values = {name: -2.0 for name in method_model.FULL_NUMERIC_FEATURES}
        reversed_values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        reverse = method_model.method_features(pd.DataFrame([reversed_values]))
        self.assertEqual(forward, reverse)

    def test_prediction_probabilities_are_normalized(self):
        artifact = joblib.load(method_model.MODEL_PATH)
        values = {name: 0.0 for name in method_model.FULL_NUMERIC_FEATURES}
        values.update({"weight_class": "Lightweight", "stance_pair": "Orthodox | Southpaw"})
        result = method_model.predict_method(artifact, pd.DataFrame([values]))
        self.assertEqual(set(result), set(method_model.CLASSES))
        self.assertTrue(all(value >= 0 for value in result.values()))
        self.assertAlmostEqual(sum(result.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
