import unittest

import numpy as np
import pandas as pd

from model_estimators import CatBoostAdapter, MarketResidualModel


class ModelEstimatorTests(unittest.TestCase):
    def test_catboost_challenger_handles_time_ordered_categories(self):
        frame = pd.DataFrame({
            "numeric": np.linspace(-1, 1, 80),
            "division": ["Lightweight", "Welterweight"] * 40,
        })
        target = np.array([0, 1] * 40)
        model = CatBoostAdapter(
            categorical_columns=("division",),
            iterations=30,
            depth=2,
        ).fit(frame, target)
        probability = model.predict_proba(frame)[:, 1]
        self.assertEqual(len(probability), len(frame))
        self.assertTrue(np.logical_and(probability > 0, probability < 1).all())

    def test_market_residual_returns_valid_probabilities(self):
        frame = pd.DataFrame({"age_diff": [-3, -1, 1, 3] * 10})
        model_probability = np.array([0.3, 0.4, 0.6, 0.7] * 10)
        market_probability = np.array([0.4, 0.45, 0.55, 0.6] * 10)
        target = np.array([0, 0, 1, 1] * 10)
        residual = MarketResidualModel(("age_diff",), l2=10).fit(
            model_probability,
            market_probability,
            frame,
            target,
        )
        probability = residual.predict_probability(
            model_probability,
            market_probability,
            frame,
        )
        self.assertTrue(np.logical_and(probability > 0, probability < 1).all())


if __name__ == "__main__":
    unittest.main()
