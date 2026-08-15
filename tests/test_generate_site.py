import unittest

import generate_site


class SiteGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = generate_site.build_site_data()

    def test_elo_probability_is_complementary(self):
        red = generate_site.elo_probability(1620, 1510)
        blue = generate_site.elo_probability(1510, 1620)
        self.assertAlmostEqual(red + blue, 1.0)
        self.assertGreater(red, 0.5)

    def test_american_odds_conversion(self):
        self.assertAlmostEqual(generate_site.american_to_decimal(150), 2.5)
        self.assertAlmostEqual(generate_site.american_to_decimal(-200), 1.5)
        self.assertAlmostEqual(generate_site.american_to_implied(-200), 2 / 3)

    def test_payload_has_event_predictions_and_rankings(self):
        self.assertIsNotNone(self.payload["next_event"])
        self.assertGreater(len(self.payload["predictions"]), 0)
        self.assertGreater(len(self.payload["rankings"]), 2_000)

    def test_prediction_probabilities_and_recommendations_are_consistent(self):
        for prediction in self.payload["predictions"]:
            self.assertAlmostEqual(
                prediction["red_probability"] + prediction["blue_probability"],
                1.0,
                places=3,
            )
            self.assertIn(
                prediction["predicted_winner"],
                {prediction["fighter_red"], prediction["fighter_blue"]},
            )
            if prediction["recommendation"]:
                self.assertIn(
                    prediction["recommendation"],
                    {prediction["fighter_red"], prediction["fighter_blue"]},
                )

    def test_rankings_are_sorted_and_uniquely_ranked(self):
        rankings = self.payload["rankings"]
        self.assertEqual([row["rank"] for row in rankings], list(range(1, len(rankings) + 1)))
        self.assertEqual(
            [row["elo"] for row in rankings],
            sorted((row["elo"] for row in rankings), reverse=True),
        )


if __name__ == "__main__":
    unittest.main()
