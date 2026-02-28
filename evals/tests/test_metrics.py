from __future__ import annotations

import unittest

from evals.metrics import rank_models, summarize_rows, wilson_interval


class MetricsTests(unittest.TestCase):
    def test_wilson_interval_bounds(self) -> None:
        low, high = wilson_interval(7, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)

    def test_summarize_rows_basic(self) -> None:
        rows = [
            {"outcome": "win", "reward": 100, "turns": 8, "hp_left": 20},
            {"outcome": "timeout", "reward": -10, "turns": 12, "hp_left": 5},
            {"outcome": "invalid_state", "reward": -100, "turns": 1, "hp_left": 0},
            {"outcome": "loss", "reward": -50, "turns": 9, "hp_left": 0},
        ]
        metrics = summarize_rows(rows)
        self.assertEqual(metrics["episodes"], 4)
        self.assertEqual(metrics["wins"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 0.25)
        self.assertAlmostEqual(metrics["timeout_rate"], 0.25)
        self.assertAlmostEqual(metrics["invalid_state_rate"], 0.25)

    def test_rank_models_no_clear_winner(self) -> None:
        summaries = [
            {
                "model": "m1",
                "win_rate": 0.55,
                "win_rate_ci_low": 0.40,
                "win_rate_ci_high": 0.70,
                "timeout_rate": 0.10,
                "median_turns": 8,
                "mean_hp_left": 10,
            },
            {
                "model": "m2",
                "win_rate": 0.50,
                "win_rate_ci_low": 0.38,
                "win_rate_ci_high": 0.64,
                "timeout_rate": 0.08,
                "median_turns": 7,
                "mean_hp_left": 11,
            },
        ]
        result = rank_models(summaries, no_clear_winner_gap=0.10)
        self.assertEqual(result["decision"], "no_clear_winner")
        self.assertIsNone(result["winner_model"])

    def test_rank_models_with_winner(self) -> None:
        summaries = [
            {
                "model": "m1",
                "win_rate": 0.80,
                "win_rate_ci_low": 0.70,
                "win_rate_ci_high": 0.88,
                "timeout_rate": 0.05,
                "median_turns": 7,
                "mean_hp_left": 15,
            },
            {
                "model": "m2",
                "win_rate": 0.60,
                "win_rate_ci_low": 0.48,
                "win_rate_ci_high": 0.70,
                "timeout_rate": 0.05,
                "median_turns": 7,
                "mean_hp_left": 15,
            },
        ]
        result = rank_models(summaries, no_clear_winner_gap=0.10)
        self.assertEqual(result["decision"], "winner")
        self.assertEqual(result["winner_model"], "m1")


if __name__ == "__main__":
    unittest.main()
