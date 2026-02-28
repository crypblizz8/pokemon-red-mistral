from __future__ import annotations

import unittest

from pokemon.phase3_metrics import (
    aggregate_episode_rows,
    build_v2_critique_prompt,
    compare_versions,
)


class Phase3MetricsTests(unittest.TestCase):
    def test_aggregate_episode_rows(self) -> None:
        rows = [
            {
                "unique_tiles": 10,
                "furthest_map_id": 2,
                "episode_len": 100,
                "episode_return": 5.0,
                "stuck_ratio": 0.2,
                "valid_transition_rate": 0.6,
                "new_map_discovery_count": 2,
                "ping_pong_ratio": 0.1,
                "map_ids_seen": [1, 2],
            },
            {
                "unique_tiles": 14,
                "furthest_map_id": 3,
                "episode_len": 120,
                "episode_return": 6.0,
                "stuck_ratio": 0.1,
                "valid_transition_rate": 0.8,
                "new_map_discovery_count": 3,
                "ping_pong_ratio": 0.05,
                "map_ids_seen": [2, 3],
            },
        ]
        out = aggregate_episode_rows(rows)
        self.assertEqual(out["episodes"], 2)
        self.assertEqual(out["unique_tiles"], 12.0)
        self.assertEqual(out["furthest_map_id"], 3)
        self.assertEqual(out["avg_episode_len"], 110.0)
        self.assertEqual(out["avg_return"], 5.5)
        self.assertEqual(out["stuck_ratio"], 0.15)
        self.assertEqual(out["valid_transition_rate"], 0.7)
        self.assertEqual(out["new_map_discovery_count"], 2.5)
        self.assertEqual(out["ping_pong_ratio"], 0.075)
        self.assertEqual(out["map_ids_seen"], [1, 2, 3])

    def test_compare_versions_passes_on_positive_delta(self) -> None:
        v1 = {
            "unique_tiles": 12.0,
            "furthest_map_id": 3,
            "avg_return": 2.0,
            "avg_episode_len": 100,
            "stuck_ratio": 0.2,
            "valid_transition_rate": 0.4,
            "new_map_discovery_count": 2.0,
            "ping_pong_ratio": 0.1,
        }
        v2 = {
            "unique_tiles": 13.0,
            "furthest_map_id": 3,
            "avg_return": 2.0,
            "avg_episode_len": 90,
            "stuck_ratio": 0.1,
            "valid_transition_rate": 0.6,
            "new_map_discovery_count": 2.4,
            "ping_pong_ratio": 0.05,
        }
        out = compare_versions(v1, v2)
        self.assertEqual(out["delta_unique_tiles"], 1.0)
        self.assertEqual(out["delta_valid_transition_rate"], 0.2)
        self.assertEqual(out["delta_new_map_discovery_count"], 0.4)
        self.assertEqual(out["delta_ping_pong_ratio"], -0.05)
        self.assertTrue(out["phase3_pass"])

    def test_compare_versions_fails_when_no_core_improvement(self) -> None:
        v1 = {
            "unique_tiles": 12.0,
            "furthest_map_id": 3,
            "avg_return": 2.0,
            "avg_episode_len": 100,
            "stuck_ratio": 0.2,
            "valid_transition_rate": 0.5,
            "new_map_discovery_count": 2.5,
            "ping_pong_ratio": 0.1,
        }
        v2 = {
            "unique_tiles": 11.0,
            "furthest_map_id": 3,
            "avg_return": 1.5,
            "avg_episode_len": 95,
            "stuck_ratio": 0.1,
            "valid_transition_rate": 0.6,
            "new_map_discovery_count": 2.0,
            "ping_pong_ratio": 0.08,
        }
        out = compare_versions(v1, v2)
        self.assertFalse(out["phase3_pass"])

    def test_build_v2_prompt_contains_metrics(self) -> None:
        text = build_v2_critique_prompt(
            {
                "unique_tiles": 12.0,
                "furthest_map_id": 3,
                "avg_episode_len": 100.0,
                "avg_return": 4.5,
                "stuck_ratio": 0.3,
                "valid_transition_rate": 0.7,
                "new_map_discovery_count": 3.0,
                "ping_pong_ratio": 0.04,
            }
        )
        self.assertIn("unique_tiles", text)
        self.assertIn("reward_v2.py", text)
        self.assertIn("valid_transition_rate", text)


if __name__ == "__main__":
    unittest.main()
