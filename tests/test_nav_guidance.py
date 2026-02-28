from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pokemon.nav_guidance import load_guidance_profile, score_transition


class NavGuidanceTests(unittest.TestCase):
    def test_kanto_early_valid_transition_scores_positive(self) -> None:
        profile = load_guidance_profile("kanto_early")
        prev_obs = {"map_id": 12}
        curr_obs = {"map_id": 1}
        ctx = {
            "map_changed": True,
            "new_map_discovery": True,
            "ping_pong_event": False,
            "battle_frame": False,
        }
        out = score_transition(prev_obs, curr_obs, ctx, profile)
        self.assertTrue(out["valid_map_transition"])
        self.assertGreater(out["guidance_reward"], 0.0)
        self.assertIn("valid_transition", out["guidance_tags"])

    def test_kanto_early_invalid_transition_penalized(self) -> None:
        profile = load_guidance_profile("kanto_early")
        prev_obs = {"map_id": 12}
        curr_obs = {"map_id": 99}
        ctx = {
            "map_changed": True,
            "new_map_discovery": False,
            "ping_pong_event": False,
            "battle_frame": False,
        }
        out = score_transition(prev_obs, curr_obs, ctx, profile)
        self.assertFalse(out["valid_map_transition"])
        self.assertLess(out["guidance_reward"], 0.0)
        self.assertIn("invalid_transition", out["guidance_tags"])

    def test_ping_pong_penalty_applies(self) -> None:
        profile = load_guidance_profile("kanto_early")
        prev_obs = {"map_id": 12}
        curr_obs = {"map_id": 1}
        ctx = {
            "map_changed": True,
            "new_map_discovery": False,
            "ping_pong_event": True,
            "battle_frame": False,
        }
        out = score_transition(prev_obs, curr_obs, ctx, profile)
        self.assertIn("ping_pong", out["guidance_tags"])
        self.assertGreaterEqual(out["loop_penalty"], 0.0)

    def test_custom_guidance_json_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "guidance.json"
            payload = {
                "name": "custom_local",
                "adjacency": {"2": [3]},
                "valid_transition_bonus": 0.4,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            profile = load_guidance_profile("none", path)
            self.assertEqual(profile["name"], "custom")
            prev_obs = {"map_id": 2}
            curr_obs = {"map_id": 3}
            ctx = {
                "map_changed": True,
                "new_map_discovery": False,
                "ping_pong_event": False,
                "battle_frame": False,
            }
            out = score_transition(prev_obs, curr_obs, ctx, profile)
            self.assertTrue(out["valid_map_transition"])
            self.assertGreaterEqual(out["guidance_reward"], 0.4)


if __name__ == "__main__":
    unittest.main()
