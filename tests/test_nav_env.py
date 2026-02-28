from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pokemon.nav_env import PokemonNavEnv


def _reward_fn(_prev: dict, _curr: dict, ctx: dict) -> float:
    return 1.0 if bool(ctx.get("entered_new_tile", False)) else 0.0


class FakeEmulator:
    def __init__(
        self,
        _rom_path: Path,
        state_path: Path,
        window: str = "null",
        emulation_speed: int = 0,
    ) -> None:
        del window, emulation_speed
        self.state_name = Path(state_path).name
        self.last_button = ""
        self.stopped = False
        self._idx = 0
        self._sequences = {
            "start.state": [
                {"x": 1, "y": 1, "map_id": 1, "badges": 0, "hp": 10, "level": 5, "in_battle": 0},
                {"x": 2, "y": 1, "map_id": 1, "badges": 0, "hp": 10, "level": 5, "in_battle": 0},
                {"x": 3, "y": 1, "map_id": 1, "badges": 0, "hp": 10, "level": 5, "in_battle": 1},
            ],
            "transition.state": [
                {"x": 1, "y": 1, "map_id": 12, "badges": 0, "hp": 10, "level": 5, "in_battle": 0},
                {"x": 1, "y": 1, "map_id": 1, "badges": 0, "hp": 10, "level": 5, "in_battle": 0},
                {"x": 1, "y": 1, "map_id": 12, "badges": 0, "hp": 10, "level": 5, "in_battle": 0},
            ],
            "stuck.state": [
                {"x": 4, "y": 4, "map_id": 2, "badges": 0, "hp": 12, "level": 6, "in_battle": 0},
                {"x": 4, "y": 4, "map_id": 2, "badges": 0, "hp": 12, "level": 6, "in_battle": 0},
                {"x": 4, "y": 4, "map_id": 2, "badges": 0, "hp": 12, "level": 6, "in_battle": 0},
            ],
        }

    def reset(self, state_path: Path | None = None) -> None:
        if state_path is not None:
            self.state_name = Path(state_path).name
        self._idx = 0

    def get_nav_state(self) -> dict:
        seq = self._sequences[self.state_name]
        row = seq[min(self._idx, len(seq) - 1)]
        return dict(row)

    def press(self, button: str, frames: int = 8) -> None:
        del frames
        self.last_button = button
        seq = self._sequences[self.state_name]
        if self._idx < len(seq) - 1:
            self._idx += 1

    def tick(self, frames: int = 1) -> None:
        del frames

    def stop(self) -> None:
        self.stopped = True


class NavEnvTests(unittest.TestCase):
    def test_reset_observation_shape(self) -> None:
        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("start.state"),
            reward_fn=_reward_fn,
            max_episode_steps=4,
            emulator_factory=FakeEmulator,
        )
        try:
            obs, info = env.reset(seed=1)
            self.assertEqual(obs.shape, (8,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(int(obs[7]), 1)  # visited_count
            self.assertEqual(info["map_id"], 1)
        finally:
            env.close()

    def test_action_button_mapping(self) -> None:
        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("start.state"),
            reward_fn=_reward_fn,
            max_episode_steps=4,
            emulator_factory=FakeEmulator,
        )
        try:
            env.reset(seed=1)
            _, _, _, _, info = env.step(0)
            self.assertEqual(info["action_button"], "up")
            _, _, _, _, info = env.step(5)
            self.assertEqual(info["action_button"], "b")
        finally:
            env.close()

    def test_new_tile_count_ignores_battle_frame(self) -> None:
        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("start.state"),
            reward_fn=_reward_fn,
            max_episode_steps=2,
            emulator_factory=FakeEmulator,
        )
        try:
            env.reset(seed=1)
            _, _, _, _, info = env.step(0)
            self.assertEqual(info["visited_count"], 2)
            _, _, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            summary = info.get("episode_summary", {})
            self.assertEqual(summary.get("unique_tiles"), 2)
        finally:
            env.close()

    def test_no_progress_truncates_episode(self) -> None:
        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("stuck.state"),
            reward_fn=_reward_fn,
            max_episode_steps=10,
            no_progress_limit=2,
            emulator_factory=FakeEmulator,
        )
        try:
            env.reset(seed=2)
            _, _, _, truncated, _ = env.step(0)
            self.assertFalse(truncated)
            _, _, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            summary = info.get("episode_summary", {})
            self.assertEqual(summary.get("episode_len"), 2)
        finally:
            env.close()

    def test_guidance_ctx_and_summary_metrics(self) -> None:
        captured_ctx: list[dict] = []

        def reward_with_capture(_prev: dict, _curr: dict, ctx: dict) -> float:
            captured_ctx.append(dict(ctx))
            return float(ctx.get("guidance_weight", 0.0) * ctx.get("guidance_reward", 0.0))

        def scorer(_prev: dict, _curr: dict, _ctx: dict) -> dict:
            return {"guidance_reward": 1.0, "valid_map_transition": True}

        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("transition.state"),
            reward_fn=reward_with_capture,
            max_episode_steps=2,
            guidance_scorer=scorer,
            guidance_weight=0.25,
            emulator_factory=FakeEmulator,
        )
        try:
            env.reset(seed=3)
            _, reward, _, _, _ = env.step(0)
            self.assertAlmostEqual(reward, 0.25)
            _, _, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            self.assertGreaterEqual(len(captured_ctx), 2)
            self.assertIn("guidance_reward", captured_ctx[0])
            self.assertIn("valid_map_transition", captured_ctx[0])
            summary = info.get("episode_summary", {})
            self.assertGreaterEqual(float(summary.get("valid_transition_rate", 0.0)), 1.0)
            self.assertGreaterEqual(float(summary.get("ping_pong_ratio", 0.0)), 0.0)
            self.assertEqual(float(summary.get("new_map_discovery_count", 0.0)), 1.0)
        finally:
            env.close()

    def test_dpad_action_set_reduces_action_space(self) -> None:
        env = PokemonNavEnv(
            rom_path=Path("dummy.gb"),
            state_path=Path("start.state"),
            reward_fn=_reward_fn,
            action_buttons=("up", "down", "left", "right"),
            emulator_factory=FakeEmulator,
        )
        try:
            env.reset(seed=4)
            self.assertEqual(env.action_space.n, 4)
            _, _, _, _, info = env.step(3)
            self.assertEqual(info["action_button"], "right")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
