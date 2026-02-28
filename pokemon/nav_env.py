from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pokemon.emulator import PokemonEmulator

RewardFn = Callable[[Dict[str, int], Dict[str, int], Dict[str, object]], float]
GuidanceScorer = Callable[[Dict[str, int], Dict[str, int], Dict[str, object]], Dict[str, object]]


class PokemonNavEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}
    ACTION_BUTTONS: Tuple[str, ...] = ("up", "down", "left", "right", "a", "b", "start", "select")

    def __init__(
        self,
        *,
        rom_path: Path,
        state_path: Path,
        reward_fn: RewardFn,
        window: str = "null",
        emulation_speed: int = 0,
        action_hold_frames: int = 6,
        post_action_ticks: int = 10,
        max_episode_steps: int = 512,
        no_progress_limit: int = 80,
        guidance_scorer: GuidanceScorer | None = None,
        guidance_weight: float = 0.0,
        action_buttons: Tuple[str, ...] | None = None,
        emulator_factory: Callable[..., PokemonEmulator] = PokemonEmulator,
    ) -> None:
        super().__init__()
        self.rom_path = Path(rom_path)
        self.default_state_path = Path(state_path)
        self.active_state_path = Path(state_path)
        self.reward_fn = reward_fn
        self.window = window
        self.emulation_speed = int(emulation_speed)
        self.action_hold_frames = max(1, int(action_hold_frames))
        self.post_action_ticks = max(0, int(post_action_ticks))
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.no_progress_limit = max(1, int(no_progress_limit))
        self.guidance_scorer = guidance_scorer
        self.guidance_weight = float(guidance_weight)
        if action_buttons is None:
            action_buttons = self.ACTION_BUTTONS
        self.action_buttons = tuple(str(btn).lower().strip() for btn in action_buttons)
        if not self.action_buttons:
            raise ValueError("action_buttons must not be empty")

        self.observation_space = spaces.Box(low=0.0, high=65535.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_buttons))

        self._emulator_factory = emulator_factory
        self.emu = self._emulator_factory(
            self.rom_path,
            self.default_state_path,
            window=self.window,
            emulation_speed=self.emulation_speed,
        )

        self.episode_steps = 0
        self.no_progress_steps = 0
        self.episode_return = 0.0
        self.visited_tiles: set[Tuple[int, int, int]] = set()
        self.visited_tile_hits: Dict[Tuple[int, int, int], int] = {}
        self.map_ids_seen: set[int] = set()
        self.last_obs_dict: Dict[str, int] = {}
        self.last_episode_summary: Dict[str, object] | None = None
        self.completed_episodes: List[Dict[str, object]] = []
        self.valid_map_transitions = 0
        self.map_transition_attempts = 0
        self.new_map_discoveries = 0
        self.ping_pong_events = 0
        self.map_sequence: List[int] = []

    def set_reward_fn(self, fn: RewardFn) -> None:
        self.reward_fn = fn

    def _tile_key(self, state: Dict[str, int]) -> Tuple[int, int, int]:
        return (int(state.get("map_id", 0)), int(state.get("x", 0)), int(state.get("y", 0)))

    def _record_tile(self, state: Dict[str, int]) -> tuple[bool, int]:
        if int(state.get("in_battle", 0)) != 0:
            return False, 0
        key = self._tile_key(state)
        previous_hits = int(self.visited_tile_hits.get(key, 0))
        self.visited_tile_hits[key] = previous_hits + 1
        was_new = key not in self.visited_tiles
        self.visited_tiles.add(key)
        return was_new, max(0, previous_hits)

    def _obs_vector(self, state: Dict[str, int]) -> np.ndarray:
        return np.asarray(
            [
                float(int(state.get("x", 0))),
                float(int(state.get("y", 0))),
                float(int(state.get("map_id", 0))),
                float(int(state.get("badges", 0))),
                float(int(state.get("hp", 0))),
                float(int(state.get("level", 0))),
                float(int(state.get("in_battle", 0))),
                float(len(self.visited_tiles)),
            ],
            dtype=np.float32,
        )

    def _episode_summary(self) -> Dict[str, object]:
        furthest_map = max(self.map_ids_seen) if self.map_ids_seen else 0
        map_ids_seen = sorted(self.map_ids_seen)
        stuck_ratio = self.no_progress_steps / max(1, self.episode_steps)
        valid_transition_rate = self.valid_map_transitions / max(1, self.map_transition_attempts)
        ping_pong_ratio = self.ping_pong_events / max(1, self.map_transition_attempts)
        return {
            "episode_len": int(self.episode_steps),
            "episode_return": round(float(self.episode_return), 4),
            "unique_tiles": int(len(self.visited_tiles)),
            "furthest_map_id": int(furthest_map),
            "stuck_ratio": round(float(stuck_ratio), 4),
            "valid_transition_rate": round(float(valid_transition_rate), 4),
            "new_map_discovery_count": int(self.new_map_discoveries),
            "ping_pong_ratio": round(float(ping_pong_ratio), 4),
            "map_ids_seen": map_ids_seen,
            "state_path": str(self.active_state_path),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.active_state_path = self.default_state_path
        if options and isinstance(options.get("state_path"), (str, Path)):
            self.active_state_path = Path(options["state_path"])

        self.emu.reset(state_path=self.active_state_path)
        state = self.emu.get_nav_state()

        self.episode_steps = 0
        self.no_progress_steps = 0
        self.episode_return = 0.0
        self.visited_tiles = set()
        self.visited_tile_hits = {}
        self.map_ids_seen = {int(state.get("map_id", 0))}
        self.last_episode_summary = None
        self.valid_map_transitions = 0
        self.map_transition_attempts = 0
        self.new_map_discoveries = 0
        self.ping_pong_events = 0
        self.map_sequence = [int(state.get("map_id", 0))]
        self._record_tile(state)
        self.last_obs_dict = dict(state)

        obs = self._obs_vector(state)
        info = {
            "map_id": int(state.get("map_id", 0)),
            "x": int(state.get("x", 0)),
            "y": int(state.get("y", 0)),
            "visited_count": len(self.visited_tiles),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError(f"Action out of range: {action}")

        prev = dict(self.last_obs_dict) if self.last_obs_dict else self.emu.get_nav_state()
        button = self.action_buttons[action_id]
        self.emu.press(button, frames=self.action_hold_frames)
        if self.post_action_ticks > 0:
            self.emu.tick(self.post_action_ticks)
        curr = self.emu.get_nav_state()

        self.episode_steps += 1
        curr_map_id = int(curr.get("map_id", 0))
        new_map_discovery = curr_map_id not in self.map_ids_seen
        if new_map_discovery:
            self.new_map_discoveries += 1
        self.map_ids_seen.add(curr_map_id)

        prev_map_id = int(prev.get("map_id", -1))
        map_changed = bool(prev_map_id != curr_map_id)
        moved = map_changed or int(prev.get("x", -1)) != int(curr.get("x", -1)) or int(
            prev.get("y", -1)
        ) != int(curr.get("y", -1))
        if moved:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        ping_pong_event = False
        if map_changed:
            self.map_transition_attempts += 1
            self.map_sequence.append(curr_map_id)
            if len(self.map_sequence) >= 3:
                if self.map_sequence[-1] == self.map_sequence[-3] and self.map_sequence[-1] != self.map_sequence[-2]:
                    ping_pong_event = True
                    self.ping_pong_events += 1
            if len(self.map_sequence) > 8:
                self.map_sequence = self.map_sequence[-8:]

        entered_new_tile, revisit_count = self._record_tile(curr)
        battle_frame = int(curr.get("in_battle", 0)) != 0
        if battle_frame:
            entered_new_tile = False
            revisit_count = 0

        ctx: Dict[str, object] = {
            "entered_new_tile": bool(entered_new_tile),
            "revisit_count": int(revisit_count),
            "moved": bool(moved),
            "no_progress_steps": int(self.no_progress_steps),
            "battle_frame": bool(battle_frame),
            "map_changed": bool(map_changed),
            "badge_delta": int(curr.get("badges", 0)) - int(prev.get("badges", 0)),
            "hp_delta": int(curr.get("hp", 0)) - int(prev.get("hp", 0)),
            "visited_count": int(len(self.visited_tiles)),
            "episode_steps": int(self.episode_steps),
            "new_map_discovery": bool(new_map_discovery),
            "ping_pong_event": bool(ping_pong_event),
            "guidance_weight": float(self.guidance_weight),
            "valid_map_transition": False,
            "guidance_reward": 0.0,
            "frontier_map_bonus": 0.0,
            "loop_penalty": 0.0,
            "priority_map_bonus": 0.0,
            "guidance_tags": [],
        }
        if self.guidance_scorer is not None:
            guidance = self.guidance_scorer(prev, curr, ctx)
            if isinstance(guidance, dict):
                ctx.update(guidance)
        if map_changed and bool(ctx.get("valid_map_transition", False)):
            self.valid_map_transitions += 1

        reward = float(self.reward_fn(prev, curr, ctx))
        self.episode_return += reward
        self.last_obs_dict = dict(curr)

        terminated = False
        truncated = bool(
            self.episode_steps >= self.max_episode_steps
            or self.no_progress_steps >= self.no_progress_limit
        )
        obs = self._obs_vector(curr)

        info: Dict[str, object] = {
            "action_button": button,
            "visited_count": len(self.visited_tiles),
            "map_id": int(curr.get("map_id", 0)),
            "x": int(curr.get("x", 0)),
            "y": int(curr.get("y", 0)),
        }
        if terminated or truncated:
            summary = self._episode_summary()
            self.last_episode_summary = summary
            self.completed_episodes.append(summary)
            info["episode_summary"] = summary

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.emu.stop()
