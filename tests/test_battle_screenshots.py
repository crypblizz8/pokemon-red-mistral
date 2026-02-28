from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run as run_module
from configs import (
    DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES,
    DEFAULT_BATTLE_SCREENSHOTS_DIR,
    DEFAULT_BATTLE_SCREENSHOTS_ENABLED,
)
from pokemon.battle_agent import MistralBattleAgent
from pokemon.battle_screenshots import BattleScreenshotConfig, BattleScreenshotRecorder
from run import build_parser, check_dependencies, run_battle_episode, run_battle_loop, save_battle_results


class _FakeScreen:
    def __init__(self) -> None:
        self.image = Image.new("RGBA", (160, 144), (120, 80, 40, 255))


class _FakePyBoy:
    def __init__(self) -> None:
        self.screen = _FakeScreen()


class _FakeBattleEmulator:
    def __init__(self, *, start_battle: bool, resolve_on_wait: bool) -> None:
        self.start_battle = bool(start_battle)
        self.resolve_on_wait = bool(resolve_on_wait)
        self.pyboy = _FakePyBoy()
        self._in_battle = False
        self.turns = 0
        self.player_hp = 20
        self.enemy_hp = 15

    def reset(self, state_path: Path | None = None) -> None:
        del state_path
        self._in_battle = False
        self.turns = 0
        self.player_hp = 20
        self.enemy_hp = 15

    def wait_for_battle(self, timeout: int = 120) -> bool:
        del timeout
        if not self.start_battle:
            return False
        self._in_battle = True
        return True

    def seek_battle(self, max_steps: int = 120) -> bool:
        del max_steps
        if not self.start_battle:
            return False
        self._in_battle = True
        return True

    def in_battle(self) -> bool:
        return bool(self._in_battle)

    def execute_move(self, slot: int) -> bool:
        del slot
        if not self._in_battle:
            return False
        self.turns += 1
        self.enemy_hp = max(0, self.enemy_hp - 5)
        return True

    def wait_for_battle_end(self, timeout: int = 1200) -> bool:
        del timeout
        if not self._in_battle:
            return True
        if self.resolve_on_wait and self.turns >= 1:
            self._in_battle = False
            return True
        return False

    def get_battle_state(self) -> dict:
        return {
            "in_battle": int(self._in_battle),
            "player_hp": int(self.player_hp),
            "player_max_hp": 20,
            "player_level": 8,
            "player_species": 4,
            "enemy_hp": int(self.enemy_hp),
            "enemy_level": 4,
            "enemy_species": 19,
            "move_ids": [10, 45, 52, 0],
            "move_pps": [10, 10, 10, 0],
            "legal_slots": [0, 1, 2],
        }

    def build_phase2_state(self, turn: int) -> dict:
        base = self.get_battle_state()
        return {
            "turn": int(turn),
            "in_battle": int(base["in_battle"]),
            "player_hp": int(base["player_hp"]),
            "player_max_hp": 20,
            "player_level": 8,
            "enemy_hp": int(base["enemy_hp"]),
            "enemy_level": 4,
            "player_species_id": 4,
            "enemy_species_id": 19,
            "move_ids": [10, 45, 52, 0],
            "move_pps": [10, 10, 10, 0],
            "legal_slots": [0, 1, 2],
            "moves": [
                {"slot": 0, "power": 40, "effectiveness": 1.0},
                {"slot": 1, "power": 0, "effectiveness": 1.0},
                {"slot": 2, "power": 40, "effectiveness": 1.0},
                {"slot": 3, "power": 0, "effectiveness": 1.0},
            ],
        }


class BattleScreenshotRecorderTests(unittest.TestCase):
    def test_recorder_uses_deterministic_names_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screenshots"
            recorder = BattleScreenshotRecorder(
                BattleScreenshotConfig(enabled=True, root_dir=root, retain_battles=25, phase="phase1")
            )
            emu = _FakeBattleEmulator(start_battle=True, resolve_on_wait=True)

            recorder.start_episode(episode=1, state_path=Path("a.state"), state_index=0, policy_mode="heuristic", model="m")
            recorder.capture_event("battle_start", emu, state={"in_battle": 1, "player_hp": 20, "enemy_hp": 15})
            recorder.capture_event(
                "turn_post_action",
                emu,
                turn=1,
                chosen_slot=0,
                state={"in_battle": 1, "player_hp": 19, "enemy_hp": 10},
            )
            recorder.capture_event("battle_end", emu, state={"in_battle": 0, "player_hp": 19, "enemy_hp": 0})
            recorder.finish_episode(outcome="win", turns=1, hp_left=19, reward=99.0)

            recorder.start_episode(episode=2, state_path=Path("b.state"), state_index=1, policy_mode="heuristic", model="m")
            recorder.capture_event("battle_start", emu, state={"in_battle": 1, "player_hp": 20, "enemy_hp": 15})
            recorder.capture_event("battle_end", emu, state={"in_battle": 0, "player_hp": 20, "enemy_hp": 0})
            recorder.finish_episode(outcome="win", turns=1, hp_left=20, reward=100.0)

            battle_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
            self.assertEqual(len(battle_dirs), 2)
            self.assertEqual(battle_dirs[0].name, "battle_000001_ep001_phase1")
            self.assertEqual(battle_dirs[1].name, "battle_000002_ep002_phase1")
            self.assertTrue((battle_dirs[0] / "000_battle_start.png").exists())
            self.assertTrue((battle_dirs[0] / "001_turn_001_post_action.png").exists())
            self.assertTrue((battle_dirs[0] / "999_battle_end.png").exists())

            manifest = json.loads((battle_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
            for key in {
                "battle_id",
                "phase",
                "episode",
                "state_path",
                "state_index",
                "policy_mode",
                "model",
                "outcome",
                "turns",
                "hp_left",
                "reward",
                "started_at",
                "ended_at",
                "frames",
            }:
                self.assertIn(key, manifest)
            self.assertEqual(manifest["frames"][0]["event"], "battle_start")
            self.assertEqual(manifest["frames"][-1]["event"], "battle_end")

    def test_retention_and_index_keep_latest_battles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screenshots"
            recorder = BattleScreenshotRecorder(
                BattleScreenshotConfig(enabled=True, root_dir=root, retain_battles=25, phase="phase2")
            )
            emu = _FakeBattleEmulator(start_battle=True, resolve_on_wait=True)

            for episode in range(1, 31):
                recorder.start_episode(
                    episode=episode,
                    state_path=Path(f"state_{episode}.state"),
                    state_index=episode - 1,
                    policy_mode="heuristic",
                    model="m",
                )
                recorder.capture_event("battle_start", emu, state={"in_battle": 1, "player_hp": 20, "enemy_hp": 15})
                recorder.capture_event("battle_end", emu, state={"in_battle": 0, "player_hp": 20, "enemy_hp": 0})
                recorder.finish_episode(outcome="win", turns=1, hp_left=20, reward=100.0)

            battle_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
            self.assertEqual(len(battle_dirs), 25)
            ids = [int(re.match(r"^battle_(\d+)_", p.name).group(1)) for p in battle_dirs]
            self.assertEqual(ids[0], 6)
            self.assertEqual(ids[-1], 30)

            lines = (root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 25)
            first = json.loads(lines[0])
            self.assertEqual(first["battle_id"], 6)


class BattleScreenshotIntegrationTests(unittest.TestCase):
    def _make_recorder(self, root: Path, phase: str) -> BattleScreenshotRecorder:
        return BattleScreenshotRecorder(
            BattleScreenshotConfig(enabled=True, root_dir=root, retain_battles=25, phase=phase)
        )

    def test_phase1_smoke_captures_start_turn_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screens"
            emu = _FakeBattleEmulator(start_battle=True, resolve_on_wait=True)
            agent = MistralBattleAgent(api_key="dummy", model="mistral-large-latest", policy_mode="heuristic")

            row, _ = run_battle_episode(
                emu=emu,
                agent=agent,
                screenshot_recorder=self._make_recorder(root, "phase1"),
                state_path=Path("battle.state"),
                state_index=0,
                episode=1,
                max_turns=3,
                turn_tick_budget=20,
                battle_wait_ticks=20,
                battle_search_steps=20,
                exploration_epsilon=0.0,
                phase2=False,
                llm_turn_interval=3,
                max_decision_calls=10,
            )
            self.assertEqual(row["outcome"], "win")
            battle_dir = [p for p in root.iterdir() if p.is_dir()][0]
            self.assertTrue((battle_dir / "000_battle_start.png").exists())
            self.assertTrue((battle_dir / "001_turn_001_post_action.png").exists())
            self.assertTrue((battle_dir / "999_battle_end.png").exists())

    def test_phase2_smoke_records_and_saves_results_with_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screens"
            results_path = Path(tmpdir) / "battle_results.json"
            emu = _FakeBattleEmulator(start_battle=True, resolve_on_wait=True)
            agent = MistralBattleAgent(api_key="dummy", model="mistral-large-latest", policy_mode="heuristic")

            row, _ = run_battle_episode(
                emu=emu,
                agent=agent,
                screenshot_recorder=self._make_recorder(root, "phase2"),
                state_path=Path("battle.state"),
                state_index=0,
                episode=1,
                max_turns=3,
                turn_tick_budget=20,
                battle_wait_ticks=20,
                battle_search_steps=20,
                exploration_epsilon=0.0,
                phase2=True,
                llm_turn_interval=3,
                max_decision_calls=10,
            )
            self.assertEqual(row["outcome"], "win")
            save_battle_results(results_path, "mistral-large-latest", agent, phase="phase2")
            saved = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["episodes"]), 1)
            battle_dir = [p for p in root.iterdir() if p.is_dir()][0]
            self.assertTrue((battle_dir / "manifest.json").exists())

    def test_timeout_after_battle_start_still_captures_battle_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screens"
            emu = _FakeBattleEmulator(start_battle=True, resolve_on_wait=False)
            agent = MistralBattleAgent(api_key="dummy", model="mistral-large-latest", policy_mode="heuristic")

            row, _ = run_battle_episode(
                emu=emu,
                agent=agent,
                screenshot_recorder=self._make_recorder(root, "phase1"),
                state_path=Path("battle.state"),
                state_index=0,
                episode=1,
                max_turns=1,
                turn_tick_budget=1,
                battle_wait_ticks=20,
                battle_search_steps=20,
                exploration_epsilon=0.0,
                phase2=False,
                llm_turn_interval=3,
                max_decision_calls=10,
            )
            self.assertEqual(row["outcome"], "timeout")
            battle_dir = [p for p in root.iterdir() if p.is_dir()][0]
            self.assertTrue((battle_dir / "999_battle_end.png").exists())
            manifest = json.loads((battle_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["outcome"], "timeout")

    def test_non_battle_episode_creates_no_screenshot_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "screens"
            emu = _FakeBattleEmulator(start_battle=False, resolve_on_wait=False)
            agent = MistralBattleAgent(api_key="dummy", model="mistral-large-latest", policy_mode="heuristic")

            row, _ = run_battle_episode(
                emu=emu,
                agent=agent,
                screenshot_recorder=self._make_recorder(root, "phase2"),
                state_path=Path("battle.state"),
                state_index=0,
                episode=1,
                max_turns=1,
                turn_tick_budget=1,
                battle_wait_ticks=1,
                battle_search_steps=1,
                exploration_epsilon=0.0,
                phase2=True,
                llm_turn_interval=3,
                max_decision_calls=10,
            )
            self.assertEqual(row["outcome"], "timeout")
            if root.exists():
                dirs = [p for p in root.iterdir() if p.is_dir()]
                self.assertEqual(dirs, [])


class BattleScreenshotCliTests(unittest.TestCase):
    def test_parser_defaults_include_screenshot_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.battle_screenshots, DEFAULT_BATTLE_SCREENSHOTS_ENABLED)
        self.assertEqual(args.battle_screenshots_dir, DEFAULT_BATTLE_SCREENSHOTS_DIR)
        self.assertEqual(
            args.battle_screenshot_retain_battles,
            DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES,
        )

    def test_invalid_retain_value_fails_early(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--phase1", "--battle-screenshot-retain-battles", "0"])
        output = io.StringIO()
        with redirect_stdout(output):
            rc = run_battle_loop(args, phase="phase1")
        self.assertEqual(rc, 1)
        self.assertIn("battle_screenshot_retain_battles must be >= 1", output.getvalue())

    def test_dependency_check_reports_missing_pillow(self) -> None:
        original_find_spec = run_module.importlib.util.find_spec

        def fake_find_spec(name: str):
            if name == "PIL":
                return None
            return original_find_spec(name)

        with patch("run.importlib.util.find_spec", side_effect=fake_find_spec):
            result = check_dependencies()
        self.assertFalse(result.passed)
        self.assertIn("pillow", result.message.lower())


if __name__ == "__main__":
    unittest.main()
