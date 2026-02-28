from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pokemon.route_executor import RouteValidationError, load_route_script, run_phase4_route


class FakeAgent:
    def __init__(self) -> None:
        self.llm_decision_calls = 0
        self.llm_reflection_calls = 0
        self.legality_fallback_count = 0
        self.budget_fallback_count = 0
        self.pick_calls = 0

    def start_episode_context(self, state_label: str, state_index: int) -> None:
        del state_label, state_index

    def record_turn_decision(self, state: dict, chosen_slot: int) -> None:
        del state, chosen_slot

    def pick_move(self, state: dict, use_llm: bool = True, budget_fallback: bool = False) -> int:
        del state, use_llm, budget_fallback
        self.pick_calls += 1
        return 0


class FakeEmulator:
    def __init__(
        self,
        battle_flag_on_first_move: int = 0,
        *,
        battle_triggers: int = 1,
        party_species_ids: list[int] | None = None,
        active_species_id: int | None = None,
        active_level: int = 16,
        player_hp: int = 20,
        player_max_hp: int = 20,
    ) -> None:
        self._battle_flag_on_first_move = int(battle_flag_on_first_move)
        self._initial_battle_triggers = (
            max(0, int(battle_triggers)) if self._battle_flag_on_first_move != 0 else 0
        )
        self._battle_triggers_remaining = int(self._initial_battle_triggers)
        self._battle_triggered = False
        self._battle_flag = 0
        self.run_attempts = 0
        self.party_species_ids = [int(v) for v in (party_species_ids or [176])]
        self.active_species_id = (
            int(active_species_id)
            if active_species_id is not None
            else int(self.party_species_ids[0] if self.party_species_ids else 0)
        )
        self.active_level = int(active_level)
        self.player_hp = int(player_hp)
        self.player_max_hp = max(1, int(player_max_hp))
        self.state = {
            "x": 0,
            "y": 0,
            "map_id": 12,
            "badges": 0,
            "in_battle": 0,
        }

    def reset(self, state_path: Path | None = None) -> None:
        del state_path
        self._battle_triggered = False
        self._battle_flag = 0
        self._battle_triggers_remaining = int(self._initial_battle_triggers)
        self.run_attempts = 0
        self.state.update({"x": 0, "y": 0, "map_id": 12, "badges": 0, "in_battle": 0})

    def get_nav_state(self) -> dict:
        return dict(self.state)

    def in_battle(self) -> bool:
        return int(self.state.get("in_battle", 0)) != 0

    def battle_flag(self) -> int:
        return int(self._battle_flag) if self.in_battle() else 0

    def is_trainer_battle(self) -> bool:
        return self.battle_flag() == 2

    def has_badge(self, badge_bit: int = 0) -> bool:
        return (int(self.state.get("badges", 0)) & (1 << int(badge_bit))) != 0

    def get_party_snapshot(self) -> dict:
        levels = [int(self.active_level) for _ in self.party_species_ids]
        return {
            "party_species_ids": [int(v) for v in self.party_species_ids],
            "party_levels": levels,
            "active_species_id": int(self.active_species_id),
            "active_level": int(self.active_level),
            "party_count": len(self.party_species_ids),
        }

    def validate_single_species(self, required_species_id: int) -> tuple[bool, str, dict]:
        required = int(required_species_id)
        snap = self.get_party_snapshot()
        party_count = int(snap["party_count"])
        if party_count != 1:
            return False, f"party_size_{party_count}", snap
        if int(snap["active_species_id"]) != required:
            return False, f"active_species_{snap['active_species_id']}_expected_{required}", snap
        if int(snap["party_species_ids"][0]) != required:
            return False, f"party_species_mismatch_expected_{required}", snap
        return True, "ok", snap

    def press(self, button: str, frames: int = 6) -> None:
        del frames
        if self.in_battle():
            return
        if button == "right":
            if self._battle_flag_on_first_move != 0 and self._battle_triggers_remaining > 0:
                self._battle_triggered = True
                self._battle_triggers_remaining -= 1
                self._battle_flag = self._battle_flag_on_first_move
                self.state["in_battle"] = self._battle_flag
                return
            self.state["x"] = int(self.state.get("x", 0)) + 1
            return
        if button == "up":
            self.state["y"] = int(self.state.get("y", 0)) - 1
            return
        if button == "left":
            self.state["x"] = int(self.state.get("x", 0)) - 1
            return
        if button == "down":
            self.state["y"] = int(self.state.get("y", 0)) + 1

    def tick(self, frames: int = 1) -> None:
        del frames

    def attempt_run(self, timeout_ticks: int = 220) -> bool:
        del timeout_ticks
        self.run_attempts += 1
        if self.in_battle() and not self.is_trainer_battle():
            self.state["in_battle"] = 0
            self._battle_flag = 0
            return True
        return False

    def build_phase2_state(self, turn: int) -> dict:
        return {
            "turn": int(turn),
            "player_hp": int(self.player_hp),
            "player_max_hp": int(self.player_max_hp),
            "legal_slots": [0],
            "move_ids": [10, 0, 0, 0],
            "move_pps": [10, 0, 0, 0],
            "moves": [{"slot": 0, "power": 40, "effectiveness": 1.0}],
        }

    def get_battle_state(self) -> dict:
        return {
            "player_hp": int(self.player_hp),
            "player_max_hp": int(self.player_max_hp),
            "enemy_hp": 12,
            "enemy_level": 4,
            "enemy_species": 153,
            "move_ids": [10, 0, 0, 0],
            "move_pps": [10, 0, 0, 0],
            "legal_slots": [0],
        }

    def execute_move(self, slot: int) -> bool:
        del slot
        if self.in_battle():
            self.state["in_battle"] = 0
            self._battle_flag = 0
        return True

    def wait_for_battle_end(self, timeout: int = 1200) -> bool:
        del timeout
        return not self.in_battle()

    def stop(self) -> None:
        return None


class StuckEmulator(FakeEmulator):
    def __init__(self) -> None:
        super().__init__(battle_flag_on_first_move=0)

    def press(self, button: str, frames: int = 6) -> None:
        del button, frames
        # no movement
        return


class CornerPocketEmulator(FakeEmulator):
    def __init__(self) -> None:
        super().__init__(battle_flag_on_first_move=0)

    def press(self, button: str, frames: int = 6) -> None:
        del frames
        if self.in_battle():
            return
        x = int(self.state.get("x", 0))
        y = int(self.state.get("y", 0))
        if button == "up":
            self.state["y"] = max(0, y - 1)
            return
        if button == "down":
            self.state["y"] = min(1, y + 1)
            return
        if button in {"left", "right"}:
            # Horizontal edge is blocked, so requested forward movement cannot progress.
            self.state["x"] = x
            return


class TransitionThenStallEmulator(FakeEmulator):
    def __init__(self) -> None:
        super().__init__(battle_flag_on_first_move=0)

    def reset(self, state_path: Path | None = None) -> None:
        super().reset(state_path)
        self.state.update({"x": 10, "y": 0, "map_id": 12, "badges": 0, "in_battle": 0})

    def press(self, button: str, frames: int = 6) -> None:
        del frames
        if self.in_battle():
            return
        x = int(self.state.get("x", 0))
        if button == "up" and int(self.state.get("map_id", 0)) == 12:
            # Simulate crossing Route 1 -> Viridian then stalling in-city.
            self.state.update({"map_id": 1, "x": 21, "y": 35})
            return
        if button == "left" and int(self.state.get("map_id", 0)) == 1:
            self.state["x"] = max(34, x - 1)
            return
        if button == "right" and int(self.state.get("map_id", 0)) == 1:
            self.state["x"] = min(35, x + 1)
            return


class WaypointGridEmulator(FakeEmulator):
    def __init__(self) -> None:
        super().__init__(battle_flag_on_first_move=0)

    def reset(self, state_path: Path | None = None) -> None:
        super().reset(state_path)
        self.state.update({"x": 0, "y": 0, "map_id": 12, "badges": 0, "in_battle": 0})


class ForestCorridorEmulator(FakeEmulator):
    def __init__(self) -> None:
        super().__init__(battle_flag_on_first_move=0)

    def reset(self, state_path: Path | None = None) -> None:
        super().reset(state_path)
        self.state.update({"x": 4, "y": 30, "map_id": 51, "badges": 0, "in_battle": 0})

    def press(self, button: str, frames: int = 6) -> None:
        del frames
        if self.in_battle():
            return
        map_id = int(self.state.get("map_id", 0))
        x = int(self.state.get("x", 0))
        y = int(self.state.get("y", 0))
        if map_id == 51:
            if button == "up" and (x, y) == (8, 30):
                self.state.update({"map_id": 2, "x": 5, "y": 5})
                return
            if button == "right" and y == 30 and x < 8:
                self.state["x"] = x + 1
                return
            if button == "left" and y == 30 and x > 4:
                self.state["x"] = x - 1
                return
            return
        super().press(button, frames=1)


def _route_payload() -> dict:
    return {
        "name": "test_route",
        "targets": {"gym_entrance": {"map_id": 12, "x": 1}},
        "steps": [
            {"type": "checkpoint", "name": "route_start", "expected_map_id": 12},
            {"type": "move", "direction": "right", "steps": 1, "hold_frames": 1},
        ],
    }


class RouteExecutorTests(unittest.TestCase):
    def test_route_validation_fails_on_unknown_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            route_path.write_text(
                json.dumps({"steps": [{"type": "teleport"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(RouteValidationError):
                load_route_script(route_path)

    def test_route_validation_accepts_waypoint_and_traverse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "route_schema_ok",
                        "steps": [
                            {"type": "checkpoint", "name": "start", "expected_map_id": 12},
                            {"type": "waypoint", "map_id": 12, "x": 3, "y": 2},
                            {"type": "traverse_until_map", "target_map_id": 2},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            route = load_route_script(route_path)
            self.assertEqual(str(route["steps"][1]["type"]), "waypoint")
            self.assertEqual(str(route["steps"][2]["type"]), "traverse_until_map")

    def test_waypoint_step_reaches_target_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "waypoint_route",
                        "targets": {"gym_entrance": {"map_id": 12, "x": 3, "y": 2}},
                        "steps": [
                            {"type": "checkpoint", "name": "route_start", "expected_map_id": 12},
                            {
                                "type": "waypoint",
                                "map_id": 12,
                                "x": 3,
                                "y": 2,
                                "radius": 0,
                                "max_seek_steps": 200,
                                "hold_frames": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            emu = WaypointGridEmulator()
            agent = FakeAgent()
            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=400,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
            )
            self.assertEqual(results["run_status"], "success")
            self.assertTrue(results["target_reached"])

    def test_traverse_until_map_step_exits_to_target_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "forest_traverse_route",
                        "targets": {"gym_entrance": {"map_id": 2}},
                        "steps": [
                            {"type": "checkpoint", "name": "forest_start", "expected_map_id": 51},
                            {
                                "type": "traverse_until_map",
                                "target_map_id": 2,
                                "mode": "wall_follow_ccw",
                                "max_steps": 600,
                                "hold_frames": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            emu = ForestCorridorEmulator()
            agent = FakeAgent()
            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=1200,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
            )
            self.assertEqual(results["run_status"], "success")
            self.assertTrue(results["target_reached"])

    def test_wild_battle_attempts_run_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(battle_flag_on_first_move=1)
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="hybrid",
                target="gym_entrance",
                wild_run_enabled=True,
            )

            self.assertEqual(results["run_status"], "success")
            self.assertTrue(results["target_reached"])
            self.assertEqual(results["wild_battles"], 1)
            self.assertEqual(results["wild_run_attempts"], 1)
            self.assertEqual(results["wild_run_successes"], 1)
            self.assertEqual(emu.run_attempts, 1)
            self.assertEqual(agent.pick_calls, 0)

    def test_trainer_battle_uses_battle_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(battle_flag_on_first_move=2)
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="hybrid",
                target="gym_entrance",
                wild_run_enabled=True,
            )

            self.assertEqual(results["run_status"], "success")
            self.assertEqual(results["trainer_battles"], 1)
            self.assertEqual(results["wild_run_attempts"], 0)
            self.assertGreaterEqual(agent.pick_calls, 1)

    def test_hp_gated_farm_fights_when_hp_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=1,
                battle_triggers=1,
                player_hp=18,
                player_max_hp=20,
            )
            agent = FakeAgent()
            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=240,
                policy_mode="hybrid",
                target="gym_entrance",
                wild_run_enabled=True,
                wild_battle_mode="hp_gated_farm",
                farm_hp_threshold=0.45,
                farm_max_consecutive_fights=3,
            )
            self.assertEqual(results["run_status"], "success")
            self.assertGreaterEqual(int(results["wild_fights_committed"]), 1)
            self.assertEqual(int(results["wild_run_attempts"]), 0)
            self.assertEqual(emu.run_attempts, 0)
            self.assertGreaterEqual(agent.pick_calls, 1)

    def test_hp_gated_farm_runs_when_hp_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=1,
                battle_triggers=1,
                player_hp=5,
                player_max_hp=20,
            )
            agent = FakeAgent()
            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=240,
                policy_mode="hybrid",
                target="gym_entrance",
                wild_run_enabled=True,
                wild_battle_mode="hp_gated_farm",
                farm_hp_threshold=0.45,
                farm_max_consecutive_fights=3,
            )
            self.assertEqual(results["run_status"], "success")
            self.assertGreaterEqual(int(results["wild_run_attempts"]), 1)
            self.assertEqual(int(results["wild_fights_committed"]), 0)
            self.assertEqual(agent.pick_calls, 0)
            self.assertIn('"event": "wild_mode_decision"', timeline_path.read_text(encoding="utf-8"))

    def test_hp_gated_farm_consecutive_fight_cap_forces_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=1,
                battle_triggers=2,
                player_hp=18,
                player_max_hp=20,
            )
            agent = FakeAgent()
            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=320,
                policy_mode="hybrid",
                target="gym_entrance",
                wild_run_enabled=True,
                wild_battle_mode="hp_gated_farm",
                farm_hp_threshold=0.45,
                farm_max_consecutive_fights=1,
            )
            self.assertEqual(results["run_status"], "success")
            self.assertEqual(int(results["wild_fights_committed"]), 1)
            self.assertGreaterEqual(int(results["wild_run_attempts"]), 1)
            self.assertGreaterEqual(int(results["wild_runs_forced"]), 1)

    def test_recovery_exhaustion_fails_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "stuck_route",
                        "targets": {"gym_entrance": {"map_id": 12, "x": 5}},
                        "steps": [
                            {"type": "checkpoint", "name": "route_start", "expected_map_id": 12},
                            {"type": "move", "direction": "up", "steps": 2, "hold_frames": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            emu = StuckEmulator()
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                no_progress_limit=1,
            )

            self.assertEqual(results["run_status"], "failed")
            self.assertEqual(results["failure_reason"], "recovery_exhausted")
            self.assertGreaterEqual(int(results["recovery_events"]), 2)

    def test_corner_pocket_sideways_motion_triggers_recovery_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "corner_pocket_route",
                        "targets": {"gym_entrance": {"map_id": 12, "x": 5}},
                        "steps": [
                            {"type": "checkpoint", "name": "route_start", "expected_map_id": 12},
                            {"type": "move", "direction": "right", "steps": 2, "hold_frames": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            emu = CornerPocketEmulator()
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=8000,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                no_progress_limit=9999,
            )

            self.assertEqual(results["run_status"], "failed")
            self.assertEqual(results["failure_reason"], "recovery_exhausted")
            self.assertGreaterEqual(int(results["recovery_events"]), 2)

    def test_move_step_exits_when_next_checkpoint_map_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(
                json.dumps(
                    {
                        "name": "transition_then_stall_route",
                        "targets": {"gym_entrance": {"map_id": 1}},
                        "steps": [
                            {"type": "checkpoint", "name": "route_start", "expected_map_id": 12},
                            {"type": "move", "direction": "up", "steps": 80, "hold_frames": 1},
                            {"type": "checkpoint", "name": "viridian", "expected_map_id": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            emu = TransitionThenStallEmulator()
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=500,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                no_progress_limit=9999,
            )

            self.assertEqual(results["run_status"], "success")
            self.assertTrue(results["target_reached"])
            self.assertLess(int(results["steps_executed"]), 120)

    def test_party_constraint_fails_on_multi_party(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=0,
                party_species_ids=[176, 165],
                active_species_id=176,
            )
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                required_species_id=176,
                single_pokemon_only=True,
                enforce_party_constraint=True,
            )

            self.assertEqual(results["run_status"], "failed")
            self.assertEqual(results["failure_reason"], "party_constraint_failed")
            self.assertFalse(bool(results.get("party_constraint_passed", True)))
            timeline_text = timeline_path.read_text(encoding="utf-8")
            self.assertIn('"event": "constraint_failed"', timeline_text)

    def test_party_constraint_fails_on_wrong_active_species(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=0,
                party_species_ids=[176],
                active_species_id=177,
            )
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                required_species_id=176,
                single_pokemon_only=True,
                enforce_party_constraint=True,
            )

            self.assertEqual(results["run_status"], "failed")
            self.assertEqual(results["failure_reason"], "party_constraint_failed")
            self.assertFalse(bool(results.get("party_constraint_passed", True)))

    def test_party_constraint_passes_for_solo_charmander(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_path = Path(tmpdir) / "route.json"
            timeline_path = Path(tmpdir) / "timeline.jsonl"
            route_path.write_text(json.dumps(_route_payload()), encoding="utf-8")
            emu = FakeEmulator(
                battle_flag_on_first_move=0,
                party_species_ids=[176],
                active_species_id=176,
                active_level=18,
            )
            agent = FakeAgent()

            results = run_phase4_route(
                emu=emu,
                agent=agent,
                start_state_path=Path("dummy.state"),
                route_script_path=route_path,
                timeline_path=timeline_path,
                max_steps=200,
                policy_mode="heuristic",
                target="gym_entrance",
                wild_run_enabled=False,
                required_species_id=176,
                single_pokemon_only=True,
                enforce_party_constraint=True,
            )

            self.assertEqual(results["run_status"], "success")
            self.assertTrue(results["target_reached"])
            self.assertTrue(bool(results.get("party_constraint_passed", False)))
            self.assertEqual(int(results.get("active_species_id_final", 0)), 176)


if __name__ == "__main__":
    unittest.main()
