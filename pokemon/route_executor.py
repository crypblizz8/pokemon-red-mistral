from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


VALID_BUTTONS = {"a", "b", "start", "select", "up", "down", "left", "right"}
VALID_DIRECTIONS = {"up", "down", "left", "right"}
VALID_WAIT_CONDITIONS = {"map_id_is", "not_in_battle", "in_battle"}
VALID_TARGETS = {"gym_entrance", "brock_badge"}
VALID_TRAVERSE_MODES = {"wall_follow_ccw"}
MAP_ID_ALIASES: Dict[int, set[int]] = {
    51: {37, 38},
}


class RouteValidationError(ValueError):
    pass


class RouteExecutionError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_field(row: Dict[str, object], key: str, *, min_value: int | None = None) -> int:
    if key not in row:
        raise RouteValidationError(f"missing required field: {key}")
    try:
        value = int(row[key])
    except Exception as exc:
        raise RouteValidationError(f"field {key} must be int") from exc
    if min_value is not None and value < min_value:
        raise RouteValidationError(f"field {key} must be >= {min_value}")
    return value


def _normalize_targets(raw: object) -> Dict[str, Dict[str, int]]:
    targets: Dict[str, Dict[str, int]] = {
        "gym_entrance": {"map_id": 2},
        "brock_badge": {"badge_bit": 0},
    }
    if not isinstance(raw, dict):
        return targets

    gym = raw.get("gym_entrance")
    if isinstance(gym, dict):
        map_id = int(gym.get("map_id", targets["gym_entrance"]["map_id"]))
        gym_out: Dict[str, int] = {"map_id": map_id}
        if "x" in gym:
            gym_out["x"] = int(gym["x"])
        if "y" in gym:
            gym_out["y"] = int(gym["y"])
        targets["gym_entrance"] = gym_out

    badge = raw.get("brock_badge")
    if isinstance(badge, dict):
        targets["brock_badge"] = {"badge_bit": int(badge.get("badge_bit", 0))}
    return targets


def _validate_step(raw: Dict[str, object], idx: int) -> Dict[str, object]:
    step_type = str(raw.get("type", "")).strip().lower()
    if not step_type:
        raise RouteValidationError(f"route step {idx}: missing required field: type")

    if step_type == "checkpoint":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise RouteValidationError(f"route step {idx}: checkpoint.name is required")
        expected_map_id = _int_field(raw, "expected_map_id", min_value=0)
        raw_allowed_map_ids = raw.get("allowed_map_ids", [])
        if raw_allowed_map_ids is None:
            raw_allowed_map_ids = []
        if not isinstance(raw_allowed_map_ids, list):
            raise RouteValidationError(
                f"route step {idx}: checkpoint.allowed_map_ids must be an array of ints"
            )
        allowed_map_ids: List[int] = []
        seen_allowed: set[int] = set()
        for raw_value in raw_allowed_map_ids:
            try:
                map_id = int(raw_value)
            except Exception as exc:
                raise RouteValidationError(
                    f"route step {idx}: checkpoint.allowed_map_ids must be ints"
                ) from exc
            if map_id < 0:
                raise RouteValidationError(
                    f"route step {idx}: checkpoint.allowed_map_ids must be >= 0"
                )
            if map_id == expected_map_id or map_id in seen_allowed:
                continue
            seen_allowed.add(map_id)
            allowed_map_ids.append(map_id)
        return {
            "type": "checkpoint",
            "name": name,
            "expected_map_id": expected_map_id,
            "allowed_map_ids": allowed_map_ids,
        }

    if step_type == "move":
        direction = str(raw.get("direction", "")).strip().lower()
        if direction not in VALID_DIRECTIONS:
            raise RouteValidationError(
                f"route step {idx}: invalid direction={direction!r} (allowed: {sorted(VALID_DIRECTIONS)})"
            )
        steps = _int_field(raw, "steps", min_value=1)
        hold_frames = int(raw.get("hold_frames", 6))
        if hold_frames < 1:
            raise RouteValidationError(f"route step {idx}: hold_frames must be >= 1")
        return {
            "type": "move",
            "direction": direction,
            "steps": steps,
            "hold_frames": hold_frames,
        }

    if step_type == "press":
        button = str(raw.get("button", "")).strip().lower()
        if button not in VALID_BUTTONS:
            raise RouteValidationError(
                f"route step {idx}: invalid button={button!r} (allowed: {sorted(VALID_BUTTONS)})"
            )
        count = _int_field(raw, "count", min_value=1)
        hold_frames = int(raw.get("hold_frames", 6))
        if hold_frames < 1:
            raise RouteValidationError(f"route step {idx}: hold_frames must be >= 1")
        return {
            "type": "press",
            "button": button,
            "count": count,
            "hold_frames": hold_frames,
        }

    if step_type == "wait_until":
        condition = str(raw.get("condition", "")).strip().lower()
        if condition not in VALID_WAIT_CONDITIONS:
            raise RouteValidationError(
                f"route step {idx}: invalid condition={condition!r} (allowed: {sorted(VALID_WAIT_CONDITIONS)})"
            )
        timeout_ticks = _int_field(raw, "timeout_ticks", min_value=1)
        out: Dict[str, object] = {
            "type": "wait_until",
            "condition": condition,
            "timeout_ticks": timeout_ticks,
        }
        if condition == "map_id_is":
            out["value"] = _int_field(raw, "value", min_value=0)
        return out

    if step_type == "interact":
        button = str(raw.get("button", "")).strip().lower()
        if button not in VALID_BUTTONS:
            raise RouteValidationError(
                f"route step {idx}: invalid button={button!r} (allowed: {sorted(VALID_BUTTONS)})"
            )
        attempts = _int_field(raw, "attempts", min_value=1)
        timeout_ticks = _int_field(raw, "timeout_ticks", min_value=1)
        hold_frames = int(raw.get("hold_frames", 6))
        if hold_frames < 1:
            raise RouteValidationError(f"route step {idx}: hold_frames must be >= 1")
        return {
            "type": "interact",
            "button": button,
            "attempts": attempts,
            "timeout_ticks": timeout_ticks,
            "hold_frames": hold_frames,
        }

    if step_type == "waypoint":
        map_id = _int_field(raw, "map_id", min_value=0)
        x = _int_field(raw, "x", min_value=0)
        y = _int_field(raw, "y", min_value=0)
        radius = int(raw.get("radius", 1))
        if radius < 0:
            raise RouteValidationError(f"route step {idx}: radius must be >= 0")
        max_seek_steps = int(raw.get("max_seek_steps", 1200))
        if max_seek_steps < 1:
            raise RouteValidationError(f"route step {idx}: max_seek_steps must be >= 1")
        hold_frames = int(raw.get("hold_frames", 6))
        if hold_frames < 1:
            raise RouteValidationError(f"route step {idx}: hold_frames must be >= 1")
        transition_preferred = bool(raw.get("transition_preferred", True))
        return {
            "type": "waypoint",
            "map_id": map_id,
            "x": x,
            "y": y,
            "radius": radius,
            "max_seek_steps": max_seek_steps,
            "hold_frames": hold_frames,
            "transition_preferred": transition_preferred,
        }

    if step_type == "traverse_until_map":
        target_map_id = _int_field(raw, "target_map_id", min_value=0)
        mode = str(raw.get("mode", "wall_follow_ccw")).strip().lower()
        if mode not in VALID_TRAVERSE_MODES:
            raise RouteValidationError(
                f"route step {idx}: invalid mode={mode!r} (allowed: {sorted(VALID_TRAVERSE_MODES)})"
            )
        max_steps = int(raw.get("max_steps", 7000))
        if max_steps < 1:
            raise RouteValidationError(f"route step {idx}: max_steps must be >= 1")
        hold_frames = int(raw.get("hold_frames", 6))
        if hold_frames < 1:
            raise RouteValidationError(f"route step {idx}: hold_frames must be >= 1")
        return {
            "type": "traverse_until_map",
            "target_map_id": target_map_id,
            "mode": mode,
            "max_steps": max_steps,
            "hold_frames": hold_frames,
        }

    raise RouteValidationError(f"route step {idx}: unknown step type={step_type!r}")


def load_route_script(path: Path) -> Dict[str, object]:
    route_path = Path(path)
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_steps = payload
        route_name = route_path.stem
        raw_targets: object = {}
    elif isinstance(payload, dict):
        raw_steps = payload.get("steps")
        route_name = str(payload.get("name", route_path.stem)).strip() or route_path.stem
        raw_targets = payload.get("targets", {})
    else:
        raise RouteValidationError(f"route script must be object/list: {route_path}")

    if not isinstance(raw_steps, list) or not raw_steps:
        raise RouteValidationError("route script must include non-empty steps array")

    steps: List[Dict[str, object]] = []
    for idx, row in enumerate(raw_steps):
        if not isinstance(row, dict):
            raise RouteValidationError(f"route step {idx}: each step must be an object")
        steps.append(_validate_step(row, idx))

    return {
        "name": route_name,
        "steps": steps,
        "targets": _normalize_targets(raw_targets),
    }


class _TimelineWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def emit(self, event: str, payload: Dict[str, object]) -> None:
        row = {"ts": _utc_now(), "event": event}
        row.update(payload)
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class _Runtime:
    def __init__(
        self,
        *,
        emu: object,
        agent: object,
        route: Dict[str, object],
        route_script_path: Path,
        start_state_path: Path,
        timeline: _TimelineWriter,
        max_steps: int,
        policy_mode: str,
        target: str,
        phase4_scope: str,
        wild_run_enabled: bool,
        wild_battle_mode: str,
        farm_hp_threshold: float,
        farm_max_consecutive_fights: int,
        no_progress_limit: int,
        llm_turn_interval: int,
        max_decision_calls: int,
        turn_tick_budget: int,
        max_battle_turns: int,
        required_species_id: int | None = None,
        single_pokemon_only: bool = False,
        enforce_party_constraint: bool = False,
    ) -> None:
        if target not in VALID_TARGETS:
            raise RouteExecutionError(f"invalid phase4 target: {target}")
        self.emu = emu
        self.agent = agent
        self.route = route
        self.route_script_path = Path(route_script_path)
        self.start_state_path = Path(start_state_path)
        self.timeline = timeline
        self.max_steps = max(1, int(max_steps))
        self.policy_mode = str(policy_mode)
        self.target = target
        raw_scope = str(phase4_scope).strip().lower()
        self.phase4_scope = raw_scope if raw_scope in {"route_only", "integrated"} else "integrated"
        self.wild_run_enabled = bool(wild_run_enabled)
        self.wild_battle_mode = str(wild_battle_mode).strip().lower() or "run_first"
        if self.wild_battle_mode not in {"run_first", "hp_gated_farm"}:
            self.wild_battle_mode = "run_first"
        self.farm_hp_threshold = min(1.0, max(0.0, float(farm_hp_threshold)))
        self.farm_max_consecutive_fights = max(1, int(farm_max_consecutive_fights))
        self.no_progress_limit = max(1, int(no_progress_limit))
        self.llm_turn_interval = max(1, int(llm_turn_interval))
        self.max_decision_calls = max(0, int(max_decision_calls))
        self.turn_tick_budget = max(1, int(turn_tick_budget))
        self.max_battle_turns = max(1, int(max_battle_turns))
        self.required_species_id = (
            int(required_species_id) if required_species_id is not None else None
        )
        self.single_pokemon_only = bool(single_pokemon_only)
        self.enforce_party_constraint = bool(enforce_party_constraint)

        self.steps: List[Dict[str, object]] = list(route.get("steps", []))
        self.targets: Dict[str, Dict[str, int]] = dict(route.get("targets", {}))
        self.steps_executed = 0
        self.checkpoints_reached = 0
        self.battles_fought = 0
        self.wild_battles = 0
        self.trainer_battles = 0
        self.wild_run_attempts = 0
        self.wild_run_successes = 0
        self.wild_fights_committed = 0
        self.wild_runs_forced = 0
        self.consecutive_wild_fights = 0
        self.recovery_events = 0
        self.current_block_start = 0
        self.last_checkpoint_step = -1
        self.current_checkpoint_name = "start"
        self.current_checkpoint_expected_map_id = -1
        self.current_checkpoint_allowed_map_ids: List[int] = []
        self.block_retry_used = False
        self.no_progress_steps = 0
        self.last_position: Tuple[int, int, int] | None = None
        self.last_observed_map_id: int | None = None
        self.observed_map_transitions: Dict[str, int] = {}
        self.forest_probe_edges: Dict[str, int] = {}
        self.forest_probe_phase_counts: Dict[str, int] = {}
        self.forest_probe_samples: List[Dict[str, object]] = []
        self.rng = random.Random(7)
        self.blocked_edges: set[Tuple[int, int, int, str]] = set()
        self.edge_fail_counts: Dict[Tuple[int, int, int, str], int] = {}
        self.recent_positions: List[Tuple[int, int, int]] = []
        self.party_constraint_passed = True
        self.party_constraint_failure_reason = ""
        self.last_party_snapshot: Dict[str, object] = {}
        self.battle_turns_total = 0
        self.route2_regression_count = 0
        self._suspend_route2_regression_checks = False
        self.traverse_attempt_counts: Dict[Tuple[str, int], int] = {}
        self.forest_script_recovery_count = 0
        self.forest_script_cycle_count = 0
        self.forest_script_variant = 0
        self.route2_lane_fail_counts: Dict[int, int] = {}
        self.route2_gate_x_fail_counts: Dict[int, int] = {}
        self.generic_recovery_count = 0
        self.pewter_waypoint_recovery_count = 0

    def _reset_wild_fight_streak(self) -> None:
        self.consecutive_wild_fights = 0

    def _record_map_transition(self, current_map_id: int) -> None:
        current = int(current_map_id)
        if self.last_observed_map_id is None:
            self.last_observed_map_id = current
            return
        previous = int(self.last_observed_map_id)
        if previous != current:
            edge = f"{previous}->{current}"
            self.observed_map_transitions[edge] = int(self.observed_map_transitions.get(edge, 0)) + 1
        self.last_observed_map_id = current

    def _record_forest_probe_action(
        self,
        *,
        button: str,
        phase: str,
        before: Tuple[int, int, int],
        after: Tuple[int, int, int],
    ) -> None:
        if not (
            self.phase4_scope == "route_only"
            and str(self.current_checkpoint_name)
            in {"Forest_To_Pewter", "Forest_Exit_Gate", "Viridian_Forest_Upper"}
        ):
            return
        phase_name = str(phase).strip() or "unknown"
        from_map = int(before[0])
        to_map = int(after[0])
        if phase_name == "unknown" and from_map == to_map:
            return
        self.forest_probe_phase_counts[phase_name] = int(
            self.forest_probe_phase_counts.get(phase_name, 0)
        ) + 1
        edge_key = f"{from_map}->{to_map}"
        self.forest_probe_edges[edge_key] = int(self.forest_probe_edges.get(edge_key, 0)) + 1
        if len(self.forest_probe_samples) >= 240:
            return
        self.forest_probe_samples.append(
            {
                "phase": phase_name,
                "button": str(button),
                "from_map": from_map,
                "from_x": int(before[1]),
                "from_y": int(before[2]),
                "to_map": to_map,
                "to_x": int(after[1]),
                "to_y": int(after[2]),
            }
        )

    def _nav_state(self) -> Dict[str, int]:
        raw = self.emu.get_nav_state()
        return {
            "x": int(raw.get("x", 0)),
            "y": int(raw.get("y", 0)),
            "map_id": int(raw.get("map_id", 0)),
            "badges": int(raw.get("badges", 0)),
            "in_battle": int(raw.get("in_battle", 0)),
        }

    def _position_key(self) -> Tuple[int, int, int]:
        nav = self._nav_state()
        return (int(nav["map_id"]), int(nav["x"]), int(nav["y"]))

    def _reset_nav_memory(self) -> None:
        self.blocked_edges.clear()
        self.edge_fail_counts.clear()
        self.recent_positions.clear()

    def _rotate_int_sequence(self, values: List[int], offset: int) -> List[int]:
        if not values:
            return []
        size = len(values)
        shift = int(offset) % size
        if shift == 0:
            return list(values)
        return list(values[shift:]) + list(values[:shift])

    def _bump_forest_variant(self, step: int = 1) -> None:
        self.forest_script_variant = (int(self.forest_script_variant) + max(1, int(step))) % 32

    def _edge_key(self, pos: Tuple[int, int, int], action: str) -> Tuple[int, int, int, str]:
        map_id, x, y = pos
        return (int(map_id), int(x), int(y), str(action))

    def _is_edge_blocked(self, pos: Tuple[int, int, int], action: str) -> bool:
        return self._edge_key(pos, action) in self.blocked_edges

    def _record_move_outcome(
        self,
        *,
        pos_before: Tuple[int, int, int],
        action: str,
        pos_after: Tuple[int, int, int],
    ) -> bool:
        edge = self._edge_key(pos_before, action)
        moved = pos_after != pos_before
        if moved:
            self.edge_fail_counts.pop(edge, None)
            self.blocked_edges.discard(edge)
            self.recent_positions.append(pos_after)
            if len(self.recent_positions) > 24:
                self.recent_positions = self.recent_positions[-24:]
            return True

        fail_count = int(self.edge_fail_counts.get(edge, 0)) + 1
        self.edge_fail_counts[edge] = fail_count
        if fail_count >= 5:
            self.blocked_edges.add(edge)
        return False

    def _target_reached(self) -> bool:
        if self.target == "brock_badge":
            badge_cfg = self.targets.get("brock_badge", {"badge_bit": 0})
            badge_bit = int(badge_cfg.get("badge_bit", 0))
            if hasattr(self.emu, "has_badge"):
                return bool(self.emu.has_badge(badge_bit))
            nav = self._nav_state()
            return (int(nav.get("badges", 0)) & (1 << badge_bit)) != 0

        gym_cfg = self.targets.get("gym_entrance", {"map_id": 2})
        nav = self._nav_state()
        if not self._map_matches(
            int(nav.get("map_id", -1)),
            int(gym_cfg.get("map_id", 2)),
        ):
            return False
        if "x" in gym_cfg and int(nav.get("x", -1)) != int(gym_cfg["x"]):
            return False
        if "y" in gym_cfg and int(nav.get("y", -1)) != int(gym_cfg["y"]):
            return False
        return True

    def _map_matches(self, actual_map_id: int, expected_map_id: int) -> bool:
        actual = int(actual_map_id)
        expected = int(expected_map_id)
        if actual == expected:
            return True
        aliases = MAP_ID_ALIASES.get(expected, set())
        return actual in aliases

    def _canonical_route2_map_id(self, map_id: int) -> int:
        value = int(map_id)
        if value in {37, 38}:
            return 51
        if value == 47:
            return 50
        return value

    def _is_route2_progress_checkpoint(self, checkpoint_name: str) -> bool:
        return str(checkpoint_name) in {
            "Route2",
            "Route2_GateHouse",
            "Viridian_Forest",
            "Forest_To_Pewter",
            "Forest_Exit_Gate",
            "Forest_Exit",
        }

    def _emit(self, event: str, **payload: object) -> None:
        nav = self._nav_state()
        base = {
            "map_id": int(nav.get("map_id", 0)),
            "x": int(nav.get("x", 0)),
            "y": int(nav.get("y", 0)),
            "in_battle": int(nav.get("in_battle", 0)),
            "steps_executed": int(self.steps_executed),
            "checkpoint": self.current_checkpoint_name,
        }
        base.update(payload)
        self.timeline.emit(event, base)

    def _checkpoint_map_allowed(self, map_id: int) -> bool:
        current_map = int(map_id)
        expected = int(self.current_checkpoint_expected_map_id)
        if expected >= 0 and self._map_matches(current_map, expected):
            return True
        for allowed_map_id in self.current_checkpoint_allowed_map_ids:
            if self._map_matches(current_map, int(allowed_map_id)):
                return True
        return False

    def _party_constraint_status(self) -> Dict[str, object]:
        if not self.enforce_party_constraint:
            return {"passed": True, "reason": "disabled", "snapshot": {}}

        snapshot: Dict[str, object] = {}
        if hasattr(self.emu, "get_party_snapshot"):
            try:
                raw = self.emu.get_party_snapshot()
            except Exception as exc:
                return {
                    "passed": False,
                    "reason": f"party_snapshot_error:{exc}",
                    "snapshot": {},
                }
            if isinstance(raw, dict):
                snapshot = dict(raw)
        else:
            return {
                "passed": False,
                "reason": "party_snapshot_unavailable",
                "snapshot": {},
            }

        # Fast path when emulator already exposes single-species validation.
        if (
            self.single_pokemon_only
            and self.required_species_id is not None
            and hasattr(self.emu, "validate_single_species")
        ):
            try:
                passed, reason, validated_snapshot = self.emu.validate_single_species(
                    int(self.required_species_id)
                )
            except Exception as exc:
                return {
                    "passed": False,
                    "reason": f"party_validation_error:{exc}",
                    "snapshot": snapshot,
                }
            if isinstance(validated_snapshot, dict):
                snapshot = dict(validated_snapshot)
            return {
                "passed": bool(passed),
                "reason": str(reason),
                "snapshot": snapshot,
            }

        party_count = int(snapshot.get("party_count", 0))
        party_species = [int(v) for v in snapshot.get("party_species_ids", [])]
        active_species_id = int(snapshot.get("active_species_id", 0))

        if self.single_pokemon_only and party_count != 1:
            return {
                "passed": False,
                "reason": f"party_size_{party_count}",
                "snapshot": snapshot,
            }
        if self.required_species_id is not None and active_species_id != int(self.required_species_id):
            return {
                "passed": False,
                "reason": (
                    f"active_species_{active_species_id}_expected_{int(self.required_species_id)}"
                ),
                "snapshot": snapshot,
            }
        if self.required_species_id is not None and party_species:
            if any(int(v) != int(self.required_species_id) for v in party_species):
                return {
                    "passed": False,
                    "reason": f"party_species_not_all_{int(self.required_species_id)}",
                    "snapshot": snapshot,
                }
        return {"passed": True, "reason": "ok", "snapshot": snapshot}

    def _validate_party_constraint(self, *, stage: str, step_index: int) -> None:
        if not self.enforce_party_constraint:
            return
        status = self._party_constraint_status()
        passed = bool(status.get("passed", False))
        reason = str(status.get("reason", "unknown"))
        snapshot = status.get("snapshot", {})
        if isinstance(snapshot, dict):
            self.last_party_snapshot = dict(snapshot)
        self._emit(
            "constraint_check",
            constraint_stage=stage,
            constraint_passed=passed,
            constraint_reason=reason,
            route_step_index=int(step_index),
            party_snapshot=self.last_party_snapshot,
        )
        if passed:
            return
        self.party_constraint_passed = False
        self.party_constraint_failure_reason = reason
        self._emit(
            "constraint_failed",
            constraint_stage=stage,
            constraint_reason=reason,
            route_step_index=int(step_index),
            party_snapshot=self.last_party_snapshot,
        )
        raise RouteExecutionError("party_constraint_failed")

    def _mark_checkpoint_reached(
        self,
        *,
        step_index: int,
        checkpoint_name: str,
        expected_map_id: int,
        allowed_map_ids: List[int] | None = None,
    ) -> None:
        self.checkpoints_reached += 1
        self.current_checkpoint_name = str(checkpoint_name)
        self.current_checkpoint_expected_map_id = int(expected_map_id)
        self.current_checkpoint_allowed_map_ids = [int(v) for v in (allowed_map_ids or [])]
        self.last_checkpoint_step = step_index
        self.current_block_start = step_index + 1
        self.block_retry_used = False
        self.generic_recovery_count = 0
        self.pewter_waypoint_recovery_count = 0
        self.no_progress_steps = 0
        self._reset_wild_fight_streak()
        self.last_position = self._position_key()
        self._reset_nav_memory()
        if self.current_checkpoint_name not in {
            "Forest_To_Pewter",
            "Forest_Exit_Gate",
            "Forest_Exit",
            "Viridian_Forest_Upper",
        }:
            self.forest_script_recovery_count = 0
            self.forest_script_cycle_count = 0
            self.forest_script_variant = 0
            self.route2_lane_fail_counts.clear()
            self.route2_gate_x_fail_counts.clear()
        if not self._is_route2_progress_checkpoint(self.current_checkpoint_name):
            self.route2_regression_count = 0
        self._emit(
            "checkpoint_reached",
            route_step_index=step_index,
            checkpoint_name=self.current_checkpoint_name,
            expected_map_id=expected_map_id,
            allowed_map_ids=self.current_checkpoint_allowed_map_ids,
        )

    def _consume_step(self, amount: int = 1) -> None:
        self.steps_executed += max(1, int(amount))
        if self.steps_executed >= self.max_steps:
            raise TimeoutError("max_steps_exceeded")

    def _battle_kind(self) -> str:
        if hasattr(self.emu, "is_trainer_battle"):
            return "trainer" if bool(self.emu.is_trainer_battle()) else "wild"
        if hasattr(self.emu, "battle_flag"):
            flag = int(self.emu.battle_flag())
            return "trainer" if flag == 2 else "wild"
        return "wild"

    def _wild_hp_ratio(self) -> float:
        state: Dict[str, object] = {}
        if hasattr(self.emu, "get_battle_state"):
            try:
                raw = self.emu.get_battle_state()
                if isinstance(raw, dict):
                    state = dict(raw)
            except Exception:
                state = {}
        hp_now = int(state.get("player_hp", 0))
        hp_max = int(state.get("player_max_hp", 0))
        if hp_max <= 0:
            return 1.0 if hp_now > 0 else 0.0
        return min(1.0, max(0.0, float(hp_now) / float(max(1, hp_max))))

    def _wild_battle_decision(self, *, step_index: int) -> str:
        mode = self.wild_battle_mode
        hp_ratio = self._wild_hp_ratio()
        decision = "fight"

        if self.wild_run_enabled:
            if mode == "run_first":
                decision = "run"
            elif mode == "hp_gated_farm":
                should_fight = (
                    hp_ratio >= self.farm_hp_threshold
                    and self.consecutive_wild_fights < self.farm_max_consecutive_fights
                )
                decision = "fight" if should_fight else "run"
                if decision == "run":
                    self.wild_runs_forced += 1

        self._emit(
            "wild_mode_decision",
            route_step_index=step_index,
            mode=mode,
            decision=decision,
            hp_pct=round(hp_ratio * 100.0, 2),
            consecutive_fights=int(self.consecutive_wild_fights),
        )
        return decision

    def _execute_recovery_actions(self, step_index: int) -> None:
        del step_index
        for button in ("b", "b", "b", "left", "right", "down", "up"):
            self.emu.press(button, frames=2)
            self._consume_step(1)
            if self.emu.in_battle():
                self._handle_battle_interrupt(step_index=-1)

    def _trigger_recovery(self, *, reason: str, step_index: int, **extra: object) -> int:
        if (
            str(reason) == "forest_to_pewter_script_failed"
            and str(self.current_checkpoint_name) in {"Forest_To_Pewter", "Forest_Exit_Gate", "Forest_Exit"}
        ):
            self.recovery_events += 1
            self.forest_script_recovery_count += 1
            attempt = int(self.forest_script_recovery_count)
            max_attempts = 8
            self._emit(
                "recovery_triggered",
                reason=reason,
                route_step_index=step_index,
                attempt=attempt,
                max_attempts=max_attempts,
                **extra,
            )
            self._execute_recovery_actions(step_index)
            if attempt >= max_attempts:
                raise RouteExecutionError("recovery_exhausted")
            self.block_retry_used = False
            self.no_progress_steps = 0
            self.generic_recovery_count = 0
            self._reset_wild_fight_streak()
            self.last_position = self._position_key()
            self._reset_nav_memory()
            self._bump_forest_variant(step=3)
            self.forest_script_cycle_count = 0
            return self.current_block_start

        if (
            str(reason) == "forest_to_pewter_script_cycle_limit"
            and str(self.current_checkpoint_name) in {"Forest_To_Pewter", "Forest_Exit_Gate", "Forest_Exit"}
        ):
            self._bump_forest_variant(step=2)

        if (
            self.phase4_scope == "route_only"
            and str(self.current_checkpoint_name) == "Pewter_City"
            and str(reason) in {"waypoint_stuck", "waypoint_timeout", "waypoint_map_lost"}
        ):
            self.recovery_events += 1
            self.pewter_waypoint_recovery_count += 1
            attempt = int(self.pewter_waypoint_recovery_count)
            max_attempts = 10
            self._emit(
                "recovery_triggered",
                reason=reason,
                route_step_index=step_index,
                attempt=attempt,
                max_attempts=max_attempts,
                **extra,
            )
            self._execute_recovery_actions(step_index)
            if attempt >= max_attempts:
                raise RouteExecutionError("recovery_exhausted")
            self.block_retry_used = True
            self.no_progress_steps = 0
            self._reset_wild_fight_streak()
            self.last_position = self._position_key()
            self._reset_nav_memory()
            return self.current_block_start

        extended_recovery = (
            self.phase4_scope == "integrated"
            and str(self.current_checkpoint_name)
            in {"Route2", "Route2_GateHouse", "Viridian_Forest", "Forest_To_Pewter", "Forest_Exit_Gate"}
            and str(reason)
            in {
                "traverse_timeout",
                "waypoint_map_mismatch",
                "waypoint_map_lost",
                "traverse_loop_stuck",
                "traverse_gatehouse_map_lost",
            }
        )
        if extended_recovery:
            self.recovery_events += 1
            self.generic_recovery_count += 1
            attempt = int(self.generic_recovery_count)
            max_attempts = 4
            self._emit(
                "recovery_triggered",
                reason=reason,
                route_step_index=step_index,
                attempt=attempt,
                max_attempts=max_attempts,
                **extra,
            )
            self._execute_recovery_actions(step_index)
            if attempt >= max_attempts:
                raise RouteExecutionError("recovery_exhausted")
            self.block_retry_used = True
            self.no_progress_steps = 0
            self._reset_wild_fight_streak()
            self.last_position = self._position_key()
            self._reset_nav_memory()
            self.forest_script_cycle_count = 0
            return self.current_block_start

        self.recovery_events += 1
        attempt = 2 if self.block_retry_used else 1
        self._emit(
            "recovery_triggered",
            reason=reason,
            route_step_index=step_index,
            attempt=attempt,
            **extra,
        )
        self._execute_recovery_actions(step_index)
        if self.block_retry_used:
            raise RouteExecutionError("recovery_exhausted")
        self.block_retry_used = True
        self.no_progress_steps = 0
        self._reset_wild_fight_streak()
        self.last_position = self._position_key()
        self._reset_nav_memory()
        self.forest_script_cycle_count = 0
        return self.current_block_start

    def _handle_route2_map_regression(self, *, step_index: int, map_id: int) -> int | None:
        if self._suspend_route2_regression_checks:
            return None
        if self.phase4_scope != "route_only":
            return None
        if not self._is_route2_progress_checkpoint(self.current_checkpoint_name):
            return None
        canonical_current = self._canonical_route2_map_id(int(map_id))
        canonical_expected = self._canonical_route2_map_id(int(self.current_checkpoint_expected_map_id))
        if self._map_matches(canonical_current, canonical_expected):
            return None
        next_map_id = self._next_checkpoint_expected_map_id(step_index)
        canonical_next = (
            self._canonical_route2_map_id(int(next_map_id))
            if next_map_id is not None
            else None
        )
        if canonical_next is not None and self._map_matches(canonical_current, canonical_next):
            return None

        progression = [13, 50, 51, 2]
        expected_idx = progression.index(canonical_expected) if canonical_expected in progression else -1
        current_idx = progression.index(canonical_current) if canonical_current in progression else -1
        if current_idx > expected_idx >= 0:
            return None
        if current_idx == -1 and int(map_id) not in {0, 1, 12, 44}:
            return None

        if (
            self.current_checkpoint_name == "Viridian_Forest"
            and canonical_expected == 51
            and canonical_current in {50, 13}
        ):
            reentered = self._seek_expected_map(
                expected_map_id=51,
                step_index=step_index,
                max_seek_steps=420,
                suppress_regression=True,
            )
            if reentered:
                self.route2_regression_count = 0
                self.no_progress_steps = 0
                self.last_position = self._position_key()
                self._reset_nav_memory()
                self._emit(
                    "route2_forest_reentry",
                    route_step_index=step_index,
                    reentry_from_map_id=int(map_id),
                )
                return self.current_block_start

        if (
            self.current_checkpoint_name in {"Forest_To_Pewter", "Forest_Exit_Gate", "Forest_Exit"}
            and canonical_expected == 51
            and canonical_current in {50, 13}
        ):
            return None

        self.route2_regression_count += 1
        if self.route2_regression_count >= 2:
            self._emit(
                "recovery_triggered",
                reason="route2_map_regression_exhausted",
                route_step_index=step_index,
                attempt=int(self.route2_regression_count),
                regressed_map_id=int(map_id),
                regressed_map_id_canonical=int(canonical_current),
                checkpoint_expected_map_id=int(canonical_expected),
                checkpoint_next_map_id=int(canonical_next if canonical_next is not None else -1),
            )
            raise RouteExecutionError("route2_map_regression_exhausted")

        self.recovery_events += 1
        self._emit(
            "recovery_triggered",
            reason="route2_map_regression",
            route_step_index=step_index,
            attempt=int(self.route2_regression_count),
            regressed_map_id=int(map_id),
            regressed_map_id_canonical=int(canonical_current),
            checkpoint_expected_map_id=int(canonical_expected),
            checkpoint_next_map_id=int(canonical_next if canonical_next is not None else -1),
        )
        self._execute_recovery_actions(step_index)
        self.no_progress_steps = 0
        self._reset_wild_fight_streak()
        self.last_position = self._position_key()
        self._reset_nav_memory()
        return self.current_block_start

    def _after_world_progress(self, *, step_index: int, suppress_no_progress: bool = False) -> int | None:
        nav = self._nav_state()
        self._record_map_transition(int(nav.get("map_id", 0)))
        if int(nav.get("in_battle", 0)) != 0:
            self.no_progress_steps = 0
            return self._handle_battle_interrupt(step_index=step_index)

        redirect = self._handle_route2_map_regression(
            step_index=step_index,
            map_id=int(nav.get("map_id", 0)),
        )
        if redirect is not None:
            return redirect

        pos = (int(nav["map_id"]), int(nav["x"]), int(nav["y"]))
        if self.last_position is None or pos != self.last_position:
            if self.last_position is not None and int(pos[0]) != int(self.last_position[0]):
                self._reset_wild_fight_streak()
            self.last_position = pos
            self.no_progress_steps = 0
            return None

        self.no_progress_steps += 1
        if suppress_no_progress:
            return None
        if self.no_progress_steps < self.no_progress_limit:
            return None
        return self._trigger_recovery(
            reason="no_progress",
            step_index=step_index,
            no_progress_steps=self.no_progress_steps,
        )

    def _run_battle_agent(self) -> tuple[bool, int]:
        battle_index = self.battles_fought
        if hasattr(self.agent, "start_episode_context"):
            self.agent.start_episode_context(
                state_label=f"phase4:{self.current_checkpoint_name}",
                state_index=battle_index,
            )

        turn = 0
        while self.emu.in_battle() and turn < self.max_battle_turns:
            turn += 1
            if hasattr(self.emu, "build_phase2_state"):
                state = self.emu.build_phase2_state(turn=turn)
            else:
                state = self.emu.get_battle_state()
                state["turn"] = turn

            use_llm = self.policy_mode == "hybrid" and ((turn - 1) % self.llm_turn_interval == 0)
            budget_fallback = False
            decision_calls = int(getattr(self.agent, "llm_decision_calls", 0))
            if use_llm and decision_calls >= self.max_decision_calls:
                use_llm = False
                budget_fallback = True

            slot = int(self.agent.pick_move(state, use_llm=use_llm, budget_fallback=budget_fallback))
            if hasattr(self.agent, "record_turn_decision"):
                self.agent.record_turn_decision(state, slot)

            committed = bool(self.emu.execute_move(slot))
            self._consume_step(1)
            if not committed:
                self.emu.tick(2)
                self._consume_step(1)

            if self.emu.wait_for_battle_end(timeout=self.turn_tick_budget):
                break

        return (not self.emu.in_battle(), int(turn))

    def _handle_battle_interrupt(self, *, step_index: int) -> int | None:
        self._validate_party_constraint(stage="battle_enter", step_index=step_index)
        kind = self._battle_kind()
        self.battles_fought += 1
        if kind == "trainer":
            self.trainer_battles += 1
        else:
            self.wild_battles += 1
        self._emit("battle_enter", battle_kind=kind, route_step_index=step_index)

        run_decision = "fight"
        if kind == "wild":
            run_decision = self._wild_battle_decision(step_index=step_index)

        if kind == "wild" and self.phase4_scope == "route_only":
            max_attempts = 3
            for _ in range(max_attempts):
                self.wild_run_attempts += 1
                self._emit(
                    "wild_run_attempt",
                    route_step_index=step_index,
                    attempt=self.wild_run_attempts,
                )
                run_success = False
                if hasattr(self.emu, "attempt_run"):
                    run_success = bool(self.emu.attempt_run(timeout_ticks=320))
                if run_success and not self.emu.in_battle():
                    self.wild_run_successes += 1
                    self._emit("battle_exit", battle_kind=kind, result="run_success")
                    self._validate_party_constraint(stage="battle_exit", step_index=step_index)
                    self.no_progress_steps = 0
                    self._reset_wild_fight_streak()
                    self.last_position = self._position_key()
                    return None
            raise RouteExecutionError("wild_run_blocked_route_only")

        if kind == "wild" and self.wild_run_enabled and run_decision == "run":
            max_attempts = 3
            for _ in range(max_attempts):
                self.wild_run_attempts += 1
                self._emit(
                    "wild_run_attempt",
                    route_step_index=step_index,
                    attempt=self.wild_run_attempts,
                )
                run_success = False
                if hasattr(self.emu, "attempt_run"):
                    run_success = bool(self.emu.attempt_run(timeout_ticks=320))
                if run_success and not self.emu.in_battle():
                    self.wild_run_successes += 1
                    self._emit("battle_exit", battle_kind=kind, result="run_success")
                    self._validate_party_constraint(stage="battle_exit", step_index=step_index)
                    self.no_progress_steps = 0
                    self._reset_wild_fight_streak()
                    self.last_position = self._position_key()
                    if self.phase4_scope == "route_only":
                        return None
                    return self.current_block_start

        battle_resolved, battle_turns = self._run_battle_agent()
        self.battle_turns_total += int(battle_turns)
        if not battle_resolved:
            raise RouteExecutionError("battle_unresolved")
        self._emit(
            "battle_exit",
            battle_kind=kind,
            result="battle_agent",
            battle_turns=int(battle_turns),
        )
        self._validate_party_constraint(stage="battle_exit", step_index=step_index)
        if kind == "wild":
            self.wild_fights_committed += 1
            self.consecutive_wild_fights += 1
        self.no_progress_steps = 0
        self.last_position = self._position_key()
        if self.phase4_scope == "route_only":
            return None
        return self.current_block_start

    def _seek_expected_map(
        self,
        *,
        expected_map_id: int,
        step_index: int,
        max_seek_steps: int = 5000,
        suppress_regression: bool = False,
    ) -> bool:
        transition_hint = {
            (12, 1): "up",
            (1, 12): "down",
            (1, 13): "up",
            (12, 13): "up",
            (13, 1): "down",
            (13, 50): "up",
            (50, 13): "down",
            (13, 47): "down",
            (47, 13): "up",
            (47, 50): "up",
            (50, 47): "down",
            (47, 51): "down",
            (51, 47): "up",
            (13, 0): "up",
            (0, 13): "down",
            (13, 2): "up",
            (47, 2): "up",
            (2, 13): "down",
            (2, 47): "down",
            (50, 2): "up",
            (2, 50): "down",
            (50, 51): "up",
            (51, 50): "down",
            (0, 51): "up",
            (51, 0): "down",
            (51, 2): "up",
            (38, 2): "up",
            (0, 2): "up",
            (37, 2): "up",
        }
        action_tries: Dict[Tuple[int, int, int, str], int] = {}
        pos_visits: Dict[Tuple[int, int, int], int] = {}
        unique_positions: set[Tuple[int, int, int]] = set()
        last_unique_count = 0
        last_unique_step = 0
        prev_suspend_regression = bool(self._suspend_route2_regression_checks)
        if suppress_regression:
            self._suspend_route2_regression_checks = True
        try:
            for i in range(max_seek_steps):
                nav = self._nav_state()
                curr_map_id = int(nav.get("map_id", 0))
                if self._map_matches(curr_map_id, expected_map_id):
                    return True

                preferred = transition_hint.get((curr_map_id, expected_map_id), "up")
                pos = self._position_key()
                pos_visits[pos] = int(pos_visits.get(pos, 0)) + 1
                unique_positions.add(pos)
                if len(unique_positions) > last_unique_count:
                    last_unique_count = len(unique_positions)
                    last_unique_step = i
                elif i > 0 and (i - last_unique_step) >= 192:
                    self.blocked_edges.clear()
                    self.edge_fail_counts.clear()
                    last_unique_step = i
                    self._emit(
                        "seek_unstuck_reset",
                        route_step_index=step_index,
                        expected_map_id=expected_map_id,
                        seek_steps=i,
                        seek_unique_positions=len(unique_positions),
                        seek_current_map_id=curr_map_id,
                    )
                    redirect = self._press_button(
                        "b",
                        hold_frames=2,
                        step_index=step_index,
                        suppress_no_progress=True,
                    )
                    if redirect is not None:
                        continue

                if i > 0 and i % 256 == 0:
                    self._emit(
                        "seek_progress",
                        route_step_index=step_index,
                        expected_map_id=expected_map_id,
                        seek_steps=i,
                        seek_unique_positions=len(unique_positions),
                        seek_current_map_id=curr_map_id,
                    )
                sweep_dir = None
                if preferred == "up" and int(pos[2]) <= 2:
                    sweep_dir = "right" if ((i // 18) % 2 == 0) else "left"
                elif preferred == "down" and int(pos[2]) >= 253:
                    sweep_dir = "right" if ((i // 18) % 2 == 0) else "left"
                elif preferred == "left" and int(pos[1]) <= 2:
                    sweep_dir = "up" if ((i // 18) % 2 == 0) else "down"
                elif preferred == "right" and int(pos[1]) >= 253:
                    sweep_dir = "up" if ((i // 18) % 2 == 0) else "down"

                if i % 41 == 0:
                    action = "b"
                else:
                    orth_1, orth_2 = self._orthogonal_directions(preferred, i)
                    opposite = self._opposite_direction(preferred)
                    candidates: List[str] = []
                    if sweep_dir is not None:
                        candidates.extend([preferred, sweep_dir, orth_1, orth_2, opposite])
                    else:
                        candidates.extend([preferred, orth_1, orth_2, opposite])

                    action = preferred
                    best_score: Tuple[float, float] | None = None
                    for candidate in candidates:
                        if not self._is_edge_blocked(pos, candidate):
                            edge = self._edge_key(pos, candidate)
                            tries = float(action_tries.get(edge, 0))
                            # Lower score is better: prefer less-tried edges while biasing toward hinted direction.
                            bias = 0.0
                            if candidate == preferred:
                                bias -= 0.35
                            elif candidate == opposite:
                                bias += 0.9
                            else:
                                bias += 0.25
                            if sweep_dir is not None and candidate == sweep_dir:
                                bias -= 0.15
                            if pos_visits[pos] > 8 and candidate == opposite:
                                bias += 0.6
                            score = (tries + bias, self.rng.random())
                            if best_score is None or score < best_score:
                                best_score = score
                                action = candidate
                    if best_score is None:
                        action = preferred

                if action != "b":
                    edge = self._edge_key(pos, action)
                    action_tries[edge] = int(action_tries.get(edge, 0)) + 1

                hold_frames = 2 if action == "b" else 6
                if action == "b":
                    redirect = self._press_button(
                        action,
                        hold_frames=hold_frames,
                        step_index=step_index,
                        suppress_no_progress=True,
                    )
                    if redirect is not None:
                        continue
                else:
                    redirect, moved, _, _ = self._attempt_nav_action(
                        action=action,
                        hold_frames=hold_frames,
                        step_index=step_index,
                    )
                    if redirect is not None:
                        continue
                    if not moved and pos_visits[pos] >= 6:
                        # If repeatedly stuck on this tile, lower opposite-edge penalty next rounds.
                        opposite_edge = self._edge_key(pos, self._opposite_direction(preferred))
                        action_tries[opposite_edge] = max(0, int(action_tries.get(opposite_edge, 0)) - 1)
        finally:
            self._suspend_route2_regression_checks = prev_suspend_regression
        self._emit(
            "seek_failed",
            route_step_index=step_index,
            expected_map_id=expected_map_id,
            seek_steps=max_seek_steps,
            seek_unique_positions=len(unique_positions),
            seek_current_map_id=int(self._nav_state().get("map_id", -1)),
        )
        return self._map_matches(int(self._nav_state().get("map_id", -1)), int(expected_map_id))

    def _press_button(
        self,
        button: str,
        hold_frames: int,
        step_index: int,
        *,
        suppress_no_progress: bool = False,
        forest_phase: str = "",
    ) -> int | None:
        before = self._position_key()
        self.emu.press(button, frames=max(1, int(hold_frames)))
        self._consume_step(1)
        after_nav = self._nav_state()
        after = (
            int(after_nav.get("map_id", 0)),
            int(after_nav.get("x", 0)),
            int(after_nav.get("y", 0)),
        )
        self._record_forest_probe_action(
            button=button,
            phase=str(forest_phase),
            before=before,
            after=after,
        )
        return self._after_world_progress(
            step_index=step_index,
            suppress_no_progress=suppress_no_progress,
        )

    def _opposite_direction(self, direction: str) -> str:
        opposite = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }
        return opposite.get(direction, "down")

    def _right_direction(self, direction: str) -> str:
        right_of = {
            "up": "right",
            "right": "down",
            "down": "left",
            "left": "up",
        }
        return right_of.get(direction, "right")

    def _left_direction(self, direction: str) -> str:
        left_of = {
            "up": "left",
            "left": "down",
            "down": "right",
            "right": "up",
        }
        return left_of.get(direction, "left")

    def _orthogonal_directions(self, direction: str, cycle: int) -> Tuple[str, str]:
        if direction in {"up", "down"}:
            first, second = "left", "right"
        else:
            first, second = "up", "down"
        if cycle % 2 == 0:
            return second, first
        return first, second

    def _is_forward_progress(
        self,
        direction: str,
        pos_before: Tuple[int, int, int],
        pos_after: Tuple[int, int, int],
    ) -> bool:
        if pos_after[0] != pos_before[0]:
            return True
        dx = int(pos_after[1]) - int(pos_before[1])
        dy = int(pos_after[2]) - int(pos_before[2])
        if direction == "up":
            return dy < 0
        if direction == "down":
            return dy > 0
        if direction == "left":
            return dx < 0
        if direction == "right":
            return dx > 0
        return False

    def _biased_side_preference(
        self,
        *,
        direction: str,
        pos: Tuple[int, int, int],
        default_prefer_right: bool,
    ) -> bool:
        forward_edge = self._edge_key(pos, direction)
        forward_fails = int(self.edge_fail_counts.get(forward_edge, 0))
        if forward_fails < 2:
            return bool(default_prefer_right)

        map_id = int(pos[0])
        x_now = int(pos[1])
        y_now = int(pos[2])
        same_map = [p for p in self.recent_positions if int(p[0]) == map_id]
        if len(same_map) < 4:
            return bool(default_prefer_right)

        xs = [int(p[1]) for p in same_map]
        ys = [int(p[2]) for p in same_map]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if direction in {"up", "down"}:
            # If pinned on the left side of explored corridor, bias to right escape and vice versa.
            if x_now <= (min_x + 1):
                return True
            if x_now >= (max_x - 1):
                return False
            return bool(default_prefer_right)

        # Horizontal travel: same idea with vertical corridor bounds.
        if y_now <= (min_y + 1):
            return True
        if y_now >= (max_y - 1):
            return False
        return bool(default_prefer_right)

    def _select_roomba_action(
        self,
        *,
        direction: str,
        pos: Tuple[int, int, int],
        prefer_right: bool,
    ) -> str:
        side_pref_right = self._biased_side_preference(
            direction=direction,
            pos=pos,
            default_prefer_right=prefer_right,
        )
        side_1 = self._right_direction(direction) if side_pref_right else self._left_direction(direction)
        side_2 = self._left_direction(direction) if side_pref_right else self._right_direction(direction)
        sequence = [
            direction,
            side_1,
            side_1,
            direction,
            side_1,
            side_2,
            direction,
            self._opposite_direction(direction),
        ]
        for action in sequence:
            if not self._is_edge_blocked(pos, action):
                return action
        return direction

    def _attempt_nav_action(
        self,
        *,
        action: str,
        hold_frames: int,
        step_index: int,
    ) -> tuple[int | None, bool, Tuple[int, int, int], Tuple[int, int, int]]:
        before = self._position_key()
        redirect = self._press_button(
            button=action,
            hold_frames=hold_frames,
            step_index=step_index,
            suppress_no_progress=True,
        )
        if redirect is not None:
            return redirect, False, before, before
        after = self._position_key()
        moved = self._record_move_outcome(
            pos_before=before,
            action=action,
            pos_after=after,
        )
        return None, moved, before, after

    def _corner_escape(
        self,
        *,
        direction: str,
        hold_frames: int,
        step_index: int,
        prefer_right: bool,
    ) -> tuple[int | None, bool]:
        back = self._opposite_direction(direction)
        side_primary = self._right_direction(direction) if prefer_right else self._left_direction(direction)
        side_secondary = self._left_direction(direction) if prefer_right else self._right_direction(direction)
        gained_forward = False

        # 1) Back out of the corner pocket.
        for _ in range(4):
            redirect, moved, before, after = self._attempt_nav_action(
                action=back,
                hold_frames=hold_frames,
                step_index=step_index,
            )
            if redirect is not None:
                return redirect, gained_forward
            if moved and self._is_forward_progress(direction, before, after):
                gained_forward = True

        # 2) Slide laterally toward the freer side.
        for action in (side_primary, side_primary, side_secondary):
            redirect, moved, before, after = self._attempt_nav_action(
                action=action,
                hold_frames=hold_frames,
                step_index=step_index,
            )
            if redirect is not None:
                return redirect, gained_forward
            if moved and self._is_forward_progress(direction, before, after):
                gained_forward = True

        # 3) Re-attempt forward approach from new alignment.
        for _ in range(3):
            redirect, moved, before, after = self._attempt_nav_action(
                action=direction,
                hold_frames=hold_frames,
                step_index=step_index,
            )
            if redirect is not None:
                return redirect, gained_forward
            if moved and self._is_forward_progress(direction, before, after):
                gained_forward = True

        return None, gained_forward

    def _tick_and_check(self, *, step_index: int) -> int | None:
        self.emu.tick(1)
        self._consume_step(1)
        return self._after_world_progress(step_index=step_index)

    def _direction_priority_to_target(
        self,
        *,
        pos: Tuple[int, int, int],
        target_x: int,
        target_y: int,
    ) -> List[str]:
        _, x_now, y_now = pos
        dx = int(target_x) - int(x_now)
        dy = int(target_y) - int(y_now)
        ordered: List[str] = []

        def _push(direction: str) -> None:
            if direction in VALID_DIRECTIONS and direction not in ordered:
                ordered.append(direction)

        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        if abs(dx) >= abs(dy):
            if dx != 0:
                _push(horiz)
            if dy != 0:
                _push(vert)
        else:
            if dy != 0:
                _push(vert)
            if dx != 0:
                _push(horiz)
        for direction in ("up", "right", "down", "left"):
            _push(direction)
        return ordered

    def _step_distance(
        self,
        *,
        x_now: int,
        y_now: int,
        target_x: int,
        target_y: int,
        action: str,
    ) -> int:
        x_after = int(x_now)
        y_after = int(y_now)
        if action == "up":
            y_after -= 1
        elif action == "down":
            y_after += 1
        elif action == "left":
            x_after -= 1
        elif action == "right":
            x_after += 1
        return abs(int(target_x) - x_after) + abs(int(target_y) - y_after)

    def _run_waypoint(self, *, step_index: int, step: Dict[str, object]) -> int | None:
        target_map_id = int(step["map_id"])
        target_x = int(step["x"])
        target_y = int(step["y"])
        radius = max(0, int(step["radius"]))
        max_seek_steps = max(1, int(step["max_seek_steps"]))
        hold_frames = max(1, int(step["hold_frames"]))
        transition_preferred = bool(step.get("transition_preferred", True))
        next_checkpoint_map_id = self._next_checkpoint_expected_map_id(step_index)

        nav = self._nav_state()
        if not self._map_matches(int(nav.get("map_id", -1)), target_map_id):
            if not self._seek_expected_map(
                expected_map_id=target_map_id,
                step_index=step_index,
                max_seek_steps=max_seek_steps,
            ):
                return self._trigger_recovery(
                    reason="waypoint_map_mismatch",
                    step_index=step_index,
                    target_map_id=target_map_id,
                )

        action_tries: Dict[Tuple[int, int, int, str], int] = {}
        stagnant = 0
        best_distance = 1_000_000
        stagnant_without_improvement = 0
        for i in range(max_seek_steps):
            nav = self._nav_state()
            curr_map_id = int(nav.get("map_id", 0))
            if (
                next_checkpoint_map_id is not None
                and not self._map_matches(int(next_checkpoint_map_id), int(target_map_id))
                and self._map_matches(curr_map_id, int(next_checkpoint_map_id))
            ):
                return None
            if not self._map_matches(curr_map_id, target_map_id):
                remaining = max(64, max_seek_steps - i)
                if self._seek_expected_map(
                    expected_map_id=target_map_id,
                    step_index=step_index,
                    max_seek_steps=remaining,
                ):
                    continue
                return self._trigger_recovery(
                    reason="waypoint_map_lost",
                    step_index=step_index,
                    target_map_id=target_map_id,
                )

            x_now = int(nav.get("x", 0))
            y_now = int(nav.get("y", 0))
            distance = abs(target_x - x_now) + abs(target_y - y_now)
            if distance <= radius:
                return None
            if distance < best_distance:
                best_distance = int(distance)
                stagnant_without_improvement = 0
            else:
                stagnant_without_improvement += 1

            if (
                transition_preferred
                and
                next_checkpoint_map_id is not None
                and not self._map_matches(int(next_checkpoint_map_id), int(target_map_id))
                and stagnant_without_improvement >= 72
            ):
                remaining = max(96, max_seek_steps - i)
                if self._seek_expected_map(
                    expected_map_id=int(next_checkpoint_map_id),
                    step_index=step_index,
                    max_seek_steps=remaining,
                ):
                    return None
                stagnant_without_improvement = 0

            if (
                transition_preferred
                and
                self.phase4_scope == "route_only"
                and next_checkpoint_map_id is not None
                and not self._map_matches(int(next_checkpoint_map_id), int(target_map_id))
                and i > 0
                and (i % 96 == 0 or stagnant_without_improvement >= 24)
            ):
                remaining = max(96, max_seek_steps - i)
                if self._seek_expected_map(
                    expected_map_id=int(next_checkpoint_map_id),
                    step_index=step_index,
                    max_seek_steps=min(420, remaining),
                ):
                    self._emit(
                        "waypoint_transition_preferred",
                        route_step_index=step_index,
                        waypoint_map_id=target_map_id,
                        next_checkpoint_map_id=int(next_checkpoint_map_id),
                        waypoint_seek_steps=i,
                    )
                    return None

            pos = (curr_map_id, x_now, y_now)
            candidates = self._direction_priority_to_target(
                pos=pos,
                target_x=target_x,
                target_y=target_y,
            )
            chosen = candidates[0]
            best_score: Tuple[float, float] | None = None
            for action in candidates:
                if self._is_edge_blocked(pos, action):
                    continue
                edge = self._edge_key(pos, action)
                tries = float(action_tries.get(edge, 0))
                dist = float(
                    self._step_distance(
                        x_now=x_now,
                        y_now=y_now,
                        target_x=target_x,
                        target_y=target_y,
                        action=action,
                    )
                )
                score = (dist + (0.15 * tries), self.rng.random())
                if best_score is None or score < best_score:
                    best_score = score
                    chosen = action

            edge = self._edge_key(pos, chosen)
            action_tries[edge] = int(action_tries.get(edge, 0)) + 1
            redirect, moved, _, _ = self._attempt_nav_action(
                action=chosen,
                hold_frames=hold_frames,
                step_index=step_index,
            )
            if redirect is not None:
                return redirect
            if moved:
                stagnant = 0
                continue
            stagnant += 1
            if stagnant >= 40:
                return self._trigger_recovery(
                    reason="waypoint_stuck",
                    step_index=step_index,
                    target_map_id=target_map_id,
                    target_x=target_x,
                    target_y=target_y,
                )

        return self._trigger_recovery(
            reason="waypoint_timeout",
            step_index=step_index,
            target_map_id=target_map_id,
            target_x=target_x,
            target_y=target_y,
        )

    def _wall_follow_order(self, *, heading: str, hand: str) -> List[str]:
        if hand == "left":
            return [
                self._left_direction(heading),
                heading,
                self._right_direction(heading),
                self._opposite_direction(heading),
            ]
        return [
            self._right_direction(heading),
            heading,
            self._left_direction(heading),
            self._opposite_direction(heading),
        ]

    def _run_forest_to_pewter_script(
        self,
        *,
        step_index: int,
        hold_frames: int,
        target_map_id: int,
        traverse_attempt: int,
    ) -> tuple[bool, int | None]:
        if not (
            self.phase4_scope == "route_only"
            and int(target_map_id) == 2
            and str(self.current_checkpoint_name) in {"Forest_To_Pewter", "Forest_Exit_Gate"}
        ):
            return False, None

        self.forest_script_cycle_count += 1
        if self.forest_script_cycle_count > 10:
            return True, self._trigger_recovery(
                reason="forest_to_pewter_script_cycle_limit",
                step_index=step_index,
                target_map_id=target_map_id,
                cycle_count=int(self.forest_script_cycle_count),
            )

        nav = self._nav_state()
        curr_map_id = int(nav.get("map_id", 0))
        y_now = int(nav.get("y", 0))

        if curr_map_id not in {13, 47, 50, 51}:
            return False, None

        def _script_press(button: str, phase: str) -> tuple[int | None, bool, bool]:
            before = self._position_key()
            redirect = self._press_button(
                button,
                hold_frames=hold_frames,
                step_index=step_index,
                suppress_no_progress=True,
                forest_phase=phase,
            )
            if redirect is not None:
                return redirect, False, False
            after = self._position_key()
            moved = after != before
            reached = self._map_matches(int(after[0]), target_map_id)
            return None, moved, reached

        any_progress = False
        variant_seed = (
            int(self.forest_script_variant)
            + int(traverse_attempt)
            + int(self.forest_script_cycle_count)
        ) % 32

        if curr_map_id == 51:
            self._emit(
                "forest_to_pewter_script_forest_south_exit",
                route_step_index=step_index,
                target_map_id=target_map_id,
                traverse_attempt=traverse_attempt,
                traverse_current_map_id=curr_map_id,
                traverse_current_y=y_now,
                forest_variant=variant_seed,
            )
            progressed = False
            forest_budget = 320

            # Phase-rotated pre-probe before committing to the legacy south-exit routine.
            if (variant_seed % 3) != 0:
                north_lanes = self._rotate_int_sequence(
                    [12, 13, 14, 15, 16, 17, 11, 18, 10, 19],
                    offset=variant_seed,
                )
                for align_x in north_lanes:
                    for _ in range(8):
                        if forest_budget <= 0:
                            break
                        nav_now = self._nav_state()
                        if int(nav_now.get("map_id", 0)) != 51:
                            break
                        x_now = int(nav_now.get("x", 0))
                        if x_now == int(align_x):
                            break
                        forest_budget -= 1
                        button = "right" if x_now < int(align_x) else "left"
                        redirect, moved, reached = _script_press(button, "forest_north_align")
                        if redirect is not None:
                            return True, redirect
                        if reached:
                            return True, None
                        if moved:
                            progressed = True
                            any_progress = True

                    stagnant = 0
                    probe_budget = 24 + (variant_seed % 5)
                    for _ in range(probe_budget):
                        if forest_budget <= 0:
                            break
                        nav_now = self._nav_state()
                        if int(nav_now.get("map_id", 0)) != 51:
                            break
                        forest_budget -= 1
                        redirect, moved, reached = _script_press("up", "forest_north_probe")
                        if redirect is not None:
                            return True, redirect
                        if reached:
                            return True, None
                        if moved:
                            progressed = True
                            any_progress = True
                            stagnant = 0
                        else:
                            stagnant += 1
                        if stagnant >= 8:
                            break
                    if int(self._nav_state().get("map_id", 0)) != 51 or forest_budget <= 0:
                        break
                if int(self._nav_state().get("map_id", 0)) != 51:
                    nav = self._nav_state()
                    curr_map_id = int(nav.get("map_id", 0))
                    y_now = int(nav.get("y", 0))

            for _ in range(36):
                if forest_budget <= 0:
                    break
                nav_now = self._nav_state()
                if int(nav_now.get("map_id", 0)) != 51:
                    break
                if int(nav_now.get("y", 0)) >= 30:
                    break
                forest_budget -= 1
                redirect, moved, reached = _script_press("down", "forest_entry_descent")
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    progressed = True
                    any_progress = True

            south_lanes = self._rotate_int_sequence([15, 14, 16, 13, 17, 12, 18], offset=variant_seed)
            for align_x in south_lanes:
                for _ in range(14):
                    if forest_budget <= 0:
                        break
                    nav_now = self._nav_state()
                    if int(nav_now.get("map_id", 0)) != 51:
                        break
                    x_now = int(nav_now.get("x", 0))
                    if x_now == int(align_x):
                        break
                    forest_budget -= 1
                    button = "right" if x_now < int(align_x) else "left"
                    redirect, moved, reached = _script_press(button, "forest_lane_align")
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    if moved:
                        progressed = True
                        any_progress = True

                stagnant = 0
                for _ in range(56):
                    if forest_budget <= 0:
                        break
                    nav_now = self._nav_state()
                    if int(nav_now.get("map_id", 0)) != 51:
                        break
                    forest_budget -= 1
                    redirect, moved, reached = _script_press("down", "forest_lane_down")
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    nav_after = self._nav_state()
                    if moved:
                        progressed = True
                        any_progress = True
                        stagnant = 0
                    else:
                        stagnant += 1
                    if int(nav_after.get("map_id", 0)) != 51:
                        break
                    if stagnant >= 12:
                        break
                if int(self._nav_state().get("map_id", 0)) != 51 or forest_budget <= 0:
                    break

            nav_after_lane = self._nav_state()
            if int(nav_after_lane.get("map_id", 0)) == 51:
                fallback = (["left"] * 6) + (["down"] * 28) + (["right"] * 12) + (["down"] * 28)
                if variant_seed % 2 == 1:
                    fallback = (["right"] * 6) + (["down"] * 28) + (["left"] * 12) + (["down"] * 28)
                for button in fallback:
                    if forest_budget <= 0:
                        break
                    forest_budget -= 1
                    redirect, moved, reached = _script_press(button, "forest_lane_fallback")
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    if moved:
                        progressed = True
                        any_progress = True
                    if int(self._nav_state().get("map_id", 0)) != 51:
                        break

            if int(self._nav_state().get("map_id", 0)) == 51:
                if progressed or any_progress:
                    # Keep iterating scripted traversal before burning the whole run.
                    return True, None
                return True, self._trigger_recovery(
                    reason="forest_to_pewter_script_failed",
                    step_index=step_index,
                    target_map_id=target_map_id,
                )

        nav = self._nav_state()
        curr_map_id = int(nav.get("map_id", 0))
        y_now = int(nav.get("y", 0))
        if curr_map_id == 47:
            self._emit(
                "forest_to_pewter_script_gatehouse_bridge",
                route_step_index=step_index,
                target_map_id=target_map_id,
                traverse_attempt=traverse_attempt,
                traverse_current_map_id=curr_map_id,
                traverse_current_y=y_now,
            )
            bridge_pattern = (
                (["up"] * 18)
                + (["left"] * 2)
                + (["up"] * 12)
                + (["right"] * 4)
                + (["up"] * 12)
                + (["left"] * 2)
                + (["up"] * 10)
            )
            for button in bridge_pattern:
                redirect, moved, reached = _script_press(button, "route2_gatehouse_bridge")
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True
                nav_after = self._nav_state()
                map_after = int(nav_after.get("map_id", 0))
                if map_after in {13, 50, 2}:
                    break
                if map_after not in {47, 13, 50, 51, 2}:
                    break
            nav = self._nav_state()
            curr_map_id = int(nav.get("map_id", 0))
            y_now = int(nav.get("y", 0))

        if curr_map_id == 50 and y_now > 20:
            self._emit(
                "forest_to_pewter_script_gate_descent",
                route_step_index=step_index,
                target_map_id=target_map_id,
                traverse_attempt=traverse_attempt,
                traverse_current_map_id=curr_map_id,
                traverse_current_y=y_now,
            )
            progressed = False
            stagnant = 0
            for _ in range(56):
                redirect, moved, reached = _script_press("down", "gate_descent")
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                nav_after = self._nav_state()
                if moved:
                    progressed = True
                    stagnant = 0
                    any_progress = True
                else:
                    stagnant += 1
                after_map = int(nav_after.get("map_id", 0))
                after_y = int(nav_after.get("y", 0))
                if after_map == 13 and after_y <= 20:
                    break
                if after_map not in {50, 13, 47}:
                    break
                if stagnant >= 14:
                    break
            if not progressed:
                return True, self._trigger_recovery(
                    reason="forest_to_pewter_script_failed",
                    step_index=step_index,
                    target_map_id=target_map_id,
                )

        nav = self._nav_state()
        curr_map_id = int(nav.get("map_id", 0))
        y_now = int(nav.get("y", 0))
        if curr_map_id not in {47, 50, 13}:
            return True, None

        if curr_map_id == 13 and y_now > 20:
            self._emit(
                "forest_to_pewter_script_route2_climb",
                route_step_index=step_index,
                target_map_id=target_map_id,
                traverse_attempt=traverse_attempt,
                traverse_current_map_id=curr_map_id,
                traverse_current_y=y_now,
            )
            climb_stagnant = 0
            for i in range(110):
                redirect, moved, reached = _script_press("up", "route2_climb_up")
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                nav_after = self._nav_state()
                map_after = int(nav_after.get("map_id", 0))
                y_after = int(nav_after.get("y", 0))
                if moved:
                    any_progress = True
                    climb_stagnant = 0
                else:
                    climb_stagnant += 1
                if map_after != 13:
                    break
                if y_after <= 20:
                    break
                if climb_stagnant >= 8:
                    nudge_cycle = ("left", "right", "right", "left", "left", "right")
                    nudge = nudge_cycle[(i // 8) % len(nudge_cycle)]
                    redirect, moved, reached = _script_press(nudge, "route2_climb_nudge")
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    if moved:
                        any_progress = True
                        climb_stagnant = 0

        # Route2 north-pocket finite-state exploration.
        self._emit(
            "forest_to_pewter_script_route2_north",
            route_step_index=step_index,
            target_map_id=target_map_id,
            traverse_attempt=traverse_attempt,
            traverse_current_map_id=curr_map_id,
            traverse_current_y=y_now,
        )
        total_budget = 840

        def _budget_press(button: str, phase: str) -> tuple[int | None, bool, bool, bool]:
            nonlocal total_budget
            if total_budget <= 0:
                return None, False, False, True
            total_budget -= 1
            redirect, moved, reached = _script_press(button, phase)
            return redirect, moved, reached, False

        nav_now = self._nav_state()
        if int(nav_now.get("map_id", 0)) == 47:
            for _ in range(96):
                redirect, moved, reached, exhausted = _budget_press("up", "route2_gatehouse_up")
                if exhausted:
                    break
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True
                nav_after = self._nav_state()
                map_after = int(nav_after.get("map_id", 0))
                if map_after in {13, 50, 2}:
                    break
                if map_after not in {47, 13, 50, 2}:
                    break

        if int(nav_now.get("map_id", 0)) == 50 and int(nav_now.get("y", 0)) <= 8:
            gate_candidates = self._rotate_int_sequence([2, 3, 4, 5, 6, 7, 8, 1, 9], offset=variant_seed)
            ranked_gate_candidates = sorted(
                gate_candidates,
                key=lambda lane: int(self.route2_gate_x_fail_counts.get(int(lane), 0)),
            )
            for gate_x in ranked_gate_candidates:
                lane_fail = 0
                for _ in range(10):
                    nav_gate = self._nav_state()
                    if int(nav_gate.get("map_id", 0)) != 50:
                        break
                    x_gate = int(nav_gate.get("x", 0))
                    if x_gate == int(gate_x):
                        break
                    button = "right" if x_gate < int(gate_x) else "left"
                    redirect, moved, reached, exhausted = _budget_press(button, "route2_gate_align")
                    if exhausted:
                        break
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    if moved:
                        any_progress = True
                for _ in range(18):
                    redirect, moved, reached, exhausted = _budget_press("up", "route2_gate_up")
                    if exhausted:
                        break
                    if redirect is not None:
                        return True, redirect
                    if reached:
                        return True, None
                    if moved:
                        any_progress = True
                    nav_after = self._nav_state()
                    map_after = int(nav_after.get("map_id", 0))
                    if map_after == 51:
                        lane_fail += 1
                        break
                    if map_after == 13:
                        break
                    if map_after not in {50, 13, 51}:
                        break
                if lane_fail > 0:
                    self.route2_gate_x_fail_counts[int(gate_x)] = (
                        int(self.route2_gate_x_fail_counts.get(int(gate_x), 0)) + lane_fail
                    )
                else:
                    prev = int(self.route2_gate_x_fail_counts.get(int(gate_x), 0))
                    if prev > 0:
                        self.route2_gate_x_fail_counts[int(gate_x)] = max(0, prev - 1)

        core_lanes = [1, 2, 3, 7, 8, 9]
        wide_lanes = [5, 11, 13, 15, 17, 19, 21, 23]
        lane_plan = core_lanes + wide_lanes
        if traverse_attempt % 2 == 0:
            lane_plan = list(reversed(lane_plan))
        lane_plan = self._rotate_int_sequence(lane_plan, offset=variant_seed)
        lane_plan = sorted(
            lane_plan,
            key=lambda lane: int(self.route2_lane_fail_counts.get(int(lane), 0)),
        )

        for align_x in lane_plan:
            lane_fail = 0
            align_budget = 12 if align_x in core_lanes else 22
            for _ in range(align_budget):
                nav_now = self._nav_state()
                map_now = int(nav_now.get("map_id", 0))
                x_now = int(nav_now.get("x", 0))
                y_now = int(nav_now.get("y", 0))
                if map_now not in {13, 47, 50}:
                    break
                if x_now == int(align_x):
                    break
                button = "right" if x_now < int(align_x) else "left"
                redirect, moved, reached, exhausted = _budget_press(button, "route2_align")
                if exhausted:
                    break
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True

            probe_budget = 18 if align_x in core_lanes else 24
            for _ in range(probe_budget):
                redirect, moved, reached, exhausted = _budget_press("up", "route2_probe_up")
                if exhausted:
                    break
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True
                nav_after = self._nav_state()
                map_after = int(nav_after.get("map_id", 0))
                y_after = int(nav_after.get("y", 0))
                if map_after in {50, 51} and map_after != map_now:
                    lane_fail += 1
                if map_after not in {13, 47, 50}:
                    break
                if map_after == 50 and y_after >= 40:
                    lane_fail += 1
                    break

            reset_budget = 3 if align_x in core_lanes else 2
            for _ in range(reset_budget):
                nav_now = self._nav_state()
                if int(nav_now.get("map_id", 0)) != 13:
                    break
                if int(nav_now.get("y", 0)) > 20:
                    break
                redirect, moved, reached, exhausted = _budget_press("down", "route2_reset")
                if exhausted:
                    break
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True

            if lane_fail >= 2:
                self.route2_lane_fail_counts[int(align_x)] = (
                    int(self.route2_lane_fail_counts.get(int(align_x), 0)) + 1
                )
            else:
                prev = int(self.route2_lane_fail_counts.get(int(align_x), 0))
                if prev > 0:
                    self.route2_lane_fail_counts[int(align_x)] = max(0, prev - 1)

            if total_budget <= 0:
                break

        sweep = (["left"] * 16) + (["right"] * 32) + (["left"] * 16)
        for button in sweep:
            redirect, moved, reached, exhausted = _budget_press(button, "route2_sweep")
            if exhausted:
                break
            if redirect is not None:
                return True, redirect
            if reached:
                return True, None
            if moved:
                any_progress = True

            redirect, moved, reached, exhausted = _budget_press("up", "route2_sweep_up")
            if exhausted:
                break
            if redirect is not None:
                return True, redirect
            if reached:
                return True, None
            if moved:
                any_progress = True

            nav_after_up = self._nav_state()
            if int(nav_after_up.get("map_id", 0)) == 13 and int(nav_after_up.get("y", 0)) <= 8:
                redirect, moved, reached, exhausted = _budget_press("down", "route2_sweep_down")
                if exhausted:
                    break
                if redirect is not None:
                    return True, redirect
                if reached:
                    return True, None
                if moved:
                    any_progress = True

        if self._seek_expected_map(
            expected_map_id=target_map_id,
            step_index=step_index,
            max_seek_steps=260,
            suppress_regression=True,
        ):
            return True, None
        if any_progress:
            self._bump_forest_variant(step=1)
            return True, None
        return True, self._trigger_recovery(
            reason="forest_to_pewter_script_failed",
            step_index=step_index,
            target_map_id=target_map_id,
        )

    def _run_traverse_until_map(self, *, step_index: int, step: Dict[str, object]) -> int | None:
        target_map_id = int(step["target_map_id"])
        mode = str(step.get("mode", "wall_follow_ccw"))
        max_steps = max(1, int(step["max_steps"]))
        hold_frames = max(1, int(step["hold_frames"]))

        nav = self._nav_state()
        if self._map_matches(int(nav.get("map_id", -1)), target_map_id):
            return None

        if mode != "wall_follow_ccw":
            raise RouteExecutionError(f"unsupported traverse mode at runtime: {mode}")

        attempt_key = (str(self.current_checkpoint_name), int(target_map_id))
        traverse_attempt = int(self.traverse_attempt_counts.get(attempt_key, 0)) + 1
        self.traverse_attempt_counts[attempt_key] = traverse_attempt
        heading_cycle = ("up", "right", "left", "down")
        heading = heading_cycle[(traverse_attempt - 1) % len(heading_cycle)]
        hand = "left"
        no_move_cycles = 0
        seen_states: Dict[Tuple[int, int, int, str, str], int] = {}
        stuck_events = 0
        unique_positions: set[Tuple[int, int, int]] = set()

        for i in range(max_steps):
            nav = self._nav_state()
            curr_map_id = int(nav.get("map_id", 0))
            if self._map_matches(curr_map_id, target_map_id):
                return None
            if (
                self.phase4_scope == "integrated"
                and int(target_map_id) == 51
                and str(self.current_checkpoint_name) == "Route2_GateHouse"
                and curr_map_id in {0, 1, 12, 43, 44}
            ):
                self._emit(
                    "traverse_gatehouse_reacquire",
                    route_step_index=step_index,
                    target_map_id=target_map_id,
                    traverse_current_map_id=curr_map_id,
                    traverse_current_x=int(nav.get("x", 0)),
                    traverse_current_y=int(nav.get("y", 0)),
                    traverse_attempt=traverse_attempt,
                    traverse_steps=i,
                )
                if self._seek_expected_map(
                    expected_map_id=50,
                    step_index=step_index,
                    max_seek_steps=640,
                    suppress_regression=True,
                ):
                    continue
                return self._trigger_recovery(
                    reason="traverse_gatehouse_map_lost",
                    step_index=step_index,
                    target_map_id=target_map_id,
                    traverse_current_map_id=curr_map_id,
                )
            if (
                self.phase4_scope == "route_only"
                and int(target_map_id) == 2
                and str(self.current_checkpoint_name) in {"Forest_To_Pewter", "Forest_Exit_Gate"}
            ):
                scripted, scripted_redirect = self._run_forest_to_pewter_script(
                    step_index=step_index,
                    hold_frames=hold_frames,
                    target_map_id=target_map_id,
                    traverse_attempt=traverse_attempt,
                )
                if scripted_redirect is not None:
                    return scripted_redirect
                if scripted:
                    after_script_map = int(self._nav_state().get("map_id", 0))
                    if self._map_matches(after_script_map, target_map_id):
                        return None
                    continue

                y_now = int(nav.get("y", 0))
                if curr_map_id in {1, 12} or (curr_map_id == 13 and y_now >= 66):
                    self._emit(
                        "traverse_offroute_reroute",
                        route_step_index=step_index,
                        target_map_id=target_map_id,
                        traverse_current_map_id=curr_map_id,
                        traverse_current_y=y_now,
                        traverse_attempt=traverse_attempt,
                    )
                    self.no_progress_steps = 0
                    self._reset_nav_memory()
                    self.last_position = self._position_key()
                    return self.current_block_start
                if curr_map_id == 50 and y_now <= 6:
                    self._emit(
                        "traverse_north_gate_push",
                        route_step_index=step_index,
                        target_map_id=target_map_id,
                        traverse_attempt=traverse_attempt,
                        traverse_current_y=y_now,
                    )
                    for push_step in range(64):
                        redirect = self._press_button(
                            "up",
                            hold_frames=hold_frames,
                            step_index=step_index,
                            suppress_no_progress=True,
                        )
                        if redirect is not None:
                            return redirect
                        after_nav = self._nav_state()
                        after_map = int(after_nav.get("map_id", 0))
                        if self._map_matches(after_map, target_map_id):
                            return None
                        if after_map not in {50, 47, 13, 2}:
                            break
                        if after_map == 13 and int(after_nav.get("y", 0)) >= 40:
                            break
                        if push_step > 0 and push_step % 16 == 0:
                            self._emit(
                                "traverse_north_gate_push_progress",
                                route_step_index=step_index,
                                target_map_id=target_map_id,
                                traverse_current_map_id=after_map,
                                traverse_current_y=int(after_nav.get("y", 0)),
                                traverse_push_steps=push_step,
                            )
                if curr_map_id in {13, 47} and y_now <= 15:
                    self._emit(
                        "traverse_route2_north_push",
                        route_step_index=step_index,
                        target_map_id=target_map_id,
                        traverse_attempt=traverse_attempt,
                        traverse_current_y=y_now,
                    )
                    for push_step in range(64):
                        redirect = self._press_button(
                            "up",
                            hold_frames=hold_frames,
                            step_index=step_index,
                            suppress_no_progress=True,
                        )
                        if redirect is not None:
                            return redirect
                        after_nav = self._nav_state()
                        after_map = int(after_nav.get("map_id", 0))
                        if self._map_matches(after_map, target_map_id):
                            return None
                        if after_map not in {13, 47, 50, 2}:
                            break
                        if after_map == 50 and int(after_nav.get("y", 0)) >= 40:
                            break
                if i > 0 and i % 256 == 0:
                    seek_budget = min(420, max(120, max_steps - i))
                    if self._seek_expected_map(
                        expected_map_id=target_map_id,
                        step_index=step_index,
                        max_seek_steps=seek_budget,
                        suppress_regression=True,
                    ):
                        return None
            if (
                self.phase4_scope == "route_only"
                and self.current_checkpoint_expected_map_id >= 0
                and self.current_checkpoint_name == "Viridian_Forest"
                and not self._map_matches(curr_map_id, int(self.current_checkpoint_expected_map_id))
            ):
                if self._seek_expected_map(
                    expected_map_id=int(self.current_checkpoint_expected_map_id),
                    step_index=step_index,
                    max_seek_steps=420,
                    suppress_regression=True,
                ):
                    continue
                return self._trigger_recovery(
                    reason="traverse_map_lost",
                    step_index=step_index,
                    target_map_id=target_map_id,
                    traverse_current_map_id=curr_map_id,
                )

            pos = (curr_map_id, int(nav.get("x", 0)), int(nav.get("y", 0)))
            unique_positions.add(pos)
            if i > 0 and i % 256 == 0:
                self._emit(
                    "traverse_progress",
                    route_step_index=step_index,
                    target_map_id=target_map_id,
                    traverse_steps=i,
                    traverse_unique_positions=len(unique_positions),
                    traverse_current_map_id=curr_map_id,
                )
            state_key = (pos[0], pos[1], pos[2], heading, hand)
            seen_count = int(seen_states.get(state_key, 0)) + 1
            seen_states[state_key] = seen_count

            ordered = self._wall_follow_order(heading=heading, hand=hand)
            action = ordered[0]
            for candidate in ordered:
                if not self._is_edge_blocked(pos, candidate):
                    action = candidate
                    break

            redirect, moved, _, after = self._attempt_nav_action(
                action=action,
                hold_frames=hold_frames,
                step_index=step_index,
            )
            if redirect is not None:
                return redirect
            if moved:
                heading = action
                no_move_cycles = 0
                stuck_events = 0
                if self._map_matches(int(after[0]), target_map_id):
                    return None
            else:
                no_move_cycles += 1

            if no_move_cycles >= 24 or seen_count >= 12:
                stuck_events += 1
                hand = "right" if hand == "left" else "left"
                no_move_cycles = 0
                seen_states.clear()
                if i % 32 == 0:
                    redirect = self._press_button(
                        "b",
                        hold_frames=2,
                        step_index=step_index,
                        suppress_no_progress=True,
                    )
                    if redirect is not None:
                        return redirect
                if stuck_events <= 10:
                    continue
                return self._trigger_recovery(
                    reason="traverse_loop_stuck",
                    step_index=step_index,
                    target_map_id=target_map_id,
                )

        return self._trigger_recovery(
            reason="traverse_timeout",
            step_index=step_index,
            target_map_id=target_map_id,
        )

    def _condition_met(self, condition: str, value: int | None = None) -> bool:
        nav = self._nav_state()
        if condition == "map_id_is":
            return self._map_matches(
                int(nav.get("map_id", -1)),
                int(value if value is not None else -1),
            )
        if condition == "not_in_battle":
            return int(nav.get("in_battle", 0)) == 0
        if condition == "in_battle":
            return int(nav.get("in_battle", 0)) != 0
        return False

    def _next_checkpoint_expected_map_id(self, step_index: int) -> int | None:
        for idx in range(int(step_index) + 1, len(self.steps)):
            row = self.steps[idx]
            row_type = str(row.get("type", ""))
            if row_type == "checkpoint":
                return int(row["expected_map_id"])
            if row_type == "waypoint":
                return int(row["map_id"])
            if row_type == "traverse_until_map":
                return int(row["target_map_id"])
        return None

    def _execute_step(self, step_index: int, step: Dict[str, object]) -> int | None:
        step_type = str(step["type"])
        if step_type == "checkpoint":
            nav = self._nav_state()
            expected_map_id = int(step["expected_map_id"])
            current_map_id = int(nav.get("map_id", -1))
            if not self._map_matches(current_map_id, expected_map_id):
                seek_ok = self._seek_expected_map(expected_map_id=expected_map_id, step_index=step_index)
                if seek_ok:
                    self._mark_checkpoint_reached(
                        step_index=step_index,
                        checkpoint_name=str(step["name"]),
                        expected_map_id=expected_map_id,
                        allowed_map_ids=[int(v) for v in step.get("allowed_map_ids", [])],
                    )
                    return None
                return self._trigger_recovery(
                    reason="checkpoint_mismatch",
                    step_index=step_index,
                    expected_map_id=expected_map_id,
                    current_map_id=current_map_id,
                )
            self._mark_checkpoint_reached(
                step_index=step_index,
                checkpoint_name=str(step["name"]),
                expected_map_id=expected_map_id,
                allowed_map_ids=[int(v) for v in step.get("allowed_map_ids", [])],
            )
            return None

        if step_type == "move":
            direction = str(step["direction"])
            hold_frames = int(step["hold_frames"])
            target_forward = int(step["steps"])
            start_map_id = int(self._nav_state().get("map_id", 0))
            next_checkpoint_map_id = self._next_checkpoint_expected_map_id(step_index)
            if (
                next_checkpoint_map_id is not None
                and not self._map_matches(start_map_id, int(next_checkpoint_map_id))
            ):
                seek_budget = max(320, target_forward * 48)
                if self._seek_expected_map(
                    expected_map_id=next_checkpoint_map_id,
                    step_index=step_index,
                    max_seek_steps=seek_budget,
                ):
                    return None
                return self._trigger_recovery(
                    reason="move_transition_mismatch",
                    step_index=step_index,
                    direction=direction,
                    expected_map_id=next_checkpoint_map_id,
                )
            forward_progress = 0
            prefer_right = True
            no_forward_cycles = 0
            hard_stuck_cycles = 0
            while forward_progress < target_forward:
                gained = False
                moved_any = False
                stagnant_presses = 0
                for _ in range(44):
                    action = self._select_roomba_action(
                        direction=direction,
                        pos=self._position_key(),
                        prefer_right=prefer_right,
                    )
                    redirect, moved, before, after = self._attempt_nav_action(
                        action=action,
                        hold_frames=hold_frames,
                        step_index=step_index,
                    )
                    if redirect is not None:
                        return redirect
                    if not moved:
                        stagnant_presses += 1
                        if not moved_any and stagnant_presses >= 12:
                            break
                        continue
                    if (
                        next_checkpoint_map_id is not None
                        and not self._map_matches(start_map_id, int(next_checkpoint_map_id))
                        and self._map_matches(int(after[0]), int(next_checkpoint_map_id))
                    ):
                        return None
                    moved_any = True
                    stagnant_presses = 0
                    if self._is_forward_progress(direction, before, after):
                        forward_progress += 1
                        gained = True
                        break

                if gained:
                    no_forward_cycles = 0
                    hard_stuck_cycles = 0
                    continue

                no_forward_cycles += 1
                if moved_any:
                    hard_stuck_cycles = 0
                else:
                    hard_stuck_cycles += 1
                prefer_right = not prefer_right

                if no_forward_cycles % 8 == 0:
                    redirect, escaped = self._corner_escape(
                        direction=direction,
                        hold_frames=hold_frames,
                        step_index=step_index,
                        prefer_right=prefer_right,
                    )
                    if redirect is not None:
                        return redirect
                    if escaped:
                        no_forward_cycles = 0
                        hard_stuck_cycles = 0
                        continue

                if hard_stuck_cycles >= 4 or no_forward_cycles >= 48:
                    return self._trigger_recovery(
                        reason="move_blocked",
                        step_index=step_index,
                        direction=direction,
                    )
            return None

        if step_type == "waypoint":
            return self._run_waypoint(step_index=step_index, step=step)

        if step_type == "traverse_until_map":
            nav = self._nav_state()
            if (
                self.phase4_scope == "route_only"
                and self.current_checkpoint_expected_map_id >= 0
                and not self._checkpoint_map_allowed(int(nav.get("map_id", -1)))
            ):
                if self._seek_expected_map(
                    expected_map_id=int(self.current_checkpoint_expected_map_id),
                    step_index=step_index,
                    max_seek_steps=640,
                    suppress_regression=True,
                ):
                    self._emit(
                        "traverse_checkpoint_reacquired",
                        route_step_index=step_index,
                        expected_map_id=int(self.current_checkpoint_expected_map_id),
                    )
                else:
                    return self._trigger_recovery(
                        reason="traverse_checkpoint_map_mismatch",
                        step_index=step_index,
                        expected_map_id=int(self.current_checkpoint_expected_map_id),
                        allowed_map_ids=self.current_checkpoint_allowed_map_ids,
                    )
            return self._run_traverse_until_map(step_index=step_index, step=step)

        if step_type == "press":
            for _ in range(int(step["count"])):
                redirect = self._press_button(
                    button=str(step["button"]),
                    hold_frames=int(step["hold_frames"]),
                    step_index=step_index,
                )
                if redirect is not None:
                    return redirect
            return None

        if step_type == "wait_until":
            condition = str(step["condition"])
            value = int(step["value"]) if "value" in step else None
            timeout_ticks = int(step["timeout_ticks"])
            for _ in range(timeout_ticks):
                if self._condition_met(condition, value):
                    return None
                redirect = self._tick_and_check(step_index=step_index)
                if redirect is not None:
                    return redirect
            if self._condition_met(condition, value):
                return None
            return self._trigger_recovery(
                reason="wait_until_timeout",
                step_index=step_index,
                condition=condition,
            )

        if step_type == "interact":
            button = str(step["button"])
            attempts = int(step["attempts"])
            timeout_ticks = int(step["timeout_ticks"])
            hold_frames = int(step["hold_frames"])
            for _ in range(attempts):
                before = self._position_key()
                redirect = self._press_button(
                    button=button,
                    hold_frames=hold_frames,
                    step_index=step_index,
                )
                if redirect is not None:
                    return redirect

                for _ in range(timeout_ticks):
                    if self._position_key() != before:
                        return None
                    redirect = self._tick_and_check(step_index=step_index)
                    if redirect is not None:
                        return redirect
            return self._trigger_recovery(
                reason="interact_timeout",
                step_index=step_index,
                button=button,
            )

        raise RouteExecutionError(f"unsupported step type at runtime: {step_type}")

    def run(self) -> Dict[str, object]:
        self.emu.reset(state_path=self.start_state_path)
        nav = self._nav_state()
        self.last_position = (
            int(nav.get("map_id", 0)),
            int(nav.get("x", 0)),
            int(nav.get("y", 0)),
        )
        self.last_observed_map_id = int(nav.get("map_id", 0))
        step_index = 0
        run_status = "failed"
        target_reached = False
        failure_reason = ""

        try:
            self._validate_party_constraint(stage="run_start", step_index=-1)
            while step_index < len(self.steps):
                if self._target_reached():
                    run_status = "success"
                    target_reached = True
                    break

                step = self.steps[step_index]
                self._emit("route_step_start", route_step_index=step_index, step_type=step["type"])
                redirect = self._execute_step(step_index, step)
                if redirect is None:
                    self._emit("route_step_end", route_step_index=step_index, status="ok")
                    step_index += 1
                else:
                    self._emit(
                        "route_step_end",
                        route_step_index=step_index,
                        status="reroute",
                        next_step_index=int(redirect),
                    )
                    step_index = max(0, int(redirect))

            if run_status != "success":
                target_reached = self._target_reached()
                if target_reached:
                    run_status = "success"
                else:
                    run_status = "failed"
                    failure_reason = "route_complete_without_target"
        except TimeoutError as exc:
            run_status = "timeout"
            target_reached = self._target_reached()
            failure_reason = str(exc)
        except RouteExecutionError as exc:
            run_status = "failed"
            target_reached = self._target_reached()
            failure_reason = str(exc)
        except Exception as exc:
            run_status = "failed"
            target_reached = self._target_reached()
            failure_reason = f"unhandled_exception:{exc.__class__.__name__}"

        final_constraint = self._party_constraint_status()
        if isinstance(final_constraint.get("snapshot"), dict):
            self.last_party_snapshot = dict(final_constraint.get("snapshot", {}))
        if self.enforce_party_constraint:
            final_passed = bool(final_constraint.get("passed", False))
            final_reason = str(final_constraint.get("reason", "unknown"))
            self._emit(
                "constraint_check",
                constraint_stage="run_end",
                constraint_passed=final_passed,
                constraint_reason=final_reason,
                route_step_index=int(step_index),
                party_snapshot=self.last_party_snapshot,
            )
            if not final_passed:
                self.party_constraint_passed = False
                if not self.party_constraint_failure_reason:
                    self.party_constraint_failure_reason = final_reason
                self._emit(
                    "constraint_failed",
                    constraint_stage="run_end",
                    constraint_reason=self.party_constraint_failure_reason,
                    route_step_index=int(step_index),
                    party_snapshot=self.last_party_snapshot,
                )
                if run_status == "success":
                    run_status = "failed"
                    failure_reason = "party_constraint_failed"
                    target_reached = False

        self._emit(
            "run_terminated",
            run_status=run_status,
            target=self.target,
            target_reached=target_reached,
            failure_reason=failure_reason,
        )

        fallback_events = int(getattr(self.agent, "legality_fallback_count", 0)) + int(
            getattr(self.agent, "budget_fallback_count", 0)
        )
        battle_turns_total = int(self.battle_turns_total)
        battles_with_turns = max(1, int(self.battles_fought))
        avg_battle_turns = round(float(battle_turns_total) / float(battles_with_turns), 4)
        final_party_species = [int(v) for v in self.last_party_snapshot.get("party_species_ids", [])]
        final_party_levels = [int(v) for v in self.last_party_snapshot.get("party_levels", [])]
        return {
            "run_status": run_status,
            "target": self.target,
            "phase4_scope": self.phase4_scope,
            "target_reached": bool(target_reached),
            "start_state": str(self.start_state_path),
            "route_script": str(self.route_script_path),
            "steps_executed": int(self.steps_executed),
            "checkpoints_reached": int(self.checkpoints_reached),
            "battles_fought": int(self.battles_fought),
            "wild_battles": int(self.wild_battles),
            "trainer_battles": int(self.trainer_battles),
            "wild_run_attempts": int(self.wild_run_attempts),
            "wild_run_successes": int(self.wild_run_successes),
            "wild_battle_mode": self.wild_battle_mode,
            "farm_hp_threshold": float(self.farm_hp_threshold),
            "farm_max_consecutive_fights": int(self.farm_max_consecutive_fights),
            "wild_fights_committed": int(self.wild_fights_committed),
            "wild_runs_forced": int(self.wild_runs_forced),
            "battle_turns_total": battle_turns_total,
            "avg_battle_turns": avg_battle_turns,
            "battle_policy_mode": self.policy_mode,
            "llm_decision_calls": int(getattr(self.agent, "llm_decision_calls", 0)),
            "llm_reflection_calls": int(getattr(self.agent, "llm_reflection_calls", 0)),
            "fallback_events": int(fallback_events),
            "recovery_events": int(self.recovery_events),
            "constraint_enforced": bool(self.enforce_party_constraint),
            "single_pokemon_only": bool(self.single_pokemon_only),
            "required_species_id": self.required_species_id,
            "party_constraint_passed": bool(self.party_constraint_passed),
            "party_constraint_failure_reason": self.party_constraint_failure_reason,
            "active_species_id_final": int(self.last_party_snapshot.get("active_species_id", 0)),
            "active_level_final": int(self.last_party_snapshot.get("active_level", 0)),
            "party_species_ids_final": final_party_species,
            "party_levels_final": final_party_levels,
            "observed_map_transitions": dict(sorted(self.observed_map_transitions.items())),
            "forest_transition_profile": {
                "edges": dict(sorted(self.forest_probe_edges.items())),
                "phase_counts": dict(sorted(self.forest_probe_phase_counts.items())),
                "samples": list(self.forest_probe_samples),
            },
            "failure_reason": failure_reason,
        }


def run_phase4_route(
    *,
    emu: object,
    agent: object,
    start_state_path: Path,
    route_script_path: Path,
    timeline_path: Path,
    max_steps: int = 15000,
    policy_mode: str = "hybrid",
    target: str = "gym_entrance",
    phase4_scope: str = "integrated",
    wild_run_enabled: bool = True,
    wild_battle_mode: str = "run_first",
    farm_hp_threshold: float = 0.45,
    farm_max_consecutive_fights: int = 3,
    no_progress_limit: int = 40,
    llm_turn_interval: int = 3,
    max_decision_calls: int = 120,
    turn_tick_budget: int = 1200,
    max_battle_turns: int = 120,
    required_species_id: int | None = None,
    single_pokemon_only: bool = False,
    enforce_party_constraint: bool = False,
) -> Dict[str, object]:
    route = load_route_script(route_script_path)
    timeline = _TimelineWriter(timeline_path)
    try:
        runtime = _Runtime(
            emu=emu,
            agent=agent,
            route=route,
            route_script_path=Path(route_script_path),
            start_state_path=Path(start_state_path),
            timeline=timeline,
            max_steps=max_steps,
            policy_mode=policy_mode,
            target=target,
            phase4_scope=phase4_scope,
            wild_run_enabled=wild_run_enabled,
            wild_battle_mode=wild_battle_mode,
            farm_hp_threshold=farm_hp_threshold,
            farm_max_consecutive_fights=farm_max_consecutive_fights,
            no_progress_limit=no_progress_limit,
            llm_turn_interval=llm_turn_interval,
            max_decision_calls=max_decision_calls,
            turn_tick_budget=turn_tick_budget,
            max_battle_turns=max_battle_turns,
            required_species_id=required_species_id,
            single_pokemon_only=single_pokemon_only,
            enforce_party_constraint=enforce_party_constraint,
        )
        return runtime.run()
    finally:
        timeline.close()
