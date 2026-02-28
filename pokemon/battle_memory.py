from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class BattleMemory:
    def __init__(self, path: Path, snapshot_keep: int = 10) -> None:
        self.path = Path(path)
        self.snapshot_keep = max(1, int(snapshot_keep))
        self.history_dir = self.path.parent / "history"

        # State-level lessons (Phase 2B baseline)
        self.rules: Dict[str, Dict[str, object]] = {}
        # Matchup-level lessons (Phase 2C)
        self.matchup_lessons: Dict[str, Dict[str, object]] = {}
        # Global lesson (Phase 2C)
        self.global_lesson: Dict[str, object] = self._empty_lesson(
            scope="global",
            key="global",
            state_label="all_states",
            enemy_species_id=0,
            player_species_id=0,
            move_ids=[],
            legal_slots=[0, 1, 2, 3],
        )

        self.rules_loaded = 0
        self.memory_hint_count = 0
        self.memory_override_count = 0
        self.repeated_mistake_events = 0
        self.decisions_logged = 0

        self.memory_hint_state_count = 0
        self.memory_hint_matchup_count = 0
        self.memory_hint_global_count = 0
        self.memory_override_state_count = 0
        self.memory_override_matchup_count = 0
        self.memory_override_global_count = 0

        self._episode_turns: List[Tuple[Dict[str, object], int]] = []
        self._load()

    def _new_payload(self) -> Dict[str, object]:
        return {
            "version": 2,
            "updated_at": "",
            # Back-compat key retained.
            "rules": [],
            "state_rules": [],
            "matchup_lessons": [],
            "global_lesson": {},
            "stats": {
                "memory_hint_count": 0,
                "memory_override_count": 0,
                "repeated_mistake_events": 0,
                "decisions_logged": 0,
                "memory_hint_state_count": 0,
                "memory_hint_matchup_count": 0,
                "memory_hint_global_count": 0,
                "memory_override_state_count": 0,
                "memory_override_matchup_count": 0,
                "memory_override_global_count": 0,
            },
        }

    def _normalize_slot_stats(self, raw: object) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if not isinstance(raw, dict):
            raw = {}
        for idx in range(4):
            key = str(idx)
            row = raw.get(key, {})
            if not isinstance(row, dict):
                row = {}
            out[key] = {
                "samples": float(_as_int(row.get("samples", 0))),
                "wins": float(_as_int(row.get("wins", 0))),
                "losses": float(_as_int(row.get("losses", 0))),
                "avg_reward": float(_as_float(row.get("avg_reward", 0.0))),
                "avg_turns": float(_as_float(row.get("avg_turns", 0.0))),
            }
        return out

    def _empty_lesson(
        self,
        *,
        scope: str,
        key: str,
        state_label: str,
        enemy_species_id: int,
        player_species_id: int,
        move_ids: List[int],
        legal_slots: List[int],
    ) -> Dict[str, object]:
        return {
            "lesson_scope": scope,
            "lesson_key": key,
            "signature": key if scope == "state" else "",
            "state_label": state_label,
            "enemy_species_id": int(enemy_species_id),
            "player_species_id": int(player_species_id),
            "move_ids": [int(v) for v in move_ids[:4]],
            "legal_slots": sorted({int(v) for v in legal_slots if 0 <= int(v) <= 3}),
            "slot_stats": self._normalize_slot_stats({}),
            "preferred_slots": [],
            "blocked_slots": [],
            "best_reward": 0.0,
            "best_turns": 0.0,
            "samples_total": 0,
            "last_updated_at": _utc_now(),
        }

    def _normalize_lesson(self, raw: object, scope: str) -> Dict[str, object] | None:
        if not isinstance(raw, dict):
            return None

        if scope == "state":
            key = str(raw.get("signature", "")).strip()
            if not key:
                key = str(raw.get("lesson_key", "")).strip()
            if not key:
                return None
        elif scope == "matchup":
            key = str(raw.get("lesson_key", "")).strip()
            if not key:
                player = _as_int(raw.get("player_species_id", 0))
                enemy = _as_int(raw.get("enemy_species_id", 0))
                key = self._matchup_key(player_species_id=player, enemy_species_id=enemy)
        else:
            key = "global"

        move_ids = [_as_int(v) for v in (raw.get("move_ids") or [])[:4]]
        legal_slots = sorted(
            {
                idx
                for idx in [_as_int(v, -1) for v in (raw.get("legal_slots") or [])[:4]]
                if 0 <= idx <= 3
            }
        )
        if scope == "global" and not legal_slots:
            legal_slots = [0, 1, 2, 3]

        preferred_slots = sorted(
            {
                idx
                for idx in [_as_int(v, -1) for v in (raw.get("preferred_slots") or [])]
                if 0 <= idx <= 3
            }
        )
        blocked_slots = sorted(
            {
                idx
                for idx in [_as_int(v, -1) for v in (raw.get("blocked_slots") or [])]
                if 0 <= idx <= 3
            }
        )

        return {
            "lesson_scope": scope,
            "lesson_key": key,
            "signature": key if scope == "state" else "",
            "state_label": str(raw.get("state_label", "unknown_state")),
            "enemy_species_id": _as_int(raw.get("enemy_species_id", 0)),
            "player_species_id": _as_int(raw.get("player_species_id", 0)),
            "move_ids": move_ids,
            "legal_slots": legal_slots,
            "slot_stats": self._normalize_slot_stats(raw.get("slot_stats")),
            "preferred_slots": preferred_slots,
            "blocked_slots": blocked_slots,
            "best_reward": float(_as_float(raw.get("best_reward", 0.0))),
            "best_turns": float(_as_float(raw.get("best_turns", 0.0))),
            "samples_total": _as_int(raw.get("samples_total", 0)),
            "last_updated_at": str(raw.get("last_updated_at", "")),
        }

    def _load(self) -> None:
        payload = self._new_payload()
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                payload = self._new_payload()

        state_rules_raw = payload.get("state_rules")
        if not isinstance(state_rules_raw, list):
            # Back-compat with old format.
            state_rules_raw = payload.get("rules", [])
        if not isinstance(state_rules_raw, list):
            state_rules_raw = []

        for raw in state_rules_raw:
            lesson = self._normalize_lesson(raw, scope="state")
            if lesson is None:
                continue
            key = str(lesson["lesson_key"])
            self.rules[key] = lesson

        matchup_raw = payload.get("matchup_lessons", [])
        if not isinstance(matchup_raw, list):
            matchup_raw = []
        for raw in matchup_raw:
            lesson = self._normalize_lesson(raw, scope="matchup")
            if lesson is None:
                continue
            key = str(lesson["lesson_key"])
            self.matchup_lessons[key] = lesson

        global_lesson = self._normalize_lesson(payload.get("global_lesson", {}), scope="global")
        if global_lesson is not None:
            self.global_lesson = global_lesson

        # If older file had no hierarchy, derive it from state rules.
        if not self.matchup_lessons and _as_int(self.global_lesson.get("samples_total", 0)) == 0:
            self._rebuild_hierarchy()

        self.rules_loaded = len(self.rules)
        stats = payload.get("stats", {})
        if isinstance(stats, dict):
            self.memory_hint_count = _as_int(stats.get("memory_hint_count", 0))
            self.memory_override_count = _as_int(stats.get("memory_override_count", 0))
            self.repeated_mistake_events = _as_int(stats.get("repeated_mistake_events", 0))
            self.decisions_logged = _as_int(stats.get("decisions_logged", 0))
            self.memory_hint_state_count = _as_int(stats.get("memory_hint_state_count", 0))
            self.memory_hint_matchup_count = _as_int(stats.get("memory_hint_matchup_count", 0))
            self.memory_hint_global_count = _as_int(stats.get("memory_hint_global_count", 0))
            self.memory_override_state_count = _as_int(stats.get("memory_override_state_count", 0))
            self.memory_override_matchup_count = _as_int(stats.get("memory_override_matchup_count", 0))
            self.memory_override_global_count = _as_int(stats.get("memory_override_global_count", 0))

    def _slot_update(self, row: Dict[str, float], reward: float, turns: int, won: bool) -> None:
        samples = int(row["samples"])
        new_samples = samples + 1
        row["avg_reward"] = ((row["avg_reward"] * samples) + reward) / max(1, new_samples)
        row["avg_turns"] = ((row["avg_turns"] * samples) + float(turns)) / max(1, new_samples)
        row["samples"] = float(new_samples)
        if won:
            row["wins"] = float(int(row["wins"]) + 1)
        else:
            row["losses"] = float(int(row["losses"]) + 1)

    def _compute_pref_block(
        self,
        slot_stats: Dict[str, Dict[str, float]],
        *,
        block_reward_gap: float,
    ) -> Tuple[List[int], List[int], float, float, int]:
        ranked: List[Tuple[int, float, float, int]] = []
        samples_total = 0
        for key, row in slot_stats.items():
            if not isinstance(row, dict):
                continue
            samples = _as_int(row.get("samples", 0))
            if samples <= 0:
                continue
            slot = _as_int(key, -1)
            if slot < 0 or slot > 3:
                continue
            avg_reward = _as_float(row.get("avg_reward", 0.0))
            avg_turns = _as_float(row.get("avg_turns", 999.0))
            samples_total += samples
            ranked.append((slot, avg_reward, avg_turns, samples))
        ranked.sort(key=lambda item: (-item[1], item[2], item[0]))

        preferred_slots = [ranked[0][0]] if ranked else []
        blocked_slots: List[int] = []
        best_reward = ranked[0][1] if ranked else 0.0
        best_turns = ranked[0][2] if ranked else 0.0
        for slot, avg_reward, _, samples in ranked:
            if slot in preferred_slots:
                continue
            if samples >= 2 and avg_reward <= (best_reward - block_reward_gap):
                blocked_slots.append(slot)

        return preferred_slots, blocked_slots, float(best_reward), float(best_turns), int(samples_total)

    def _recompute_lesson(self, lesson: Dict[str, object], *, block_reward_gap: float) -> None:
        slot_stats = lesson["slot_stats"]
        assert isinstance(slot_stats, dict)
        preferred, blocked, best_reward, best_turns, samples_total = self._compute_pref_block(
            slot_stats,
            block_reward_gap=block_reward_gap,
        )
        lesson["preferred_slots"] = preferred
        lesson["blocked_slots"] = blocked
        lesson["best_reward"] = best_reward
        lesson["best_turns"] = best_turns
        lesson["samples_total"] = samples_total
        lesson["last_updated_at"] = _utc_now()

    def _matchup_key(self, *, player_species_id: int, enemy_species_id: int) -> str:
        return f"p{int(player_species_id)}_e{int(enemy_species_id)}"

    def _ensure_rule(self, state_meta: Dict[str, object]) -> Dict[str, object]:
        signature = str(state_meta["signature"])
        existing = self.rules.get(signature)
        if existing is not None:
            return existing
        lesson = self._empty_lesson(
            scope="state",
            key=signature,
            state_label=str(state_meta.get("state_label", "unknown_state")),
            enemy_species_id=_as_int(state_meta.get("enemy_species_id", 0)),
            player_species_id=_as_int(state_meta.get("player_species_id", 0)),
            move_ids=[_as_int(v) for v in (state_meta.get("move_ids") or [])[:4]],
            legal_slots=[_as_int(v) for v in (state_meta.get("legal_slots") or [])[:4]],
        )
        lesson["signature"] = signature
        self.rules[signature] = lesson
        return lesson

    def state_signature(self, state: Dict[str, object]) -> Dict[str, object]:
        enemy_species_id = _as_int(state.get("enemy_species_id", state.get("enemy_species", 0)))
        player_species_id = _as_int(state.get("player_species_id", state.get("player_species", 0)))
        move_ids = [_as_int(v) for v in (state.get("move_ids") or [])[:4]]
        legal_slots = sorted(
            {
                idx
                for idx in [_as_int(v, -1) for v in (state.get("legal_slots") or [])[:4]]
                if 0 <= idx <= 3
            }
        )
        state_label = str(state.get("state_label", "unknown_state"))
        key = {
            "state_label": state_label,
            "enemy_species_id": enemy_species_id,
            "player_species_id": player_species_id,
            "move_ids": move_ids,
            "legal_slots": legal_slots,
        }
        return {
            "signature": json.dumps(key, sort_keys=True, separators=(",", ":")),
            "state_label": state_label,
            "enemy_species_id": enemy_species_id,
            "player_species_id": player_species_id,
            "move_ids": move_ids,
            "legal_slots": legal_slots,
        }

    def lookup_rule(self, state: Dict[str, object]) -> Dict[str, object] | None:
        signature = self.state_signature(state)
        return self.rules.get(str(signature["signature"]))

    def _lookup_matchup_lesson(self, state: Dict[str, object]) -> Dict[str, object] | None:
        state_meta = self.state_signature(state)
        key = self._matchup_key(
            player_species_id=_as_int(state_meta.get("player_species_id", 0)),
            enemy_species_id=_as_int(state_meta.get("enemy_species_id", 0)),
        )
        return self.matchup_lessons.get(key)

    def _collect_lessons(self, state: Dict[str, object]) -> List[Tuple[str, Dict[str, object]]]:
        out: List[Tuple[str, Dict[str, object]]] = []
        state_rule = self.lookup_rule(state)
        if state_rule is not None:
            out.append(("state", state_rule))
        matchup_rule = self._lookup_matchup_lesson(state)
        if matchup_rule is not None:
            out.append(("matchup", matchup_rule))
        if _as_int(self.global_lesson.get("samples_total", 0)) > 0:
            out.append(("global", self.global_lesson))
        return out

    def prompt_hint(self, state: Dict[str, object]) -> str:
        lessons = self._collect_lessons(state)
        if not lessons:
            return ""

        lines: List[str] = []
        used_scopes: set[str] = set()
        for scope, rule in lessons:
            preferred = [int(v) for v in (rule.get("preferred_slots") or []) if 0 <= _as_int(v, -1) <= 3]
            blocked = [int(v) for v in (rule.get("blocked_slots") or []) if 0 <= _as_int(v, -1) <= 3]
            if not preferred and not blocked:
                continue
            used_scopes.add(scope)
            if scope == "state":
                lines.append(f"State lesson: prefer {preferred}, avoid {blocked}.")
            elif scope == "matchup":
                lines.append(
                    "Matchup lesson: "
                    f"P{rule.get('player_species_id',0)} vs E{rule.get('enemy_species_id',0)} "
                    f"prefer {preferred}, avoid {blocked}."
                )
            else:
                lines.append(f"Global lesson: prefer {preferred}, avoid {blocked}.")

        if not lines:
            return ""

        self.memory_hint_count += 1
        if "state" in used_scopes:
            self.memory_hint_state_count += 1
        if "matchup" in used_scopes:
            self.memory_hint_matchup_count += 1
        if "global" in used_scopes:
            self.memory_hint_global_count += 1

        return "Memory guidance:\n- " + "\n- ".join(lines)

    def maybe_override_slot(self, state: Dict[str, object], chosen_slot: int) -> Tuple[int, bool]:
        legal_slots = {
            _as_int(v, -1)
            for v in (state.get("legal_slots") or [])
            if 0 <= _as_int(v, -1) <= 3
        }
        if not legal_slots:
            return int(chosen_slot), False

        for scope, rule in self._collect_lessons(state):
            blocked = {int(v) for v in (rule.get("blocked_slots") or []) if 0 <= _as_int(v, -1) <= 3}
            if int(chosen_slot) not in blocked:
                continue
            preferred = [int(v) for v in (rule.get("preferred_slots") or []) if 0 <= _as_int(v, -1) <= 3]
            for slot in preferred:
                if slot in legal_slots:
                    self.memory_override_count += 1
                    if scope == "state":
                        self.memory_override_state_count += 1
                    elif scope == "matchup":
                        self.memory_override_matchup_count += 1
                    else:
                        self.memory_override_global_count += 1
                    return slot, True

        return int(chosen_slot), False

    def start_episode(self) -> None:
        self._episode_turns = []

    def record_turn(self, state: Dict[str, object], chosen_slot: int) -> None:
        state_meta = self.state_signature(state)
        self._episode_turns.append((state_meta, int(chosen_slot)))
        self.decisions_logged += 1

    def _build_aggregate_slot_stats(
        self,
        aggregate: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        out = self._normalize_slot_stats({})
        for slot_key, row in aggregate.items():
            samples = _as_int(row.get("samples", 0))
            if samples <= 0:
                continue
            reward_sum = _as_float(row.get("reward_sum", 0.0))
            turn_sum = _as_float(row.get("turn_sum", 0.0))
            wins = _as_int(row.get("wins", 0))
            losses = _as_int(row.get("losses", 0))
            out[slot_key] = {
                "samples": float(samples),
                "wins": float(wins),
                "losses": float(losses),
                "avg_reward": float(reward_sum / max(1, samples)),
                "avg_turns": float(turn_sum / max(1, samples)),
            }
        return out

    def _accumulate_slot_stats(
        self,
        aggregate: Dict[str, Dict[str, float]],
        slot_key: str,
        row: Dict[str, float],
    ) -> None:
        samples = _as_int(row.get("samples", 0))
        if samples <= 0:
            return
        reward_sum = _as_float(row.get("avg_reward", 0.0)) * samples
        turn_sum = _as_float(row.get("avg_turns", 0.0)) * samples
        agg = aggregate.get(slot_key)
        if agg is None:
            agg = {
                "samples": 0.0,
                "wins": 0.0,
                "losses": 0.0,
                "reward_sum": 0.0,
                "turn_sum": 0.0,
            }
            aggregate[slot_key] = agg
        agg["samples"] = float(_as_int(agg.get("samples", 0)) + samples)
        agg["wins"] = float(_as_int(agg.get("wins", 0)) + _as_int(row.get("wins", 0)))
        agg["losses"] = float(_as_int(agg.get("losses", 0)) + _as_int(row.get("losses", 0)))
        agg["reward_sum"] = float(_as_float(agg.get("reward_sum", 0.0)) + reward_sum)
        agg["turn_sum"] = float(_as_float(agg.get("turn_sum", 0.0)) + turn_sum)

    def _rebuild_hierarchy(self) -> None:
        matchup_aggregates: Dict[str, Dict[str, object]] = {}
        global_aggregate: Dict[str, Dict[str, float]] = {}

        for state_rule in self.rules.values():
            slot_stats = state_rule.get("slot_stats", {})
            if not isinstance(slot_stats, dict):
                continue

            matchup_key = self._matchup_key(
                player_species_id=_as_int(state_rule.get("player_species_id", 0)),
                enemy_species_id=_as_int(state_rule.get("enemy_species_id", 0)),
            )
            matchup = matchup_aggregates.get(matchup_key)
            if matchup is None:
                matchup = {
                    "lesson_scope": "matchup",
                    "lesson_key": matchup_key,
                    "signature": "",
                    "state_label": "multi_state",
                    "enemy_species_id": _as_int(state_rule.get("enemy_species_id", 0)),
                    "player_species_id": _as_int(state_rule.get("player_species_id", 0)),
                    "move_ids": [],
                    "legal_slots": [0, 1, 2, 3],
                    "slot_stats_agg": {},
                    "preferred_slots": [],
                    "blocked_slots": [],
                    "best_reward": 0.0,
                    "best_turns": 0.0,
                    "samples_total": 0,
                    "last_updated_at": _utc_now(),
                }
                matchup_aggregates[matchup_key] = matchup

            agg_slots = matchup.get("slot_stats_agg", {})
            if not isinstance(agg_slots, dict):
                agg_slots = {}
                matchup["slot_stats_agg"] = agg_slots

            for slot_key, row in slot_stats.items():
                if not isinstance(row, dict):
                    continue
                self._accumulate_slot_stats(agg_slots, str(slot_key), row)
                self._accumulate_slot_stats(global_aggregate, str(slot_key), row)

        rebuilt_matchups: Dict[str, Dict[str, object]] = {}
        for key, raw in matchup_aggregates.items():
            slot_stats = self._build_aggregate_slot_stats(raw.get("slot_stats_agg", {}))
            lesson = {
                "lesson_scope": "matchup",
                "lesson_key": key,
                "signature": "",
                "state_label": "multi_state",
                "enemy_species_id": _as_int(raw.get("enemy_species_id", 0)),
                "player_species_id": _as_int(raw.get("player_species_id", 0)),
                "move_ids": [],
                "legal_slots": [0, 1, 2, 3],
                "slot_stats": slot_stats,
                "preferred_slots": [],
                "blocked_slots": [],
                "best_reward": 0.0,
                "best_turns": 0.0,
                "samples_total": 0,
                "last_updated_at": _utc_now(),
            }
            self._recompute_lesson(lesson, block_reward_gap=6.0)
            rebuilt_matchups[key] = lesson

        self.matchup_lessons = rebuilt_matchups

        self.global_lesson = {
            "lesson_scope": "global",
            "lesson_key": "global",
            "signature": "",
            "state_label": "all_states",
            "enemy_species_id": 0,
            "player_species_id": 0,
            "move_ids": [],
            "legal_slots": [0, 1, 2, 3],
            "slot_stats": self._build_aggregate_slot_stats(global_aggregate),
            "preferred_slots": [],
            "blocked_slots": [],
            "best_reward": 0.0,
            "best_turns": 0.0,
            "samples_total": 0,
            "last_updated_at": _utc_now(),
        }
        self._recompute_lesson(self.global_lesson, block_reward_gap=6.0)

    def finalize_episode(self, outcome: str, reward: float, turns: int) -> None:
        if not self._episode_turns:
            return
        won = str(outcome) == "win"
        failed_outcome = str(outcome) in {"loss", "timeout", "invalid_state"}
        for state_meta, chosen_slot in self._episode_turns:
            rule = self._ensure_rule(state_meta)
            slot_stats = rule.get("slot_stats", {})
            if not isinstance(slot_stats, dict):
                continue
            slot_key = str(max(0, min(3, int(chosen_slot))))
            row = slot_stats.get(slot_key, {})
            if not isinstance(row, dict):
                row = {
                    "samples": 0.0,
                    "wins": 0.0,
                    "losses": 0.0,
                    "avg_reward": 0.0,
                    "avg_turns": 0.0,
                }
                slot_stats[slot_key] = row
            prior_samples = _as_int(row.get("samples", 0))
            prior_avg_reward = _as_float(row.get("avg_reward", 0.0))
            prior_blocked = int(chosen_slot) in {int(v) for v in (rule.get("blocked_slots") or [])}
            if prior_samples >= 2 and (failed_outcome or reward <= (prior_avg_reward - 5.0)):
                self.repeated_mistake_events += 1
            if prior_blocked and failed_outcome:
                self.repeated_mistake_events += 1

            self._slot_update(row, reward=reward, turns=turns, won=won)
            rule["slot_stats"] = slot_stats
            self._recompute_lesson(rule, block_reward_gap=8.0)

        self._episode_turns = []
        self._rebuild_hierarchy()
        self.persist()

    def reflection_notes(self, limit: int = 5) -> str:
        lines: List[str] = []

        global_pref = self.global_lesson.get("preferred_slots", [])
        global_block = self.global_lesson.get("blocked_slots", [])
        if global_pref or global_block:
            lines.append(
                "global prefer="
                f"{global_pref} avoid={global_block} samples={self.global_lesson.get('samples_total',0)}"
            )

        matchup_rows = sorted(
            self.matchup_lessons.values(),
            key=lambda row: int(row.get("samples_total", 0)),
            reverse=True,
        )
        for row in matchup_rows:
            pref = row.get("preferred_slots", [])
            block = row.get("blocked_slots", [])
            if not pref and not block:
                continue
            lines.append(
                f"matchup p={row.get('player_species_id',0)} e={row.get('enemy_species_id',0)} "
                f"prefer={pref} avoid={block} samples={row.get('samples_total',0)}"
            )
            if len(lines) >= max(1, int(limit)):
                break

        if len(lines) < max(1, int(limit)):
            state_rows = sorted(
                self.rules.values(),
                key=lambda row: int(row.get("samples_total", 0)),
                reverse=True,
            )
            for row in state_rows:
                pref = row.get("preferred_slots", [])
                block = row.get("blocked_slots", [])
                if not pref and not block:
                    continue
                lines.append(
                    f"state={row.get('state_label','unknown_state')} enemy={row.get('enemy_species_id',0)} "
                    f"prefer={pref} avoid={block} samples={row.get('samples_total',0)}"
                )
                if len(lines) >= max(1, int(limit)):
                    break

        return "\n".join(lines)

    def _sorted_rule_payload(self) -> List[Dict[str, object]]:
        rows = sorted(
            self.rules.values(),
            key=lambda row: (
                str(row.get("state_label", "")),
                int(row.get("enemy_species_id", 0)),
                str(row.get("signature", "")),
            ),
        )
        return rows

    def _sorted_matchup_payload(self) -> List[Dict[str, object]]:
        rows = sorted(
            self.matchup_lessons.values(),
            key=lambda row: (
                int(row.get("player_species_id", 0)),
                int(row.get("enemy_species_id", 0)),
                str(row.get("lesson_key", "")),
            ),
        )
        return rows

    def _build_payload(self) -> Dict[str, object]:
        state_rules = self._sorted_rule_payload()
        return {
            "version": 2,
            "updated_at": _utc_now(),
            # Back-compat key retained.
            "rules": state_rules,
            "state_rules": state_rules,
            "matchup_lessons": self._sorted_matchup_payload(),
            "global_lesson": self.global_lesson,
            "stats": {
                "memory_hint_count": int(self.memory_hint_count),
                "memory_override_count": int(self.memory_override_count),
                "repeated_mistake_events": int(self.repeated_mistake_events),
                "decisions_logged": int(self.decisions_logged),
                "memory_hint_state_count": int(self.memory_hint_state_count),
                "memory_hint_matchup_count": int(self.memory_hint_matchup_count),
                "memory_hint_global_count": int(self.memory_hint_global_count),
                "memory_override_state_count": int(self.memory_override_state_count),
                "memory_override_matchup_count": int(self.memory_override_matchup_count),
                "memory_override_global_count": int(self.memory_override_global_count),
            },
        }

    def _next_snapshot_path(self) -> Path:
        pattern = re.compile(r"^battle_memory_(\d+)\.json$")
        max_seen = 0
        if self.history_dir.exists():
            for file in self.history_dir.glob("battle_memory_*.json"):
                match = pattern.match(file.name)
                if not match:
                    continue
                max_seen = max(max_seen, _as_int(match.group(1), 0))
        return self.history_dir / f"battle_memory_{max_seen + 1:03d}.json"

    def _prune_snapshots(self) -> None:
        pattern = re.compile(r"^battle_memory_(\d+)\.json$")
        rows: List[Tuple[int, Path]] = []
        for file in self.history_dir.glob("battle_memory_*.json"):
            match = pattern.match(file.name)
            if not match:
                continue
            rows.append((_as_int(match.group(1), 0), file))
        rows.sort(key=lambda item: item[0])
        keep = max(1, int(self.snapshot_keep))
        to_delete = rows[:-keep] if len(rows) > keep else []
        for _, file in to_delete:
            try:
                file.unlink()
            except Exception:
                pass

    def persist(self) -> None:
        payload = self._build_payload()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.history_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._next_snapshot_path()
        snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._prune_snapshots()

    def metrics(self) -> Dict[str, object]:
        repeated_rate = 0.0
        if self.decisions_logged > 0:
            repeated_rate = round(self.repeated_mistake_events / self.decisions_logged, 4)
        return {
            "memory_enabled": True,
            "memory_rules_loaded": int(self.rules_loaded),
            "memory_rules_written": int(len(self.rules)),
            "memory_matchup_lessons_written": int(len(self.matchup_lessons)),
            "memory_global_lesson_samples": int(_as_int(self.global_lesson.get("samples_total", 0))),
            "memory_hint_count": int(self.memory_hint_count),
            "memory_hint_state_count": int(self.memory_hint_state_count),
            "memory_hint_matchup_count": int(self.memory_hint_matchup_count),
            "memory_hint_global_count": int(self.memory_hint_global_count),
            "memory_override_count": int(self.memory_override_count),
            "memory_override_state_count": int(self.memory_override_state_count),
            "memory_override_matchup_count": int(self.memory_override_matchup_count),
            "memory_override_global_count": int(self.memory_override_global_count),
            "repeated_mistake_events": int(self.repeated_mistake_events),
            "repeated_mistake_rate": repeated_rate,
        }
