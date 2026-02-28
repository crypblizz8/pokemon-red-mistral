from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from mistralai import Mistral
except Exception:  # pragma: no cover - optional dependency in heuristic-only flows.
    Mistral = None  # type: ignore[assignment]

from pokemon.battle_memory import BattleMemory

ACTION_RE = re.compile(r"ACTION\s*:\s*([0-3])", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_message_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _condense_reply_for_reflection(reply: str, max_chars: int = 180) -> str:
    if not reply:
        return ""
    cleaned: List[str] = []
    for raw_line in str(reply).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "LLM" or upper.startswith("CACHE:") or upper.startswith("HEURISTIC:"):
            continue
        if upper.startswith("BUDGET_FALLBACK"):
            continue
        if "ACTION" in upper:
            continue
        cleaned.append(line)
    if not cleaned:
        return ""
    text = " ".join(cleaned)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


@dataclass
class EpisodeRecord:
    episode: int
    outcome: str
    turns: int
    hp_left: int
    reward: float
    move_slots: List[int]
    llm_replies: List[str]
    strategy_version: int
    state_path: str = ""
    state_index: int = -1
    policy_mode: str = "llm"
    decision_calls_used: int = 0
    fallback_events: int = 0
    illegal_move_attempts: int = 0
    pp_depletion_events: int = 0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode": self.episode,
            "outcome": self.outcome,
            "turns": self.turns,
            "hp_left": self.hp_left,
            "reward": self.reward,
            "move_slots": self.move_slots,
            "llm_replies": self.llm_replies,
            "strategy_version": self.strategy_version,
            "state_path": self.state_path,
            "state_index": self.state_index,
            "policy_mode": self.policy_mode,
            "decision_calls_used": self.decision_calls_used,
            "fallback_events": self.fallback_events,
            "illegal_move_attempts": self.illegal_move_attempts,
            "pp_depletion_events": self.pp_depletion_events,
            "created_at": self.created_at,
        }


class MistralBattleAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        policy_mode: str = "llm",
        battle_memory: Optional[BattleMemory] = None,
    ) -> None:
        self.client = Mistral(api_key=api_key) if Mistral is not None else None
        self.model = model
        self.policy_mode = policy_mode
        self.strategy = (
            "Prioritize winning safely. Choose strong damaging moves. Avoid wasting turns. "
            "Return one line with ACTION: <0-3>."
        )
        self.strategy_versions: List[str] = [self.strategy]
        self.history: List[EpisodeRecord] = []
        self.last_reply: str = ""
        self.action_cache: Dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.illegal_move_attempts = 0
        self.legality_fallback_count = 0
        self.llm_decision_calls = 0
        self.llm_reflection_calls = 0
        self.budget_fallback_count = 0
        self.battle_memory = battle_memory
        self._episode_state_label = "unknown_state"
        self._episode_state_index = -1
        self._last_enemy_hp: int | None = None
        self._last_chosen_slot: int | None = None
        self._stagnation_count = 0

    def _extract_action(self, text: str) -> Optional[int]:
        match = ACTION_RE.search(text)
        if not match:
            return None
        return int(match.group(1))

    def _legal_slots_from_state(self, state: Dict[str, object]) -> List[int]:
        legal_slots = state.get("legal_slots", [])
        if isinstance(legal_slots, list):
            out = []
            for item in legal_slots:
                try:
                    slot = int(item)
                except Exception:
                    continue
                if 0 <= slot <= 3:
                    out.append(slot)
            if out:
                return sorted(set(out))

        move_ids = state.get("move_ids", [])
        if not isinstance(move_ids, list):
            return []
        out = []
        for idx, move_id in enumerate(move_ids[:4]):
            if int(move_id) != 0:
                out.append(idx)
        return out

    def _move_info_by_slot(self, state: Dict[str, object]) -> Dict[int, Dict[str, object]]:
        move_rows = state.get("moves", [])
        if not isinstance(move_rows, list):
            return {}
        out: Dict[int, Dict[str, object]] = {}
        for row in move_rows:
            if not isinstance(row, dict):
                continue
            slot = row.get("slot")
            if slot is None:
                continue
            try:
                slot_i = int(slot)
            except Exception:
                continue
            out[slot_i] = row
        return out

    def _best_legal_slot(self, state: Dict[str, object], legal_slots: List[int]) -> int:
        if not legal_slots:
            return 0
        move_rows = self._move_info_by_slot(state)
        damaging_slots: List[int] = []
        for slot in legal_slots:
            row = move_rows.get(slot, {})
            if int(row.get("power", 0)) > 0:
                damaging_slots.append(slot)
        candidate_slots = damaging_slots if damaging_slots else legal_slots
        best_slot = candidate_slots[0]
        best_score = (-1.0, -1, 0)
        for slot in candidate_slots:
            row = move_rows.get(slot, {})
            eff = float(row.get("effectiveness", 1.0))
            power = int(row.get("power", 0))
            score = (eff, power, -slot)
            if score > best_score:
                best_score = score
                best_slot = slot
        return best_slot

    def _heuristic_move(self, state: Dict[str, object]) -> int:
        legal_slots = self._legal_slots_from_state(state)
        return self._best_legal_slot(state, legal_slots)

    def _best_alternate_slot(self, state: Dict[str, object], *, exclude_slot: int) -> int | None:
        legal_slots = [slot for slot in self._legal_slots_from_state(state) if slot != int(exclude_slot)]
        if not legal_slots:
            return None
        move_rows = self._move_info_by_slot(state)
        damaging = [slot for slot in legal_slots if int(move_rows.get(slot, {}).get("power", 0)) > 0]
        ranked = damaging if damaging else legal_slots
        best_slot = ranked[0]
        best_score = (-1.0, -1, 0)
        for slot in ranked:
            row = move_rows.get(slot, {})
            score = (float(row.get("effectiveness", 1.0)), int(row.get("power", 0)), -slot)
            if score > best_score:
                best_score = score
                best_slot = slot
        return best_slot

    def _apply_stagnation_override(self, state: Dict[str, object], chosen: int) -> tuple[int, bool]:
        enemy_hp = int(state.get("enemy_hp", -1))
        if enemy_hp < 0:
            self._last_enemy_hp = None
            self._last_chosen_slot = int(chosen)
            self._stagnation_count = 0
            return int(chosen), False

        if self._last_enemy_hp is not None and enemy_hp < self._last_enemy_hp:
            self._stagnation_count = 0
        elif (
            self._last_enemy_hp is not None
            and enemy_hp == self._last_enemy_hp
            and self._last_chosen_slot is not None
            and int(self._last_chosen_slot) == int(chosen)
        ):
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        overrode = False
        final_slot = int(chosen)
        if self._stagnation_count >= 2:
            alternate = self._best_alternate_slot(state, exclude_slot=int(chosen))
            if alternate is not None and int(alternate) != int(chosen):
                final_slot = int(alternate)
                overrode = True
                self._stagnation_count = 0

        self._last_enemy_hp = enemy_hp
        self._last_chosen_slot = final_slot
        return final_slot, overrode

    def _state_cache_key(self, state: Dict[str, object]) -> str:
        key = {
            "player_hp": int(state.get("player_hp", 0)),
            "player_max_hp": int(state.get("player_max_hp", 0)),
            "player_level": int(state.get("player_level", 0)),
            "enemy_hp": int(state.get("enemy_hp", 0)),
            "enemy_level": int(state.get("enemy_level", 0)),
            "enemy_species": int(state.get("enemy_species", 0)),
            "move_ids": [int(v) for v in state.get("move_ids", [])[:4]],
            "move_pps": [int(v) for v in state.get("move_pps", [])[:4]],
            "legal_slots": [int(v) for v in state.get("legal_slots", [])[:4]],
            "strategy_version": len(self.strategy_versions) - 1,
        }
        return json.dumps(key, sort_keys=True)

    def _sanitize_action(self, candidate: int, state: Dict[str, object]) -> tuple[int, bool]:
        legal_slots = self._legal_slots_from_state(state)
        if not legal_slots:
            return 0, False
        if candidate in legal_slots:
            return candidate, False
        self.illegal_move_attempts += 1
        self.legality_fallback_count += 1
        return self._best_legal_slot(state, legal_slots), True

    def _apply_memory_override(self, state: Dict[str, object], chosen: int) -> tuple[int, bool]:
        if self.battle_memory is None:
            return chosen, False
        return self.battle_memory.maybe_override_slot(state, chosen)

    def start_episode_context(self, state_label: str, state_index: int) -> None:
        self._episode_state_label = state_label or "unknown_state"
        self._episode_state_index = int(state_index)
        self._last_enemy_hp = None
        self._last_chosen_slot = None
        self._stagnation_count = 0
        if self.battle_memory is not None:
            self.battle_memory.start_episode()

    def record_turn_decision(self, state: Dict[str, object], chosen_slot: int) -> None:
        if self.battle_memory is None:
            return
        enriched = dict(state)
        enriched.setdefault("state_label", self._episode_state_label)
        enriched.setdefault("state_index", self._episode_state_index)
        self.battle_memory.record_turn(enriched, chosen_slot)

    def finalize_episode_memory(self, record: EpisodeRecord) -> None:
        if self.battle_memory is None:
            return
        self.battle_memory.finalize_episode(
            outcome=record.outcome,
            reward=record.reward,
            turns=record.turns,
        )

    def pick_move(
        self,
        state: Dict[str, object],
        use_llm: bool = True,
        budget_fallback: bool = False,
    ) -> int:
        if self.policy_mode == "heuristic":
            use_llm = False

        chosen: int
        tag: str

        if not use_llm:
            chosen = self._heuristic_move(state)
            if budget_fallback:
                self.budget_fallback_count += 1
                tag = "BUDGET_FALLBACK"
            elif self.policy_mode == "hybrid":
                tag = "HEURISTIC"
            else:
                tag = "HEURISTIC"
        else:
            cache_key = self._state_cache_key(state)
            cached = self.action_cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                chosen = cached
                tag = "CACHE"
            elif self.client is None:
                self.cache_misses += 1
                chosen = self._heuristic_move(state)
                tag = "LLM_UNAVAILABLE"
                self.last_reply = "LLM_UNAVAILABLE: using heuristic fallback."
            else:
                self.cache_misses += 1
                self.llm_decision_calls += 1
                system_prompt = (
                    "You are a Pokemon Red battle policy. Decide one legal move slot from 0 to 3. "
                    "Prefer super-effective and non-zero power moves when safe. "
                    "Always include line: ACTION: <0-3>\n\n"
                    f"Current strategy:\n{self.strategy}"
                )
                memory_hint = ""
                if self.battle_memory is not None:
                    memory_hint = self.battle_memory.prompt_hint(state)
                user_prompt = (
                    "Battle state JSON:\n"
                    f"{json.dumps(state, sort_keys=True)}\n\n"
                    "Return brief reasoning and final ACTION line."
                )
                if memory_hint:
                    user_prompt = f"{user_prompt}\n\n{memory_hint}"
                try:
                    response = self.client.chat.complete(
                        model=self.model,
                        max_tokens=180,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    content = _coerce_message_content(response.choices[0].message.content)
                    parsed = self._extract_action(content)
                    chosen = parsed if parsed is not None else self._heuristic_move(state)
                    self.last_reply = f"LLM\n{content}"
                    tag = "LLM"
                except Exception as exc:
                    chosen = self._heuristic_move(state)
                    self.last_reply = f"LLM_ERROR: {exc}"
                    tag = "HEURISTIC"
            chosen, memory_overrode = self._apply_memory_override(state, chosen)
            if memory_overrode:
                self.last_reply = (
                    f"{self.last_reply}\nMEMORY_OVERRIDE: ACTION: {chosen}"
                    if self.last_reply
                    else f"{tag}|MEMORY_OVERRIDE: ACTION: {chosen}"
                )
            chosen, fell_back = self._sanitize_action(chosen, state)
            chosen, stagnation_overrode = self._apply_stagnation_override(state, chosen)
            if stagnation_overrode:
                self.last_reply = (
                    f"{self.last_reply}\nSTAGNATION_OVERRIDE: ACTION: {chosen}"
                    if self.last_reply
                    else f"{tag}|STAGNATION_OVERRIDE: ACTION: {chosen}"
                )
            self.action_cache[cache_key] = chosen
            if fell_back:
                self.last_reply = f"{tag}|LEGALITY_FALLBACK: ACTION: {chosen}"
            else:
                if "LLM\n" not in self.last_reply:
                    self.last_reply = f"{tag}: ACTION: {chosen}"
            return chosen

        chosen, memory_overrode = self._apply_memory_override(state, chosen)
        if memory_overrode:
            self.last_reply = f"{tag}|MEMORY_OVERRIDE: ACTION: {chosen}"
            tag = f"{tag}|MEMORY_OVERRIDE"
        chosen, fell_back = self._sanitize_action(chosen, state)
        chosen, stagnation_overrode = self._apply_stagnation_override(state, chosen)
        if stagnation_overrode:
            self.last_reply = f"{tag}|STAGNATION_OVERRIDE: ACTION: {chosen}"
            tag = f"{tag}|STAGNATION_OVERRIDE"
        if fell_back:
            self.last_reply = f"{tag}|LEGALITY_FALLBACK: ACTION: {chosen}"
        else:
            self.last_reply = f"{tag}: ACTION: {chosen}"
        return chosen

    def compute_reward(self, outcome: str, hp_left: int, turns: int) -> float:
        if outcome == "win":
            return float(100 + min(max(hp_left, 0), 50) - (2 * turns))
        return float(-100 + (0.5 * max(hp_left, 0)))

    def record_battle(
        self,
        episode: int,
        move_slots: List[int],
        outcome: str,
        hp_left: int,
        turns: int,
        llm_replies: List[str],
        state_path: str = "",
        state_index: int = -1,
        decision_calls_used: int = 0,
        fallback_events: int = 0,
        illegal_move_attempts: int = 0,
        pp_depletion_events: int = 0,
    ) -> EpisodeRecord:
        reward = self.compute_reward(outcome=outcome, hp_left=hp_left, turns=turns)
        record = EpisodeRecord(
            episode=episode,
            outcome=outcome,
            turns=turns,
            hp_left=hp_left,
            reward=reward,
            move_slots=move_slots,
            llm_replies=llm_replies,
            strategy_version=len(self.strategy_versions) - 1,
            state_path=state_path,
            state_index=state_index,
            policy_mode=self.policy_mode,
            decision_calls_used=decision_calls_used,
            fallback_events=fallback_events,
            illegal_move_attempts=illegal_move_attempts,
            pp_depletion_events=pp_depletion_events,
        )
        self.history.append(record)
        return record

    def update_strategy(self, use_llm: bool = True) -> str:
        if not self.history:
            return self.strategy
        if not use_llm or self.client is None:
            return self.strategy

        recent = self.history[-10:]
        summary_lines = []
        for row in recent:
            state_label = (
                Path(row.state_path).name
                if row.state_path
                else (f"state_{row.state_index}" if row.state_index >= 0 else "unknown_state")
            )
            first_move = row.move_slots[0] if row.move_slots else -1
            first_note = ""
            if row.llm_replies:
                first_note = _condense_reply_for_reflection(row.llm_replies[0])
            note_suffix = f" note={first_note}" if first_note else ""
            summary_lines.append(
                f"ep={row.episode} state={state_label} outcome={row.outcome} turns={row.turns} "
                f"hp={row.hp_left} reward={row.reward:.1f} first_move={first_move} "
                f"moves={row.move_slots}{note_suffix}"
            )
        summary = "\n".join(summary_lines)

        system_prompt = (
            "You improve a battle strategy policy using reinforcement outcomes. "
            "Use both numeric outcomes and battle notes. "
            "Return concise updated strategy with 4-8 bullet points."
        )
        user_prompt = (
            f"Current strategy:\n{self.strategy}\n\n"
            f"Recent outcomes:\n{summary}\n\n"
            "Rewrite strategy to maximize win rate and efficiency. "
            "Include explicit guidance to avoid zero-power moves unless no damaging legal move exists."
        )
        if self.battle_memory is not None:
            memory_notes = self.battle_memory.reflection_notes(limit=5)
            if memory_notes:
                user_prompt = f"{user_prompt}\n\nMemory notes:\n{memory_notes}"
        self.llm_reflection_calls += 1
        try:
            response = self.client.chat.complete(
                model=self.model,
                max_tokens=260,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            new_strategy = _coerce_message_content(response.choices[0].message.content)
            if new_strategy:
                self.strategy = new_strategy
                self.strategy_versions.append(new_strategy)
                self.action_cache.clear()
        except Exception:
            # Keep prior strategy if reflection call fails.
            pass
        return self.strategy

    def summary(self) -> Dict[str, object]:
        total = len(self.history)
        wins = sum(1 for item in self.history if item.outcome == "win")
        win_rate = (wins / total) if total else 0.0
        return {
            "episodes": total,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "strategy_versions": len(self.strategy_versions),
            "policy_mode": self.policy_mode,
            "action_cache_hits": self.cache_hits,
            "action_cache_misses": self.cache_misses,
            "illegal_move_attempts": self.illegal_move_attempts,
            "legality_fallback_count": self.legality_fallback_count,
            "llm_decision_calls": self.llm_decision_calls,
            "llm_reflection_calls": self.llm_reflection_calls,
            "budget_fallback_count": self.budget_fallback_count,
        }

    def history_as_dicts(self) -> List[Dict[str, object]]:
        return [entry.to_dict() for entry in self.history]
