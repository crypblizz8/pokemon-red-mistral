from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_adjacency(raw: object) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if not isinstance(raw, dict):
        return out
    for key, values in raw.items():
        map_id = _as_int(key, -1)
        if map_id < 0:
            continue
        if not isinstance(values, list):
            continue
        out[map_id] = {_as_int(v, -1) for v in values if _as_int(v, -1) >= 0}
    return out


def _normalize_priority(raw: object) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        map_id = _as_int(key, -1)
        if map_id < 0:
            continue
        try:
            out[map_id] = float(value)
        except Exception:
            continue
    return out


def _normalize_profile(payload: Dict[str, object]) -> Dict[str, object]:
    profile_name = str(payload.get("name", "none")).strip() or "none"
    return {
        "name": profile_name,
        "adjacency": _normalize_adjacency(payload.get("adjacency", {})),
        "map_priority_bonus": _normalize_priority(payload.get("map_priority_bonus", {})),
        "valid_transition_bonus": float(payload.get("valid_transition_bonus", 0.2)),
        "invalid_transition_penalty": float(payload.get("invalid_transition_penalty", 0.05)),
        "frontier_map_bonus": float(payload.get("frontier_map_bonus", 0.3)),
        "loop_penalty": float(payload.get("loop_penalty", 0.08)),
    }


def _builtin_profile(name: str) -> Dict[str, object]:
    if name == "kanto_early":
        return _normalize_profile(
            {
                "name": "kanto_early",
                # Soft priors for early-game connectivity:
                # Pallet(0) <-> Route1(12) <-> Viridian(1) <-> Route2(13)
                "adjacency": {
                    "0": [12],
                    "12": [0, 1],
                    "1": [12, 13],
                    "13": [1],
                },
                # Small additive bonus for reaching known progression maps.
                "map_priority_bonus": {
                    "12": 0.03,
                    "1": 0.05,
                    "13": 0.06,
                },
                "valid_transition_bonus": 0.2,
                "invalid_transition_penalty": 0.05,
                "frontier_map_bonus": 0.35,
                "loop_penalty": 0.1,
            }
        )
    return _normalize_profile({"name": "none"})


def load_guidance_profile(name: str, json_path: Path | None = None) -> Dict[str, object]:
    profile_name = str(name or "none").strip().lower()
    base = _builtin_profile(profile_name)
    if json_path is None:
        return base

    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Guidance profile must be a JSON object: {path}")

    merged: Dict[str, Any] = dict(base)
    merged.update(payload)
    normalized = _normalize_profile(merged)
    if profile_name == "none":
        normalized["name"] = "custom"
    return normalized


def score_transition(
    prev_obs: Dict[str, int],
    curr_obs: Dict[str, int],
    ctx: Dict[str, object],
    profile: Dict[str, object],
) -> Dict[str, object]:
    if str(profile.get("name", "none")) == "none":
        return {
            "guidance_reward": 0.0,
            "valid_map_transition": False,
            "frontier_map_bonus": 0.0,
            "loop_penalty": 0.0,
            "guidance_tags": [],
        }

    if bool(ctx.get("battle_frame", False)):
        return {
            "guidance_reward": 0.0,
            "valid_map_transition": False,
            "frontier_map_bonus": 0.0,
            "loop_penalty": 0.0,
            "guidance_tags": ["battle_frame"],
        }

    prev_map = int(prev_obs.get("map_id", 0))
    curr_map = int(curr_obs.get("map_id", 0))
    map_changed = bool(ctx.get("map_changed", False))
    new_map_discovery = bool(ctx.get("new_map_discovery", False))
    ping_pong_event = bool(ctx.get("ping_pong_event", False))
    adjacency = profile.get("adjacency", {})
    if not isinstance(adjacency, dict):
        adjacency = {}

    guidance_reward = 0.0
    valid_map_transition = False
    frontier_bonus = 0.0
    loop_pen = 0.0
    tags: list[str] = []

    if map_changed:
        neighbors = adjacency.get(prev_map, set())
        if isinstance(neighbors, set) and neighbors:
            valid_map_transition = curr_map in neighbors
            if valid_map_transition:
                tags.append("valid_transition")
                guidance_reward += float(profile.get("valid_transition_bonus", 0.0))
            else:
                tags.append("invalid_transition")
                guidance_reward -= abs(float(profile.get("invalid_transition_penalty", 0.0)))
        if new_map_discovery:
            frontier_bonus = float(profile.get("frontier_map_bonus", 0.0))
            guidance_reward += frontier_bonus
            tags.append("frontier_map")

    priority_bonus = 0.0
    raw_priority = profile.get("map_priority_bonus", {})
    if isinstance(raw_priority, dict) and curr_map in raw_priority:
        priority_bonus = float(raw_priority[curr_map])
        guidance_reward += priority_bonus
        tags.append("priority_map")

    if ping_pong_event:
        loop_pen = abs(float(profile.get("loop_penalty", 0.0)))
        guidance_reward -= loop_pen
        tags.append("ping_pong")

    return {
        "guidance_reward": round(float(guidance_reward), 6),
        "valid_map_transition": bool(valid_map_transition),
        "frontier_map_bonus": round(float(frontier_bonus), 6),
        "loop_penalty": round(float(loop_pen), 6),
        "priority_map_bonus": round(float(priority_bonus), 6),
        "guidance_tags": tags,
    }

