from __future__ import annotations

from statistics import mean
from typing import Dict, List, Sequence


def aggregate_episode_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "episodes": 0,
            "unique_tiles": 0.0,
            "furthest_map_id": 0,
            "avg_episode_len": 0.0,
            "avg_return": 0.0,
            "stuck_ratio": 0.0,
            "valid_transition_rate": 0.0,
            "new_map_discovery_count": 0.0,
            "ping_pong_ratio": 0.0,
            "map_ids_seen": [],
        }

    unique_tiles = [float(row.get("unique_tiles", 0.0)) for row in rows]
    episode_lens = [float(row.get("episode_len", 0.0)) for row in rows]
    returns = [float(row.get("episode_return", 0.0)) for row in rows]
    stuck_ratios = [float(row.get("stuck_ratio", 0.0)) for row in rows]
    valid_transition_rates = [float(row.get("valid_transition_rate", 0.0)) for row in rows]
    new_map_discovery_counts = [float(row.get("new_map_discovery_count", 0.0)) for row in rows]
    ping_pong_ratios = [float(row.get("ping_pong_ratio", 0.0)) for row in rows]
    furthest_map = max(int(row.get("furthest_map_id", 0)) for row in rows)
    map_ids_seen = sorted(
        {
            int(map_id)
            for row in rows
            for map_id in (row.get("map_ids_seen", []) if isinstance(row.get("map_ids_seen"), list) else [])
        }
    )

    return {
        "episodes": len(rows),
        "unique_tiles": round(mean(unique_tiles), 4),
        "furthest_map_id": int(furthest_map),
        "avg_episode_len": round(mean(episode_lens), 4),
        "avg_return": round(mean(returns), 4),
        "stuck_ratio": round(mean(stuck_ratios), 4),
        "valid_transition_rate": round(mean(valid_transition_rates), 4),
        "new_map_discovery_count": round(mean(new_map_discovery_counts), 4),
        "ping_pong_ratio": round(mean(ping_pong_ratios), 4),
        "map_ids_seen": map_ids_seen,
    }


def compare_versions(v1: Dict[str, object], v2: Dict[str, object]) -> Dict[str, object]:
    delta_unique_tiles = round(float(v2.get("unique_tiles", 0.0)) - float(v1.get("unique_tiles", 0.0)), 4)
    delta_furthest_map_id = int(v2.get("furthest_map_id", 0)) - int(v1.get("furthest_map_id", 0))
    delta_avg_episode_len = round(
        float(v2.get("avg_episode_len", 0.0)) - float(v1.get("avg_episode_len", 0.0)),
        4,
    )
    delta_avg_return = round(float(v2.get("avg_return", 0.0)) - float(v1.get("avg_return", 0.0)), 4)
    delta_stuck_ratio = round(float(v2.get("stuck_ratio", 0.0)) - float(v1.get("stuck_ratio", 0.0)), 4)
    delta_valid_transition_rate = round(
        float(v2.get("valid_transition_rate", 0.0)) - float(v1.get("valid_transition_rate", 0.0)),
        4,
    )
    delta_new_map_discovery_count = round(
        float(v2.get("new_map_discovery_count", 0.0))
        - float(v1.get("new_map_discovery_count", 0.0)),
        4,
    )
    delta_ping_pong_ratio = round(
        float(v2.get("ping_pong_ratio", 0.0)) - float(v1.get("ping_pong_ratio", 0.0)),
        4,
    )

    core_metric_improved = bool(
        delta_unique_tiles > 0
        or delta_furthest_map_id > 0
        or delta_avg_return > 0
    )
    return {
        "delta_unique_tiles": delta_unique_tiles,
        "delta_furthest_map_id": delta_furthest_map_id,
        "delta_avg_episode_len": delta_avg_episode_len,
        "delta_avg_return": delta_avg_return,
        "delta_stuck_ratio": delta_stuck_ratio,
        "delta_valid_transition_rate": delta_valid_transition_rate,
        "delta_new_map_discovery_count": delta_new_map_discovery_count,
        "delta_ping_pong_ratio": delta_ping_pong_ratio,
        "phase3_pass": core_metric_improved,
    }


def build_v2_critique_prompt(v1_metrics: Dict[str, object]) -> str:
    return (
        "Phase 3 reward iteration request:\n"
        "You are optimizing a Pokemon navigation reward function for PPO.\n"
        "Current v1 metrics:\n"
        f"- unique_tiles(avg): {v1_metrics.get('unique_tiles', 0.0)}\n"
        f"- furthest_map_id: {v1_metrics.get('furthest_map_id', 0)}\n"
        f"- avg_episode_len: {v1_metrics.get('avg_episode_len', 0.0)}\n"
        f"- avg_return: {v1_metrics.get('avg_return', 0.0)}\n"
        f"- stuck_ratio: {v1_metrics.get('stuck_ratio', 0.0)}\n\n"
        f"- valid_transition_rate: {v1_metrics.get('valid_transition_rate', 0.0)}\n"
        f"- new_map_discovery_count: {v1_metrics.get('new_map_discovery_count', 0.0)}\n"
        f"- ping_pong_ratio: {v1_metrics.get('ping_pong_ratio', 0.0)}\n\n"
        "Return only Python code for reward_v2.py with function:\n"
        "def reward_fn(prev_obs, curr_obs, ctx) -> float\n"
        "Keep interface unchanged. Improve exploration without reward hacking."
    )


def build_phase3_markdown(payload: Dict[str, object]) -> str:
    versions = payload.get("versions", {})
    v1 = versions.get("v1", {}).get("aggregate", {})
    v2 = versions.get("v2", {}).get("aggregate", {})
    comparison = payload.get("comparison", {})
    critique_prompt = payload.get("v2_critique_prompt", "")
    lines: List[str] = []
    lines.append("# Phase 3 Results")
    lines.append("")
    lines.append("## V1 Metrics")
    lines.append(f"- unique_tiles(avg): {v1.get('unique_tiles', 0.0)}")
    lines.append(f"- furthest_map_id: {v1.get('furthest_map_id', 0)}")
    lines.append(f"- avg_episode_len: {v1.get('avg_episode_len', 0.0)}")
    lines.append(f"- avg_return: {v1.get('avg_return', 0.0)}")
    lines.append(f"- stuck_ratio: {v1.get('stuck_ratio', 0.0)}")
    lines.append(f"- valid_transition_rate: {v1.get('valid_transition_rate', 0.0)}")
    lines.append(f"- new_map_discovery_count: {v1.get('new_map_discovery_count', 0.0)}")
    lines.append(f"- ping_pong_ratio: {v1.get('ping_pong_ratio', 0.0)}")
    lines.append("")
    lines.append("## V2 Metrics")
    lines.append(f"- unique_tiles(avg): {v2.get('unique_tiles', 0.0)}")
    lines.append(f"- furthest_map_id: {v2.get('furthest_map_id', 0)}")
    lines.append(f"- avg_episode_len: {v2.get('avg_episode_len', 0.0)}")
    lines.append(f"- avg_return: {v2.get('avg_return', 0.0)}")
    lines.append(f"- stuck_ratio: {v2.get('stuck_ratio', 0.0)}")
    lines.append(f"- valid_transition_rate: {v2.get('valid_transition_rate', 0.0)}")
    lines.append(f"- new_map_discovery_count: {v2.get('new_map_discovery_count', 0.0)}")
    lines.append(f"- ping_pong_ratio: {v2.get('ping_pong_ratio', 0.0)}")
    lines.append("")
    lines.append("## Deltas (v2 - v1)")
    lines.append(f"- delta_unique_tiles: {comparison.get('delta_unique_tiles', 0.0)}")
    lines.append(f"- delta_furthest_map_id: {comparison.get('delta_furthest_map_id', 0)}")
    lines.append(f"- delta_avg_episode_len: {comparison.get('delta_avg_episode_len', 0.0)}")
    lines.append(f"- delta_avg_return: {comparison.get('delta_avg_return', 0.0)}")
    lines.append(f"- delta_stuck_ratio: {comparison.get('delta_stuck_ratio', 0.0)}")
    lines.append(f"- delta_valid_transition_rate: {comparison.get('delta_valid_transition_rate', 0.0)}")
    lines.append(
        f"- delta_new_map_discovery_count: {comparison.get('delta_new_map_discovery_count', 0.0)}"
    )
    lines.append(f"- delta_ping_pong_ratio: {comparison.get('delta_ping_pong_ratio', 0.0)}")
    lines.append(f"- phase3_pass: {comparison.get('phase3_pass', False)}")
    lines.append("")
    lines.append("## V2 Critique Prompt")
    lines.append("```text")
    lines.append(str(critique_prompt))
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"
