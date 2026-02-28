from __future__ import annotations

from math import sqrt
from statistics import median
from typing import Dict, List, Sequence, Tuple

WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z_score: float = WILSON_Z_95) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    p_hat = successes / total
    z2 = z_score * z_score
    denominator = 1.0 + (z2 / total)
    center = (p_hat + (z2 / (2.0 * total))) / denominator
    margin = (
        z_score * sqrt((p_hat * (1.0 - p_hat) + (z2 / (4.0 * total))) / total) / denominator
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    wins = sum(1 for row in rows if row.get("outcome") == "win")
    timeouts = sum(1 for row in rows if row.get("outcome") == "timeout")
    invalid_states = sum(1 for row in rows if row.get("outcome") == "invalid_state")

    rewards = [float(row.get("reward", 0.0)) for row in rows]
    turns = [int(row.get("turns", 0)) for row in rows]
    hp_left = [int(row.get("hp_left", 0)) for row in rows]

    win_rate = (wins / total) if total else 0.0
    timeout_rate = (timeouts / total) if total else 0.0
    invalid_state_rate = (invalid_states / total) if total else 0.0
    ci_low, ci_high = wilson_interval(wins, total)

    return {
        "episodes": total,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "win_rate_pct": round(win_rate * 100.0, 2),
        "win_rate_ci_low": round(ci_low, 4),
        "win_rate_ci_high": round(ci_high, 4),
        "timeout_rate": round(timeout_rate, 4),
        "invalid_state_rate": round(invalid_state_rate, 4),
        "mean_reward": round((sum(rewards) / total), 4) if total else 0.0,
        "median_turns": round(float(median(turns)), 4) if turns else 0.0,
        "mean_hp_left": round((sum(hp_left) / total), 4) if total else 0.0,
    }


def intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return max(a_low, b_low) <= min(a_high, b_high)


def rank_models(
    model_summaries: Sequence[Dict[str, object]],
    no_clear_winner_gap: float,
) -> Dict[str, object]:
    if not model_summaries:
        return {"decision": "no_models", "winner_model": None, "ranking": []}

    ranking = sorted(
        model_summaries,
        key=lambda row: (
            -float(row.get("win_rate", 0.0)),
            float(row.get("timeout_rate", 1.0)),
            float(row.get("median_turns", 9999.0)),
            -float(row.get("mean_hp_left", 0.0)),
        ),
    )

    best = ranking[0]
    if len(ranking) == 1:
        return {
            "decision": "winner",
            "winner_model": best.get("model"),
            "reason": "Only one model evaluated.",
            "ranking": ranking,
        }

    second = ranking[1]
    best_rate = float(best.get("win_rate", 0.0))
    second_rate = float(second.get("win_rate", 0.0))
    gap = best_rate - second_rate
    overlap = intervals_overlap(
        float(best.get("win_rate_ci_low", 0.0)),
        float(best.get("win_rate_ci_high", 0.0)),
        float(second.get("win_rate_ci_low", 0.0)),
        float(second.get("win_rate_ci_high", 0.0)),
    )

    no_clear_winner = gap < no_clear_winner_gap and overlap
    if no_clear_winner:
        return {
            "decision": "no_clear_winner",
            "winner_model": None,
            "reason": (
                "Top two models are within the configured win-rate gap and their 95% "
                "Wilson intervals overlap."
            ),
            "ranking": ranking,
        }

    return {
        "decision": "winner",
        "winner_model": best.get("model"),
        "reason": "Best held-out win rate; tie-breakers applied for timeout, turns, and hp left.",
        "ranking": ranking,
    }


def flatten_model_summary_for_csv(model_summary: Dict[str, object]) -> List[object]:
    return [
        model_summary.get("model", ""),
        model_summary.get("episodes", 0),
        model_summary.get("wins", 0),
        model_summary.get("win_rate", 0.0),
        model_summary.get("win_rate_ci_low", 0.0),
        model_summary.get("win_rate_ci_high", 0.0),
        model_summary.get("timeout_rate", 0.0),
        model_summary.get("invalid_state_rate", 0.0),
        model_summary.get("median_turns", 0.0),
        model_summary.get("mean_hp_left", 0.0),
        model_summary.get("mean_reward", 0.0),
    ]
