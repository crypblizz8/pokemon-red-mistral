from __future__ import annotations


def reward_fn(prev_obs: dict, curr_obs: dict, ctx: dict) -> float:
    del prev_obs, curr_obs
    reward = 0.0

    # Exploration-first shaping.
    if bool(ctx.get("entered_new_tile", False)):
        reward += 1.15
    if bool(ctx.get("moved", False)):
        reward += 0.12
    if bool(ctx.get("map_changed", False)):
        reward += 0.2
    if bool(ctx.get("new_map_discovery", False)):
        reward += 0.45

    no_progress_steps = int(ctx.get("no_progress_steps", 0))
    reward -= min(1.2, 0.04 * no_progress_steps)

    if bool(ctx.get("battle_frame", False)):
        reward -= 0.2
    if bool(ctx.get("ping_pong_event", False)):
        reward -= 0.15

    guidance_weight = float(ctx.get("guidance_weight", 0.0))
    guidance_reward = float(ctx.get("guidance_reward", 0.0)) + float(
        ctx.get("frontier_map_bonus", 0.0)
    )
    reward += guidance_weight * guidance_reward

    return max(-1.0, min(1.5, reward))
