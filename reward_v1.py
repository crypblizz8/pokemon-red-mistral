from __future__ import annotations


def reward_fn(prev_obs: dict, curr_obs: dict, ctx: dict) -> float:
    del prev_obs, curr_obs
    reward = 0.0
    if bool(ctx.get("entered_new_tile", False)):
        reward += 1.0
    if bool(ctx.get("moved", False)):
        reward += 0.05

    no_progress_steps = int(ctx.get("no_progress_steps", 0))
    reward -= min(0.5, 0.02 * no_progress_steps)

    if bool(ctx.get("battle_frame", False)):
        reward -= 0.1

    guidance_weight = float(ctx.get("guidance_weight", 0.0))
    guidance_reward = float(ctx.get("guidance_reward", 0.0))
    reward += guidance_weight * guidance_reward

    return max(-1.0, min(1.5, reward))
