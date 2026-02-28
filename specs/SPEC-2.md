# Pokemon Red RL Agent
**SPEC-2 - Execution Plan (Phase 1+2 Locked, Minimal Phase 3 Path)**
`API Track` · `Mistral Large` · `PyBoy` · `Stable-Baselines3 PPO`

---

## 1. Project Overview

### One-liner
Build a Pokemon Red agent that:
1. Learns battle behavior via prompt-updated policy (Phase 1+2), then
2. Reaches a minimal EUREKA loop for navigation reward iteration (Phase 3).

### Core Objective
- Must deliver: stable, demo-ready Phase 1+2.
- Must attempt and complete minimum: Phase 3 with 2 reward iterations and measurable improvement in at least one exploration metric.

### Scope
- In scope:
1. Battle RL loop with strategy reflection updates.
2. Rich battle state (move/species/type/PP-aware legality).
3. PPO navigation environment + 2-step EUREKA reward loop.
- Out of scope:
1. Fine-tuning track (Phase 4) for this spec.
2. Full 4+ EUREKA iterations unless time remains.

---

## 2. Key Changes From SPEC.md

1. Phase 4 removed from required path.
2. Phase 3 reduced to minimum viable target:
   - 2 reward versions (`reward_v1.py`, `reward_v2.py`)
   - 25-minute PPO run each
   - Comparison across 3 evaluation save-states.
3. Phase 2 now requires PP-aware move legality and fallback.
4. Reward design guardrails added:
   - No pixel-only novelty reward.
   - No extreme one-step negative penalties.
   - Anti-stall and anti-exploit constraints.
5. Added explicit cost/rate-limit controls for Mistral API usage.

---

## 3. Tech Stack

| Component | Technology |
|---|---|
| Emulator | PyBoy (headless) |
| Game | Pokemon Red ROM + save states |
| Battle policy | Mistral API (`mistral-large-latest` or current equivalent) |
| Navigation RL | Stable-Baselines3 PPO |
| Language | Python 3.11+ |
| Libraries | `pyboy`, `mistralai`, `gymnasium`, `stable-baselines3`, `numpy` |

### Required RAM Signals
- `0xD057` in-battle flag
- `0xD16C` player HP
- `0xD18D` player max HP
- `0xD18C` player level
- `0xD16B` player species
- `0xD01C-0xD01F` move IDs
- `0xCFE6` enemy HP
- `0xCFE8` enemy level
- `0xCFE5` enemy species
- `0xD356` badge bitmask

---

## 4. Interfaces and File Layout

### Files
- `pokemon/emulator.py`
- `pokemon/battle_agent.py`
- `pokemon/nav_env.py`
- `reward.py`
- `reward_v1.py`
- `reward_v2.py`
- `configs.py`
- `run.py`
- `battle_results.json`
- `exploration_results.json`
- `RESULTS.md`

### Public Interfaces
#### `PokemonEmulator`
- `read(addr: int) -> int`
- `press(button: str, frames: int = 8) -> None`
- `reset_battle() -> None`
- `reset_nav() -> None`
- `in_battle() -> bool`
- `get_battle_state() -> dict`
- `execute_move(slot: int) -> bool`
- `wait_for_battle(timeout_ticks: int) -> bool`
- `wait_for_battle_end(timeout_ticks: int) -> bool`

#### `MistralBattleAgent`
- `pick_move(state: dict) -> int`
- `compute_reward(outcome: str, hp_left: int, turns: int) -> float`
- `record_battle(episode: dict) -> None`
- `update_strategy() -> str`
- `summary() -> dict`

#### `PokemonNavEnv` (`gym.Env`)
- `reset(seed=None, options=None) -> (obs, info)`
- `step(action: int) -> (obs, reward, terminated, truncated, info)`
- `set_reward_fn(fn) -> None`

---

## 5. Phase 0 - Environment Gate (Required Before Coding)

1. Create Python 3.11 virtual environment.
2. Install pinned dependencies.
3. Validate:
   - ROM exists and loads.
   - `assets/states/battle_rattata.state` exists.
   - Mistral API key works.
   - RAM reads return sane values.
4. Add `run.py --dry-check` to perform all preflight checks.

### Success Criteria
- All preflight checks pass.
- One scripted emulator smoke run works without crash.

---

## 6. Phase 1 - Core Battle RL Loop

### Implementation
1. Run battle episodes from `assets/states/battle_rattata.state`.
2. Parse strict response format `ACTION: <0-3>`.
3. Fallback chain:
   - Parse failure -> heuristic legal move.
   - API failure/rate limit -> one retry then heuristic.
4. Reward (battle):
   - Win/loss base
   - HP remaining bonus
   - turn-efficiency term
   - clip total reward to avoid outliers.
5. Update strategy every 5 episodes via reflection over summarized history.
6. Persist per-episode logs incrementally.

### Success Criteria
- 30 complete battles run without crash.
- Strategy changes at least 3 times.
- Win rate in episodes 26-30 > episodes 1-5.
- All episodes produce structured logs.

---

## 7. Phase 2 - Richer Battle State (Required)

### Implementation
1. Decode move/species/type data from RAM-backed IDs.
2. Include in prompt:
   - Move name, power, type, effectiveness.
   - Player/enemy species.
   - HP absolute + percentage.
   - Turn count.
3. Add PP-aware legality:
   - Never select 0-PP move.
   - If chosen move illegal, pick best legal fallback.
4. Keep prompts compact and token-capped.

### Success Criteria
- Agent references move-level context in decisions.
- Illegal move execution count is zero.
- PP depletion does not deadlock action selection.
- Phase 1 performance remains stable or improves.

---

## 8. Phase 3 - Minimal EUREKA Reward Design (Must Reach)

### Target
Complete 2 reward iterations with measurable exploration trend.

### Environment
- Observation: `[x, y, map_id, badges, hp, level, in_battle, visited_count]`
- Action space: 8 buttons (`up/down/left/right/A/B/start/select`)
- Battles are skipped/handled outside nav reward logic.

### Reward v1
1. Strong first-visit tile reward on `(map_id, x, y)`.
2. Tiny revisit decay bonus (prevents hard stall on backtracking).
3. Anti-stall penalty for no-progress windows.
4. Penalty clipping to avoid catastrophic one-step events.

### EUREKA Loop (2 iterations)
1. Train PPO for 25 minutes with `reward_v1.py`.
2. Record metrics.
3. Summarize metrics for Mistral critique (compact, cost-capped).
4. Generate `reward_v2.py`.
5. Train PPO for 25 minutes with `reward_v2.py`.
6. Evaluate each reward version on 3 nearby save-states.

### Required Metrics
- Unique visited tiles
- Furthest map reached
- Episode length
- Return/stuck ratio

### Success Criteria
- `reward_v1.py` and `reward_v2.py` both executed.
- At least one core metric improves from v1 to v2.
- Results written to `exploration_results.json` and summarized in `RESULTS.md`.

---

## 9. Cost and Reliability Guardrails

1. Cap tokens and call frequency per battle episode.
2. Strategy reflection only every 5 episodes.
3. Max one retry for failed API decision calls.
4. Hard run-level API call budget with automatic heuristic fallback.
5. Log parse/API failures for diagnosis.

---

## 10. Testing Plan

### Unit tests
- Action parser (`ACTION: 0-3`) robustness.
- Type-effectiveness lookup correctness.
- PP legality and fallback behavior.
- Reward clipping behavior.

### Integration tests
- One full battle episode end-to-end.
- Five-episode smoke with one strategy update.
- Navigation env stability for fixed-step rollout.

### Acceptance checks
- Phase 1+2 criteria pass.
- Phase 3 minimum criteria pass.
- No unrecoverable crashes in demo run.

---

## 11. Timeline

### Day 1
1. Phase 0 gate.
2. Phase 1 implementation and 10-episode debug.
3. Phase 2 integration and 30-episode run.

### Day 2
1. Finalize Phase 1+2 outputs.
2. Phase 3 v1 training (25 min).
3. EUREKA rewrite to v2.
4. Phase 3 v2 training (25 min).
5. 3-state evaluation and final summary.

---

## 12. Deliverables

### Required
- Working code modules listed in Section 4.
- `battle_results.json` (30 episodes).
- `exploration_results.json` (v1 vs v2 metrics).
- `RESULTS.md` with:
  - What improved
  - What failed
  - Next iteration ideas

### Optional (time permitting)
- Extra reward versions beyond v2.
- Additional eval seeds/save-states.
- Basic trajectory heatmap or flow visualization.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Emulator timing/menu desync | Add timeout recovery + extra tick buffers |
| Reward hacking in Phase 3 | Tile-based novelty + clipped penalties + exploit checks |
| API instability/cost spikes | Retry cap + heuristic fallback + hard call budget |
| Overfitting to deterministic start | Evaluate on 3 nearby save-states |
| Scope creep | Freeze Phase 4 and enforce minimum Phase 3 target only |

---

## 14. Assumptions

1. ROM and save states are available and valid.
2. Mistral API access is active.
3. Demo-grade evidence is acceptable (not statistical publication bar).
4. Phase 4 is explicitly deferred from this execution spec.

---
