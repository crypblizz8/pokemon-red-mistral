# Pokemon Red RL Agent
**SPEC-4 - Phase 4 Route-to-Gym + Battle Integration**
`Hybrid Track` · `PyBoy` · `PPO Nav Prior` · `Mistral Battle Policy`

---

## 1. Objective

Deliver a reliable autonomous run from early-game overworld states to Pewter Gym flow, combining:

1. Route-oriented navigation (checkpointed, recovery-capable).
2. Existing battle agent for forced/trainer battles.
3. Wild-battle escape behavior (`RUN`) where legal/possible.

Primary target:
1. Reach Pewter Gym entrance consistently.
2. Optional stretch in same mode: clear Brock and obtain Boulder Badge.

---

## 2. Why SPEC-4 Exists

Phase 3 proved reward-guided exploration can improve coverage and reduce getting stuck.  
SPEC-4 converts that into mission reliability: Route 1 -> Viridian -> Route 2 -> Pewter Gym, with battle handling.

---

## 3. Scope

### In scope
1. Deterministic/semi-deterministic navigation execution with checkpoint recovery.
2. Interrupt-driven battle integration with the current battle policy.
3. Route progress and failure artifact logging.
4. A single-command run mode for rehearsal/demo.

### Out of scope
1. Claiming full end-to-end pure RL navigation.
2. New model fine-tuning work.
3. Full game progression beyond first gym in this phase.

---

## 4. Public Interfaces and Files

### New/updated files
1. `specs/SPEC-4.md`
2. `pokemon/route_executor.py` (new)
3. `assets/routes/route1_to_pewter.json` (new)
4. `run.py` (new `--phase4` mode and args)
5. `pokemon/emulator.py` (overworld helpers + optional badge check helper)
6. `artifacts/results/phase4_results.json` (new output)
7. `artifacts/results/phase4_timeline.jsonl` (new output)

### CLI contract
1. `--phase4` (bool)
2. `--phase4-start-state` (Path, default `assets/states/explore_start.state`)
3. `--phase4-route-script` (Path, default `assets/routes/route1_to_pewter.json`)
4. `--phase4-max-steps` (int, default `15000`)
5. `--phase4-policy-mode` (`hybrid|heuristic`, default `hybrid`)
6. `--phase4-results-path` (Path)
7. `--phase4-timeline-path` (Path)
8. `--phase4-target` (`gym_entrance|brock_badge`, default `gym_entrance`)
9. `--phase4-wild-run-enabled` (bool, default `true`)

Mode exclusivity:
1. `--dry-check`, `--phase1`, `--phase2`, `--phase3`, `--phase4` are mutually exclusive.

---

## 5. Route Data Schema

`assets/routes/route1_to_pewter.json` items:

1. `checkpoint`  
   fields: `name`, `expected_map_id`
2. `move`  
   fields: `direction` (`up|down|left|right`), `steps`, `hold_frames`
3. `press`  
   fields: `button` (`a|b|start|select|up|down|left|right`), `count`, `hold_frames`
4. `wait_until`  
   fields: `condition` (`map_id_is|not_in_battle|in_battle`), `value`, `timeout_ticks`
5. `interact`  
   fields: `button`, `attempts`, `timeout_ticks`

Validation rules:
1. Unknown step type fails fast before run.
2. Invalid direction/button fails fast.
3. Missing required fields fail schema validation.

---

## 6. Runtime Design

State machine:

1. `NAV_EXECUTE`
   1. Execute next route step.
   2. Track map/x/y progress window.
2. `BATTLE_INTERRUPT`
   1. Triggered when `in_battle == 1`.
   2. Branch to wild/trainer handling.
3. `RECOVERY`
   1. Triggered on no-progress timeout or route mismatch.
   2. Local nudge/backtrack then retry checkpoint.
4. `DONE`
   1. Success on target reached (`gym_entrance` map/coords or `brock_badge` bit).
   2. Failure on max steps, unrecoverable desync, or repeated recovery exhaustion.

---

## 7. Battle Integration Rules

1. If battle starts and battle class is wild and `--phase4-wild-run-enabled=true`:
   1. Attempt `RUN` first.
   2. If run fails/rejected, fallback to battle agent.
2. If trainer/forced battle:
   1. Use existing `MistralBattleAgent.pick_move`.
   2. Respect current legality fallback logic.
3. After battle end:
   1. Return to last route checkpoint.
   2. Resume execution.

Important mechanic note:
1. Gen 1 has no overworld sprint toggle.
2. `RUN` applies to battle escape only.

---

## 8. Recovery Policy

No-progress detection:
1. If `(map_id, x, y)` unchanged for `N` steps (default `N=40`), enter recovery.

Recovery actions (ordered):
1. `b` press burst to clear dialogs.
2. 2-step lateral nudge.
3. 2-step reverse nudge.
4. Retry current checkpoint block once.
5. On second failure, hard fail with reason `recovery_exhausted`.

All recovery events must be timeline-logged.

---

## 9. Metrics and Artifacts

`phase4_results.json` fields:
1. `run_status` (`success|failed|timeout`)
2. `target` (`gym_entrance|brock_badge`)
3. `target_reached` (bool)
4. `start_state`
5. `route_script`
6. `steps_executed`
7. `checkpoints_reached`
8. `battles_fought`
9. `wild_battles`
10. `trainer_battles`
11. `wild_run_attempts`
12. `wild_run_successes`
13. `battle_policy_mode`
14. `llm_decision_calls`
15. `llm_reflection_calls`
16. `fallback_events`
17. `recovery_events`
18. `failure_reason` (if any)

`phase4_timeline.jsonl` events:
1. `route_step_start`
2. `route_step_end`
3. `checkpoint_reached`
4. `battle_enter`
5. `battle_exit`
6. `wild_run_attempt`
7. `recovery_triggered`
8. `run_terminated`

---

## 10. Testing Plan

### Unit
1. Route schema parsing and validation.
2. State-machine transition tests (`NAV_EXECUTE -> BATTLE_INTERRUPT -> NAV_EXECUTE`).
3. Recovery escalation test.
4. Wild `RUN` decision branch tests.

### Integration
1. Route-only short path smoke (no forced battles).
2. Route + wild battle interrupt with run-attempt path.
3. Route + trainer battle handoff path.
4. Full rehearsal from `explore_start.state` to target.

### Manual rehearsal
1. Two consecutive runs from same state produce similar checkpoint progression.
2. Timeline explains every pause/retry.
3. Demo command completes without manual intervention.

---

## 11. Success Criteria

Minimum pass:
1. One command reaches `gym_entrance` autonomously.
2. Artifacts are produced and internally consistent.
3. At least one battle interrupt is handled without crash.

Stretch pass:
1. One command obtains Boulder Badge in same mode.

---

## 12. Sequencing

1. Implement route executor + schema + timeline logging.
2. Add `--phase4` CLI mode.
3. Integrate wild-run first branch.
4. Integrate battle agent fallback and trainer handling.
5. Run 3 rehearsal attempts and keep best artifact set.

---

## 13. Assumptions and Defaults

1. Existing Phase 1/2 battle logic remains the battle engine.
2. Existing Phase 3 reward files remain available but are not required at runtime for Phase 4 route mode.
3. Reliability and demonstrability are prioritized over pure-RL claims.
4. User-facing framing remains explicit: hybrid navigation + autonomous battle policy.

