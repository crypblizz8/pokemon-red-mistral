# Pokemon Red RL Agent
**SPEC-5 - End-Phase Brock Clear (Charmander Solo)**
`Hybrid Track` · `PyBoy` · `Route + Battle Agent` · `Validation-First`

---

## 1. Objective

Deliver a reliable end-phase run that can autonomously clear Brock and obtain the Boulder Badge with a one-Pokemon party:

1. Single battler only (Charmander species id `176`).
2. No manual inputs during run.
3. Success defined by Boulder Badge bit set.

Primary goal:
1. Reach `brock_badge` target under explicit Charmander-solo constraints.

---

## 2. Why SPEC-5 Exists

Current Phase 4 has route + battle integration, but:
1. Main demonstrated target is gym approach/entrance.
2. Brock clear is a stretch target.
3. Existing rehearsal artifacts do not show successful end-phase completion.
4. Existing battle eval set is not Brock-specific.

SPEC-5 adds the missing contract for judge-credible Brock completion under a strict "one Pokemon" condition.

---

## 3. Scope

### In scope
1. Add a dedicated Phase 5 CLI mode for Brock clear.
2. Enforce and continuously validate Charmander-only party constraint.
3. Execute route + battle loop until badge is obtained or run fails.
4. Produce explicit artifacts proving both badge success and constraint compliance.
5. Add tests for single-party validation and brock_badge target flow.

### Out of scope
1. Full game progression after first badge.
2. Party/item menu strategy beyond required scope.
3. Claiming pure-RL overworld autonomy.

---

## 4. Public Interfaces and Files

### New/updated files
1. `specs/SPEC-5.md` (new)
2. `run.py` (new `--phase5` mode + args)
3. `pokemon/route_executor.py` (phase5 validation hooks)
4. `pokemon/emulator.py` (party snapshot helpers)
5. `assets/routes/pewter_gym_to_brock_badge.json` (new route script)
6. `assets/states/before_brock.state` (valid source state; replace `.bad` placeholder)
7. `artifacts/results/phase5_results.json` (new output)
8. `artifacts/results/phase5_timeline.jsonl` (new output)
9. `tests/test_route_executor.py` (phase5 cases)

### CLI contract
1. `--phase5` (bool)
2. `--phase5-start-state` (Path, default `assets/states/before_brock.state`)
3. `--phase5-route-script` (Path, default `assets/routes/pewter_gym_to_brock_badge.json`)
4. `--phase5-max-steps` (int, default `12000`)
5. `--phase5-policy-mode` (`hybrid|heuristic`, default `hybrid`)
6. `--phase5-target` (`brock_badge`, locked default)
7. `--phase5-required-species` (int, default `176`)
8. `--phase5-single-pokemon-only` (bool, default `true`)
9. `--phase5-results-path` (Path)
10. `--phase5-timeline-path` (Path)

Mode exclusivity:
1. `--dry-check`, `--phase1`, `--phase2`, `--phase3`, `--phase4`, `--phase5` are mutually exclusive.

---

## 5. Single-Pokemon Constraint Contract

Definition of "one Pokemon":
1. Party has exactly one usable battler.
2. That battler species id equals `phase5_required_species` (default `176`, Charmander).
3. No other party member may become active during the run.

Validation checkpoints:
1. At preflight/start-state load.
2. On each battle entry.
3. On each battle exit.
4. At run termination.

Failure behavior:
1. Immediate hard-fail with `failure_reason=party_constraint_failed`.
2. Timeline event `constraint_failed` with party snapshot.

---

## 6. Runtime Design

State machine:
1. `VALIDATE_START`
   1. Load state.
   2. Verify party constraint.
2. `NAV_EXECUTE`
   1. Execute checkpointed route steps in gym flow.
3. `BATTLE_INTERRUPT`
   1. If wild and run enabled, attempt run first.
   2. If trainer/forced battle, use existing battle agent move selection.
   3. Re-validate party constraint on exit.
4. `DONE`
   1. Success when `brock_badge` bit is set.
   2. Failure on timeout, recovery exhaustion, unresolved battle, or constraint failure.

Important mechanics:
1. Agent remains move-selection only in battle (no manual intervention).
2. Success requires both badge obtained and constraint valid.

---

## 7. Route Script Requirements (Phase 5)

`assets/routes/pewter_gym_to_brock_badge.json` must:
1. Start from gym-approach/inside-gym checkpoint.
2. Include deterministic steps to engage required trainer flow and Brock.
3. Include checkpoints before trainer engagement and Brock engagement.
4. Keep target config with `brock_badge.badge_bit=0`.

Validation rules:
1. Route must contain non-empty `steps`.
2. Route must include at least one gym checkpoint and one pre-Brock checkpoint.
3. Unknown step types fail fast.

---

## 8. Results and Timeline Schema

`phase5_results.json` fields:
1. `run_status` (`success|failed|timeout`)
2. `target` (`brock_badge`)
3. `target_reached` (bool)
4. `party_constraint_passed` (bool)
5. `required_species_id` (int)
6. `active_species_id_final` (int)
7. `party_species_ids_final` (int[])
8. `start_state`
9. `route_script`
10. `steps_executed`
11. `checkpoints_reached`
12. `battles_fought`
13. `trainer_battles`
14. `wild_battles`
15. `llm_decision_calls`
16. `fallback_events`
17. `recovery_events`
18. `failure_reason` (if any)

`phase5_timeline.jsonl` required events:
1. `route_step_start`
2. `route_step_end`
3. `checkpoint_reached`
4. `battle_enter`
5. `battle_exit`
6. `constraint_check`
7. `constraint_failed` (if triggered)
8. `run_terminated`

---

## 9. Testing Plan

### Unit
1. Party parser returns stable species list and active species id from RAM.
2. Constraint check passes for `[176]` and fails for non-solo or non-176 parties.
3. `brock_badge` target check succeeds when badge bit `0` is set.

### Integration
1. Phase 5 fails fast when start state violates party constraint.
2. Phase 5 reports failure when route completes without badge.
3. Phase 5 success path writes consistent results and timeline.

### Regression
1. Existing Phase 4 tests continue to pass.
2. Route executor behavior for `gym_entrance` target remains unchanged.

---

## 10. Success Criteria

Minimum pass:
1. A valid `before_brock.state` is present and passes preflight.
2. At least 1 autonomous run reaches `brock_badge` with `party_constraint_passed=true`.
3. Artifacts show no manual inputs and no constraint violations.

Reliability pass:
1. 3 consecutive runs from same state complete with at least 2 successes.
2. All failures are classifiable (`timeout`, `recovery_exhausted`, `party_constraint_failed`, etc.).

---

## 11. Sequencing

1. Add emulator party snapshot helpers.
2. Add Phase 5 CLI wiring and preflight checks.
3. Add route + runtime constraint validation hooks.
4. Add phase5 result/timeline schema.
5. Add tests.
6. Run rehearsals and capture artifacts.

---

## 12. Assumptions and Defaults

1. Badge bit `0` corresponds to Boulder Badge.
2. Start state is positioned to reach Brock without additional game setup.
3. Existing battle move-selection logic is retained.
4. Reliability and auditability take priority over broader autonomy claims.
