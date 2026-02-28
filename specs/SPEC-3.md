# Pokemon Red RL Agent
**SPEC-3 - Phase 3B Hybrid Brock-Clear Plan**
`API Track` · `Mistral Large` · `PyBoy` · `Hybrid Autonomy`

---

## 1. Objective

Deliver a reliable, judge-credible **Phase 3B** run that reaches first gym completion from an early save-state using:

1. Automatic overworld navigation via scripted/heuristic route execution.
2. Autonomous battle decisions via Mistral battle policy.
3. Explicit disclosure that this is a hybrid system.

Primary goal: obtain Boulder Badge in a full automated run.

---

## 2. Why Phase 3B Exists

Phase 1/2 already prove battle-policy learning.
Phase 3 (EUREKA nav rewards) is broader and higher-risk.
Phase 3B is the bridge: reliable end-to-end gym clear under hackathon constraints.

---

## 3. Scope

### In scope
1. Start from one early run state and execute full path to Brock clear.
2. Deterministic route navigation with recovery logic.
3. Battle handoff to existing Mistral battle agent.
4. Artifact logging for judge verification.

### Out of scope
1. Fully policy-driven overworld navigation.
2. PPO/EUREKA reward training inside 3B.
3. Fine-tuning track work before 3B passes.

---

## 4. Public Interfaces and Files

### New/updated files
1. `specs/SPEC-3.md`
2. `pokemon/route_executor.py` (new)
3. `run.py` (new `--phase3b` mode)
4. `pokemon/emulator.py` (badge helpers)
5. `assets/routes/brock_hybrid_route.json` (new)
6. `artifacts/phase3b_results.json` (output)
7. `artifacts/phase3b_timeline.jsonl` (output)

### CLI contract
1. `--phase3b` (bool)
2. `--start-state` (Path)
3. `--route-script` (Path)
4. `--phase3b-results-path` (Path)
5. `--phase3b-max-steps` (int)
6. `--phase3b-battle-policy` (`llm|hybrid|heuristic`)

---

## 5. Runtime Design

1. Load ROM + `--start-state`.
2. Start route executor with deterministic input sequence.
3. If `in_battle == 1`, pause route and run battle loop using `MistralBattleAgent.pick_move`.
4. Resume route after battle.
5. Apply recovery logic on stuck/timeouts:
   - checkpoint timeout
   - short backtrack
   - retry from last checkpoint
6. Stop with success when Boulder Badge bit is set.
7. Persist structured results + timeline.

---

## 6. Route and Recovery Mechanism

### Route format (`brock_hybrid_route.json`)
1. `move`: direction + frames/ticks
2. `press`: button + count
3. `wait_until`: condition + timeout
4. `checkpoint`: named progress marker

### Recovery requirements
1. Detect no-progress windows (x/y/map unchanged too long).
2. Attempt local correction (nudge/backtrack).
3. Retry checkpoint once before hard fail.
4. Always log recovery attempts.

---

## 7. Metrics and Artifacts

`phase3b_results.json` fields:
1. `run_status` (`success|failed|timeout`)
2. `boulder_badge_obtained` (bool)
3. `start_state`
4. `route_script`
5. `policy_mode`
6. `model`
7. `route_steps_completed`
8. `battles_fought`
9. `battles_won`
10. `llm_decision_calls`
11. `llm_reflection_calls`
12. `fallback_events`
13. `failure_reason` (if any)

`phase3b_timeline.jsonl` must include timestamped events:
1. route step start/end
2. battle enter/exit
3. checkpoint reached
4. recovery triggered
5. termination reason

---

## 8. Success Criteria

1. One command runs from start-state to termination without manual input.
2. At least one rehearsal run obtains Boulder Badge.
3. Artifacts are generated and internally consistent.
4. Battle decisions are autonomous and logged.
5. Hybrid disclosure is explicit in README/demo narration.

---

## 9. Testing Plan

### Unit
1. Route step schema validation.
2. Badge bit parsing (`0xD356`).
3. Recovery state machine transitions.

### Integration
1. Short route-only smoke test.
2. Route + single battle interrupt/resume test.
3. Full Phase 3B rehearsal run with artifacts.

### Manual rehearsal
1. Verify determinism over two runs from same start-state.
2. Verify timeline log explains all pauses/retries.
3. Verify judge narrative matches actual behavior.

---

## 10. Sequencing After 3B

1. Add navigation reward mechanisms immediately after 3B:
   - `reward_v1.py`
   - `reward_v2.py`
   - EUREKA iteration loop
2. Keep fine-tuning as stretch-only after 3B + nav-reward baseline is stable.

---

## 11. Judge Framing (Locked)

Use explicit statement:

"We use a hybrid autonomous system:
- scripted/heuristic route navigation for reliability,
- autonomous Mistral policy for battle decisions and strategy reflection."

Do not present 3B as full policy-driven overworld autonomy.

---

## 12. Assumptions and Defaults

1. Hackathon window is 48 hours.
2. Reliability is prioritized over maximal autonomy claims.
3. Existing battle agent from Phase 1/2 is reused as-is.
4. Navigation reward learning starts after 3B.
