# Pokemon Red RL with Mistral

The project explores LLM-guided battle policy behavior in Pokemon Red, with:
- runtime battle loops in `run.py`
- isolated evaluation harness under `evals/`
- cumulative run stats in `artifacts/campaign_log.json`

## Phases
- Phase 0/1: Setup of the emualator + save states
- Phase 2: Battle(s) + Battle Memory
- Phase 3: Navigation (RL)
- Phase 4: Beating Brock with the combination of all above

## Mistral Usage
- Evals: Landed on XXX model 
- LLM reasoning learning: What does it learn from each battles. (llm_decision_calls, llm_reflection_calls)
- Policy updates on what to do for battles + navigation
- Reward mechanism learning (EUREKA reward iteration)

## Phase 2B Memory
- Phase 2 now supports persistent battle memory:
  - live file: `artifacts/memory/battle_memory.json`
  - snapshots: `artifacts/memory/history/battle_memory_###.json` (keeps last 10)
- Memory uses a hierarchy:
  - state lesson (highest priority)
  - matchup lesson (`player species x enemy species`)
  - global lesson (lowest priority)
- Default behavior for `--phase2`: memory is enabled.
- Useful flags:
  - `--disable-memory`
  - `--memory-path`
  - `--memory-snapshot-keep`
  - `--phase2b-baseline-results` (compare state-turn improvement vs baseline run)

## Phase 4 Demo
- Demo command (forced heuristic, no API dependency): `python3 run.py --phase4-demo`
- Shell shortcut: `scripts/phase4_demo.sh`
- Demo mode behavior:
  - Runs live route execution with `heuristic` battle policy.
  - Writes latest outputs to `artifacts/results/phase4_results.json` and `artifacts/results/phase4_timeline.jsonl`.
  - Writes timelapse summary to `artifacts/results/phase4_timelapse.txt`.
  - On successful live run, snapshots "last good" artifacts to:
    - `artifacts/results/phase4_last_good_results.json`
    - `artifacts/results/phase4_last_good_timeline.jsonl`
  - If live preflight/run fails and fallback is enabled, loads fallback artifacts so demo can still proceed.
- Phase 4 wild battle controls:
  - `--phase4-wild-battle-mode run_first|hp_gated_farm` (default `run_first`)
  - `--phase4-farm-hp-threshold` (default `0.45`)
  - `--phase4-farm-max-consecutive-fights` (default `3`)
  - In `hp_gated_farm`, wild battles are fought while HP is healthy and streak cap is not hit; otherwise the agent attempts `RUN` first.
