# Pokemon Red RL with Mistral

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/bc78bb5c-0ffb-4524-9b9b-b7db7fa66041" />

The project explores LLM-guided battle policy behavior in Pokemon Red, with:
- runtime battle loops in `run.py`
- isolated evaluation harness under `evals/`
- cumulative run stats in `artifacts/campaign_log.json`
- starter save-states under `assets/states/*.state` (publishable)

## Phases
- Phase 0/1: Setup of the emulator + save states
- Phase 2: Battle(s) + Battle Memory
- Phase 3: Navigation (RL)
- Phase 4: Route executor (Route1 -> Viridian -> Route2 -> Pewter)
- Phase 5: Brock-clear executor with single-Pokemon constraint checks

## Mistral Usage
- Set `.env.local`: `MISTRAL_API_KEY=...`
- Validate key/setup: `python3 run.py --dry-check --api-ping`
- Defaults: model `mistral-large-latest`; phase1 `llm`; phase2 `hybrid`; hybrid cadence `--llm-turn-interval 3`; limits `--max-decision-calls 120`, `--max-reflection-calls 5`
- Run:
  - `python3 run.py --phase1 --model mistral-large-latest`
  - `python3 run.py --phase2 --model mistral-large-latest --policy-mode hybrid`
  - `python3 run.py --phase4 --phase4-policy-mode hybrid --model mistral-large-latest`
- Eval: `python3 evals/run_phase1_eval.py --models mistral-large-latest,ministral-8b-latest` (details: `evals/README.md`)

## Relevant LoC
- `run.py` (`3053`): `200-241` key/ping, `572-609` dry-check, `743-751` hybrid fallback, `958-1237` phase1/2 loop, `2422-2538` Mistral CLI flags
- `pokemon/battle_agent.py` (`551`): `107-139` init/counters, `319-426` decision path, `469-529` reflection, `531-548` summary metrics
- `configs.py` (`87`): `72-83` model + LLM defaults
- `evals/run_phase1_eval.py`: `365-367` `--models`, `614` ranking

## Installation
```bash
# 1) Python env + deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) API key
cp .env.example .env.local
# then set: MISTRAL_API_KEY=...

# 3) ROM (legal copy only)
# put your legally dumped Pokemon Red ROM at:
# assets/pokemon_red.gb

# 4) Save states
# This repo can include shareable `.state` files in:
# assets/states/
# If you use your own states, pass:
# --state-path / --state-paths / --state-dir

# 5) Preflight
python3 run.py --dry-check --api-ping
```

Quick run commands:
- Battles: `python3 run.py --phase1` or `python3 run.py --phase2`
- Battle evals: `python3 evals/run_phase1_eval.py --models mistral-large-latest,ministral-8b-latest`
- PPO evals (Phase 3): `python3 run.py --phase3 --phase3-train-minutes 1 --phase3-eval-episodes 1`
- Demo (unfinished/experimental): `python3 run.py --phase4-demo`

## Publishing Policy
- OK to publish: `assets/states/*.state` (starter checkpoints for reproducible runs).
- Do not publish: `assets/pokemon_red.gb` or ROM-adjacent files (`*.gb`, `*.gb.*`).
