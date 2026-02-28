# Isolated Phase 1 Evals

This folder contains an isolated model-selection harness for Phase 1 battle policy evaluation.

## Isolation guarantees

- All eval code lives under `evals/`.
- All run artifacts are written under `evals/runs/<run_id>/`.
- Existing workflow files like `battle_results.json` and `run.py` behavior are unchanged.

## Default scored states

- `assets/states/before_brock.state`
- `assets/states/battle_weedle.state`
- `assets/states/battle_kakuna.state`

## Run evaluation

```bash
source .venv/bin/activate
python evals/run_phase1_eval.py \
  --models mistral-large-latest,ministral-8b-latest \
  --state-paths assets/states/before_brock.state,assets/states/battle_weedle.state,assets/states/battle_kakuna.state \
  --train-episodes 6 \
  --eval-episodes 4 \
  --max-turns 12
```

## Smoke run (quick)

```bash
source .venv/bin/activate
python evals/run_phase1_eval.py \
  --models mistral-large-latest \
  --train-episodes 2 \
  --eval-episodes 1 \
  --max-folds 1
```

## Outputs per run

- `eval_results.json`: full per-episode logs and ranking metadata
- `eval_summary.md`: compact human-readable summary
- `eval_summary.csv`: flat model comparison table
