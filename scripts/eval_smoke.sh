#!/usr/bin/env bash
# Usage: ./scripts/eval_smoke.sh
# Runs a low-cost Phase 1 evaluation smoke command with the local .venv.
# Uses the first three .state files in assets/states for LOSO.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing virtual environment at $ROOT_DIR/.venv" >&2
  echo "Create it first: uv venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
cd "$ROOT_DIR"

STATE_DIR="$ROOT_DIR/assets/states"
if [[ ! -d "$STATE_DIR" ]]; then
  echo "Missing state directory: $STATE_DIR" >&2
  echo "Create it and add at least three .state files before running eval smoke." >&2
  exit 1
fi

STATE_FILES=()
while IFS= read -r state_file; do
  STATE_FILES+=("$state_file")
done < <(find "$STATE_DIR" -maxdepth 1 -type f -name '*.state' | sort)
if [[ "${#STATE_FILES[@]}" -lt 3 ]]; then
  echo "LOSO eval smoke requires at least 3 .state files in $STATE_DIR" >&2
  echo "Found ${#STATE_FILES[@]}. Add more save states before running eval smoke." >&2
  exit 1
fi

SELECTED_STATE_PATHS=("${STATE_FILES[@]:0:3}")
STATE_PATHS_CSV="$(IFS=,; echo "${SELECTED_STATE_PATHS[*]}")"

python evals/run_phase1_eval.py \
  --models mistral-large-latest \
  --state-paths "$STATE_PATHS_CSV" \
  --train-episodes 2 \
  --eval-episodes 1 \
  --max-folds 1 \
  --max-turns 12
