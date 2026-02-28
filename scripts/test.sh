#!/usr/bin/env bash
# Usage: ./scripts/test.sh
# Runs all repository unittest suites with the local .venv.
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

python -m unittest discover -s tests
python -m unittest discover -s evals/tests
