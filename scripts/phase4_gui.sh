#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_ARGS=(
  --phase4
  --phase4-start-state assets/states/explore_eval_2.state
  --phase4-policy-mode heuristic
  --phase4-wild-battle-mode hp_gated_farm
  --phase4-farm-hp-threshold 0.45
  --phase4-farm-max-consecutive-fights 3
  --phase4-max-steps 15000
  --window SDL2
  --emulation-speed 1
)

python3 run.py "${DEFAULT_ARGS[@]}" "$@"
