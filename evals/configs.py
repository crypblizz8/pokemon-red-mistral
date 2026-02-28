from __future__ import annotations

from pathlib import Path


EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
STATES_DIR = ASSETS_DIR / "states"

ROM_PATH = ASSETS_DIR / "pokemon_red.gb"

DEFAULT_STATE_PATHS = [
    STATES_DIR / "before_brock.state",
    STATES_DIR / "battle_weedle.state",
    STATES_DIR / "battle_kakuna.state",
]

DEFAULT_MODEL = "mistral-large-latest"
DEFAULT_TRAIN_EPISODES = 6
DEFAULT_EVAL_EPISODES = 4
DEFAULT_UPDATE_EVERY = 5
DEFAULT_MAX_TURNS = 12
DEFAULT_BATTLE_WAIT_TICKS = 12000
DEFAULT_BATTLE_SEARCH_STEPS = 220
DEFAULT_TURN_TICK_BUDGET = 1200

DEFAULT_OUTPUT_ROOT = EVALS_DIR / "runs"
DEFAULT_CAMPAIGN_LOG_PATH = PROJECT_ROOT / "artifacts" / "campaign_log.json"
MIN_STATE_FILE_BYTES = 1
NO_CLEAR_WINNER_GAP = 0.10
