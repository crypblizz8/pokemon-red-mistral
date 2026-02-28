# Pokemon Red RL Agent
**SCREENSHOT - Battle Screenshot Capture for Learning**
`Phase 1+2` · `PyBoy` · `PNG Artifacts` · `Retention Capped`

---

## 1. Summary

Add an implementation-ready runtime screenshot capability for battle learning:
1. Automatically capture battle screenshots in Phase 1 and Phase 2.
2. Use event-based capture (`battle_start`, per committed turn, `battle_end`).
3. Save PNG frames and JSON metadata for analysis.
4. Retain only the latest 25 battles to control storage growth.

This spec defines a runtime project capability, not a standalone Codex skill package.

---

## 2. Scope

### In scope
1. Phase 1 and Phase 2 screenshot capture in `run.py` battle loop.
2. PNG frame storage and per-battle metadata manifests.
3. Global index file for retained battles.
4. Automatic pruning to latest `N=25` battles.
5. Tests for recorder behavior, battle-loop integration, and CLI/preflight contracts.

### Out of scope
1. Phase 4 route-executor screenshot capture.
2. GIF/video generation and export workflows.
3. Codex skill packaging under `$CODEX_HOME/skills`.

---

## 3. Public Interfaces and Contracts

### CLI additions (`run.py`)
1. `--battle-screenshots` (`parse_bool_arg`, default `true`)
2. `--battle-screenshots-dir` (`Path`, default `artifacts/screenshots/battles`)
3. `--battle-screenshot-retain-battles` (`int`, default `25`, must be `>=1`)

Validation:
1. Hard-fail with clear message if `battle_screenshot_retain_battles < 1`.

### Config additions (`configs.py`)
1. `DEFAULT_BATTLE_SCREENSHOTS_ENABLED = True`
2. `DEFAULT_BATTLE_SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots" / "battles"`
3. `DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES = 25`

### Dependency contract
1. Add `pillow>=10,<12` to `requirements.txt`.
2. Include Pillow (`PIL` / `pillow`) in preflight dependency checks.

### New module (`pokemon/battle_screenshots.py`)
1. `BattleScreenshotConfig` dataclass fields:
   - `enabled: bool`
   - `root_dir: Path`
   - `retain_battles: int`
   - `phase: str`
2. `BattleScreenshotRecorder` methods:
   - `start_episode(episode, state_path, state_index, policy_mode, model)`
   - `capture_event(event, emu, turn=None, chosen_slot=None, outcome=None, state=None)`
   - `finish_episode(outcome, turns, hp_left, reward)`
   - `prune_old_battles()`
   - `rebuild_index()`

---

## 4. File Output Contract

### Root directory
1. `artifacts/screenshots/battles/`

### Per-battle folder naming
1. `battle_000001_ep001_phase1/`
2. `battle_id` is global, numeric, monotonic within existing retained data.

### Frame naming
1. `000_battle_start.png`
2. `001_turn_001_post_action.png`
3. `...`
4. `999_battle_end.png`

### Metadata outputs
1. Per-battle `manifest.json`
2. Root-level `index.jsonl` with one retained battle summary per line

---

## 5. Capture Behavior

1. Capture only in Phase 1 and Phase 2 battle loops.
2. Capture at battle confirmation (`battle_start`).
3. Capture after each committed turn action (`turn_post_action`).
4. Capture at battle finalization (`battle_end`).
5. Skip screenshot folders for episodes that never enter battle.
6. On screenshot/manifest/index errors, log warnings and continue battle execution.

---

## 6. Manifest Schema (`manifest.json`)

Top-level required fields:
1. `battle_id`
2. `phase`
3. `episode`
4. `state_path`
5. `state_index`
6. `policy_mode`
7. `model`
8. `outcome`
9. `turns`
10. `hp_left`
11. `reward`
12. `started_at`
13. `ended_at`
14. `frames`

`frames[]` required fields:
1. `seq`
2. `event`
3. `turn` (nullable)
4. `chosen_slot` (nullable)
5. `in_battle`
6. `player_hp`
7. `enemy_hp`
8. `path`

---

## 7. Retention Policy

1. Keep only the latest `N=25` battle folders by numeric `battle_id`.
2. Prune older folders immediately after each episode finalization.
3. Rebuild `index.jsonl` from retained `manifest.json` files after pruning.

---

## 8. Integration Points (`run.py`)

1. Instantiate `BattleScreenshotRecorder` in `run_battle_loop()`.
2. Pass recorder into `run_battle_episode()` (optional parameter).
3. Trigger capture calls:
   1. After battle start is confirmed.
   2. After committed turn action + wait logic.
   3. After final outcome resolution.
4. Keep battle behavior unchanged when screenshot capture is disabled.

---

## 9. Test Coverage

### Unit tests
1. Deterministic folder/frame naming and manifest output.
2. Recorder writes PNG + manifest with required schema fields.
3. Retention pruning with 30 synthetic battles leaves 25 newest.
4. `index.jsonl` reflects retained manifests only.

### Integration tests
1. Phase 1 battle-episode smoke creates start/turn/end screenshots.
2. Phase 2 battle-episode smoke records battle output and screenshots.
3. Timeout after battle start still captures `battle_end`.
4. No-battle episode produces no screenshot folder.

### CLI/preflight tests
1. Parser defaults include screenshot args with expected defaults.
2. Retention `<1` fails with clear validation error.
3. Missing Pillow path is reported by dependency preflight.

---

## 10. Defaults and Assumptions

1. Capture is enabled by default.
2. Event-based cadence is preferred over fixed-interval sampling.
3. Storage format is PNG + JSON metadata.
4. Retention is global across Phase 1 and Phase 2 artifacts.
5. This is a runtime feature for learning/review workflows in this repo.
