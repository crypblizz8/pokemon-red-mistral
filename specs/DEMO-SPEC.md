# DEMO-SPEC: Exploration Map Broadcast (30s Pre-Rendered MVP)

## 1) Objective
Create a reliable, presentation-ready 30-second exploration broadcast that shows:

1. Live gameplay feed (left panel)
2. Incrementally stitched exploration map (right panel)
3. Live counters (unique tiles, unique maps, elapsed time)

Primary outputs:

1. `artifacts/exploration_demo.gif`
2. `artifacts/exploration_final_map.png`
3. `artifacts/exploration_trace.json`

This is optimized for stage reliability (pre-rendered media, deterministic seed), not live training.

## 2) Why This Demo
Judges should immediately see:

1. Real emulator-driven state sampling
2. Concrete world coverage growth over time
3. Quantitative metrics tied to movement behavior

This mirrors the strongest visual pattern from exploration-map broadcasts while keeping scope feasible for current repo state.

## 3) Scope
### In scope
1. Exploration-only sampling/tracking from a required overworld state.
2. Multi-map stitching via RAM coordinates + transition offset inference.
3. Broadcast composition and export to GIF + map PNG + trace JSON.
4. Python-only rendering path (no ffmpeg dependency).

### Out of scope
1. PPO training loop.
2. Battle policy quality claims.
3. Full world-accurate cartography.
4. MP4 as required output.

## 4) Prerequisites
1. ROM exists: `assets/pokemon_red.gb`
2. Required state exists: `assets/explore_start.state` (clean overworld checkpoint)
3. Python environment active with project requirements installed.

Hard requirement:
`assets/explore_start.state` must be a non-intro, non-battle start state. Existing battle-first or intro states are not acceptable for demo quality.

## 5) CLI Contract Changes (`run.py`)
Add a new mode:

1. `--exploration-demo` (bool)

Add arguments:

1. `--explore-state-path` (Path, default `assets/explore_start.state`)
2. `--demo-seconds` (int, default `30`)
3. `--demo-fps` (int, default `10`)
4. `--demo-seed` (int, default `7`)
5. `--demo-output-dir` (Path, default `artifacts`)
6. `--demo-tile-scale` (int, default `3`)
7. `--demo-sample-ticks` (int, default `4`)

Mode exclusivity:

1. `--dry-check`, `--phase1`, and `--exploration-demo` are mutually exclusive.

## 6) Config Additions (`configs.py`)
Ensure the following addresses are available as constants:

1. `RAM_ADDR_Y_POS = 0xD361`
2. `RAM_ADDR_X_POS = 0xD362` (already present)
3. `RAM_ADDR_MAP_ID = 0xD35E`
4. `RAM_ADDR_IN_BATTLE = 0xD057` (already present)

## 7) New Module: `pokemon/exploration_tracker.py`
### Data structure
`OverworldSample` fields:

1. `tick`
2. `map_id`
3. `x`
4. `y`
5. `in_battle`

### Functions/classes
1. `read_overworld_sample(emu) -> OverworldSample`
2. `MapStitcher`

`MapStitcher` responsibilities:

1. Store `map_offsets: dict[int, tuple[int, int]]`
2. Store `visited_global_tiles: set[tuple[int, int]]`
3. Store `visited_by_map: dict[int, set[tuple[int, int]]]`
4. On transition from map `A` to map `B`, infer unknown offset:
   `offset_B = offset_A + (x_A - x_B, y_A - y_B)`
5. Convert local `(map_id, x, y)` to global stitched coordinates.

### Movement policy
Implement `ExplorationPolicy`:

1. Deterministic RNG with seed (`--demo-seed`)
2. Direction persistence bias (~60%)
3. Inputs one movement at fixed cadence
4. If in battle, do not add map tiles during battle frames; press `B` to clear dialog safely

## 8) New Module: `pokemon/broadcast_renderer.py`
Implement:

1. `compose_frame(game_rgba, stitcher_state, metrics) -> PIL.Image`

Layout:

1. Left: scaled game frame (160x144 source, nearest-neighbor scale)
2. Right: stitched map canvas on dark background
3. Top/bottom text strip: elapsed, unique tiles, unique maps, current map id

Visual defaults:

1. Background `#111216`
2. Visited tile color `#b9e3a1`
3. Current position marker `#ffffff`
4. Current-map accent `#f0b24a`

## 9) Exploration Demo Pipeline (`run_exploration_demo`)
Implementation steps:

1. Preflight:
   1. Validate ROM and `explore_state_path` files
   2. Load state; verify overworld-like start (`in_battle == 0`) and non-intro sanity checks
2. Init emulator in headless mode (`window="null"`)
3. Compute run budget:
   1. `frames = demo_seconds * demo_fps`
   2. `total_ticks = frames * demo_sample_ticks`
4. Main loop:
   1. Tick emulator
   2. Apply movement policy at cadence
   3. Sample RAM every `demo_sample_ticks`
   4. Update stitcher
   5. Capture current game frame from `pyboy.screen.ndarray`
   6. Render composite frame
5. Save outputs:
   1. GIF using Pillow (`save_all=True`)
   2. Final map PNG
   3. Trace JSON
6. Print summary:
   1. paths
   2. unique tiles
   3. unique maps
   4. map IDs seen

## 10) `exploration_trace.json` Schema
Top-level fields:

1. `run_config`
2. `summary`
3. `map_offsets`
4. `samples`

`summary` fields:

1. `duration_seconds`
2. `frames`
3. `unique_tiles`
4. `unique_maps`
5. `map_ids_seen`
6. `start_map_id`
7. `end_map_id`

Each `samples[]` item:

1. `tick`
2. `frame_index`
3. `map_id`
4. `x`
5. `y`
6. `in_battle`
7. `global_x`
8. `global_y`

## 11) Acceptance Criteria
1. One command produces all 3 artifacts.
2. GIF duration is `30s +/- 1s`.
3. Stitched map visibly expands (not static).
4. `summary.unique_tiles >= 40` from valid overworld state.
5. Deterministic output under fixed seed (trace stability across repeated runs).

## 12) Tests
### Unit
1. Transition offset inference correctness.
2. Local-to-global coordinate mapping stability.
3. Deterministic action generation from fixed seed.
4. Renderer output dimensions and color mode.

### Integration
1. 5-second smoke run produces GIF/PNG/JSON.
2. Invalid start state exits with clear message.
3. Battle interruption does not crash pipeline.

### Manual rehearsal
1. Confirm GIF readability on presentation display.
2. Confirm final map legibility at a glance.
3. Rehearse 30-second narration against generated GIF.

## 13) Command Examples
Primary:

```bash
source .venv/bin/activate
python run.py --exploration-demo \
  --explore-state-path assets/explore_start.state \
  --demo-seconds 30 \
  --demo-fps 10 \
  --demo-seed 7 \
  --demo-output-dir artifacts
```

Fast smoke:

```bash
python run.py --exploration-demo \
  --explore-state-path assets/explore_start.state \
  --demo-seconds 5 \
  --demo-fps 5 \
  --demo-output-dir /tmp/explore-smoke
```

## 14) Troubleshooting
1. `in_battle` at start:
   1. Re-record `assets/explore_start.state` while standing in overworld and idle.
2. Intro-like state detected (zeroed player fields):
   1. Load existing save in emulator and create new checkpoint after gaining control.
3. Map not expanding:
   1. Increase `--demo-seconds` or reduce `--demo-sample-ticks`.
4. Jittery visuals:
   1. Increase direction persistence or lower movement cadence.

## 15) 30-Second Presentation Script
1. "This is a real emulator run sampled from RAM in real time."
2. "Left is live gameplay; right is a stitched exploration map built on-the-fly from map ID and coordinates."
3. "As coverage grows, we log measurable progress: unique tiles, maps reached, and a full trace artifact for reproducibility."

## 16) Assumptions and Defaults
1. Demo is pre-rendered, not live.
2. Visual style is stitched-map growth.
3. Scope is MVP exploration broadcast only.
4. GIF is the required media output.
5. Start state must be `assets/explore_start.state` and valid overworld.
