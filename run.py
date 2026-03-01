#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict, List, Tuple

from configs import (
    BATTLE_STATE_PATH,
    PROJECT_ROOT,
    DEFAULT_BATTLE_SCREENSHOTS_ENABLED,
    DEFAULT_BATTLE_SCREENSHOTS_DIR,
    DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES,
    DEFAULT_NAV_STATE_PATH,
    DEFAULT_CAMPAIGN_LOG_PATH,
    DEFAULT_BATTLE_SEARCH_STEPS,
    DEFAULT_BATTLE_WAIT_TICKS,
    DEFAULT_EPISODES,
    DEFAULT_LLM_TURN_INTERVAL,
    DEFAULT_MEMORY_PATH,
    DEFAULT_MEMORY_SNAPSHOT_KEEP,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_DECISION_CALLS,
    DEFAULT_MAX_REFLECTION_CALLS,
    DEFAULT_MODEL,
    DEFAULT_PHASE3_OUTPUT_PATH,
    DEFAULT_PHASE3_SUMMARY_PATH,
    DEFAULT_PHASE4_START_STATE_PATH,
    DEFAULT_PHASE4_ROUTE_SCRIPT_PATH,
    DEFAULT_PHASE4_RESULTS_PATH,
    DEFAULT_PHASE4_TIMELINE_PATH,
    DEFAULT_PHASE4_TIMELAPSE_PATH,
    DEFAULT_PHASE4_LAST_GOOD_RESULTS_PATH,
    DEFAULT_PHASE4_LAST_GOOD_TIMELINE_PATH,
    DEFAULT_PHASE4_WILD_BATTLE_MODE,
    DEFAULT_PHASE4_FARM_HP_THRESHOLD,
    DEFAULT_PHASE4_FARM_MAX_CONSECUTIVE_FIGHTS,
    DEFAULT_PHASE5_START_STATE_PATH,
    DEFAULT_PHASE5_ROUTE_SCRIPT_PATH,
    DEFAULT_PHASE5_RESULTS_PATH,
    DEFAULT_PHASE5_TIMELINE_PATH,
    DEFAULT_PHASE5_REQUIRED_SPECIES,
    DEFAULT_PHASE5_STRENGTH_MIN_LEVEL,
    DEFAULT_PHASE5_STRENGTH_RUNS,
    DEFAULT_PHASE5_STRENGTH_PASS_RATE,
    DEFAULT_PHASE5_STRENGTH_CHECK,
    DEFAULT_PHASE2B_MIN_TURN_IMPROVEMENT,
    DEFAULT_PHASE2B_TARGET_STATE_INDEX,
    DEFAULT_TURN_TICK_BUDGET,
    DEFAULT_UPDATE_EVERY,
    ENV_FILE_PATH,
    EXPECTED_PYTHON_MAJOR,
    EXPECTED_PYTHON_MINOR,
    MIN_NONEMPTY_FILE_BYTES,
    RAM_ADDR_PLAYER_LEVEL,
    RAM_ADDR_PLAYER_HP,
    RAM_ADDR_PLAYER_SPECIES,
    RAM_ADDR_X_POS,
    ROM_PATH,
)
from pokemon.battle_agent import MistralBattleAgent
from pokemon.battle_memory import BattleMemory
from pokemon.battle_screenshots import BattleScreenshotConfig, BattleScreenshotRecorder
from pokemon.campaign_log import append_campaign_log_entry, campaign_log_report
from pokemon.nav_guidance import load_guidance_profile, score_transition
from pokemon.phase3_metrics import (
    aggregate_episode_rows,
    build_phase3_markdown,
    build_v2_critique_prompt,
    compare_versions,
)
from pokemon.route_executor import RouteValidationError, load_route_script, run_phase4_route

DEFAULT_PHASE4_FOREST_PROFILE_PATH = Path("artifacts/results/forest_transition_profile.json")
DEFAULT_PHASE4_FOREST_PROBE_STEPS = 5000


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def format_repo_relative(path: Path, *, root: Path = PROJECT_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def load_env_file(path: Path) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def check_python_version(allow_newer_python: bool) -> CheckResult:
    major, minor = sys.version_info[:2]
    current = f"{major}.{minor}.{sys.version_info.micro}"
    expected = f"{EXPECTED_PYTHON_MAJOR}.{EXPECTED_PYTHON_MINOR}.x"

    if allow_newer_python:
        if (major, minor) < (EXPECTED_PYTHON_MAJOR, EXPECTED_PYTHON_MINOR):
            return CheckResult(
                "python",
                False,
                f"Python {current} is too old. Need >= {expected}.",
            )
        return CheckResult("python", True, f"Python {current} is compatible (>= {expected}).")

    if (major, minor) != (EXPECTED_PYTHON_MAJOR, EXPECTED_PYTHON_MINOR):
        return CheckResult(
            "python",
            False,
            f"Python {current} does not match required {expected}. "
            "Use pyenv local 3.11.14 (or another 3.11.x).",
        )
    return CheckResult("python", True, f"Python {current} matches required {expected}.")


def check_dependencies() -> CheckResult:
    requirements = [
        ("pyboy", "pyboy"),
        ("PIL", "pillow"),
        ("mistralai", "mistralai"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("stable_baselines3", "stable-baselines3"),
    ]
    missing: List[str] = []
    detected: List[str] = []

    for module_name, dist_name in requirements:
        if importlib.util.find_spec(module_name) is None:
            missing.append(dist_name)
            continue

        try:
            version = importlib_metadata.version(dist_name)
            detected.append(f"{dist_name}=={version}")
        except Exception:
            detected.append(dist_name)

    if missing:
        return CheckResult(
            "deps",
            False,
            "Missing packages: " + ", ".join(missing),
        )
    return CheckResult("deps", True, "Installed: " + ", ".join(detected))


def check_phase4_demo_dependencies() -> CheckResult:
    if importlib.util.find_spec("pyboy") is None:
        return CheckResult(
            "phase4_demo_deps",
            False,
            "Missing package: pyboy (required for live demo execution).",
        )
    return CheckResult("phase4_demo_deps", True, "pyboy import detected.")


def check_file_exists_nonempty(path: Path, label: str) -> CheckResult:
    if not path.exists():
        return CheckResult(label, False, f"Missing file: {path}")
    size = path.stat().st_size
    if size < MIN_NONEMPTY_FILE_BYTES:
        return CheckResult(label, False, f"File exists but is empty: {path}")
    return CheckResult(label, True, f"Found {path} ({size} bytes)")


def check_mistral_api_key_present() -> CheckResult:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    placeholder_values = {
        "",
        "replace_me",
        "your_key_here",
        "changeme",
        "set_me",
    }
    if key.lower() in placeholder_values:
        return CheckResult(
            "mistral_key",
            False,
            "MISTRAL_API_KEY is missing. Set it in environment or .env.local.",
        )
    return CheckResult("mistral_key", True, "MISTRAL_API_KEY is present.")


def check_mistral_api_ping() -> CheckResult:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        return CheckResult("mistral_ping", False, "Cannot ping API: MISTRAL_API_KEY missing.")

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status != 200:
                return CheckResult(
                    "mistral_ping",
                    False,
                    f"Unexpected status from Mistral API: {response.status}",
                )
    except urllib.error.HTTPError as exc:
        return CheckResult("mistral_ping", False, f"Mistral API HTTP {exc.code}: {exc.reason}")
    except Exception as exc:  # network and transport issues
        return CheckResult("mistral_ping", False, f"Mistral API ping failed: {exc}")

    return CheckResult("mistral_ping", True, "Mistral API key validated with /v1/models.")


def check_pyboy_smoke(rom_path: Path, state_path: Path) -> CheckResult:
    try:
        from pyboy import PyBoy
    except Exception as exc:
        return CheckResult("pyboy_smoke", False, f"pyboy import failed: {exc}")

    pyboy = None
    try:
        try:
            pyboy = PyBoy(str(rom_path), window="null")
        except TypeError:
            # Compatibility fallback for older PyBoy constructor signatures.
            pyboy = PyBoy(str(rom_path), window_type="headless")

        if hasattr(pyboy, "set_emulation_speed"):
            pyboy.set_emulation_speed(0)

        for _ in range(1000):
            pyboy.tick()

        # PyBoy v1 used get_memory_value; v2 exposes memory via index access.
        if hasattr(pyboy, "get_memory_value"):
            x_pos = int(pyboy.get_memory_value(RAM_ADDR_X_POS))
            hp = int(pyboy.get_memory_value(RAM_ADDR_PLAYER_HP))
        else:
            x_pos = int(pyboy.memory[RAM_ADDR_X_POS])
            hp = int(pyboy.memory[RAM_ADDR_PLAYER_HP])

        if not (0 <= x_pos <= 255):
            return CheckResult("pyboy_smoke", False, f"x position RAM read out of byte range: {x_pos}")
        if not (0 <= hp <= 255):
            return CheckResult("pyboy_smoke", False, f"player HP RAM read out of byte range: {hp}")

        with state_path.open("rb") as handle:
            pyboy.load_state(handle)
        for _ in range(30):
            pyboy.tick()

        if hasattr(pyboy, "get_memory_value"):
            state_hp = int(pyboy.get_memory_value(RAM_ADDR_PLAYER_HP))
            state_level = int(pyboy.get_memory_value(0xD18C))
            state_species = int(pyboy.get_memory_value(0xD16B))
        else:
            state_hp = int(pyboy.memory[RAM_ADDR_PLAYER_HP])
            state_level = int(pyboy.memory[0xD18C])
            state_species = int(pyboy.memory[0xD16B])

        if state_hp == 0 and state_level == 0 and state_species == 0:
            return CheckResult(
                "pyboy_smoke",
                False,
                "Loaded save-state appears to be intro/title flow. "
                "Create a new state from overworld or active battle.",
            )
    except Exception as exc:
        return CheckResult("pyboy_smoke", False, f"PyBoy smoke test failed: {exc}")
    finally:
        if pyboy is not None:
            try:
                pyboy.stop(save=False)
            except TypeError:
                pyboy.stop()
            except Exception:
                pass

    return CheckResult(
        "pyboy_smoke",
        True,
        f"PyBoy booted ROM, RAM reads OK (x={x_pos}, hp={hp}), state load OK.",
    )


def print_result(result: CheckResult) -> None:
    icon = "PASS" if result.passed else "FAIL"
    print(f"[{icon}] {result.name}: {result.message}")


def parse_bool_arg(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw!r}")


def parse_state_paths_arg(raw: str) -> List[Path]:
    parts = [item.strip() for item in raw.split(",")]
    return [Path(item) for item in parts if item]


def resolve_phase1_state_paths(args: argparse.Namespace) -> List[Path]:
    if args.state_paths:
        candidates = parse_state_paths_arg(args.state_paths)
    elif args.state_dir:
        state_dir = Path(args.state_dir)
        candidates = sorted(state_dir.glob("*.state"))
    else:
        candidates = [args.state_path]

    resolved: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(path)
    return resolved


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    resolved: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(path)
    return resolved


def resolve_phase3_eval_state_paths(args: argparse.Namespace) -> List[Path]:
    eval_paths = parse_state_paths_arg(args.phase3_eval_state_paths)
    merged = [Path(args.nav_state_path), *eval_paths]
    return _dedupe_paths([Path(path) for path in merged])


def load_reward_function(path: Path) -> Callable[[dict, dict, dict], float]:
    path = Path(path)
    module_name = f"phase3_reward_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_fn = getattr(module, "reward_fn", None)
    if not callable(reward_fn):
        raise RuntimeError(f"{path} must define callable reward_fn(prev_obs, curr_obs, ctx)")
    return reward_fn


def check_nav_state_ready(rom_path: Path, state_path: Path, label: str) -> CheckResult:
    emu = None
    try:
        from pokemon.emulator import PokemonEmulator

        emu = PokemonEmulator(
            rom_path=rom_path,
            state_path=state_path,
            window="null",
            emulation_speed=0,
        )
        emu.reset(state_path=state_path)
        nav = emu.get_nav_state()
        level = int(emu.read(RAM_ADDR_PLAYER_LEVEL))
        species = int(emu.read(RAM_ADDR_PLAYER_SPECIES))
        if int(nav.get("in_battle", 1)) != 0:
            return CheckResult(label, False, f"{state_path} starts in battle; expected overworld.")
        if int(nav.get("hp", 0)) == 0 and level == 0 and species == 0:
            return CheckResult(
                label,
                False,
                f"{state_path} looks like intro/title flow; expected active overworld state.",
            )
        return CheckResult(
            label,
            True,
            (
                f"map_id={int(nav.get('map_id', 0))} "
                f"x={int(nav.get('x', 0))} y={int(nav.get('y', 0))} "
                f"in_battle={int(nav.get('in_battle', 0))}"
            ),
        )
    except Exception as exc:
        return CheckResult(label, False, f"State validation failed: {exc}")
    finally:
        if emu is not None:
            emu.stop()


def pick_episode_state_index(episode: int, state_count: int, rotation: str) -> int:
    if state_count <= 1:
        return 0
    if rotation == "random":
        return random.randrange(state_count)
    return (episode - 1) % state_count


def compute_window_win_rate(records: List[Dict[str, object]]) -> float:
    if not records:
        return 0.0
    wins = sum(1 for row in records if row.get("outcome") == "win")
    return round(wins / len(records), 4)


def compute_window_avg_reward(records: List[Dict[str, object]]) -> float:
    if not records:
        return 0.0
    rewards = [float(row.get("reward", 0.0)) for row in records]
    return round(sum(rewards) / len(rewards), 4)


def compute_phase1_metrics(agent: MistralBattleAgent) -> Dict[str, object]:
    episodes = agent.history_as_dicts()
    first_five = episodes[:5]
    last_five = episodes[-5:] if len(episodes) >= 5 else episodes
    first_win_rate = compute_window_win_rate(first_five)
    last_win_rate = compute_window_win_rate(last_five)
    combos = {
        (row.get("outcome"), row.get("turns"), row.get("hp_left"))
        for row in episodes
    }
    used_states = sorted(
        {
            str(row.get("state_path"))
            for row in episodes
            if isinstance(row.get("state_path"), str) and row.get("state_path")
        }
    )
    return {
        "first_5_win_rate": first_win_rate,
        "last_5_win_rate": last_win_rate,
        "improved_last5_vs_first5": bool(last_win_rate > first_win_rate),
        "unique_outcome_turn_hp_combos": len(combos),
        "unique_state_paths_used": len(used_states),
        "state_paths_used": used_states,
    }


def compute_phase2_metrics(agent: MistralBattleAgent) -> Dict[str, object]:
    episodes = agent.history_as_dicts()
    first_five = episodes[:5]
    last_five = episodes[-5:] if len(episodes) >= 5 else episodes
    first_win_rate = compute_window_win_rate(first_five)
    last_win_rate = compute_window_win_rate(last_five)
    first_avg_reward = compute_window_avg_reward(first_five)
    last_avg_reward = compute_window_avg_reward(last_five)
    pp_depletion_total = sum(int(row.get("pp_depletion_events", 0)) for row in episodes)
    return {
        "illegal_move_attempts_total": int(agent.illegal_move_attempts),
        "legality_fallback_total": int(agent.legality_fallback_count),
        "pp_depletion_events": int(pp_depletion_total),
        "llm_decision_calls": int(agent.llm_decision_calls),
        "llm_reflection_calls": int(agent.llm_reflection_calls),
        "budget_fallback_total": int(agent.budget_fallback_count),
        "first_5_win_rate": first_win_rate,
        "last_5_win_rate": last_win_rate,
        "first_5_avg_reward": first_avg_reward,
        "last_5_avg_reward": last_avg_reward,
        "improved_last5_vs_first5": bool(last_win_rate > first_win_rate),
        "improved_last5_avg_reward": bool(last_avg_reward > first_avg_reward),
    }


def _state_avg_turns(records: List[Dict[str, object]], state_index: int) -> float:
    rows = [
        row
        for row in records
        if int(row.get("state_index", -1)) == int(state_index)
        and str(row.get("outcome", "")) == "win"
    ]
    if not rows:
        return 0.0
    turns = [float(row.get("turns", 0.0)) for row in rows]
    return round(sum(turns) / len(turns), 4)


def _load_results_episodes(path: Path) -> List[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("episodes", [])
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, object]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def compute_phase2b_metrics(
    agent: MistralBattleAgent,
    baseline_results: Path | None,
    target_state_index: int,
    min_turn_improvement: float,
) -> Dict[str, object]:
    history_rows = agent.history_as_dicts()
    target_state_avg_turns = _state_avg_turns(history_rows, target_state_index)
    out: Dict[str, object] = {
        "memory_enabled": bool(agent.battle_memory is not None),
        "target_state_index": int(target_state_index),
        "target_state_avg_turns": target_state_avg_turns,
    }
    if agent.battle_memory is not None:
        out.update(agent.battle_memory.metrics())
    else:
        out.update(
            {
                "memory_rules_loaded": 0,
                "memory_rules_written": 0,
                "memory_hint_count": 0,
                "memory_override_count": 0,
                "repeated_mistake_events": 0,
                "repeated_mistake_rate": 0.0,
            }
        )

    if baseline_results is None:
        return out

    baseline_rows = _load_results_episodes(baseline_results)
    baseline_avg_turns = _state_avg_turns(baseline_rows, target_state_index)
    improvement = round(baseline_avg_turns - target_state_avg_turns, 4)
    gate_pass = bool(improvement >= float(min_turn_improvement)) if baseline_avg_turns > 0 else False
    out["baseline_target_state_avg_turns"] = baseline_avg_turns
    out["target_state_turn_improvement"] = improvement
    out["phase2b_gate_pass"] = gate_pass
    out["phase2b_min_turn_improvement"] = float(min_turn_improvement)
    return out


def run_dry_check(args: argparse.Namespace) -> int:
    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    results = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_file_exists_nonempty(args.state_path, "battle_state"),
        check_mistral_api_key_present(),
    ]

    if args.api_ping:
        results.append(check_mistral_api_ping())

    if results[1].passed and results[2].passed and results[3].passed:
        results.append(check_pyboy_smoke(args.rom_path, args.state_path))
    else:
        results.append(
            CheckResult(
                "pyboy_smoke",
                False,
                "Skipped: dependency/ROM/state prerequisites failed.",
            )
        )

    for result in results:
        print_result(result)

    all_pass = all(result.passed for result in results)
    print()
    if all_pass:
        print("Phase 0 dry-check passed.")
        return 0

    print("Phase 0 dry-check failed. Fix FAIL checks and re-run.")
    return 1


def run_battle_episode(
    emu: object,
    agent: MistralBattleAgent,
    screenshot_recorder: BattleScreenshotRecorder | None,
    state_path: Path,
    state_index: int,
    episode: int,
    max_turns: int,
    turn_tick_budget: int,
    battle_wait_ticks: int,
    battle_search_steps: int,
    exploration_epsilon: float,
    phase2: bool,
    llm_turn_interval: int,
    max_decision_calls: int,
) -> Tuple[Dict[str, object], str]:
    start_decision_calls = agent.llm_decision_calls
    start_fallback_count = agent.legality_fallback_count + agent.budget_fallback_count
    start_illegal_attempts = agent.illegal_move_attempts
    pp_depletion_events = 0
    state_label = state_path.name
    agent.start_episode_context(state_label=state_label, state_index=state_index)
    if screenshot_recorder is not None:
        screenshot_recorder.start_episode(
            episode=episode,
            state_path=state_path,
            state_index=state_index,
            policy_mode=agent.policy_mode,
            model=agent.model,
        )

    emu.reset(state_path=state_path)
    starting_state = emu.get_battle_state()
    likely_intro_state = (
        int(starting_state.get("player_hp", 0)) == 0
        and int(starting_state.get("player_level", 0)) == 0
        and int(starting_state.get("player_species", 0)) == 0
    )
    if likely_intro_state and not emu.in_battle():
        record = agent.record_battle(
            episode=episode,
            move_slots=[],
            outcome="invalid_state",
            hp_left=0,
            turns=1,
            llm_replies=[
                "Loaded save-state looks like intro/title flow (no active player stats). "
                "Create a new state in overworld or in-battle."
            ],
            state_path=str(state_path),
            state_index=state_index,
            decision_calls_used=0,
            fallback_events=0,
            illegal_move_attempts=0,
            pp_depletion_events=0,
        )
        agent.finalize_episode_memory(record)
        if screenshot_recorder is not None:
            screenshot_recorder.finish_episode(
                outcome=record.outcome,
                turns=record.turns,
                hp_left=record.hp_left,
                reward=record.reward,
            )
        return record.to_dict(), "Invalid save-state (intro/title), episode skipped"

    if not emu.wait_for_battle(timeout=120):
        # For phase2, keep strict battle-only starts and avoid overworld wandering.
        if not phase2:
            emu.seek_battle(max_steps=battle_search_steps)

    if not emu.wait_for_battle(timeout=battle_wait_ticks):
        record = agent.record_battle(
            episode=episode,
            move_slots=[],
            outcome="timeout",
            hp_left=0,
            turns=1,
            llm_replies=["Battle did not start within timeout."],
            state_path=str(state_path),
            state_index=state_index,
            decision_calls_used=0,
            fallback_events=0,
            illegal_move_attempts=0,
            pp_depletion_events=0,
        )
        agent.finalize_episode_memory(record)
        if screenshot_recorder is not None:
            screenshot_recorder.finish_episode(
                outcome=record.outcome,
                turns=record.turns,
                hp_left=record.hp_left,
                reward=record.reward,
            )
        return record.to_dict(), "Battle start timeout"

    if screenshot_recorder is not None:
        screenshot_recorder.capture_event(
            "battle_start",
            emu,
            state=emu.get_battle_state(),
        )

    move_slots: List[int] = []
    llm_replies: List[str] = []

    turns_taken = 0
    for turn_idx in range(max_turns):
        if not emu.in_battle():
            break

        turn_number = turn_idx + 1
        if phase2:
            state = emu.build_phase2_state(turn=turn_number)
            state["state_label"] = state_label
            state["state_index"] = state_index
            moves = state.get("moves", [])
            if isinstance(moves, list):
                depleted = [
                    row
                    for row in moves
                    if isinstance(row, dict)
                    and int(row.get("id", 0)) != 0
                    and int(row.get("pp_current", 0)) == 0
                ]
                if depleted:
                    pp_depletion_events += 1
        else:
            state = emu.get_battle_state()
            state["turn"] = turn_number

        use_llm = agent.policy_mode == "llm"
        if agent.policy_mode == "hybrid":
            interval = max(1, int(llm_turn_interval))
            use_llm = (turn_number % interval) == 1
        budget_fallback = False
        if use_llm and agent.llm_decision_calls >= max_decision_calls:
            use_llm = False
            budget_fallback = True

        slot = agent.pick_move(state, use_llm=use_llm, budget_fallback=budget_fallback)
        if exploration_epsilon > 0 and random.random() < exploration_epsilon:
            legal_slots_raw = state.get("legal_slots", [])
            legal_slots = [int(v) for v in legal_slots_raw if isinstance(v, int)]
            if not legal_slots:
                legal_slots = [
                    idx
                    for idx, move_id in enumerate(state.get("move_ids", [])[:4])
                    if int(move_id) != 0
                ]
            if legal_slots:
                slot = random.choice(legal_slots)
            else:
                slot = random.randint(0, 3)
            agent.last_reply = (
                f"{agent.last_reply}\nEXPLORATION_OVERRIDE: ACTION: {slot}"
                if agent.last_reply
                else f"EXPLORATION_OVERRIDE: ACTION: {slot}"
            )
        agent.record_turn_decision(state, slot)
        move_slots.append(slot)
        llm_replies.append(agent.last_reply)
        committed = emu.execute_move(slot)
        if not committed:
            llm_replies[-1] = (
                f"{llm_replies[-1]}\nINPUT_RETRY_EXHAUSTED"
                if llm_replies[-1]
                else "INPUT_RETRY_EXHAUSTED"
            )
        turns_taken += 1

        battle_ended = emu.wait_for_battle_end(timeout=turn_tick_budget)
        if committed and screenshot_recorder is not None:
            screenshot_recorder.capture_event(
                "turn_post_action",
                emu,
                turn=turn_number,
                chosen_slot=slot,
            )
        if battle_ended:
            break

    final_state = emu.get_battle_state()
    player_hp = int(final_state.get("player_hp", 0))
    outcome = "timeout"
    if not emu.in_battle():
        outcome = "win" if player_hp > 0 else "loss"

    if screenshot_recorder is not None:
        screenshot_recorder.capture_event(
            "battle_end",
            emu,
            outcome=outcome,
            state=final_state,
        )

    record = agent.record_battle(
        episode=episode,
        move_slots=move_slots,
        outcome=outcome,
        hp_left=player_hp,
        turns=max(turns_taken, 1),
        llm_replies=llm_replies,
        state_path=str(state_path),
        state_index=state_index,
        decision_calls_used=agent.llm_decision_calls - start_decision_calls,
        fallback_events=(
            agent.legality_fallback_count + agent.budget_fallback_count - start_fallback_count
        ),
        illegal_move_attempts=agent.illegal_move_attempts - start_illegal_attempts,
        pp_depletion_events=pp_depletion_events,
    )
    agent.finalize_episode_memory(record)
    if screenshot_recorder is not None:
        screenshot_recorder.finish_episode(
            outcome=record.outcome,
            turns=record.turns,
            hp_left=record.hp_left,
            reward=record.reward,
        )
    info = (
        f"episode={episode} state={state_index} outcome={outcome} "
        f"turns={record.turns} hp={player_hp} reward={record.reward:.1f}"
    )
    return record.to_dict(), info


def save_battle_results(
    path: Path,
    model: str,
    agent: MistralBattleAgent,
    phase: str,
    phase2b_metrics: Dict[str, object] | None = None,
) -> None:
    phase1_metrics = compute_phase1_metrics(agent)
    phase2_metrics = compute_phase2_metrics(agent)
    payload = {
        "model": model,
        "phase": phase,
        "summary": agent.summary(),
        "phase1_metrics": phase1_metrics,
        "phase2_metrics": phase2_metrics,
        "phase2b_metrics": phase2b_metrics or {},
        "strategy_versions": agent.strategy_versions,
        "episodes": agent.history_as_dicts(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_campaign_report(path: Path) -> None:
    report = campaign_log_report(path)
    print("[CAMPAIGN]")
    print(f"- path: {report['path']}")
    print(f"- real_battles: {report['real_battles']}")
    print(f"- simulations: {report['simulations']}")
    print(f"- combined_total: {report['combined']}")
    print(f"- movement_steps: {report.get('movement_steps', 0)}")
    print(
        f"- movement_steps_phase3: {report.get('phase3_steps', 0)} "
        f"movement_steps_phase4: {report.get('phase4_steps', 0)}"
    )
    print(f"- log_entries: {report['entries']}")
    if report["updated_at"]:
        print(f"- updated_at: {report['updated_at']}")


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _log_movement_campaign_run(
    campaign_log_path: Path,
    *,
    phase: str,
    source: str,
    movement_steps: int,
    metadata: Dict[str, object],
) -> None:
    try:
        append_campaign_log_entry(
            campaign_log_path,
            kind="simulation",
            count=1,
            source=source,
            movement_steps=max(0, _safe_int(movement_steps)),
            phase=phase,
            metadata=metadata,
        )
        print_campaign_report(campaign_log_path)
    except Exception as exc:
        print(f"[WARN] campaign log update failed: {exc}")


def _episode_outcome_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts = {
        "win": 0,
        "loss": 0,
        "timeout": 0,
        "invalid_state": 0,
        "other": 0,
    }
    for row in rows:
        outcome = str(row.get("outcome", ""))
        if outcome in counts:
            counts[outcome] += 1
        else:
            counts["other"] += 1
    return counts


def resolve_incremented_results_path(requested_path: Path) -> Path:
    target_dir = requested_path.expanduser().resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = requested_path.suffix or ".json"
    raw_prefix = requested_path.stem.strip() or "battle"
    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_prefix).strip("_") or "battle"

    # Keep a single battle sequence even if older files used battle_resultsN naming.
    if prefix in {"battle", "battle_results"}:
        prefix = "battle"
        legacy_battle_stem = re.compile(r"^battle(?:_results)?_?(\d+)$", re.IGNORECASE)
    else:
        legacy_battle_stem = None

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")

    max_seen = 0
    if legacy_battle_stem is not None:
        for candidate in target_dir.glob(f"*{suffix}"):
            match = legacy_battle_stem.match(candidate.stem)
            if not match:
                continue
            max_seen = max(max_seen, int(match.group(1)))
    else:
        for candidate in target_dir.glob(f"{prefix}_*{suffix}"):
            match = pattern.match(candidate.name)
            if not match:
                continue
            max_seen = max(max_seen, int(match.group(1)))

    return target_dir / f"{prefix}_{max_seen + 1}{suffix}"


def run_battle_loop(args: argparse.Namespace, phase: str) -> int:
    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    if not (0.0 <= args.exploration_epsilon <= 1.0):
        print("[FAIL] exploration_epsilon must be between 0.0 and 1.0.")
        return 1

    if args.llm_turn_interval < 1:
        print("[FAIL] llm_turn_interval must be >= 1.")
        return 1
    if args.max_decision_calls < 0:
        print("[FAIL] max_decision_calls must be >= 0.")
        return 1
    if args.max_reflection_calls < 0:
        print("[FAIL] max_reflection_calls must be >= 0.")
        return 1
    if args.memory_snapshot_keep < 1:
        print("[FAIL] memory_snapshot_keep must be >= 1.")
        return 1
    if args.phase2b_min_turn_improvement < 0:
        print("[FAIL] phase2b_min_turn_improvement must be >= 0.")
        return 1
    if args.battle_screenshot_retain_battles < 1:
        print("[FAIL] battle_screenshot_retain_battles must be >= 1.")
        return 1

    state_paths = resolve_phase1_state_paths(args)
    if not state_paths:
        print("[FAIL] battle_state: no state files resolved.")
        print("Use --state-path, --state-paths, or --state-dir with .state files.")
        return 1
    if phase == "phase2" and len(state_paths) < 4:
        print("[FAIL] phase2 requires at least 4 state files for mixed evaluation.")
        print("Provide 3+ wild states plus at least 1 trainer/gym-adjacent battle state.")
        return 1
    if args.phase2b_baseline_results and not Path(args.phase2b_baseline_results).exists():
        print(f"[FAIL] baseline results file not found: {args.phase2b_baseline_results}")
        return 1

    prereqs: List[CheckResult] = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_mistral_api_key_present(),
    ]
    for idx, path in enumerate(state_paths):
        prereqs.append(check_file_exists_nonempty(path, f"battle_state[{idx}]"))

    for item in prereqs:
        print_result(item)
    if not all(item.passed for item in prereqs):
        print(f"\n{phase.capitalize()} start blocked by failed preflight checks.")
        return 1

    policy_mode = args.policy_mode or ("hybrid" if phase == "phase2" else "llm")
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    results_path = resolve_incremented_results_path(args.results_path)
    print(f"[INFO] results output: {results_path}")
    memory_enabled = phase == "phase2" and not args.disable_memory
    memory: BattleMemory | None = None
    if memory_enabled:
        memory = BattleMemory(
            path=args.memory_path,
            snapshot_keep=args.memory_snapshot_keep,
        )
        print(
            "[MEMORY] enabled "
            f"path={format_repo_relative(args.memory_path)} "
            f"loaded_rules={memory.rules_loaded}"
        )
    from pokemon.emulator import PokemonEmulator

    emu = PokemonEmulator(
        args.rom_path,
        state_paths[0],
        window=args.window,
        emulation_speed=args.emulation_speed,
    )
    agent = MistralBattleAgent(
        api_key=api_key,
        model=args.model,
        policy_mode=policy_mode,
        battle_memory=memory,
    )
    screenshot_recorder = BattleScreenshotRecorder(
        BattleScreenshotConfig(
            enabled=bool(args.battle_screenshots),
            root_dir=args.battle_screenshots_dir,
            retain_battles=args.battle_screenshot_retain_battles,
            phase=phase,
        )
    )
    if args.battle_screenshots:
        print(
            "[SCREENSHOTS] enabled "
            f"path={format_repo_relative(args.battle_screenshots_dir)} "
            f"retain_battles={args.battle_screenshot_retain_battles}"
        )
    completed_rows: List[Dict[str, object]] = []
    phase2b_metrics: Dict[str, object] = {}

    try:
        for episode in range(1, args.episodes + 1):
            state_index = pick_episode_state_index(
                episode=episode,
                state_count=len(state_paths),
                rotation=args.state_rotation,
            )
            selected_state = state_paths[state_index]
            row, info = run_battle_episode(
                emu=emu,
                agent=agent,
                screenshot_recorder=screenshot_recorder,
                state_path=selected_state,
                state_index=state_index,
                episode=episode,
                max_turns=args.max_turns,
                turn_tick_budget=args.turn_tick_budget,
                battle_wait_ticks=args.battle_wait_ticks,
                battle_search_steps=args.battle_search_steps,
                exploration_epsilon=args.exploration_epsilon,
                phase2=(phase == "phase2"),
                llm_turn_interval=args.llm_turn_interval,
                max_decision_calls=args.max_decision_calls,
            )
            print(f"[EP] {info}")
            completed_rows.append(row)
            if phase == "phase2":
                baseline_path = (
                    Path(args.phase2b_baseline_results)
                    if args.phase2b_baseline_results
                    else None
                )
                phase2b_metrics = compute_phase2b_metrics(
                    agent=agent,
                    baseline_results=baseline_path,
                    target_state_index=args.phase2b_target_state_index,
                    min_turn_improvement=args.phase2b_min_turn_improvement,
                )
            save_battle_results(
                results_path,
                args.model,
                agent,
                phase=phase,
                phase2b_metrics=phase2b_metrics,
            )

            if episode % args.update_every == 0 and episode < args.episodes:
                if policy_mode not in {"llm", "hybrid"}:
                    print(f"[STRATEGY] skipped after episode {episode} (policy_mode={policy_mode})")
                elif agent.llm_reflection_calls >= args.max_reflection_calls:
                    print(
                        f"[STRATEGY] skipped after episode {episode} "
                        "(reflection budget exhausted)"
                    )
                else:
                    old_strategy = agent.strategy
                    new_strategy = agent.update_strategy(use_llm=True)
                    changed = "yes" if new_strategy != old_strategy else "no"
                    print(
                        f"[STRATEGY] updated after episode {episode} "
                        f"(changed={changed}, versions={len(agent.strategy_versions)})"
                    )

        summary = agent.summary()
        phase_label = "Phase 2" if phase == "phase2" else "Phase 1"
        print(
            f"\n{phase_label} complete: "
            f"episodes={summary['episodes']} wins={summary['wins']} "
            f"win_rate={summary['win_rate']} strategy_versions={summary['strategy_versions']} "
            f"policy_mode={summary['policy_mode']} cache_hits={summary['action_cache_hits']} "
            f"cache_misses={summary['action_cache_misses']} "
            f"illegal_moves={summary['illegal_move_attempts']} "
            f"legality_fallbacks={summary['legality_fallback_count']} "
            f"llm_decisions={summary['llm_decision_calls']} "
            f"llm_reflections={summary['llm_reflection_calls']} "
            f"budget_fallbacks={summary['budget_fallback_count']}"
        )
        metrics = compute_phase1_metrics(agent)
        print(
            "[PHASE1] first5_win_rate="
            f"{metrics['first_5_win_rate']} last5_win_rate={metrics['last_5_win_rate']} "
            f"improved={metrics['improved_last5_vs_first5']} "
            f"state_paths_used={metrics['unique_state_paths_used']} "
            f"unique_outcome_turn_hp_combos={metrics['unique_outcome_turn_hp_combos']}"
        )
        if phase == "phase2":
            phase2_metrics = compute_phase2_metrics(agent)
            print(
                "[PHASE2] illegal_move_attempts_total="
                f"{phase2_metrics['illegal_move_attempts_total']} "
                f"legality_fallback_total={phase2_metrics['legality_fallback_total']} "
                f"pp_depletion_events={phase2_metrics['pp_depletion_events']} "
                f"llm_decision_calls={phase2_metrics['llm_decision_calls']} "
                f"budget_fallback_total={phase2_metrics['budget_fallback_total']} "
                f"first5_avg_reward={phase2_metrics['first_5_avg_reward']} "
                f"last5_avg_reward={phase2_metrics['last_5_avg_reward']}"
            )
            phase2b_metrics = compute_phase2b_metrics(
                agent=agent,
                baseline_results=(
                    Path(args.phase2b_baseline_results)
                    if args.phase2b_baseline_results
                    else None
                ),
                target_state_index=args.phase2b_target_state_index,
                min_turn_improvement=args.phase2b_min_turn_improvement,
            )
            print(
                "[PHASE2B] memory_enabled="
                f"{phase2b_metrics['memory_enabled']} "
                f"rules_loaded={phase2b_metrics['memory_rules_loaded']} "
                f"rules_written={phase2b_metrics['memory_rules_written']} "
                f"matchup_lessons={phase2b_metrics.get('memory_matchup_lessons_written', 0)} "
                f"global_samples={phase2b_metrics.get('memory_global_lesson_samples', 0)} "
                f"hints={phase2b_metrics['memory_hint_count']} "
                f"hints_state={phase2b_metrics.get('memory_hint_state_count', 0)} "
                f"hints_matchup={phase2b_metrics.get('memory_hint_matchup_count', 0)} "
                f"hints_global={phase2b_metrics.get('memory_hint_global_count', 0)} "
                f"overrides={phase2b_metrics['memory_override_count']} "
                f"overrides_state={phase2b_metrics.get('memory_override_state_count', 0)} "
                f"overrides_matchup={phase2b_metrics.get('memory_override_matchup_count', 0)} "
                f"overrides_global={phase2b_metrics.get('memory_override_global_count', 0)} "
                f"repeated_mistake_rate={phase2b_metrics['repeated_mistake_rate']} "
                f"state{args.phase2b_target_state_index}_avg_turns="
                f"{phase2b_metrics['target_state_avg_turns']}"
            )
            if "baseline_target_state_avg_turns" in phase2b_metrics:
                print(
                    "[PHASE2B_GATE] baseline_state_avg_turns="
                    f"{phase2b_metrics['baseline_target_state_avg_turns']} "
                    f"improvement={phase2b_metrics['target_state_turn_improvement']} "
                    f"min_required={phase2b_metrics['phase2b_min_turn_improvement']} "
                    f"pass={phase2b_metrics['phase2b_gate_pass']}"
                )
            save_battle_results(
                results_path,
                args.model,
                agent,
                phase=phase,
                phase2b_metrics=phase2b_metrics,
            )
        print(f"Saved results: {results_path}")
    finally:
        if completed_rows:
            outcome_counts = _episode_outcome_counts(completed_rows)
            try:
                append_campaign_log_entry(
                    args.campaign_log_path,
                    kind="real_battle",
                    count=len(completed_rows),
                    source=phase,
                    metadata={
                        "phase": phase,
                        "model": args.model,
                        "policy_mode": policy_mode,
                        "episodes_logged": len(completed_rows),
                        "wins": outcome_counts["win"],
                        "losses": outcome_counts["loss"],
                        "timeouts": outcome_counts["timeout"],
                        "invalid_states": outcome_counts["invalid_state"],
                        "results_path": format_repo_relative(results_path),
                    },
                )
                print_campaign_report(args.campaign_log_path)
            except Exception as exc:
                print(f"[WARN] campaign log update failed: {exc}")
        emu.stop()

    return 0


def run_phase1(args: argparse.Namespace) -> int:
    return run_battle_loop(args, phase="phase1")


def run_phase2(args: argparse.Namespace) -> int:
    return run_battle_loop(args, phase="phase2")


def _train_ppo_wallclock(model: object, train_minutes: int, chunk_timesteps: int = 2048) -> Dict[str, object]:
    deadline = time.monotonic() + max(0, int(train_minutes)) * 60
    iterations = 0
    trained_timesteps = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        model.learn(total_timesteps=chunk_timesteps, reset_num_timesteps=False)
        trained_timesteps += int(chunk_timesteps)
        iterations += 1
    duration_seconds = round(time.monotonic() - started, 2)
    return {
        "train_minutes_target": int(train_minutes),
        "train_seconds_actual": duration_seconds,
        "learn_iterations": iterations,
        "timesteps_requested": trained_timesteps,
    }


def _evaluate_phase3_model(
    *,
    model: object,
    rom_path: Path,
    state_paths: List[Path],
    reward_fn: Callable[[dict, dict, dict], float],
    guidance_scorer: Callable[[dict, dict, dict], dict] | None,
    guidance_weight: float,
    eval_episodes: int,
    seed: int,
    window: str,
    emulation_speed: int,
    max_episode_steps: int,
    no_progress_limit: int,
    action_buttons: Tuple[str, ...],
) -> List[Dict[str, object]]:
    from pokemon.nav_env import PokemonNavEnv

    rows: List[Dict[str, object]] = []
    for state_index, state_path in enumerate(state_paths):
        for episode in range(eval_episodes):
            env = PokemonNavEnv(
                rom_path=rom_path,
                state_path=state_path,
                reward_fn=reward_fn,
                window=window,
                emulation_speed=emulation_speed,
                max_episode_steps=max_episode_steps,
                no_progress_limit=max(10, int(no_progress_limit)),
                guidance_scorer=guidance_scorer,
                guidance_weight=guidance_weight,
                action_buttons=action_buttons,
            )
            try:
                obs, _ = env.reset(
                    seed=int(seed) + (state_index * 1000) + episode,
                    options={"state_path": state_path},
                )
                terminated = False
                truncated = False
                info: Dict[str, object] = {}
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(int(action))
                summary = info.get("episode_summary")
                if not isinstance(summary, dict):
                    summary = env.last_episode_summary or {}
                out_row = dict(summary)
                out_row["state_index"] = state_index
                out_row["state_path"] = format_repo_relative(state_path)
                rows.append(out_row)
            finally:
                env.close()
    return rows


def run_phase3(args: argparse.Namespace) -> int:
    from pokemon.nav_env import PokemonNavEnv

    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    if args.phase3_train_minutes < 0:
        print("[FAIL] phase3_train_minutes must be >= 0.")
        return 1
    if args.phase3_eval_episodes < 1:
        print("[FAIL] phase3_eval_episodes must be >= 1.")
        return 1
    if args.phase3_guidance_weight < 0:
        print("[FAIL] phase3_guidance_weight must be >= 0.")
        return 1
    if args.phase3_no_progress_limit < 1:
        print("[FAIL] phase3_no_progress_limit must be >= 1.")
        return 1

    eval_state_paths = resolve_phase3_eval_state_paths(args)
    if len(eval_state_paths) < 3:
        print("[FAIL] phase3 requires at least 3 unique states (start + 2 eval states).")
        return 1

    prereqs: List[CheckResult] = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_file_exists_nonempty(args.phase3_reward_v1, "reward_v1_file"),
        check_file_exists_nonempty(args.phase3_reward_v2, "reward_v2_file"),
    ]
    if args.phase3_guidance_json is not None:
        prereqs.append(check_file_exists_nonempty(args.phase3_guidance_json, "phase3_guidance_json"))
    for idx, state_path in enumerate(eval_state_paths):
        prereqs.append(check_file_exists_nonempty(state_path, f"nav_state[{idx}]"))
        prereqs.append(check_nav_state_ready(args.rom_path, state_path, f"nav_state_check[{idx}]"))

    for item in prereqs:
        print_result(item)
    if not all(item.passed for item in prereqs):
        print("\nPhase3 start blocked by failed preflight checks.")
        return 1

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        print(f"[FAIL] failed to import stable-baselines3 PPO: {exc}")
        return 1

    try:
        reward_v1_fn = load_reward_function(args.phase3_reward_v1)
        reward_v2_fn = load_reward_function(args.phase3_reward_v2)
    except Exception as exc:
        print(f"[FAIL] reward module load failed: {exc}")
        return 1
    try:
        guidance_profile = load_guidance_profile(
            args.phase3_guidance_profile,
            args.phase3_guidance_json,
        )
    except Exception as exc:
        print(f"[FAIL] guidance profile load failed: {exc}")
        return 1

    guidance_scorer: Callable[[dict, dict, dict], dict] | None = None
    if str(guidance_profile.get("name", "none")) != "none":
        guidance_scorer = lambda prev, curr, ctx: score_transition(
            prev_obs=prev,
            curr_obs=curr,
            ctx=ctx,
            profile=guidance_profile,
        )
    print(
        "[PHASE3] guidance "
        f"profile={guidance_profile.get('name', 'none')} "
        f"weight={args.phase3_guidance_weight}"
    )
    action_buttons: Tuple[str, ...]
    if args.phase3_action_set == "dpad":
        action_buttons = ("up", "down", "left", "right")
    else:
        action_buttons = ("up", "down", "left", "right", "a", "b", "start", "select")
    print(f"[PHASE3] action_set={args.phase3_action_set} buttons={','.join(action_buttons)}")

    max_episode_steps = 512
    no_progress_limit = max(10, int(args.phase3_no_progress_limit))
    versions: Dict[str, Dict[str, object]] = {}
    critique_prompt = ""

    for version_name, reward_path, reward_fn in [
        ("v1", Path(args.phase3_reward_v1), reward_v1_fn),
        ("v2", Path(args.phase3_reward_v2), reward_v2_fn),
    ]:
        print(
            f"[PHASE3] training {version_name} "
            f"reward={format_repo_relative(reward_path)} "
            f"minutes={args.phase3_train_minutes}"
        )
        train_env = PokemonNavEnv(
            rom_path=args.rom_path,
            state_path=args.nav_state_path,
            reward_fn=reward_fn,
            window=args.window,
            emulation_speed=args.emulation_speed,
            max_episode_steps=max_episode_steps,
            no_progress_limit=no_progress_limit,
            guidance_scorer=guidance_scorer,
            guidance_weight=args.phase3_guidance_weight,
            action_buttons=action_buttons,
        )
        try:
            model = PPO(
                "MlpPolicy",
                train_env,
                seed=args.phase3_seed,
                n_steps=1024,
                batch_size=256,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=0,
            )
            train_stats = _train_ppo_wallclock(model, train_minutes=args.phase3_train_minutes)
            eval_rows = _evaluate_phase3_model(
                model=model,
                rom_path=args.rom_path,
                state_paths=eval_state_paths,
                reward_fn=reward_fn,
                guidance_scorer=guidance_scorer,
                guidance_weight=args.phase3_guidance_weight,
                eval_episodes=args.phase3_eval_episodes,
                seed=args.phase3_seed,
                window=args.window,
                emulation_speed=args.emulation_speed,
                max_episode_steps=max_episode_steps,
                no_progress_limit=no_progress_limit,
                action_buttons=action_buttons,
            )
        finally:
            train_env.close()

        aggregate = aggregate_episode_rows(eval_rows)
        versions[version_name] = {
            "reward_path": format_repo_relative(reward_path),
            "train_stats": train_stats,
            "episodes": eval_rows,
            "aggregate": aggregate,
        }
        print(
            f"[PHASE3] {version_name} unique_tiles={aggregate.get('unique_tiles', 0.0)} "
            f"furthest_map={aggregate.get('furthest_map_id', 0)} "
            f"avg_return={aggregate.get('avg_return', 0.0)} "
            f"stuck_ratio={aggregate.get('stuck_ratio', 0.0)}"
        )
        if version_name == "v1":
            critique_prompt = build_v2_critique_prompt(aggregate)
            print("[PHASE3] v1 critique prompt for v2 reward update:")
            print(critique_prompt)

    comparison = compare_versions(
        versions.get("v1", {}).get("aggregate", {}),
        versions.get("v2", {}).get("aggregate", {}),
    )
    payload: Dict[str, object] = {
        "phase": "phase3",
        "config": {
            "rom_path": format_repo_relative(args.rom_path),
            "nav_state_path": format_repo_relative(args.nav_state_path),
            "eval_state_paths": [format_repo_relative(path) for path in eval_state_paths],
            "train_minutes": int(args.phase3_train_minutes),
            "eval_episodes": int(args.phase3_eval_episodes),
            "seed": int(args.phase3_seed),
            "no_progress_limit": int(no_progress_limit),
            "guidance_profile": str(guidance_profile.get("name", "none")),
            "guidance_weight": float(args.phase3_guidance_weight),
            "guidance_json": (
                format_repo_relative(args.phase3_guidance_json)
                if args.phase3_guidance_json is not None
                else ""
            ),
            "action_set": str(args.phase3_action_set),
        },
        "versions": versions,
        "comparison": comparison,
        "v2_critique_prompt": critique_prompt,
    }

    output_path = Path(args.phase3_output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = Path(args.phase3_summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_phase3_markdown(payload), encoding="utf-8")

    print(
        "[PHASE3] complete "
        f"phase3_pass={comparison.get('phase3_pass', False)} "
        f"output={format_repo_relative(output_path)} "
        f"summary={format_repo_relative(summary_path)}"
    )
    version_names = sorted(str(name) for name in versions.keys())
    train_timesteps_total = 0
    eval_steps_total = 0
    eval_rows_total = 0
    for version_payload in versions.values():
        if not isinstance(version_payload, dict):
            continue
        train_stats = version_payload.get("train_stats")
        if isinstance(train_stats, dict):
            train_timesteps_total += _safe_int(train_stats.get("timesteps_requested"))
        episode_rows = version_payload.get("episodes")
        if isinstance(episode_rows, list):
            eval_rows_total += len(episode_rows)
            for row in episode_rows:
                if isinstance(row, dict):
                    eval_steps_total += _safe_int(row.get("episode_len"))
    movement_steps = train_timesteps_total + eval_steps_total
    _log_movement_campaign_run(
        args.campaign_log_path,
        phase="phase3",
        source="phase3",
        movement_steps=movement_steps,
        metadata={
            "phase": "phase3",
            "model": "ppo",
            "reward_versions": version_names,
            "train_minutes": int(args.phase3_train_minutes),
            "eval_episodes_config": int(args.phase3_eval_episodes),
            "eval_state_count": len(eval_state_paths),
            "eval_rows_logged": int(eval_rows_total),
            "train_timesteps": int(train_timesteps_total),
            "eval_steps": int(eval_steps_total),
            "movement_steps": int(movement_steps),
            "phase3_pass": bool(comparison.get("phase3_pass", False)),
            "output_path": format_repo_relative(output_path),
            "summary_path": format_repo_relative(summary_path),
        },
    )
    return 0


def _load_phase4_results_payload(path: Path) -> Dict[str, object] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _count_timeline_events(path: Path) -> int:
    try:
        return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _parse_iso_ts(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_timeline_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _format_phase4_timelapse(results: Dict[str, object], timeline_path: Path) -> str:
    rows = _load_timeline_rows(timeline_path)
    header = (
        f"phase4_timelapse status={results.get('run_status', 'failed')} "
        f"target_reached={bool(results.get('target_reached', False))} "
        f"steps_executed={int(results.get('steps_executed', 0))} "
        f"checkpoints_reached={int(results.get('checkpoints_reached', 0))}"
    )
    if not rows:
        return f"{header}\n(no timeline events found)\n"

    start_ts = _parse_iso_ts(str(rows[0].get("ts", "")))
    keep = {
        "checkpoint_reached",
        "battle_enter",
        "battle_exit",
        "recovery_triggered",
        "run_terminated",
    }
    lines: List[str] = [header, ""]
    for row in rows:
        event = str(row.get("event", "")).strip()
        if event not in keep:
            continue
        ts = _parse_iso_ts(str(row.get("ts", "")))
        rel = "t+?"
        if start_ts is not None and ts is not None:
            rel = f"t+{max(0, int((ts - start_ts).total_seconds()))}s"
        prefix = (
            f"{rel} | {event} | "
            f"map={int(row.get('map_id', 0))} "
            f"x={int(row.get('x', 0))} y={int(row.get('y', 0))}"
        )
        if event == "checkpoint_reached":
            suffix = f" | checkpoint={row.get('checkpoint_name', 'unknown')}"
        elif event == "battle_enter":
            suffix = f" | kind={row.get('battle_kind', 'unknown')}"
        elif event == "battle_exit":
            suffix = f" | result={row.get('result', 'unknown')}"
        elif event == "recovery_triggered":
            suffix = f" | reason={row.get('reason', 'unknown')}"
        else:
            suffix = (
                f" | run_status={row.get('run_status', 'unknown')} "
                f"reason={row.get('failure_reason', '')}"
            )
        lines.append(prefix + suffix)
    if len(lines) == 2:
        lines.append("(no checkpoint/battle/recovery/termination events found)")
    return "\n".join(lines) + "\n"


def _write_phase4_timelapse(results: Dict[str, object], timeline_path: Path, timelapse_path: Path) -> None:
    timelapse_path.parent.mkdir(parents=True, exist_ok=True)
    timelapse_path.write_text(
        _format_phase4_timelapse(results, timeline_path),
        encoding="utf-8",
    )


def _write_phase4_forest_profile(results: Dict[str, object], profile_path: Path) -> None:
    raw = results.get("forest_transition_profile", {})
    payload: Dict[str, object]
    if isinstance(raw, dict):
        payload = dict(raw)
    else:
        payload = {}
    payload.setdefault("edges", {})
    payload.setdefault("phase_counts", {})
    payload.setdefault("samples", [])
    payload["run_status"] = str(results.get("run_status", "failed"))
    payload["target_reached"] = bool(results.get("target_reached", False))
    payload["failure_reason"] = str(results.get("failure_reason", ""))
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_phase4_forest_probe_route_payload(max_steps: int) -> Dict[str, object]:
    return {
        "name": "phase4_forest_probe",
        "targets": {"gym_entrance": {"map_id": 2}},
        "steps": [
            {
                "type": "checkpoint",
                "name": "Forest_To_Pewter",
                "expected_map_id": 51,
                "allowed_map_ids": [50, 13, 47],
            },
            {
                "type": "traverse_until_map",
                "target_map_id": 2,
                "mode": "wall_follow_ccw",
                "max_steps": int(max(1, max_steps)),
                "hold_frames": 6,
            },
        ],
    }


def _phase4_run_succeeded(results: Dict[str, object]) -> bool:
    run_status = str(results.get("run_status", "")).strip().lower()
    target_reached = bool(results.get("target_reached", False))
    return run_status == "success" or target_reached


def _print_phase4_summary(prefix: str, results: Dict[str, object], results_path: Path, timeline_path: Path) -> None:
    run_status = str(results.get("run_status", "failed"))
    target_reached = bool(results.get("target_reached", False))
    timeline_events = _count_timeline_events(timeline_path)
    print(
        f"{prefix} "
        f"status={run_status} "
        f"target_reached={target_reached} "
        f"steps_executed={int(results.get('steps_executed', 0))} "
        f"checkpoints_reached={int(results.get('checkpoints_reached', 0))} "
        f"timeline_events={timeline_events} "
        f"results={format_repo_relative(results_path)} "
        f"timeline={format_repo_relative(timeline_path)}"
    )
    failure_reason = str(results.get("failure_reason", "")).strip()
    if failure_reason:
        print(f"{prefix} failure_reason={failure_reason}")


def _persist_phase4_last_good(
    *,
    source_results_path: Path,
    source_timeline_path: Path,
    target_results_path: Path,
    target_timeline_path: Path,
) -> None:
    target_results_path.parent.mkdir(parents=True, exist_ok=True)
    target_timeline_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source_results_path, target_results_path)
    if source_timeline_path.exists() and source_timeline_path.stat().st_size > 0:
        copyfile(source_timeline_path, target_timeline_path)


def _attempt_phase4_demo_fallback(args: argparse.Namespace, reason: str) -> int:
    candidates = [
        (
            Path(args.phase4_demo_fallback_results_path),
            Path(args.phase4_demo_fallback_timeline_path),
            True,
            "last_good",
        ),
        (
            Path(args.phase4_results_path),
            Path(args.phase4_timeline_path),
            False,
            "latest",
        ),
    ]
    for results_path, timeline_path, require_success, label in candidates:
        payload = _load_phase4_results_payload(results_path)
        if payload is None:
            continue
        if require_success and not _phase4_run_succeeded(payload):
            continue
        print(f"[PHASE4_DEMO] fallback_loaded source={label} reason={reason}")
        _print_phase4_summary(
            "[PHASE4_DEMO]",
            payload,
            results_path,
            timeline_path,
        )
        timelapse_path = Path(args.phase4_timelapse_path).expanduser().resolve()
        _write_phase4_timelapse(payload, timeline_path, timelapse_path)
        print(f"[PHASE4_DEMO] timelapse={format_repo_relative(timelapse_path)}")
        return 0

    print(
        "[PHASE4_DEMO] no fallback artifacts available. "
        f"Expected {format_repo_relative(Path(args.phase4_demo_fallback_results_path))} "
        "or latest phase4 results."
    )
    return 1


def run_phase4_demo(args: argparse.Namespace) -> int:
    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    if args.phase4_max_steps < 1:
        print("[FAIL] phase4_max_steps must be >= 1.")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="invalid_max_steps")
        return 1
    if str(args.phase4_wild_battle_mode) not in {"run_first", "hp_gated_farm"}:
        print("[FAIL] phase4_wild_battle_mode must be one of: run_first, hp_gated_farm.")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="invalid_wild_battle_mode")
        return 1
    if not (0.0 <= float(args.phase4_farm_hp_threshold) <= 1.0):
        print("[FAIL] phase4_farm_hp_threshold must be between 0 and 1.")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="invalid_farm_hp_threshold")
        return 1
    if int(args.phase4_farm_max_consecutive_fights) < 1:
        print("[FAIL] phase4_farm_max_consecutive_fights must be >= 1.")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="invalid_farm_fight_cap")
        return 1

    try:
        route_preview = load_route_script(args.phase4_route_script)
    except RouteValidationError as exc:
        print(f"[FAIL] phase4 route validation failed: {exc}")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="route_validation_failed")
        return 1
    except Exception as exc:
        print(f"[FAIL] unable to load phase4 route script: {exc}")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="route_load_failed")
        return 1

    prereqs: List[CheckResult] = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_phase4_demo_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_file_exists_nonempty(args.phase4_start_state, "phase4_start_state"),
        check_file_exists_nonempty(args.phase4_route_script, "phase4_route_script"),
    ]
    if all(item.passed for item in prereqs):
        prereqs.append(check_nav_state_ready(args.rom_path, args.phase4_start_state, "phase4_start_state_check"))
    else:
        prereqs.append(
            CheckResult(
                "phase4_start_state_check",
                False,
                "Skipped: dependency/ROM/state prerequisites failed.",
            )
        )

    for item in prereqs:
        print_result(item)
    if not all(item.passed for item in prereqs):
        print("\nPhase4 demo live run blocked by failed preflight checks.")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="preflight_failed")
        return 1

    print(
        "[PHASE4_DEMO] live_run "
        f"name={route_preview.get('name', 'unknown')} "
        f"steps={len(route_preview.get('steps', []))} "
        f"target={args.phase4_target} "
        f"scope={args.phase4_scope} "
        "policy_mode=heuristic "
        f"wild_run_enabled={bool(args.phase4_wild_run_enabled)} "
        f"wild_battle_mode={args.phase4_wild_battle_mode} "
        f"farm_hp_threshold={float(args.phase4_farm_hp_threshold):.2f} "
        f"farm_max_consecutive_fights={int(args.phase4_farm_max_consecutive_fights)}"
    )

    results_path = Path(args.phase4_results_path).expanduser().resolve()
    timeline_path = Path(args.phase4_timeline_path).expanduser().resolve()
    timelapse_path = Path(args.phase4_timelapse_path).expanduser().resolve()
    fallback_results_path = Path(args.phase4_demo_fallback_results_path).expanduser().resolve()
    fallback_timeline_path = Path(args.phase4_demo_fallback_timeline_path).expanduser().resolve()

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip() or "phase4_demo_heuristic_key"
    from pokemon.emulator import PokemonEmulator

    emu = PokemonEmulator(
        args.rom_path,
        args.phase4_start_state,
        window=args.window,
        emulation_speed=args.emulation_speed,
    )
    agent = MistralBattleAgent(
        api_key=api_key,
        model=args.model,
        policy_mode="heuristic",
    )

    try:
        results = run_phase4_route(
            emu=emu,
            agent=agent,
            start_state_path=args.phase4_start_state,
            route_script_path=args.phase4_route_script,
            timeline_path=timeline_path,
            max_steps=args.phase4_max_steps,
            policy_mode="heuristic",
            target=args.phase4_target,
            phase4_scope=str(args.phase4_scope),
            wild_run_enabled=bool(args.phase4_wild_run_enabled),
            wild_battle_mode=str(args.phase4_wild_battle_mode),
            farm_hp_threshold=float(args.phase4_farm_hp_threshold),
            farm_max_consecutive_fights=int(args.phase4_farm_max_consecutive_fights),
            llm_turn_interval=args.llm_turn_interval,
            max_decision_calls=args.max_decision_calls,
            turn_tick_budget=args.turn_tick_budget,
        )
    except Exception as exc:
        print(f"[PHASE4_DEMO] live run failed: {exc}")
        if bool(args.phase4_demo_allow_fallback):
            return _attempt_phase4_demo_fallback(args, reason="live_exception")
        return 1
    finally:
        emu.stop()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_phase4_timelapse(results, timeline_path, timelapse_path)
    _print_phase4_summary("[PHASE4_DEMO]", results, results_path, timeline_path)
    print(f"[PHASE4_DEMO] timelapse={format_repo_relative(timelapse_path)}")
    _log_movement_campaign_run(
        args.campaign_log_path,
        phase="phase4",
        source="phase4_demo",
        movement_steps=_safe_int(results.get("steps_executed", 0)),
        metadata={
            "phase": "phase4",
            "source_mode": "demo",
            "target": str(args.phase4_target),
            "scope": str(args.phase4_scope),
            "policy_mode": "heuristic",
            "run_status": str(results.get("run_status", "failed")),
            "target_reached": bool(results.get("target_reached", False)),
            "steps_executed": _safe_int(results.get("steps_executed", 0)),
            "battles_fought": _safe_int(results.get("battles_fought", 0)),
            "wild_battles": _safe_int(results.get("wild_battles", 0)),
            "trainer_battles": _safe_int(results.get("trainer_battles", 0)),
            "wild_run_attempts": _safe_int(results.get("wild_run_attempts", 0)),
            "wild_run_successes": _safe_int(results.get("wild_run_successes", 0)),
            "results_path": format_repo_relative(results_path),
            "timeline_path": format_repo_relative(timeline_path),
            "timelapse_path": format_repo_relative(timelapse_path),
            "failure_reason": str(results.get("failure_reason", "")).strip(),
        },
    )

    if _phase4_run_succeeded(results):
        _persist_phase4_last_good(
            source_results_path=results_path,
            source_timeline_path=timeline_path,
            target_results_path=fallback_results_path,
            target_timeline_path=fallback_timeline_path,
        )
        print(
            "[PHASE4_DEMO] last_good_updated "
            f"results={format_repo_relative(fallback_results_path)} "
            f"timeline={format_repo_relative(fallback_timeline_path)}"
        )
        return 0

    if bool(args.phase4_demo_allow_fallback):
        return _attempt_phase4_demo_fallback(args, reason="live_run_unsuccessful")
    return 1


def run_phase4(args: argparse.Namespace) -> int:
    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    if args.phase4_max_steps < 1:
        print("[FAIL] phase4_max_steps must be >= 1.")
        return 1
    if args.phase4_policy_mode not in {"hybrid", "heuristic"}:
        print("[FAIL] phase4_policy_mode must be one of: hybrid, heuristic.")
        return 1
    if str(args.phase4_wild_battle_mode) not in {"run_first", "hp_gated_farm"}:
        print("[FAIL] phase4_wild_battle_mode must be one of: run_first, hp_gated_farm.")
        return 1
    if not (0.0 <= float(args.phase4_farm_hp_threshold) <= 1.0):
        print("[FAIL] phase4_farm_hp_threshold must be between 0 and 1.")
        return 1
    if int(args.phase4_farm_max_consecutive_fights) < 1:
        print("[FAIL] phase4_farm_max_consecutive_fights must be >= 1.")
        return 1
    if int(args.phase4_forest_probe_steps) < 1:
        print("[FAIL] phase4_forest_probe_steps must be >= 1.")
        return 1

    route_script_path = Path(args.phase4_route_script)
    probe_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if bool(args.phase4_forest_probe):
        probe_tmpdir = tempfile.TemporaryDirectory(prefix="phase4_forest_probe_")
        route_script_path = Path(probe_tmpdir.name) / "route.json"
        probe_payload = _build_phase4_forest_probe_route_payload(
            int(args.phase4_forest_probe_steps)
        )
        route_script_path.write_text(json.dumps(probe_payload, indent=2), encoding="utf-8")

    try:
        route_preview = load_route_script(route_script_path)
    except RouteValidationError as exc:
        print(f"[FAIL] phase4 route validation failed: {exc}")
        if probe_tmpdir is not None:
            probe_tmpdir.cleanup()
        return 1
    except Exception as exc:
        print(f"[FAIL] unable to load phase4 route script: {exc}")
        if probe_tmpdir is not None:
            probe_tmpdir.cleanup()
        return 1

    prereqs: List[CheckResult] = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_file_exists_nonempty(args.phase4_start_state, "phase4_start_state"),
        check_file_exists_nonempty(route_script_path, "phase4_route_script"),
        check_nav_state_ready(args.rom_path, args.phase4_start_state, "phase4_start_state_check"),
    ]
    if args.phase4_policy_mode == "hybrid":
        prereqs.append(check_mistral_api_key_present())

    for item in prereqs:
        print_result(item)
    if not all(item.passed for item in prereqs):
        print("\nPhase4 start blocked by failed preflight checks.")
        if probe_tmpdir is not None:
            probe_tmpdir.cleanup()
        return 1

    print(
        "[PHASE4] route "
        f"name={route_preview.get('name', 'unknown')} "
        f"steps={len(route_preview.get('steps', []))} "
        f"target={args.phase4_target} "
        f"scope={args.phase4_scope} "
        f"policy_mode={args.phase4_policy_mode} "
        f"wild_run_enabled={bool(args.phase4_wild_run_enabled)} "
        f"wild_battle_mode={args.phase4_wild_battle_mode} "
        f"farm_hp_threshold={float(args.phase4_farm_hp_threshold):.2f} "
        f"farm_max_consecutive_fights={int(args.phase4_farm_max_consecutive_fights)}"
    )
    if bool(args.phase4_forest_probe):
        print(
            "[PHASE4] forest_probe "
            f"enabled=true probe_steps={int(args.phase4_forest_probe_steps)} "
            f"profile={format_repo_relative(Path(args.phase4_forest_profile_path).expanduser().resolve())}"
        )

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        api_key = "phase4_heuristic_key"
    from pokemon.emulator import PokemonEmulator

    emu = PokemonEmulator(
        args.rom_path,
        args.phase4_start_state,
        window=args.window,
        emulation_speed=args.emulation_speed,
    )
    agent = MistralBattleAgent(
        api_key=api_key,
        model=args.model,
        policy_mode=args.phase4_policy_mode,
    )

    try:
        results = run_phase4_route(
            emu=emu,
            agent=agent,
            start_state_path=args.phase4_start_state,
            route_script_path=route_script_path,
            timeline_path=args.phase4_timeline_path,
            max_steps=args.phase4_max_steps,
            policy_mode=args.phase4_policy_mode,
            target=args.phase4_target,
            phase4_scope=str(args.phase4_scope),
            wild_run_enabled=bool(args.phase4_wild_run_enabled),
            wild_battle_mode=str(args.phase4_wild_battle_mode),
            farm_hp_threshold=float(args.phase4_farm_hp_threshold),
            farm_max_consecutive_fights=int(args.phase4_farm_max_consecutive_fights),
            llm_turn_interval=args.llm_turn_interval,
            max_decision_calls=args.max_decision_calls,
            turn_tick_budget=args.turn_tick_budget,
        )
    finally:
        emu.stop()
        if probe_tmpdir is not None:
            probe_tmpdir.cleanup()

    results_path = Path(args.phase4_results_path).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    timeline_path = Path(args.phase4_timeline_path).expanduser().resolve()
    timelapse_path = Path(args.phase4_timelapse_path).expanduser().resolve()
    _write_phase4_timelapse(results, timeline_path, timelapse_path)
    forest_profile_path = Path(args.phase4_forest_profile_path).expanduser().resolve()
    _write_phase4_forest_profile(results, forest_profile_path)
    run_status = str(results.get("run_status", "failed"))
    target_reached = bool(results.get("target_reached", False))
    print(
        "[PHASE4] complete "
        f"status={run_status} target_reached={target_reached} "
        f"results={format_repo_relative(results_path)} "
        f"timeline={format_repo_relative(timeline_path)}"
    )
    print(f"[PHASE4] timelapse={format_repo_relative(timelapse_path)}")
    print(f"[PHASE4] forest_profile={format_repo_relative(forest_profile_path)}")
    failure_reason = str(results.get("failure_reason", "")).strip()
    if failure_reason:
        print(f"[PHASE4] failure_reason={failure_reason}")
    _log_movement_campaign_run(
        args.campaign_log_path,
        phase="phase4",
        source="phase4",
        movement_steps=_safe_int(results.get("steps_executed", 0)),
        metadata={
            "phase": "phase4",
            "source_mode": "standard",
            "target": str(args.phase4_target),
            "scope": str(args.phase4_scope),
            "policy_mode": str(args.phase4_policy_mode),
            "run_status": run_status,
            "target_reached": bool(target_reached),
            "steps_executed": _safe_int(results.get("steps_executed", 0)),
            "battles_fought": _safe_int(results.get("battles_fought", 0)),
            "wild_battles": _safe_int(results.get("wild_battles", 0)),
            "trainer_battles": _safe_int(results.get("trainer_battles", 0)),
            "wild_run_attempts": _safe_int(results.get("wild_run_attempts", 0)),
            "wild_run_successes": _safe_int(results.get("wild_run_successes", 0)),
            "results_path": format_repo_relative(results_path),
            "timeline_path": format_repo_relative(timeline_path),
            "timelapse_path": format_repo_relative(timelapse_path),
            "forest_profile_path": format_repo_relative(forest_profile_path),
            "failure_reason": failure_reason,
        },
    )

    return 0 if run_status == "success" else 1


def _write_phase5_early_timeline(
    timeline_path: Path,
    *,
    target: str,
    failure_reason: str,
    strength_gate: Dict[str, object],
) -> None:
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(),
        "event": "run_terminated",
        "run_status": "failed",
        "target": str(target),
        "target_reached": False,
        "failure_reason": str(failure_reason),
        "strength_gate": strength_gate,
    }
    timeline_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def run_phase5_strength_probe(args: argparse.Namespace) -> Dict[str, object]:
    from pokemon.emulator import PokemonEmulator

    mode = str(args.phase5_strength_check).strip().lower()
    min_level = max(1, int(args.phase5_strength_min_level))
    sample_runs = max(1, int(args.phase5_strength_runs))
    required_pass_rate = float(args.phase5_strength_pass_rate)

    output: Dict[str, object] = {
        "mode": mode,
        "min_level": min_level,
        "sample_runs": sample_runs,
        "required_pass_rate": required_pass_rate,
        "sample_size": 0,
        "observed_pass_rate": 0.0,
        "avg_turns": 0.0,
        "active_level": 0,
        "active_species_id": 0,
        "party_species_ids": [],
        "meets_level_gate": False,
        "meets_empirical_gate": mode == "off",
        "passed": mode == "off",
    }

    emu = PokemonEmulator(
        args.rom_path,
        args.phase5_start_state,
        window="null",
        emulation_speed=0,
    )
    try:
        emu.reset(state_path=args.phase5_start_state)
        snapshot: Dict[str, object] = {}
        if hasattr(emu, "get_party_snapshot"):
            raw = emu.get_party_snapshot()
            if isinstance(raw, dict):
                snapshot = dict(raw)
        active_level = int(snapshot.get("active_level", 0))
        active_species_id = int(snapshot.get("active_species_id", 0))
        party_species_ids = [int(v) for v in snapshot.get("party_species_ids", [])]
        meets_level_gate = active_level >= min_level

        output["active_level"] = active_level
        output["active_species_id"] = active_species_id
        output["party_species_ids"] = party_species_ids
        output["meets_level_gate"] = bool(meets_level_gate)
        if mode == "off":
            return output

        wins = 0
        turn_samples: List[int] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for run_idx in range(sample_runs):
                probe_timeline = tmpdir_path / f"probe_{run_idx + 1}.jsonl"
                probe_agent = MistralBattleAgent(
                    api_key="phase5_strength_probe_key",
                    model=args.model,
                    policy_mode="heuristic",
                )
                run_out = run_phase4_route(
                    emu=emu,
                    agent=probe_agent,
                    start_state_path=args.phase5_start_state,
                    route_script_path=args.phase5_route_script,
                    timeline_path=probe_timeline,
                    max_steps=min(int(args.phase5_max_steps), 6000),
                    policy_mode="heuristic",
                    target=args.phase5_target,
                    phase4_scope="integrated",
                    wild_run_enabled=True,
                    llm_turn_interval=max(1, int(args.llm_turn_interval)),
                    max_decision_calls=0,
                    turn_tick_budget=max(1, int(args.turn_tick_budget)),
                    required_species_id=int(args.phase5_required_species),
                    single_pokemon_only=bool(args.phase5_single_pokemon_only),
                    enforce_party_constraint=bool(args.phase5_single_pokemon_only),
                )
                success = (
                    str(run_out.get("run_status", "")).strip().lower() == "success"
                    and bool(run_out.get("target_reached", False))
                )
                if success:
                    wins += 1
                battle_turns_total = int(run_out.get("battle_turns_total", 0))
                if battle_turns_total > 0:
                    turn_samples.append(battle_turns_total)

        observed_pass_rate = round(float(wins) / float(sample_runs), 4)
        avg_turns = round(float(sum(turn_samples)) / float(len(turn_samples)), 4) if turn_samples else 0.0
        meets_empirical = observed_pass_rate >= required_pass_rate
        output["sample_size"] = int(sample_runs)
        output["observed_pass_rate"] = observed_pass_rate
        output["avg_turns"] = avg_turns
        output["meets_empirical_gate"] = bool(meets_empirical)
        output["passed"] = bool(meets_level_gate and meets_empirical)
        return output
    finally:
        emu.stop()


def run_phase5(args: argparse.Namespace) -> int:
    loaded = load_env_file(ENV_FILE_PATH)
    if loaded:
        print(f"[INFO] Loaded {len(loaded)} env var(s) from {ENV_FILE_PATH.name}.")

    if args.phase5_max_steps < 1:
        print("[FAIL] phase5_max_steps must be >= 1.")
        return 1
    if args.phase5_policy_mode not in {"hybrid", "heuristic"}:
        print("[FAIL] phase5_policy_mode must be one of: hybrid, heuristic.")
        return 1
    if args.phase5_strength_runs < 1:
        print("[FAIL] phase5_strength_runs must be >= 1.")
        return 1
    if args.phase5_strength_min_level < 1:
        print("[FAIL] phase5_strength_min_level must be >= 1.")
        return 1
    if not (0.0 <= float(args.phase5_strength_pass_rate) <= 1.0):
        print("[FAIL] phase5_strength_pass_rate must be between 0 and 1.")
        return 1

    try:
        route_preview = load_route_script(args.phase5_route_script)
    except RouteValidationError as exc:
        print(f"[FAIL] phase5 route validation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] unable to load phase5 route script: {exc}")
        return 1

    prereqs: List[CheckResult] = [
        check_python_version(allow_newer_python=args.allow_newer_python),
        check_dependencies(),
        check_file_exists_nonempty(args.rom_path, "rom_file"),
        check_file_exists_nonempty(args.phase5_start_state, "phase5_start_state"),
        check_file_exists_nonempty(args.phase5_route_script, "phase5_route_script"),
        check_nav_state_ready(args.rom_path, args.phase5_start_state, "phase5_start_state_check"),
    ]
    if args.phase5_policy_mode == "hybrid":
        prereqs.append(check_mistral_api_key_present())

    for item in prereqs:
        print_result(item)
    if not all(item.passed for item in prereqs):
        print("\nPhase5 start blocked by failed preflight checks.")
        return 1

    print(
        "[PHASE5] route "
        f"name={route_preview.get('name', 'unknown')} "
        f"steps={len(route_preview.get('steps', []))} "
        f"target={args.phase5_target} "
        f"policy_mode={args.phase5_policy_mode} "
        f"required_species={int(args.phase5_required_species)} "
        f"single_pokemon_only={bool(args.phase5_single_pokemon_only)}"
    )

    strength_gate = run_phase5_strength_probe(args)
    print(
        "[PHASE5] strength_gate "
        f"mode={strength_gate.get('mode', 'unknown')} "
        f"active_level={strength_gate.get('active_level', 0)} "
        f"meets_level={strength_gate.get('meets_level_gate', False)} "
        f"pass_rate={strength_gate.get('observed_pass_rate', 0.0)} "
        f"required={strength_gate.get('required_pass_rate', 0.0)} "
        f"meets_empirical={strength_gate.get('meets_empirical_gate', False)} "
        f"passed={strength_gate.get('passed', False)}"
    )

    results_path = Path(args.phase5_results_path).expanduser().resolve()
    timeline_path = Path(args.phase5_timeline_path).expanduser().resolve()
    mode = str(args.phase5_strength_check).strip().lower()
    if mode == "strict" and not bool(strength_gate.get("passed", False)):
        results_path.parent.mkdir(parents=True, exist_ok=True)
        failure_payload: Dict[str, object] = {
            "run_status": "failed",
            "target": str(args.phase5_target),
            "target_reached": False,
            "start_state": str(args.phase5_start_state),
            "route_script": str(args.phase5_route_script),
            "required_species_id": int(args.phase5_required_species),
            "party_constraint_passed": False,
            "active_species_id_final": int(strength_gate.get("active_species_id", 0)),
            "active_level_final": int(strength_gate.get("active_level", 0)),
            "party_species_ids_final": [int(v) for v in strength_gate.get("party_species_ids", [])],
            "strength_gate": strength_gate,
            "failure_reason": "strength_gate_failed",
        }
        results_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
        _write_phase5_early_timeline(
            timeline_path,
            target=args.phase5_target,
            failure_reason="strength_gate_failed",
            strength_gate=strength_gate,
        )
        print(
            "[PHASE5] complete "
            "status=failed target_reached=False "
            f"results={format_repo_relative(results_path)} "
            f"timeline={format_repo_relative(timeline_path)}"
        )
        print("[PHASE5] failure_reason=strength_gate_failed")
        return 1

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        api_key = "phase5_heuristic_key"
    from pokemon.emulator import PokemonEmulator

    emu = PokemonEmulator(
        args.rom_path,
        args.phase5_start_state,
        window=args.window,
        emulation_speed=args.emulation_speed,
    )
    agent = MistralBattleAgent(
        api_key=api_key,
        model=args.model,
        policy_mode=args.phase5_policy_mode,
    )

    try:
        results = run_phase4_route(
            emu=emu,
            agent=agent,
            start_state_path=args.phase5_start_state,
            route_script_path=args.phase5_route_script,
            timeline_path=timeline_path,
            max_steps=args.phase5_max_steps,
            policy_mode=args.phase5_policy_mode,
            target=args.phase5_target,
            phase4_scope="integrated",
            wild_run_enabled=True,
            llm_turn_interval=args.llm_turn_interval,
            max_decision_calls=args.max_decision_calls,
            turn_tick_budget=args.turn_tick_budget,
            required_species_id=int(args.phase5_required_species),
            single_pokemon_only=bool(args.phase5_single_pokemon_only),
            enforce_party_constraint=bool(args.phase5_single_pokemon_only),
        )
    finally:
        emu.stop()

    results["required_species_id"] = int(args.phase5_required_species)
    results["strength_gate"] = strength_gate
    strength_warning = mode == "warn" and not bool(strength_gate.get("passed", False))
    results["strength_gate_warning"] = bool(strength_warning)
    if strength_warning:
        results["strength_gate_warning_reason"] = "strength_gate_failed"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    run_status = str(results.get("run_status", "failed"))
    target_reached = bool(results.get("target_reached", False))
    print(
        "[PHASE5] complete "
        f"status={run_status} target_reached={target_reached} "
        f"results={format_repo_relative(results_path)} "
        f"timeline={format_repo_relative(timeline_path)}"
    )
    if strength_warning:
        print("[PHASE5] warning=strength_gate_failed_continue")
    failure_reason = str(results.get("failure_reason", "")).strip()
    if failure_reason:
        print(f"[PHASE5] failure_reason={failure_reason}")

    return 0 if run_status == "success" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokemon Red RL runner and setup checks.")
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Run Phase 0 environment preflight checks.",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=ROM_PATH,
        help=f"Path to Pokemon Red ROM (default: {ROM_PATH}).",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=BATTLE_STATE_PATH,
        help=f"Path to battle save state (default: {BATTLE_STATE_PATH}).",
    )
    parser.add_argument(
        "--state-paths",
        type=str,
        default="",
        help="Comma-separated .state paths for Phase 1 episode rotation.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory containing .state files for Phase 1 rotation.",
    )
    parser.add_argument(
        "--state-rotation",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
        help="How to choose state per episode when multiple states are provided.",
    )
    parser.add_argument(
        "--api-ping",
        action="store_true",
        help="Validate MISTRAL_API_KEY by making a live API call.",
    )
    parser.add_argument(
        "--allow-newer-python",
        action="store_true",
        help="Allow Python versions newer than 3.11.x (default is strict 3.11.x).",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1 30-episode battle loop.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run Phase 2 richer-state battle loop.",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Run Phase 3 navigation PPO loop (reward_v1 -> reward_v2).",
    )
    parser.add_argument(
        "--phase4",
        action="store_true",
        help="Run Phase 4 route executor (Route1 -> Viridian -> Route2 -> Pewter).",
    )
    parser.add_argument(
        "--phase5",
        action="store_true",
        help="Run Phase 5 Brock-clear executor from before_brock state.",
    )
    parser.add_argument(
        "--phase4-demo",
        action="store_true",
        help=(
            "Run Phase 4 demo mode: force heuristic battle policy, run live if possible, "
            "and fallback to cached artifacts when needed."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Mistral model for decisions/reflection (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default=None,
        choices=["llm", "heuristic", "hybrid"],
        help=(
            "Decision policy for turn actions. "
            "Defaults: phase1=llm, phase2=hybrid."
        ),
    )
    parser.add_argument(
        "--llm-turn-interval",
        type=int,
        default=DEFAULT_LLM_TURN_INTERVAL,
        help=(
            "In hybrid mode, call LLM on turns where turn %% interval == 1 "
            f"(default: {DEFAULT_LLM_TURN_INTERVAL})."
        ),
    )
    parser.add_argument(
        "--max-decision-calls",
        type=int,
        default=DEFAULT_MAX_DECISION_CALLS,
        help=f"Hard run-level cap on LLM decision calls (default: {DEFAULT_MAX_DECISION_CALLS}).",
    )
    parser.add_argument(
        "--max-reflection-calls",
        type=int,
        default=DEFAULT_MAX_REFLECTION_CALLS,
        help=f"Hard run-level cap on LLM reflection calls (default: {DEFAULT_MAX_REFLECTION_CALLS}).",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable Phase 2 battle-memory layer (enabled by default for phase2).",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=DEFAULT_MEMORY_PATH,
        help=f"Path to live battle memory JSON (default: {DEFAULT_MEMORY_PATH}).",
    )
    parser.add_argument(
        "--memory-snapshot-keep",
        type=int,
        default=DEFAULT_MEMORY_SNAPSHOT_KEEP,
        help=(
            "How many versioned memory snapshots to keep under memory/history "
            f"(default: {DEFAULT_MEMORY_SNAPSHOT_KEEP})."
        ),
    )
    parser.add_argument(
        "--phase2b-baseline-results",
        type=Path,
        default=None,
        help="Optional baseline battle_*.json to compare Phase 2B state-turn improvement.",
    )
    parser.add_argument(
        "--phase2b-target-state-index",
        type=int,
        default=DEFAULT_PHASE2B_TARGET_STATE_INDEX,
        help=(
            "State index used by Phase 2B turn-improvement gate "
            f"(default: {DEFAULT_PHASE2B_TARGET_STATE_INDEX})."
        ),
    )
    parser.add_argument(
        "--phase2b-min-turn-improvement",
        type=float,
        default=DEFAULT_PHASE2B_MIN_TURN_IMPROVEMENT,
        help=(
            "Minimum avg-turn improvement required vs baseline for Phase 2B gate "
            f"(default: {DEFAULT_PHASE2B_MIN_TURN_IMPROVEMENT})."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of battle episodes (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=DEFAULT_UPDATE_EVERY,
        help=f"Strategy update cadence (default: {DEFAULT_UPDATE_EVERY}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns per episode before timeout (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--battle-wait-ticks",
        type=int,
        default=DEFAULT_BATTLE_WAIT_TICKS,
        help=f"Ticks to wait for battle start (default: {DEFAULT_BATTLE_WAIT_TICKS}).",
    )
    parser.add_argument(
        "--battle-search-steps",
        type=int,
        default=DEFAULT_BATTLE_SEARCH_STEPS,
        help=f"Overworld movement steps to trigger battle before timeout (default: {DEFAULT_BATTLE_SEARCH_STEPS}).",
    )
    parser.add_argument(
        "--turn-tick-budget",
        type=int,
        default=DEFAULT_TURN_TICK_BUDGET,
        help=f"Ticks to wait for battle end after each move (default: {DEFAULT_TURN_TICK_BUDGET}).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("artifacts/results/battles/battle.json"),
        help=(
            "Directory + filename prefix anchor for results. "
            "Example: artifacts/results/battles/battle.json -> battle_1.json, battle_2.json."
        ),
    )
    parser.add_argument(
        "--campaign-log-path",
        type=Path,
        default=DEFAULT_CAMPAIGN_LOG_PATH,
        help=f"Path to cumulative campaign log JSON (default: {DEFAULT_CAMPAIGN_LOG_PATH}).",
    )
    parser.add_argument(
        "--campaign-report",
        action="store_true",
        help="Print cumulative campaign log totals and exit.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="null",
        choices=["null", "SDL2"],
        help="Emulator window backend (default: null). Use SDL2 for live visualization.",
    )
    parser.add_argument(
        "--emulation-speed",
        type=int,
        default=0,
        help="PyBoy emulation speed (0=uncapped, 1=real-time, >1 faster).",
    )
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=0.0,
        help="Probability of overriding LLM action with random legal move (default: 0.0).",
    )
    parser.add_argument(
        "--battle-screenshots",
        type=parse_bool_arg,
        default=DEFAULT_BATTLE_SCREENSHOTS_ENABLED,
        help=(
            "Enable battle screenshot capture for Phase 1/2 "
            f"(default: {DEFAULT_BATTLE_SCREENSHOTS_ENABLED})."
        ),
    )
    parser.add_argument(
        "--battle-screenshots-dir",
        type=Path,
        default=DEFAULT_BATTLE_SCREENSHOTS_DIR,
        help=(
            "Output directory for battle screenshot artifacts "
            f"(default: {DEFAULT_BATTLE_SCREENSHOTS_DIR})."
        ),
    )
    parser.add_argument(
        "--battle-screenshot-retain-battles",
        type=int,
        default=DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES,
        help=(
            "How many recent battle screenshot folders to keep "
            f"(default: {DEFAULT_BATTLE_SCREENSHOT_RETAIN_BATTLES})."
        ),
    )
    parser.add_argument(
        "--nav-state-path",
        type=Path,
        default=DEFAULT_NAV_STATE_PATH,
        help=f"Phase 3 train start-state path (default: {DEFAULT_NAV_STATE_PATH}).",
    )
    parser.add_argument(
        "--phase3-eval-state-paths",
        type=str,
        default="assets/states/explore_eval_1.state,assets/states/explore_eval_2.state",
        help="Comma-separated Phase 3 eval state paths (start-state is always included).",
    )
    parser.add_argument(
        "--phase3-train-minutes",
        type=int,
        default=25,
        help="Wall-clock PPO training minutes per reward version (default: 25).",
    )
    parser.add_argument(
        "--phase3-eval-episodes",
        type=int,
        default=3,
        help="Evaluation episodes per state and reward version (default: 3).",
    )
    parser.add_argument(
        "--phase3-no-progress-limit",
        type=int,
        default=60,
        help="Truncate nav episodes after this many no-progress steps (default: 60).",
    )
    parser.add_argument(
        "--phase3-seed",
        type=int,
        default=7,
        help="Random seed used for PPO training/evaluation (default: 7).",
    )
    parser.add_argument(
        "--phase3-guidance-profile",
        type=str,
        default="kanto_early",
        choices=["none", "kanto_early"],
        help="Soft movement guidance profile for navigation reward shaping.",
    )
    parser.add_argument(
        "--phase3-guidance-weight",
        type=float,
        default=0.4,
        help="Multiplier for guidance_reward in reward_fn (default: 0.4).",
    )
    parser.add_argument(
        "--phase3-action-set",
        type=str,
        default="dpad",
        choices=["dpad", "full"],
        help="Phase 3 action space: dpad-only (recommended) or full buttons.",
    )
    parser.add_argument(
        "--phase3-guidance-json",
        type=Path,
        default=None,
        help="Optional JSON profile overlay for guidance (adjacency/weights).",
    )
    parser.add_argument(
        "--phase3-reward-v1",
        type=Path,
        default=Path("reward_v1.py"),
        help="Path to reward_v1.py module (must expose reward_fn).",
    )
    parser.add_argument(
        "--phase3-reward-v2",
        type=Path,
        default=Path("reward_v2.py"),
        help="Path to reward_v2.py module (must expose reward_fn).",
    )
    parser.add_argument(
        "--phase3-output-path",
        type=Path,
        default=DEFAULT_PHASE3_OUTPUT_PATH,
        help=f"Phase 3 results JSON path (default: {DEFAULT_PHASE3_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--phase3-summary-path",
        type=Path,
        default=DEFAULT_PHASE3_SUMMARY_PATH,
        help=f"Phase 3 markdown summary path (default: {DEFAULT_PHASE3_SUMMARY_PATH}).",
    )
    parser.add_argument(
        "--phase4-start-state",
        type=Path,
        default=DEFAULT_PHASE4_START_STATE_PATH,
        help=f"Phase 4 start-state path (default: {DEFAULT_PHASE4_START_STATE_PATH}).",
    )
    parser.add_argument(
        "--phase4-route-script",
        type=Path,
        default=DEFAULT_PHASE4_ROUTE_SCRIPT_PATH,
        help=f"Phase 4 route JSON path (default: {DEFAULT_PHASE4_ROUTE_SCRIPT_PATH}).",
    )
    parser.add_argument(
        "--phase4-max-steps",
        type=int,
        default=15000,
        help="Max route runtime steps before timeout (default: 15000).",
    )
    parser.add_argument(
        "--phase4-policy-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "heuristic"],
        help="Phase 4 battle policy mode (default: hybrid).",
    )
    parser.add_argument(
        "--phase4-scope",
        type=str,
        default="route_only",
        choices=["route_only", "integrated"],
        help="Phase 4 execution scope: route-only 4A or integrated 4B behavior.",
    )
    parser.add_argument(
        "--phase4-results-path",
        type=Path,
        default=DEFAULT_PHASE4_RESULTS_PATH,
        help=f"Phase 4 results JSON path (default: {DEFAULT_PHASE4_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--phase4-timeline-path",
        type=Path,
        default=DEFAULT_PHASE4_TIMELINE_PATH,
        help=f"Phase 4 timeline JSONL path (default: {DEFAULT_PHASE4_TIMELINE_PATH}).",
    )
    parser.add_argument(
        "--phase4-timelapse-path",
        type=Path,
        default=DEFAULT_PHASE4_TIMELAPSE_PATH,
        help=f"Phase 4 timelapse text path (default: {DEFAULT_PHASE4_TIMELAPSE_PATH}).",
    )
    parser.add_argument(
        "--phase4-forest-profile-path",
        type=Path,
        default=DEFAULT_PHASE4_FOREST_PROFILE_PATH,
        help=(
            "Phase 4 forest transition profile JSON output path "
            f"(default: {DEFAULT_PHASE4_FOREST_PROFILE_PATH})."
        ),
    )
    parser.add_argument(
        "--phase4-forest-probe",
        type=parse_bool_arg,
        default=False,
        help=(
            "Run a focused forest transition probe route instead of the full phase4 route "
            "(default: false)."
        ),
    )
    parser.add_argument(
        "--phase4-forest-probe-steps",
        type=int,
        default=DEFAULT_PHASE4_FOREST_PROBE_STEPS,
        help=(
            "Max traverse steps for forest probe route generation "
            f"(default: {DEFAULT_PHASE4_FOREST_PROBE_STEPS})."
        ),
    )
    parser.add_argument(
        "--phase4-target",
        type=str,
        default="gym_entrance",
        choices=["gym_entrance", "brock_badge"],
        help="Phase 4 success target (default: gym_entrance).",
    )
    parser.add_argument(
        "--phase4-wild-run-enabled",
        type=parse_bool_arg,
        default=True,
        help="Attempt RUN first in wild battles (default: true).",
    )
    parser.add_argument(
        "--phase4-wild-battle-mode",
        type=str,
        default=DEFAULT_PHASE4_WILD_BATTLE_MODE,
        choices=["run_first", "hp_gated_farm"],
        help=(
            "Phase 4 wild-battle policy mode. "
            f"(default: {DEFAULT_PHASE4_WILD_BATTLE_MODE})."
        ),
    )
    parser.add_argument(
        "--phase4-farm-hp-threshold",
        type=float,
        default=DEFAULT_PHASE4_FARM_HP_THRESHOLD,
        help=(
            "In hp_gated_farm mode, run when HP ratio falls below this threshold "
            f"(default: {DEFAULT_PHASE4_FARM_HP_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--phase4-farm-max-consecutive-fights",
        type=int,
        default=DEFAULT_PHASE4_FARM_MAX_CONSECUTIVE_FIGHTS,
        help=(
            "In hp_gated_farm mode, force RUN once this many wild fights happen in a row "
            f"(default: {DEFAULT_PHASE4_FARM_MAX_CONSECUTIVE_FIGHTS})."
        ),
    )
    parser.add_argument(
        "--phase5-start-state",
        type=Path,
        default=DEFAULT_PHASE5_START_STATE_PATH,
        help=f"Phase 5 start-state path (default: {DEFAULT_PHASE5_START_STATE_PATH}).",
    )
    parser.add_argument(
        "--phase5-route-script",
        type=Path,
        default=DEFAULT_PHASE5_ROUTE_SCRIPT_PATH,
        help=f"Phase 5 route JSON path (default: {DEFAULT_PHASE5_ROUTE_SCRIPT_PATH}).",
    )
    parser.add_argument(
        "--phase5-max-steps",
        type=int,
        default=12000,
        help="Max Phase 5 route runtime steps before timeout (default: 12000).",
    )
    parser.add_argument(
        "--phase5-policy-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "heuristic"],
        help="Phase 5 battle policy mode (default: hybrid).",
    )
    parser.add_argument(
        "--phase5-target",
        type=str,
        default="brock_badge",
        choices=["brock_badge"],
        help="Phase 5 success target (fixed: brock_badge).",
    )
    parser.add_argument(
        "--phase5-required-species",
        type=int,
        default=DEFAULT_PHASE5_REQUIRED_SPECIES,
        help=(
            "Required active species id for party-constraint checks "
            f"(default: {DEFAULT_PHASE5_REQUIRED_SPECIES})."
        ),
    )
    parser.add_argument(
        "--phase5-single-pokemon-only",
        type=parse_bool_arg,
        default=True,
        help="Require solo party during Phase 5 route execution (default: true).",
    )
    parser.add_argument(
        "--phase5-strength-min-level",
        type=int,
        default=DEFAULT_PHASE5_STRENGTH_MIN_LEVEL,
        help=(
            "Minimum active level for strength gate "
            f"(default: {DEFAULT_PHASE5_STRENGTH_MIN_LEVEL})."
        ),
    )
    parser.add_argument(
        "--phase5-strength-runs",
        type=int,
        default=DEFAULT_PHASE5_STRENGTH_RUNS,
        help=f"Number of probe runs for strength gate (default: {DEFAULT_PHASE5_STRENGTH_RUNS}).",
    )
    parser.add_argument(
        "--phase5-strength-pass-rate",
        type=float,
        default=DEFAULT_PHASE5_STRENGTH_PASS_RATE,
        help=(
            "Required probe win rate for strength gate "
            f"(default: {DEFAULT_PHASE5_STRENGTH_PASS_RATE})."
        ),
    )
    parser.add_argument(
        "--phase5-strength-check",
        type=str,
        default=DEFAULT_PHASE5_STRENGTH_CHECK,
        choices=["off", "warn", "strict"],
        help=(
            "Strength gate behavior: off=skip, warn=continue with warning, "
            f"strict=block when gate fails (default: {DEFAULT_PHASE5_STRENGTH_CHECK})."
        ),
    )
    parser.add_argument(
        "--phase5-results-path",
        type=Path,
        default=DEFAULT_PHASE5_RESULTS_PATH,
        help=f"Phase 5 results JSON path (default: {DEFAULT_PHASE5_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--phase5-timeline-path",
        type=Path,
        default=DEFAULT_PHASE5_TIMELINE_PATH,
        help=f"Phase 5 timeline JSONL path (default: {DEFAULT_PHASE5_TIMELINE_PATH}).",
    )
    parser.add_argument(
        "--phase4-demo-fallback-results-path",
        type=Path,
        default=DEFAULT_PHASE4_LAST_GOOD_RESULTS_PATH,
        help=(
            "Path to last known-good Phase 4 results JSON for demo fallback "
            f"(default: {DEFAULT_PHASE4_LAST_GOOD_RESULTS_PATH})."
        ),
    )
    parser.add_argument(
        "--phase4-demo-fallback-timeline-path",
        type=Path,
        default=DEFAULT_PHASE4_LAST_GOOD_TIMELINE_PATH,
        help=(
            "Path to last known-good Phase 4 timeline JSONL for demo fallback "
            f"(default: {DEFAULT_PHASE4_LAST_GOOD_TIMELINE_PATH})."
        ),
    )
    parser.add_argument(
        "--phase4-demo-allow-fallback",
        type=parse_bool_arg,
        default=True,
        help="Allow demo mode to present fallback artifacts if live run is blocked (default: true).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selected_modes = [
        bool(args.dry_check),
        bool(args.phase1),
        bool(args.phase2),
        bool(args.phase3),
        bool(args.phase4),
        bool(args.phase5),
        bool(args.phase4_demo),
    ]
    if sum(selected_modes) > 1:
        print(
            "[FAIL] choose only one of --dry-check, --phase1, --phase2, "
            "--phase3, --phase4, --phase5, or --phase4-demo."
        )
        return 1

    if args.dry_check:
        return run_dry_check(args)
    if args.campaign_report:
        print_campaign_report(args.campaign_log_path)
        return 0
    if args.phase3:
        return run_phase3(args)
    if args.phase4_demo:
        return run_phase4_demo(args)
    if args.phase4:
        return run_phase4(args)
    if args.phase5:
        return run_phase5(args)
    if args.phase2:
        return run_phase2(args)
    if args.phase1:
        return run_phase1(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
