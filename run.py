#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from configs import (
    BATTLE_STATE_PATH,
    PROJECT_ROOT,
    DEFAULT_CAMPAIGN_LOG_PATH,
    DEFAULT_BATTLE_SEARCH_STEPS,
    DEFAULT_BATTLE_WAIT_TICKS,
    DEFAULT_EPISODES,
    DEFAULT_LLM_TURN_INTERVAL,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_DECISION_CALLS,
    DEFAULT_MAX_REFLECTION_CALLS,
    DEFAULT_MODEL,
    DEFAULT_TURN_TICK_BUDGET,
    DEFAULT_UPDATE_EVERY,
    ENV_FILE_PATH,
    EXPECTED_PYTHON_MAJOR,
    EXPECTED_PYTHON_MINOR,
    MIN_NONEMPTY_FILE_BYTES,
    RAM_ADDR_PLAYER_HP,
    RAM_ADDR_X_POS,
    ROM_PATH,
)
from pokemon.battle_agent import MistralBattleAgent
from pokemon.campaign_log import append_campaign_log_entry, campaign_log_report
from pokemon.emulator import PokemonEmulator


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
    emu: PokemonEmulator,
    agent: MistralBattleAgent,
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
        return record.to_dict(), "Battle start timeout"

    move_slots: List[int] = []
    llm_replies: List[str] = []

    turns_taken = 0
    for turn_idx in range(max_turns):
        if not emu.in_battle():
            break

        turn_number = turn_idx + 1
        if phase2:
            state = emu.build_phase2_state(turn=turn_number)
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

        if emu.wait_for_battle_end(timeout=turn_tick_budget):
            break

    final_state = emu.get_battle_state()
    player_hp = int(final_state.get("player_hp", 0))
    outcome = "timeout"
    if not emu.in_battle():
        outcome = "win" if player_hp > 0 else "loss"

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
    info = (
        f"episode={episode} state={state_index} outcome={outcome} "
        f"turns={record.turns} hp={player_hp} reward={record.reward:.1f}"
    )
    return record.to_dict(), info


def save_battle_results(path: Path, model: str, agent: MistralBattleAgent, phase: str) -> None:
    phase1_metrics = compute_phase1_metrics(agent)
    phase2_metrics = compute_phase2_metrics(agent)
    payload = {
        "model": model,
        "phase": phase,
        "summary": agent.summary(),
        "phase1_metrics": phase1_metrics,
        "phase2_metrics": phase2_metrics,
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
    print(f"- log_entries: {report['entries']}")
    if report["updated_at"]:
        print(f"- updated_at: {report['updated_at']}")


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

    state_paths = resolve_phase1_state_paths(args)
    if not state_paths:
        print("[FAIL] battle_state: no state files resolved.")
        print("Use --state-path, --state-paths, or --state-dir with .state files.")
        return 1
    if phase == "phase2" and len(state_paths) < 4:
        print("[FAIL] phase2 requires at least 4 state files for mixed evaluation.")
        print("Provide 3+ wild states plus at least 1 trainer/gym-adjacent battle state.")
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
    )
    completed_rows: List[Dict[str, object]] = []

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
            save_battle_results(results_path, args.model, agent, phase=phase)

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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_check:
        return run_dry_check(args)
    if args.campaign_report:
        print_campaign_report(args.campaign_log_path)
        return 0
    if args.phase1 and args.phase2:
        print("[FAIL] choose only one of --phase1 or --phase2.")
        return 1
    if args.phase2:
        return run_phase2(args)
    if args.phase1:
        return run_phase1(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
