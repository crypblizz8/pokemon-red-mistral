#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.configs import (  # noqa: E402
    DEFAULT_BATTLE_SEARCH_STEPS,
    DEFAULT_BATTLE_WAIT_TICKS,
    DEFAULT_CAMPAIGN_LOG_PATH,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STATE_PATHS,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_TURN_TICK_BUDGET,
    DEFAULT_UPDATE_EVERY,
    MIN_STATE_FILE_BYTES,
    NO_CLEAR_WINNER_GAP,
    ROM_PATH,
)
from evals.metrics import flatten_model_summary_for_csv, rank_models, summarize_rows  # noqa: E402
from pokemon.campaign_log import append_campaign_log_entry, campaign_log_report  # noqa: E402


@dataclass(frozen=True)
class Fold:
    fold_id: str
    heldout_state: Path
    train_states: Tuple[Path, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def format_repo_relative(path: Path, *, root: Path = PROJECT_ROOT) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def parse_csv_values(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_models(raw: str) -> List[str]:
    models = parse_csv_values(raw)
    if not models:
        raise ValueError("No models provided. Pass --models with at least one model id.")
    seen: set[str] = set()
    deduped: List[str] = []
    for model in models:
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(model)
    return deduped


def resolve_state_paths(raw: str) -> List[Path]:
    raw_paths = parse_csv_values(raw)
    if not raw_paths:
        raise ValueError("No states provided. Pass --state-paths with one or more .state paths.")
    deduped: List[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        resolved = str(Path(raw_path).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(Path(resolved))
    return deduped


def validate_state_files(state_paths: Sequence[Path]) -> None:
    if len(state_paths) < 3:
        raise ValueError(
            f"LOSO requires at least 3 states; received {len(state_paths)}. "
            "Add more paths to --state-paths."
        )

    for path in state_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing state file: {path}")
        size = path.stat().st_size
        if size < MIN_STATE_FILE_BYTES:
            raise ValueError(f"State file is empty: {path}")


def fold_label(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(letters):
        return letters[index]
    return f"F{index + 1}"


def build_loso_folds(state_paths: Sequence[Path]) -> List[Fold]:
    folds: List[Fold] = []
    for idx, heldout in enumerate(state_paths):
        train_states = tuple(path for path in state_paths if path != heldout)
        folds.append(
            Fold(
                fold_id=fold_label(idx),
                heldout_state=heldout,
                train_states=train_states,
            )
        )
    return folds


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


def run_single_episode(
    emu,
    agent,
    state_path: Path,
    state_index: int,
    episode_number: int,
    max_turns: int,
    turn_tick_budget: int,
    battle_wait_ticks: int,
    battle_search_steps: int,
) -> Tuple[Dict[str, object], str]:
    emu.reset(state_path=state_path)
    starting_state = emu.get_battle_state()
    likely_intro_state = (
        int(starting_state.get("player_hp", 0)) == 0
        and int(starting_state.get("player_level", 0)) == 0
        and int(starting_state.get("player_species", 0)) == 0
    )
    resolved_state_path = str(state_path.resolve())

    if likely_intro_state and not emu.in_battle():
        record = agent.record_battle(
            episode=episode_number,
            move_slots=[],
            outcome="invalid_state",
            hp_left=0,
            turns=1,
            llm_replies=[
                "Loaded save-state looks like intro/title flow (no active player stats). "
                "Create a new state in overworld or in-battle."
            ],
            state_path=resolved_state_path,
            state_index=state_index,
        )
        return record.to_dict(), "invalid_state"

    if not emu.wait_for_battle(timeout=120):
        emu.seek_battle(max_steps=battle_search_steps)

    if not emu.wait_for_battle(timeout=battle_wait_ticks):
        record = agent.record_battle(
            episode=episode_number,
            move_slots=[],
            outcome="timeout",
            hp_left=0,
            turns=1,
            llm_replies=["Battle did not start within timeout."],
            state_path=resolved_state_path,
            state_index=state_index,
        )
        return record.to_dict(), "timeout"

    move_slots: List[int] = []
    llm_replies: List[str] = []
    turns_taken = 0

    for turn_idx in range(max_turns):
        if not emu.in_battle():
            break

        state = emu.get_battle_state()
        state["turn"] = turn_idx + 1
        slot = agent.pick_move(state)
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
        episode=episode_number,
        move_slots=move_slots,
        outcome=outcome,
        hp_left=player_hp,
        turns=max(turns_taken, 1),
        llm_replies=llm_replies,
        state_path=resolved_state_path,
        state_index=state_index,
    )
    info = f"outcome={record.outcome} turns={record.turns} hp={record.hp_left} reward={record.reward:.1f}"
    return record.to_dict(), info


def create_episode_row(
    base_row: Dict[str, object],
    *,
    run_id: str,
    model: str,
    fold_id: str,
    heldout_state: Path,
    split: str,
) -> Dict[str, object]:
    row = dict(base_row)
    row["run_id"] = run_id
    row["model"] = model
    row["fold_id"] = fold_id
    row["heldout_state"] = str(heldout_state.resolve())
    row["split"] = split
    return row


def ensure_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    header = [
        "model",
        "episodes",
        "wins",
        "win_rate",
        "win_rate_ci_low",
        "win_rate_ci_high",
        "timeout_rate",
        "invalid_state_rate",
        "median_turns",
        "mean_hp_left",
        "mean_reward",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(flatten_model_summary_for_csv(row))


def format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "0.00%"


def build_summary_markdown(
    run_id: str,
    run_dir: Path,
    ranking: Dict[str, object],
    model_summaries: Sequence[Dict[str, object]],
    fold_summaries: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# Phase 1 LOSO Model Eval Summary")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Output directory: `{run_dir}`")
    lines.append(f"- Decision: `{ranking.get('decision', 'unknown')}`")
    winner = ranking.get("winner_model")
    lines.append(f"- Winner: `{winner if winner else 'none'}`")
    lines.append(f"- Reason: {ranking.get('reason', 'n/a')}")
    lines.append("")
    lines.append("## Model Ranking")
    lines.append("")
    lines.append(
        "| Model | Wins/Eps | Win Rate | 95% CI | Timeout Rate | Invalid-State Rate | Median Turns | Mean HP Left | Mean Reward |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in model_summaries:
        ci_low = float(row.get("win_rate_ci_low", 0.0))
        ci_high = float(row.get("win_rate_ci_high", 0.0))
        lines.append(
            "| "
            f"{row.get('model', '')} | "
            f"{row.get('wins', 0)}/{row.get('episodes', 0)} | "
            f"{format_pct(row.get('win_rate', 0.0))} | "
            f"{ci_low:.2%} - {ci_high:.2%} | "
            f"{format_pct(row.get('timeout_rate', 0.0))} | "
            f"{format_pct(row.get('invalid_state_rate', 0.0))} | "
            f"{row.get('median_turns', 0)} | "
            f"{row.get('mean_hp_left', 0)} | "
            f"{row.get('mean_reward', 0)} |"
        )

    lines.append("")
    lines.append("## Fold Details (Eval Split Only)")
    lines.append("")
    lines.append("| Model | Fold | Heldout State | Wins/Eps | Win Rate | Timeout Rate | Invalid-State Rate |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in fold_summaries:
        lines.append(
            "| "
            f"{row.get('model', '')} | "
            f"{row.get('fold_id', '')} | "
            f"{Path(str(row.get('heldout_state', ''))).name} | "
            f"{row.get('wins', 0)}/{row.get('episodes', 0)} | "
            f"{format_pct(row.get('win_rate', 0.0))} | "
            f"{format_pct(row.get('timeout_rate', 0.0))} | "
            f"{format_pct(row.get('invalid_state_rate', 0.0))} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Phase 1 LOSO model evaluation harness.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODEL,
        help="Comma-separated model ids to evaluate.",
    )
    parser.add_argument(
        "--state-paths",
        type=str,
        default=",".join(str(path) for path in DEFAULT_STATE_PATHS),
        help="Comma-separated .state paths used for LOSO folds.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=DEFAULT_TRAIN_EPISODES,
        help=f"Training episodes per fold (default: {DEFAULT_TRAIN_EPISODES}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Eval episodes on heldout state per fold (default: {DEFAULT_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns per episode (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="null",
        choices=["null", "SDL2"],
        help="Emulator window backend.",
    )
    parser.add_argument(
        "--emulation-speed",
        type=int,
        default=0,
        help="PyBoy emulation speed (0=uncapped).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run id. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=ROM_PATH,
        help=f"Path to Pokemon Red ROM (default: {ROM_PATH}).",
    )
    parser.add_argument(
        "--update-every",
        type=int,
        default=DEFAULT_UPDATE_EVERY,
        help=f"Strategy update cadence during train split (default: {DEFAULT_UPDATE_EVERY}).",
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
        help=f"Overworld movement steps to seek battle (default: {DEFAULT_BATTLE_SEARCH_STEPS}).",
    )
    parser.add_argument(
        "--turn-tick-budget",
        type=int,
        default=DEFAULT_TURN_TICK_BUDGET,
        help=f"Ticks to wait for battle end after move (default: {DEFAULT_TURN_TICK_BUDGET}).",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Optional fold cap for quick smoke runs (0 = all folds).",
    )
    parser.add_argument(
        "--campaign-log-path",
        type=Path,
        default=DEFAULT_CAMPAIGN_LOG_PATH,
        help=f"Path to cumulative campaign log JSON (default: {DEFAULT_CAMPAIGN_LOG_PATH}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.train_episodes <= 0:
        print("train-episodes must be > 0")
        return 1
    if args.eval_episodes <= 0:
        print("eval-episodes must be > 0")
        return 1
    if args.max_turns <= 0:
        print("max-turns must be > 0")
        return 1
    if args.update_every <= 0:
        print("update-every must be > 0")
        return 1

    load_env_file(PROJECT_ROOT / ".env.local")
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        print("MISTRAL_API_KEY is missing. Set it in .env.local or shell environment.")
        return 1

    try:
        models = parse_models(args.models)
        state_paths = resolve_state_paths(args.state_paths)
        validate_state_files(state_paths)
    except Exception as exc:
        print(exc)
        return 1
    if not args.rom_path.exists():
        print(f"Missing ROM file: {args.rom_path.resolve()}")
        return 1

    folds = build_loso_folds(state_paths)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    if not folds:
        print("No folds to run.")
        return 1

    output_root = ensure_output_root(args.output_root)
    run_id = args.run_id.strip() or default_run_id()
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Run directory already exists: {run_dir}")
        print("Pass --run-id with a unique value or omit it for timestamp-based ids.")
        return 1

    from pokemon.battle_agent import MistralBattleAgent
    from pokemon.emulator import PokemonEmulator

    all_episode_rows: List[Dict[str, object]] = []
    fold_summaries: List[Dict[str, object]] = []
    model_summaries: List[Dict[str, object]] = []

    for model in models:
        print(f"\n[MODEL] {model}")
        model_eval_rows: List[Dict[str, object]] = []

        for fold in folds:
            print(
                f"[FOLD {fold.fold_id}] heldout={fold.heldout_state.name} "
                f"train={[path.name for path in fold.train_states]}"
            )
            emu = PokemonEmulator(
                args.rom_path,
                fold.train_states[0],
                window=args.window,
                emulation_speed=args.emulation_speed,
            )
            agent = MistralBattleAgent(api_key=api_key, model=model)
            fold_eval_rows: List[Dict[str, object]] = []

            try:
                for episode in range(1, args.train_episodes + 1):
                    state_index = (episode - 1) % len(fold.train_states)
                    state_path = fold.train_states[state_index]
                    row, info = run_single_episode(
                        emu=emu,
                        agent=agent,
                        state_path=state_path,
                        state_index=state_index,
                        episode_number=episode,
                        max_turns=args.max_turns,
                        turn_tick_budget=args.turn_tick_budget,
                        battle_wait_ticks=args.battle_wait_ticks,
                        battle_search_steps=args.battle_search_steps,
                    )
                    enriched = create_episode_row(
                        row,
                        run_id=run_id,
                        model=model,
                        fold_id=fold.fold_id,
                        heldout_state=fold.heldout_state,
                        split="train",
                    )
                    all_episode_rows.append(enriched)
                    print(f"  [TRAIN {episode}/{args.train_episodes}] {info}")

                    if episode % args.update_every == 0 and episode < args.train_episodes:
                        old_strategy = agent.strategy
                        new_strategy = agent.update_strategy()
                        changed = "yes" if old_strategy != new_strategy else "no"
                        print(
                            f"  [TRAIN] strategy updated at episode {episode} (changed={changed})"
                        )

                for episode in range(1, args.eval_episodes + 1):
                    row, info = run_single_episode(
                        emu=emu,
                        agent=agent,
                        state_path=fold.heldout_state,
                        state_index=0,
                        episode_number=episode,
                        max_turns=args.max_turns,
                        turn_tick_budget=args.turn_tick_budget,
                        battle_wait_ticks=args.battle_wait_ticks,
                        battle_search_steps=args.battle_search_steps,
                    )
                    enriched = create_episode_row(
                        row,
                        run_id=run_id,
                        model=model,
                        fold_id=fold.fold_id,
                        heldout_state=fold.heldout_state,
                        split="eval",
                    )
                    all_episode_rows.append(enriched)
                    model_eval_rows.append(enriched)
                    fold_eval_rows.append(enriched)
                    print(f"  [EVAL {episode}/{args.eval_episodes}] {info}")
            finally:
                emu.stop()

            fold_metrics = summarize_rows(fold_eval_rows)
            fold_summaries.append(
                {
                    "model": model,
                    "fold_id": fold.fold_id,
                    "heldout_state": str(fold.heldout_state.resolve()),
                    **fold_metrics,
                }
            )

        model_summary = summarize_rows(model_eval_rows)
        model_summary["model"] = model
        model_summaries.append(model_summary)

    ranking = rank_models(model_summaries, no_clear_winner_gap=NO_CLEAR_WINNER_GAP)
    ordered_model_summaries = list(ranking.get("ranking", []))

    payload = {
        "run_meta": {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "rom_path": str(args.rom_path.resolve()),
            "models": models,
            "state_paths": [str(path.resolve()) for path in state_paths],
            "fold_count": len(folds),
            "train_episodes_per_fold": args.train_episodes,
            "eval_episodes_per_fold": args.eval_episodes,
            "update_every": args.update_every,
            "max_turns": args.max_turns,
            "window": args.window,
            "emulation_speed": args.emulation_speed,
            "no_clear_winner_gap": NO_CLEAR_WINNER_GAP,
        },
        "ranking": ranking,
        "model_summaries": ordered_model_summaries,
        "fold_summaries": fold_summaries,
        "episodes": all_episode_rows,
    }

    results_path = run_dir / "eval_results.json"
    summary_md_path = run_dir / "eval_summary.md"
    summary_csv_path = run_dir / "eval_summary.csv"

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv_path, ordered_model_summaries)
    summary_md = build_summary_markdown(
        run_id=run_id,
        run_dir=run_dir,
        ranking=ranking,
        model_summaries=ordered_model_summaries,
        fold_summaries=fold_summaries,
    )
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print("\nEvaluation complete.")
    print(f"- results: {results_path}")
    print(f"- summary: {summary_md_path}")
    print(f"- csv:     {summary_csv_path}")
    print(f"- decision: {ranking.get('decision')}")
    print(f"- winner:   {ranking.get('winner_model')}")

    if all_episode_rows:
        split_summary = {
            "train": sum(1 for row in all_episode_rows if row.get("split") == "train"),
            "eval": sum(1 for row in all_episode_rows if row.get("split") == "eval"),
        }
        try:
            append_campaign_log_entry(
                args.campaign_log_path,
                kind="simulation",
                count=len(all_episode_rows),
                source="phase1_eval_loso",
                metadata={
                    "run_id": run_id,
                    "models": models,
                    "fold_count": len(folds),
                    "train_episodes_per_fold": args.train_episodes,
                    "eval_episodes_per_fold": args.eval_episodes,
                    "split_counts": split_summary,
                    "output_dir": format_repo_relative(run_dir),
                },
            )
            report = campaign_log_report(args.campaign_log_path)
            print(
                "[CAMPAIGN] "
                f"real_battles={report['real_battles']} "
                f"simulations={report['simulations']} "
                f"combined_total={report['combined']}"
            )
        except Exception as exc:
            print(f"[WARN] campaign log update failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
