from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from PIL import Image
except Exception:  # pragma: no cover - handled by preflight and runtime warnings
    Image = None


BATTLE_DIR_RE = re.compile(r"^battle_(\d+)_ep(\d+)_(.+)$")


@dataclass
class BattleScreenshotConfig:
    enabled: bool
    root_dir: Path
    retain_battles: int
    phase: str


class BattleScreenshotRecorder:
    def __init__(self, config: BattleScreenshotConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self.root_dir = Path(config.root_dir)
        self.retain_battles = max(1, int(config.retain_battles))
        self.phase = str(config.phase).strip().lower() or "phase"
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._active = False
        self._battle_id: int | None = None
        self._battle_dir: Path | None = None
        self._episode = 0
        self._state_path = ""
        self._state_index = -1
        self._policy_mode = ""
        self._model = ""
        self._started_at = ""
        self._ended_at = ""
        self._frames: List[Dict[str, Any]] = []
        self._turn_frame_seq = 1

    def start_episode(
        self,
        episode: int,
        state_path: Path,
        state_index: int,
        policy_mode: str,
        model: str,
    ) -> None:
        self._reset_episode_state()
        self._episode = int(episode)
        self._state_path = str(state_path)
        self._state_index = int(state_index)
        self._policy_mode = str(policy_mode)
        self._model = str(model)

    def capture_event(
        self,
        event: str,
        emu: object,
        turn: int | None = None,
        chosen_slot: int | None = None,
        outcome: str | None = None,
        state: Dict[str, Any] | None = None,
    ) -> None:
        del outcome  # Reserved for future filenames/metadata variants.
        if not self.enabled:
            return
        label = str(event).strip().lower()
        if label not in {"battle_start", "turn_post_action", "battle_end"}:
            return
        if label == "battle_start":
            self._ensure_episode_dir()
        elif not self._active:
            return
        if self._battle_dir is None:
            return

        try:
            image = self._extract_image(emu)
            if image is None:
                raise RuntimeError("Pillow image is not available")
            if label == "battle_start":
                frame_prefix = 0
            elif label == "battle_end":
                frame_prefix = 999
            else:
                frame_prefix = self._turn_frame_seq
                self._turn_frame_seq += 1
            filename = self._build_frame_filename(
                event=label,
                frame_prefix=frame_prefix,
                turn=turn,
            )
            file_path = self._battle_dir / filename
            image.save(file_path, format="PNG")
            snapshot = self._extract_state_snapshot(emu, state)
            self._frames.append(
                {
                    "seq": int(frame_prefix),
                    "event": label,
                    "turn": int(turn) if turn is not None else None,
                    "chosen_slot": int(chosen_slot) if chosen_slot is not None else None,
                    "in_battle": int(snapshot.get("in_battle", 0)),
                    "player_hp": int(snapshot.get("player_hp", 0)),
                    "enemy_hp": int(snapshot.get("enemy_hp", 0)),
                    "path": filename,
                }
            )
        except Exception as exc:
            print(f"[WARN] screenshot capture failed ({label}): {exc}")

    def finish_episode(self, outcome: str, turns: int, hp_left: int, reward: float) -> None:
        if not self.enabled:
            self._reset_episode_state()
            return
        if not self._active or self._battle_dir is None or self._battle_id is None:
            self._reset_episode_state()
            return
        self._ended_at = _iso_utc_now()
        manifest = {
            "battle_id": int(self._battle_id),
            "phase": self.phase,
            "episode": int(self._episode),
            "state_path": self._state_path,
            "state_index": int(self._state_index),
            "policy_mode": self._policy_mode,
            "model": self._model,
            "outcome": str(outcome),
            "turns": int(turns),
            "hp_left": int(hp_left),
            "reward": float(reward),
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "frames": list(self._frames),
        }
        try:
            (self._battle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] screenshot manifest write failed: {exc}")

        self.prune_old_battles()
        self.rebuild_index()
        self._reset_episode_state()

    def prune_old_battles(self) -> None:
        if not self.enabled or not self.root_dir.exists():
            return
        ranked = self._list_battle_dirs_sorted()
        if len(ranked) <= self.retain_battles:
            return
        to_remove = ranked[: len(ranked) - self.retain_battles]
        for battle_id, folder in to_remove:
            try:
                shutil.rmtree(folder)
            except Exception as exc:
                print(f"[WARN] failed to prune screenshot battle {battle_id}: {exc}")

    def rebuild_index(self) -> None:
        if not self.enabled:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for battle_id, folder in self._list_battle_dirs_sorted():
            manifest_path = folder / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] failed reading screenshot manifest for index ({folder.name}): {exc}")
                continue
            summary = {
                "battle_id": int(payload.get("battle_id", battle_id)),
                "phase": str(payload.get("phase", self.phase)),
                "episode": int(payload.get("episode", 0)),
                "outcome": str(payload.get("outcome", "")),
                "turns": int(payload.get("turns", 0)),
                "frame_count": len(payload.get("frames", [])),
                "battle_dir": folder.name,
                "manifest_path": str(manifest_path),
            }
            lines.append(json.dumps(summary))
        (self.root_dir / "index.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    def _ensure_episode_dir(self) -> None:
        if self._active:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        phase_slug = re.sub(r"[^a-z0-9_-]+", "_", self.phase).strip("_") or "phase"
        battle_id = self._next_battle_id()
        while True:
            folder_name = f"battle_{battle_id:06d}_ep{self._episode:03d}_{phase_slug}"
            battle_dir = self.root_dir / folder_name
            if not battle_dir.exists():
                break
            battle_id += 1
        battle_dir.mkdir(parents=True, exist_ok=False)
        self._battle_id = int(battle_id)
        self._battle_dir = battle_dir
        self._started_at = _iso_utc_now()
        self._active = True

    def _next_battle_id(self) -> int:
        max_seen = 0
        for battle_id, _ in self._list_battle_dirs_sorted():
            if battle_id > max_seen:
                max_seen = battle_id
        return max_seen + 1

    def _list_battle_dirs_sorted(self) -> List[tuple[int, Path]]:
        if not self.root_dir.exists():
            return []
        items: List[tuple[int, Path]] = []
        for child in self.root_dir.iterdir():
            if not child.is_dir():
                continue
            match = BATTLE_DIR_RE.match(child.name)
            if not match:
                continue
            items.append((int(match.group(1)), child))
        items.sort(key=lambda row: row[0])
        return items

    def _build_frame_filename(self, event: str, frame_prefix: int, turn: int | None) -> str:
        if event == "battle_start":
            suffix = "battle_start"
        elif event == "battle_end":
            suffix = "battle_end"
        else:
            suffix_turn = int(turn) if turn is not None else 0
            suffix = f"turn_{suffix_turn:03d}_post_action"
        return f"{int(frame_prefix):03d}_{suffix}.png"

    def _extract_image(self, emu: object):
        if Image is None:
            return None
        pyboy = getattr(emu, "pyboy", None)
        if pyboy is None:
            raise RuntimeError("emulator has no pyboy attribute")
        screen = getattr(pyboy, "screen", None)
        if screen is None:
            raise RuntimeError("emulator pyboy has no screen")
        image = getattr(screen, "image", None)
        if image is not None and hasattr(image, "copy"):
            return image.copy()
        ndarray = getattr(screen, "ndarray", None)
        if ndarray is None:
            raise RuntimeError("screen has neither image nor ndarray")
        return Image.fromarray(ndarray, mode="RGBA")

    def _extract_state_snapshot(self, emu: object, state: Dict[str, Any] | None) -> Dict[str, int]:
        if isinstance(state, dict):
            in_battle = int(state.get("in_battle", 0))
            player_hp = int(state.get("player_hp", state.get("hp", 0)))
            enemy_hp = int(state.get("enemy_hp", 0))
            return {
                "in_battle": in_battle,
                "player_hp": player_hp,
                "enemy_hp": enemy_hp,
            }
        try:
            raw = emu.get_battle_state()
        except Exception:
            raw = {}
        return {
            "in_battle": int(raw.get("in_battle", 0)),
            "player_hp": int(raw.get("player_hp", raw.get("hp", 0))),
            "enemy_hp": int(raw.get("enemy_hp", 0)),
        }


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
