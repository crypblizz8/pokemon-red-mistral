from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pyboy import PyBoy
from pyboy.utils import WindowEvent

from configs import (
    RAM_ADDR_BADGES,
    DEFAULT_BATTLE_SEARCH_STEPS,
    DEFAULT_BATTLE_WAIT_TICKS,
    DEFAULT_BUTTON_HOLD_FRAMES,
    DEFAULT_POST_INPUT_TICKS,
    RAM_ADDR_ENEMY_HP,
    RAM_ADDR_ENEMY_LEVEL,
    RAM_ADDR_ENEMY_SPECIES,
    RAM_ADDR_IN_BATTLE,
    RAM_ADDR_PLAYER_HP,
    RAM_ADDR_PLAYER_LEVEL,
    RAM_ADDR_PLAYER_MAX_HP,
    RAM_ADDR_PLAYER_MOVE_1,
    RAM_ADDR_PLAYER_MOVE_2,
    RAM_ADDR_PLAYER_MOVE_3,
    RAM_ADDR_PLAYER_MOVE_4,
    RAM_ADDR_PLAYER_MOVE_PP_1,
    RAM_ADDR_PLAYER_MOVE_PP_2,
    RAM_ADDR_PLAYER_MOVE_PP_3,
    RAM_ADDR_PLAYER_MOVE_PP_4,
    RAM_ADDR_PLAYER_SPECIES,
    RAM_ADDR_PARTY_COUNT,
    RAM_ADDR_PARTY_SPECIES_START,
    RAM_ADDR_PARTY_MON_START,
    PARTY_MON_STRUCT_SIZE,
    PARTY_MON_LEVEL_OFFSET,
    MAX_PARTY_SIZE,
    RAM_ADDR_MAP_ID,
    RAM_ADDR_X_POS,
    RAM_ADDR_Y_POS,
)
from pokemon.gen1_data import effectiveness, move_meta, species_meta


BUTTON_EVENTS = {
    "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    "select": (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
}


class PokemonEmulator:
    def __init__(
        self,
        rom_path: Path,
        state_path: Path,
        window: str = "null",
        emulation_speed: int = 0,
    ) -> None:
        self.rom_path = Path(rom_path)
        self.state_path = Path(state_path)
        self.window = window
        self.emulation_speed = emulation_speed
        self.pyboy = self._create_pyboy()
        self.pp_source_valid = True
        self.pp_warning_emitted = False
        self._enemy_hp_peak = 1
        if hasattr(self.pyboy, "set_emulation_speed"):
            self.pyboy.set_emulation_speed(self.emulation_speed)

    def _create_pyboy(self) -> PyBoy:
        try:
            return PyBoy(str(self.rom_path), window=self.window)
        except TypeError:
            # Older PyBoy versions use window_type instead.
            return PyBoy(str(self.rom_path), window_type="headless")

    def _tick(self, frames: int = 1) -> None:
        for _ in range(max(frames, 1)):
            self.pyboy.tick()

    def tick(self, frames: int = 1) -> None:
        self._tick(frames=frames)

    def _read_byte(self, addr: int) -> int:
        if hasattr(self.pyboy, "get_memory_value"):
            return int(self.pyboy.get_memory_value(addr))
        return int(self.pyboy.memory[addr])

    def read(self, addr: int) -> int:
        return self._read_byte(addr)

    def _read_word_be(self, high_addr: int) -> int:
        high = self._read_byte(high_addr)
        low = self._read_byte(high_addr + 1)
        return (high << 8) | low

    def _player_hp_word(self) -> int:
        return self._read_word_be(RAM_ADDR_PLAYER_HP)

    def press(self, button: str, frames: int = DEFAULT_BUTTON_HOLD_FRAMES) -> None:
        key = button.lower().strip()
        if key not in BUTTON_EVENTS:
            raise ValueError(f"Unsupported button: {button}")
        press_evt, release_evt = BUTTON_EVENTS[key]
        self.pyboy.send_input(press_evt)
        self._tick(frames)
        self.pyboy.send_input(release_evt)
        self._tick(1)

    def reset(self, state_path: Path | None = None) -> None:
        target_state = Path(state_path) if state_path is not None else self.state_path
        with target_state.open("rb") as handle:
            self.pyboy.load_state(handle)
        self._tick(30)
        self.pp_source_valid = True
        self.pp_warning_emitted = False
        enemy_hp = self._read_word_be(RAM_ADDR_ENEMY_HP)
        self._enemy_hp_peak = max(1, enemy_hp)

    def in_battle(self) -> bool:
        # Gen1 battle flag is non-zero in battle; trainer battles commonly use value 2.
        return self.read(RAM_ADDR_IN_BATTLE) != 0

    def battle_flag(self) -> int:
        return int(self.read(RAM_ADDR_IN_BATTLE))

    def is_trainer_battle(self) -> bool:
        return self.battle_flag() == 2

    def is_wild_battle(self) -> bool:
        return self.in_battle() and not self.is_trainer_battle()

    def has_badge(self, badge_bit: int = 0) -> bool:
        bit = max(0, int(badge_bit))
        badges = int(self.read(RAM_ADDR_BADGES))
        return (badges & (1 << bit)) != 0

    def get_party_snapshot(self) -> Dict[str, object]:
        raw_count = int(self.read(RAM_ADDR_PARTY_COUNT))
        party_count = max(0, min(int(raw_count), int(MAX_PARTY_SIZE)))

        species_ids: List[int] = []
        levels: List[int] = []
        for idx in range(party_count):
            species_addr = int(RAM_ADDR_PARTY_SPECIES_START) + idx
            species_id = int(self.read(species_addr))
            if species_id in {0, 0xFF}:
                continue
            species_ids.append(species_id)
            level_addr = (
                int(RAM_ADDR_PARTY_MON_START)
                + (idx * int(PARTY_MON_STRUCT_SIZE))
                + int(PARTY_MON_LEVEL_OFFSET)
            )
            levels.append(max(0, int(self.read(level_addr))))

        active_species_id = int(self.read(RAM_ADDR_PLAYER_SPECIES))
        active_level = int(self.read(RAM_ADDR_PLAYER_LEVEL))

        # Fallback for malformed/mid-transition memory where party list is empty but lead data exists.
        if not species_ids and active_species_id > 0:
            species_ids = [active_species_id]
            levels = [max(0, active_level)]
            party_count = 1

        return {
            "party_species_ids": [int(v) for v in species_ids],
            "party_levels": [int(v) for v in levels],
            "active_species_id": int(active_species_id),
            "active_level": int(active_level),
            "party_count": int(party_count),
        }

    def validate_single_species(self, required_species_id: int) -> tuple[bool, str, Dict[str, object]]:
        required = int(required_species_id)
        snapshot = self.get_party_snapshot()
        party_count = int(snapshot.get("party_count", 0))
        party_species = [int(v) for v in snapshot.get("party_species_ids", [])]
        active_species_id = int(snapshot.get("active_species_id", 0))

        if party_count < 1:
            return False, "empty_party", snapshot
        if party_count != 1:
            return False, f"party_size_{party_count}", snapshot
        if active_species_id != required:
            return False, f"active_species_{active_species_id}_expected_{required}", snapshot
        if not party_species or party_species[0] != required:
            return False, f"party_species_mismatch_expected_{required}", snapshot
        return True, "ok", snapshot

    def _move_ids(self) -> List[int]:
        return [
            self.read(RAM_ADDR_PLAYER_MOVE_1),
            self.read(RAM_ADDR_PLAYER_MOVE_2),
            self.read(RAM_ADDR_PLAYER_MOVE_3),
            self.read(RAM_ADDR_PLAYER_MOVE_4),
        ]

    def _move_pps(self) -> List[int]:
        return [
            self.read(RAM_ADDR_PLAYER_MOVE_PP_1),
            self.read(RAM_ADDR_PLAYER_MOVE_PP_2),
            self.read(RAM_ADDR_PLAYER_MOVE_PP_3),
            self.read(RAM_ADDR_PLAYER_MOVE_PP_4),
        ]

    def _validate_pp_source(self) -> None:
        if not self.pp_source_valid:
            return
        for pp in self._move_pps():
            if pp < 0 or pp > 99:
                self.pp_source_valid = False
                if not self.pp_warning_emitted:
                    print(
                        "[WARN] PP RAM values out of expected range; "
                        "falling back to move_id-only legality."
                    )
                    self.pp_warning_emitted = True
                return

    def get_move_pps(self) -> List[int]:
        self._validate_pp_source()
        pps = self._move_pps()
        return [max(0, int(pp)) for pp in pps]

    def get_legal_move_slots(self) -> List[int]:
        move_ids = self._move_ids()
        if not self.pp_source_valid:
            return [idx for idx, move_id in enumerate(move_ids[:4]) if int(move_id) != 0]

        move_pps = self.get_move_pps()
        return [
            idx
            for idx, move_id in enumerate(move_ids[:4])
            if int(move_id) != 0 and int(move_pps[idx]) > 0
        ]

    def get_battle_state(self) -> Dict[str, int | List[int]]:
        return {
            "in_battle": int(self.in_battle()),
            "player_hp": self._player_hp_word(),
            "player_max_hp": self._read_word_be(RAM_ADDR_PLAYER_MAX_HP),
            "player_level": self.read(RAM_ADDR_PLAYER_LEVEL),
            "player_species": self.read(RAM_ADDR_PLAYER_SPECIES),
            "enemy_hp": self._read_word_be(RAM_ADDR_ENEMY_HP),
            "enemy_level": self.read(RAM_ADDR_ENEMY_LEVEL),
            "enemy_species": self.read(RAM_ADDR_ENEMY_SPECIES),
            "move_ids": self._move_ids(),
            "move_pps": self.get_move_pps(),
            "legal_slots": self.get_legal_move_slots(),
        }

    def get_nav_state(self) -> Dict[str, int]:
        return {
            "x": int(self.read(RAM_ADDR_X_POS)),
            "y": int(self.read(RAM_ADDR_Y_POS)),
            "map_id": int(self.read(RAM_ADDR_MAP_ID)),
            "badges": int(self.read(RAM_ADDR_BADGES)),
            "hp": int(self._player_hp_word()),
            "level": int(self.read(RAM_ADDR_PLAYER_LEVEL)),
            "in_battle": int(self.in_battle()),
        }

    def build_phase2_state(self, turn: int) -> Dict[str, object]:
        base = self.get_battle_state()
        player_hp = int(base.get("player_hp", 0))
        player_max_hp = max(1, int(base.get("player_max_hp", 1)))
        enemy_hp = max(0, int(base.get("enemy_hp", 0)))
        self._enemy_hp_peak = max(self._enemy_hp_peak, enemy_hp, 1)

        player_species_id = int(base.get("player_species", 0))
        enemy_species_id = int(base.get("enemy_species", 0))
        player_species = species_meta(player_species_id)
        enemy_species = species_meta(enemy_species_id)
        enemy_types = [str(t) for t in enemy_species.get("types", ["normal"])[:2]]

        move_ids = [int(v) for v in base.get("move_ids", [])[:4]]
        move_pps = [int(v) for v in base.get("move_pps", [])[:4]]
        moves: List[Dict[str, object]] = []
        legal_slots = self.get_legal_move_slots()

        for idx in range(4):
            move_id = move_ids[idx] if idx < len(move_ids) else 0
            pp_cur = move_pps[idx] if idx < len(move_pps) else 0
            meta = move_meta(move_id)
            move_type = str(meta.get("type", "normal"))
            eff = effectiveness(move_type, enemy_types)
            moves.append(
                {
                    "slot": idx,
                    "id": move_id,
                    "name": str(meta.get("name", f"move_{move_id}")),
                    "type": move_type,
                    "power": int(meta.get("power", 0)),
                    "pp_current": max(0, pp_cur),
                    "pp_max": int(meta.get("pp_max", 0)),
                    "effectiveness": float(eff),
                    "legal": idx in legal_slots,
                }
            )

        return {
            "turn": int(turn),
            "in_battle": int(base.get("in_battle", 0)),
            "player_hp": player_hp,
            "player_max_hp": player_max_hp,
            "player_hp_pct": round(100.0 * player_hp / player_max_hp, 2),
            "player_level": int(base.get("player_level", 0)),
            "player_species_id": player_species_id,
            "player_species_name": str(player_species.get("name", f"species_{player_species_id}")),
            "player_types": [str(t) for t in player_species.get("types", ["normal"])[:2]],
            "enemy_hp": enemy_hp,
            "enemy_hp_pct": round(100.0 * enemy_hp / max(1, self._enemy_hp_peak), 2),
            "enemy_level": int(base.get("enemy_level", 0)),
            "enemy_species_id": enemy_species_id,
            "enemy_species_name": str(enemy_species.get("name", f"species_{enemy_species_id}")),
            "enemy_types": enemy_types,
            "move_ids": move_ids,
            "move_pps": move_pps,
            "moves": moves,
            "legal_slots": legal_slots,
            "pp_source_valid": bool(self.pp_source_valid),
        }

    def execute_move(self, slot: int) -> bool:
        if not self.in_battle():
            return False

        slot = max(0, min(3, int(slot)))
        pp_before = self.get_move_pps()
        player_hp_before = self._player_hp_word()
        enemy_hp_before = self._read_word_be(RAM_ADDR_ENEMY_HP)

        for _ in range(2):
            self._normalize_to_fight_menu()
            self._choose_move_slot(slot)
            if self._action_committed(slot, pp_before, player_hp_before, enemy_hp_before):
                return True
        return False

    def _normalize_to_fight_menu(self) -> None:
        # Clear nested menus/dialog and bias cursor to top-left command option (FIGHT).
        self._normalize_to_command_menu()
        self.press("a")
        self._tick(DEFAULT_POST_INPUT_TICKS)

    def _normalize_to_command_menu(self) -> None:
        # Clear nested menus/dialog and anchor to top-left in the battle command menu.
        for _ in range(3):
            self.press("b")
            self._tick(max(1, DEFAULT_POST_INPUT_TICKS // 3))
        self.press("up", frames=2)
        self.press("left", frames=2)
        self._tick(2)

    def _choose_move_slot(self, slot: int) -> None:
        # Move cursor memory can persist across turns; anchor to top-left first.
        self.press("up", frames=2)
        self.press("left", frames=2)
        self._tick(2)
        if slot in (1, 3):
            self.press("right", frames=2)
        if slot in (2, 3):
            self.press("down", frames=2)
        self._tick(4)
        self.press("a")
        self._tick(DEFAULT_POST_INPUT_TICKS)

    def _action_committed(
        self,
        slot: int,
        pp_before: List[int],
        player_hp_before: int,
        enemy_hp_before: int,
        timeout: int = 180,
    ) -> bool:
        slot_pp_before = pp_before[slot] if 0 <= slot < len(pp_before) else 0
        for _ in range(timeout):
            if not self.in_battle():
                return True
            self._tick(1)

            player_hp_now = self._player_hp_word()
            enemy_hp_now = self._read_word_be(RAM_ADDR_ENEMY_HP)
            if player_hp_now != player_hp_before or enemy_hp_now != enemy_hp_before:
                return True

            if slot_pp_before > 0:
                pp_now = self.get_move_pps()
                if 0 <= slot < len(pp_now) and pp_now[slot] < slot_pp_before:
                    return True
        return False

    def wait_for_battle(self, timeout: int = DEFAULT_BATTLE_WAIT_TICKS) -> bool:
        for _ in range(timeout):
            if self.in_battle():
                return True
            self.pyboy.tick()
        return False

    def wait_for_battle_end(self, timeout: int = DEFAULT_BATTLE_WAIT_TICKS) -> bool:
        for _ in range(timeout):
            if not self.in_battle():
                return True
            self.pyboy.tick()
        return False

    def seek_battle(self, max_steps: int = DEFAULT_BATTLE_SEARCH_STEPS) -> bool:
        """Move around to trigger a wild/trainer battle from overworld states."""
        if self.in_battle():
            return True

        walk_pattern = ["up", "down", "left", "right"]
        for i in range(max_steps):
            self.press(walk_pattern[i % len(walk_pattern)], frames=6)
            self._tick(10)
            if self.in_battle():
                return True
        return False

    def attempt_run(self, timeout_ticks: int = 240) -> bool:
        """
        Attempt to flee from battle by selecting RUN.
        Returns True if battle exited, False otherwise.
        """
        if not self.in_battle():
            return True

        # Battle text boxes and command menus can interleave unpredictably.
        # This loop advances text and repeatedly re-selects RUN from command menu.
        total_ticks = max(80, int(timeout_ticks))
        cycle_budget = max(4, total_ticks // 20)
        for _ in range(cycle_budget):
            for button in ("a", "b", "up", "left", "down", "right", "a"):
                if not self.in_battle():
                    return True
                self.press(button, frames=2)
                self._tick(3)
            for _ in range(20):
                if not self.in_battle():
                    return True
                self._tick(1)
        return not self.in_battle()

    def stop(self) -> None:
        try:
            self.pyboy.stop(save=False)
        except TypeError:
            self.pyboy.stop()
