from __future__ import annotations

import unittest
import sys
import types

from configs import (
    PARTY_MON_LEVEL_OFFSET,
    PARTY_MON_STRUCT_SIZE,
    RAM_ADDR_PARTY_COUNT,
    RAM_ADDR_PARTY_MON_START,
    RAM_ADDR_PARTY_SPECIES_START,
    RAM_ADDR_PLAYER_LEVEL,
    RAM_ADDR_PLAYER_SPECIES,
)

if "pyboy" not in sys.modules:
    pyboy_stub = types.ModuleType("pyboy")

    class _DummyPyBoy:  # pragma: no cover - import shim only
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    pyboy_stub.PyBoy = _DummyPyBoy
    sys.modules["pyboy"] = pyboy_stub

if "pyboy.utils" not in sys.modules:
    utils_stub = types.ModuleType("pyboy.utils")

    class _WindowEvent:  # pragma: no cover - import shim only
        PRESS_BUTTON_A = 0
        RELEASE_BUTTON_A = 1
        PRESS_BUTTON_B = 2
        RELEASE_BUTTON_B = 3
        PRESS_BUTTON_START = 4
        RELEASE_BUTTON_START = 5
        PRESS_BUTTON_SELECT = 6
        RELEASE_BUTTON_SELECT = 7
        PRESS_ARROW_UP = 8
        RELEASE_ARROW_UP = 9
        PRESS_ARROW_DOWN = 10
        RELEASE_ARROW_DOWN = 11
        PRESS_ARROW_LEFT = 12
        RELEASE_ARROW_LEFT = 13
        PRESS_ARROW_RIGHT = 14
        RELEASE_ARROW_RIGHT = 15

    utils_stub.WindowEvent = _WindowEvent
    sys.modules["pyboy.utils"] = utils_stub

from pokemon.emulator import PokemonEmulator


class EmulatorPartyHelpersTests(unittest.TestCase):
    def _stub_emulator(self, memory: dict[int, int]) -> PokemonEmulator:
        emu = object.__new__(PokemonEmulator)
        emu.read = lambda addr: int(memory.get(int(addr), 0))  # type: ignore[method-assign]
        return emu

    def test_get_party_snapshot_single_mon(self) -> None:
        memory = {
            RAM_ADDR_PARTY_COUNT: 1,
            RAM_ADDR_PARTY_SPECIES_START: 176,
            RAM_ADDR_PLAYER_SPECIES: 176,
            RAM_ADDR_PLAYER_LEVEL: 16,
            RAM_ADDR_PARTY_MON_START + PARTY_MON_LEVEL_OFFSET: 16,
        }
        emu = self._stub_emulator(memory)

        snap = emu.get_party_snapshot()

        self.assertEqual(snap["party_count"], 1)
        self.assertEqual(snap["party_species_ids"], [176])
        self.assertEqual(snap["party_levels"], [16])
        self.assertEqual(snap["active_species_id"], 176)
        self.assertEqual(snap["active_level"], 16)

    def test_validate_single_species_passes_for_solo_charmander(self) -> None:
        memory = {
            RAM_ADDR_PARTY_COUNT: 1,
            RAM_ADDR_PARTY_SPECIES_START: 176,
            RAM_ADDR_PLAYER_SPECIES: 176,
            RAM_ADDR_PLAYER_LEVEL: 18,
            RAM_ADDR_PARTY_MON_START + PARTY_MON_LEVEL_OFFSET: 18,
        }
        emu = self._stub_emulator(memory)

        passed, reason, snap = emu.validate_single_species(176)

        self.assertTrue(passed)
        self.assertEqual(reason, "ok")
        self.assertEqual(snap["party_species_ids"], [176])

    def test_validate_single_species_fails_when_party_has_multiple(self) -> None:
        memory = {
            RAM_ADDR_PARTY_COUNT: 2,
            RAM_ADDR_PARTY_SPECIES_START + 0: 176,
            RAM_ADDR_PARTY_SPECIES_START + 1: 165,
            RAM_ADDR_PLAYER_SPECIES: 176,
            RAM_ADDR_PLAYER_LEVEL: 16,
            RAM_ADDR_PARTY_MON_START + PARTY_MON_LEVEL_OFFSET: 16,
            RAM_ADDR_PARTY_MON_START + PARTY_MON_STRUCT_SIZE + PARTY_MON_LEVEL_OFFSET: 12,
        }
        emu = self._stub_emulator(memory)

        passed, reason, _ = emu.validate_single_species(176)

        self.assertFalse(passed)
        self.assertEqual(reason, "party_size_2")

    def test_validate_single_species_fails_on_wrong_active_species(self) -> None:
        memory = {
            RAM_ADDR_PARTY_COUNT: 1,
            RAM_ADDR_PARTY_SPECIES_START: 176,
            RAM_ADDR_PLAYER_SPECIES: 177,
            RAM_ADDR_PLAYER_LEVEL: 16,
            RAM_ADDR_PARTY_MON_START + PARTY_MON_LEVEL_OFFSET: 16,
        }
        emu = self._stub_emulator(memory)

        passed, reason, _ = emu.validate_single_species(176)

        self.assertFalse(passed)
        self.assertIn("active_species_177_expected_176", reason)


if __name__ == "__main__":
    unittest.main()
