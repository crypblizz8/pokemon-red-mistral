from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pokemon.battle_memory import BattleMemory


class Phase2MemoryTests(unittest.TestCase):
    def test_memory_roundtrip_and_rule_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory" / "battle_memory.json"
            memory = BattleMemory(path=memory_path, snapshot_keep=10)
            state = {
                "state_label": "battle_first_trainer.state",
                "enemy_species_id": 153,
                "player_species_id": 4,
                "move_ids": [10, 45, 52, 0],
                "legal_slots": [0, 1, 2],
            }

            memory.start_episode()
            memory.record_turn(state, 0)
            memory.finalize_episode(outcome="loss", reward=-100.0, turns=10)

            memory.start_episode()
            memory.record_turn(state, 0)
            memory.finalize_episode(outcome="loss", reward=-90.0, turns=9)

            memory.start_episode()
            memory.record_turn(state, 1)
            memory.finalize_episode(outcome="win", reward=110.0, turns=4)

            self.assertTrue(memory_path.exists())
            reloaded = BattleMemory(path=memory_path, snapshot_keep=10)
            self.assertGreaterEqual(reloaded.rules_loaded, 1)
            rule = reloaded.lookup_rule(state)
            self.assertIsNotNone(rule)
            assert rule is not None
            self.assertIn(1, rule["preferred_slots"])
            self.assertIn(0, rule["blocked_slots"])

    def test_memory_snapshot_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory" / "battle_memory.json"
            memory = BattleMemory(path=memory_path, snapshot_keep=2)
            state = {
                "state_label": "battle_rattata.state",
                "enemy_species_id": 165,
                "player_species_id": 4,
                "move_ids": [10, 45, 52, 0],
                "legal_slots": [0, 1, 2],
            }

            for idx in range(4):
                memory.start_episode()
                memory.record_turn(state, idx % 2)
                memory.finalize_episode(outcome="loss", reward=-100.0 + idx, turns=10 - idx)

            history_dir = memory_path.parent / "history"
            snapshots = sorted(history_dir.glob("battle_memory_*.json"))
            self.assertLessEqual(len(snapshots), 2)

    def test_matchup_lesson_applies_without_state_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory" / "battle_memory.json"
            memory = BattleMemory(path=memory_path, snapshot_keep=10)

            state_a = {
                "state_label": "state_a.state",
                "enemy_species_id": 153,
                "player_species_id": 4,
                "move_ids": [10, 45, 52, 0],
                "legal_slots": [0, 1, 2],
            }
            state_b = {
                "state_label": "state_b.state",
                "enemy_species_id": 153,
                "player_species_id": 4,
                "move_ids": [10, 45, 52, 0],
                "legal_slots": [0, 1, 2],
            }

            # Teach the same matchup that slot 0 is bad and slot 1 is better.
            memory.start_episode()
            memory.record_turn(state_a, 0)
            memory.finalize_episode(outcome="loss", reward=-100.0, turns=10)
            memory.start_episode()
            memory.record_turn(state_a, 0)
            memory.finalize_episode(outcome="loss", reward=-90.0, turns=9)
            memory.start_episode()
            memory.record_turn(state_b, 1)
            memory.finalize_episode(outcome="win", reward=110.0, turns=4)

            # New state label with same matchup should use matchup lesson.
            unseen_state = {
                "state_label": "state_new.state",
                "enemy_species_id": 153,
                "player_species_id": 4,
                "move_ids": [10, 45, 52, 0],
                "legal_slots": [0, 1, 2],
            }
            self.assertIsNone(memory.lookup_rule(unseen_state))
            hint = memory.prompt_hint(unseen_state)
            self.assertIn("Matchup lesson", hint)

            slot, changed = memory.maybe_override_slot(unseen_state, 0)
            self.assertTrue(changed)
            self.assertEqual(slot, 1)


if __name__ == "__main__":
    unittest.main()
