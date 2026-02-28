from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from pokemon.battle_agent import MistralBattleAgent, _condense_reply_for_reflection
from pokemon.battle_memory import BattleMemory
from pokemon.gen1_data import effectiveness, move_meta, species_meta


class Phase2LogicTests(unittest.TestCase):
    def test_type_effectiveness_examples(self) -> None:
        self.assertEqual(effectiveness("water", ["rock"]), 2.0)
        self.assertEqual(effectiveness("normal", ["ghost"]), 0.0)
        self.assertEqual(effectiveness("fire", ["water"]), 0.5)

    def test_unknown_metadata_fallbacks(self) -> None:
        move = move_meta(999)
        species = species_meta(999)
        self.assertEqual(move["type"], "normal")
        self.assertEqual(move["power"], 0)
        self.assertEqual(species["types"], ["normal"])

    def test_pick_move_enforces_legality(self) -> None:
        agent = MistralBattleAgent(api_key="dummy", model="mistral-large-latest", policy_mode="llm")
        state = {
            "player_hp": 10,
            "player_max_hp": 20,
            "player_level": 8,
            "enemy_hp": 12,
            "enemy_level": 5,
            "enemy_species": 112,
            "move_ids": [10, 45, 52, 0],
            "move_pps": [10, 10, 10, 0],
            "legal_slots": [1],
            "moves": [
                {"slot": 0, "effectiveness": 1.0, "power": 40},
                {"slot": 1, "effectiveness": 1.0, "power": 0},
                {"slot": 2, "effectiveness": 2.0, "power": 40},
                {"slot": 3, "effectiveness": 1.0, "power": 0},
            ],
        }
        cache_key = agent._state_cache_key(state)
        agent.action_cache[cache_key] = 2  # force an illegal cached action

        slot = agent.pick_move(state, use_llm=True)
        self.assertEqual(slot, 1)
        self.assertGreaterEqual(agent.illegal_move_attempts, 1)
        self.assertGreaterEqual(agent.legality_fallback_count, 1)

    def test_heuristic_prefers_damaging_move_over_status(self) -> None:
        agent = MistralBattleAgent(
            api_key="dummy",
            model="mistral-large-latest",
            policy_mode="heuristic",
        )
        state = {
            "player_hp": 20,
            "player_max_hp": 20,
            "player_level": 8,
            "enemy_hp": 12,
            "enemy_level": 5,
            "enemy_species": 153,
            "move_ids": [10, 45, 81, 0],  # scratch, growl, string_shot
            "move_pps": [10, 10, 10, 0],
            "legal_slots": [0, 1, 2],
            "moves": [
                {"slot": 0, "effectiveness": 1.0, "power": 40},
                {"slot": 1, "effectiveness": 1.0, "power": 0},
                {"slot": 2, "effectiveness": 2.0, "power": 0},
                {"slot": 3, "effectiveness": 1.0, "power": 0},
            ],
        }
        slot = agent.pick_move(state, use_llm=False)
        self.assertEqual(slot, 0)

    def test_condense_reply_strips_action_lines(self) -> None:
        reply = "LLM\nEmber is super-effective into bug targets.\nACTION: 2"
        condensed = _condense_reply_for_reflection(reply)
        self.assertIn("Ember is super-effective", condensed)
        self.assertNotIn("ACTION", condensed.upper())

    def test_memory_override_applies_known_good_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = BattleMemory(path=Path(tmpdir) / "battle_memory.json")
            state = {
                "state_label": "battle_first_trainer.state",
                "player_species_id": 4,
                "enemy_species_id": 153,
                "player_hp": 20,
                "player_max_hp": 20,
                "player_level": 8,
                "enemy_hp": 12,
                "enemy_level": 5,
                "move_ids": [10, 45, 81, 0],
                "move_pps": [10, 10, 10, 0],
                "legal_slots": [0, 1, 2],
                "moves": [
                    {"slot": 0, "effectiveness": 2.0, "power": 40},
                    {"slot": 1, "effectiveness": 1.0, "power": 40},
                    {"slot": 2, "effectiveness": 1.0, "power": 0},
                    {"slot": 3, "effectiveness": 1.0, "power": 0},
                ],
            }
            signature = memory.state_signature(state)["signature"]
            memory.rules[str(signature)] = {
                "signature": str(signature),
                "state_label": "battle_first_trainer.state",
                "enemy_species_id": 153,
                "player_species_id": 4,
                "move_ids": [10, 45, 81, 0],
                "legal_slots": [0, 1, 2],
                "slot_stats": memory._normalize_slot_stats({}),
                "preferred_slots": [1],
                "blocked_slots": [0],
                "best_reward": 0.0,
                "best_turns": 0.0,
                "samples_total": 5,
                "last_updated_at": "",
            }

            agent = MistralBattleAgent(
                api_key="dummy",
                model="mistral-large-latest",
                policy_mode="heuristic",
                battle_memory=memory,
            )
            slot = agent.pick_move(state, use_llm=False)
            self.assertEqual(slot, 1)
            self.assertIn("MEMORY_OVERRIDE", agent.last_reply)
            self.assertGreaterEqual(memory.memory_override_count, 1)

    def test_stagnation_override_switches_to_alternate_damaging_move(self) -> None:
        agent = MistralBattleAgent(
            api_key="dummy",
            model="mistral-large-latest",
            policy_mode="heuristic",
        )
        agent.start_episode_context("battle.state", 0)
        state = {
            "enemy_hp": 20,
            "legal_slots": [0, 1],
            "move_ids": [10, 52, 0, 0],
            "move_pps": [10, 10, 0, 0],
            "moves": [
                {"slot": 0, "effectiveness": 1.0, "power": 40},
                {"slot": 1, "effectiveness": 1.0, "power": 35},
                {"slot": 2, "effectiveness": 1.0, "power": 0},
                {"slot": 3, "effectiveness": 1.0, "power": 0},
            ],
        }
        first = agent.pick_move(state, use_llm=False)
        second = agent.pick_move(state, use_llm=False)
        third = agent.pick_move(state, use_llm=False)
        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(third, 1)
        self.assertIn("STAGNATION_OVERRIDE", agent.last_reply)

    def test_stagnation_override_not_used_when_enemy_hp_drops(self) -> None:
        agent = MistralBattleAgent(
            api_key="dummy",
            model="mistral-large-latest",
            policy_mode="heuristic",
        )
        agent.start_episode_context("battle.state", 0)
        state_a = {
            "enemy_hp": 20,
            "legal_slots": [0, 1],
            "move_ids": [10, 52, 0, 0],
            "move_pps": [10, 10, 0, 0],
            "moves": [
                {"slot": 0, "effectiveness": 1.0, "power": 40},
                {"slot": 1, "effectiveness": 1.0, "power": 35},
                {"slot": 2, "effectiveness": 1.0, "power": 0},
                {"slot": 3, "effectiveness": 1.0, "power": 0},
            ],
        }
        state_b = dict(state_a)
        state_b["enemy_hp"] = 15
        first = agent.pick_move(state_a, use_llm=False)
        second = agent.pick_move(state_b, use_llm=False)
        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertNotIn("STAGNATION_OVERRIDE", agent.last_reply)


if __name__ == "__main__":
    unittest.main()
