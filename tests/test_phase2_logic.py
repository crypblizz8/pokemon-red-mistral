from __future__ import annotations

import unittest

from pokemon.battle_agent import MistralBattleAgent, _condense_reply_for_reflection
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


if __name__ == "__main__":
    unittest.main()
