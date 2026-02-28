from __future__ import annotations

from typing import Dict, List

GEN1_TYPES: List[str] = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
]


TYPE_CHART: Dict[str, Dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0},
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
    },
    "water": {
        "fire": 2.0,
        "water": 0.5,
        "grass": 0.5,
        "ground": 2.0,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 0.5,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "ghost": 0.0,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "bug": 2.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
    },
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5},
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 2.0,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
    },
    "ghost": {"normal": 0.0, "psychic": 0.0, "ghost": 2.0},
    "dragon": {"dragon": 2.0},
}


MOVE_DATA: Dict[int, Dict[str, object]] = {
    1: {"name": "pound", "type": "normal", "power": 40, "pp_max": 35},
    2: {"name": "karate_chop", "type": "normal", "power": 50, "pp_max": 25},
    5: {"name": "mega_punch", "type": "normal", "power": 80, "pp_max": 20},
    8: {"name": "ice_punch", "type": "ice", "power": 75, "pp_max": 15},
    9: {"name": "thunder_punch", "type": "electric", "power": 75, "pp_max": 15},
    10: {"name": "scratch", "type": "normal", "power": 40, "pp_max": 35},
    14: {"name": "swords_dance", "type": "normal", "power": 0, "pp_max": 30},
    15: {"name": "cut", "type": "normal", "power": 50, "pp_max": 30},
    18: {"name": "whirlwind", "type": "normal", "power": 0, "pp_max": 20},
    20: {"name": "bind", "type": "normal", "power": 15, "pp_max": 20},
    21: {"name": "slam", "type": "normal", "power": 80, "pp_max": 20},
    22: {"name": "vine_whip", "type": "grass", "power": 45, "pp_max": 10},
    24: {"name": "double_kick", "type": "fighting", "power": 30, "pp_max": 30},
    25: {"name": "mega_kick", "type": "normal", "power": 120, "pp_max": 5},
    30: {"name": "horn_attack", "type": "normal", "power": 65, "pp_max": 25},
    31: {"name": "fury_attack", "type": "normal", "power": 15, "pp_max": 20},
    33: {"name": "tackle", "type": "normal", "power": 40, "pp_max": 35},
    35: {"name": "wrap", "type": "normal", "power": 15, "pp_max": 20},
    38: {"name": "double_edge", "type": "normal", "power": 100, "pp_max": 15},
    39: {"name": "tail_whip", "type": "normal", "power": 0, "pp_max": 30},
    43: {"name": "leer", "type": "normal", "power": 0, "pp_max": 30},
    45: {"name": "growl", "type": "normal", "power": 0, "pp_max": 40},
    52: {"name": "ember", "type": "fire", "power": 40, "pp_max": 25},
    53: {"name": "flamethrower", "type": "fire", "power": 95, "pp_max": 15},
    55: {"name": "water_gun", "type": "water", "power": 40, "pp_max": 25},
    57: {"name": "surf", "type": "water", "power": 95, "pp_max": 15},
    58: {"name": "ice_beam", "type": "ice", "power": 95, "pp_max": 10},
    59: {"name": "blizzard", "type": "ice", "power": 120, "pp_max": 5},
    64: {"name": "peck", "type": "flying", "power": 35, "pp_max": 35},
    65: {"name": "drill_peck", "type": "flying", "power": 80, "pp_max": 20},
    71: {"name": "absorb", "type": "grass", "power": 20, "pp_max": 20},
    72: {"name": "mega_drain", "type": "grass", "power": 40, "pp_max": 10},
    73: {"name": "leech_seed", "type": "grass", "power": 0, "pp_max": 10},
    75: {"name": "razor_leaf", "type": "grass", "power": 55, "pp_max": 25},
    76: {"name": "solar_beam", "type": "grass", "power": 120, "pp_max": 10},
    81: {"name": "string_shot", "type": "bug", "power": 0, "pp_max": 40},
    82: {"name": "dragon_rage", "type": "dragon", "power": 0, "pp_max": 10},
    84: {"name": "thunder_shock", "type": "electric", "power": 40, "pp_max": 30},
    85: {"name": "thunderbolt", "type": "electric", "power": 95, "pp_max": 15},
    86: {"name": "thunder_wave", "type": "electric", "power": 0, "pp_max": 20},
    87: {"name": "thunder", "type": "electric", "power": 120, "pp_max": 10},
    88: {"name": "rock_throw", "type": "rock", "power": 50, "pp_max": 15},
    89: {"name": "earthquake", "type": "ground", "power": 100, "pp_max": 10},
    91: {"name": "dig", "type": "ground", "power": 100, "pp_max": 10},
    93: {"name": "confusion", "type": "psychic", "power": 50, "pp_max": 25},
    94: {"name": "psychic", "type": "psychic", "power": 90, "pp_max": 10},
    98: {"name": "quick_attack", "type": "normal", "power": 40, "pp_max": 30},
    102: {"name": "super_fang", "type": "normal", "power": 0, "pp_max": 10},
    103: {"name": "slam", "type": "normal", "power": 80, "pp_max": 20},
    116: {"name": "focus_energy", "type": "normal", "power": 0, "pp_max": 30},
    117: {"name": "bide", "type": "normal", "power": 0, "pp_max": 10},
    121: {"name": "egg_bomb", "type": "normal", "power": 100, "pp_max": 10},
    126: {"name": "fire_blast", "type": "fire", "power": 120, "pp_max": 5},
    129: {"name": "swift", "type": "normal", "power": 60, "pp_max": 20},
    130: {"name": "skull_bash", "type": "normal", "power": 100, "pp_max": 15},
    145: {"name": "bubble", "type": "water", "power": 20, "pp_max": 30},
    146: {"name": "dizzy_punch", "type": "normal", "power": 70, "pp_max": 10},
    147: {"name": "spore", "type": "grass", "power": 0, "pp_max": 15},
    148: {"name": "flash", "type": "normal", "power": 0, "pp_max": 20},
}


SPECIES_DATA: Dict[int, Dict[str, object]] = {
    1: {"name": "rhydon", "types": ["ground", "rock"]},
    2: {"name": "kangaskhan", "types": ["normal"]},
    3: {"name": "nidoran_m", "types": ["poison"]},
    4: {"name": "clefairy", "types": ["normal"]},
    5: {"name": "spearow", "types": ["normal", "flying"]},
    6: {"name": "voltorb", "types": ["electric"]},
    9: {"name": "ivysaur", "types": ["grass", "poison"]},
    10: {"name": "exeggutor", "types": ["grass", "psychic"]},
    12: {"name": "nidoqueen", "types": ["poison", "ground"]},
    14: {"name": "gengar", "types": ["ghost", "poison"]},
    16: {"name": "nidoking", "types": ["poison", "ground"]},
    21: {"name": "mew", "types": ["psychic"]},
    28: {"name": "blastoise", "types": ["water"]},
    35: {"name": "growlithe", "types": ["fire"]},
    36: {"name": "onix", "types": ["rock", "ground"]},
    37: {"name": "fearow", "types": ["normal", "flying"]},
    38: {"name": "pidgey", "types": ["normal", "flying"]},
    39: {"name": "slowbro", "types": ["water", "psychic"]},
    40: {"name": "kadabra", "types": ["psychic"]},
    41: {"name": "graveler", "types": ["rock", "ground"]},
    53: {"name": "magnemite", "types": ["electric"]},
    58: {"name": "pikachu", "types": ["electric"]},
    60: {"name": "sandshrew", "types": ["ground"]},
    64: {"name": "meowth", "types": ["normal"]},
    66: {"name": "nidoran_f", "types": ["poison"]},
    84: {"name": "pikachu_alt", "types": ["electric"]},
    96: {"name": "drowzee", "types": ["psychic"]},
    98: {"name": "krabby", "types": ["water"]},
    100: {"name": "voltorb_alt", "types": ["electric"]},
    109: {"name": "ekans", "types": ["poison"]},
    112: {"name": "weedle", "types": ["bug", "poison"]},
    113: {"name": "kakuna", "types": ["bug", "poison"]},
    114: {"name": "beedrill", "types": ["bug", "poison"]},
    123: {"name": "caterpie", "types": ["bug"]},
    124: {"name": "metapod", "types": ["bug"]},
    125: {"name": "butterfree", "types": ["bug", "flying"]},
    129: {"name": "magikarp", "types": ["water"]},
    130: {"name": "gyarados", "types": ["water", "flying"]},
    133: {"name": "vaporeon", "types": ["water"]},
    134: {"name": "jolteon", "types": ["electric"]},
    135: {"name": "flareon", "types": ["fire"]},
    144: {"name": "articuno", "types": ["ice", "flying"]},
    145: {"name": "zapdos", "types": ["electric", "flying"]},
    146: {"name": "moltres", "types": ["fire", "flying"]},
    150: {"name": "mewtwo", "types": ["psychic"]},
    153: {"name": "bulbasaur", "types": ["grass", "poison"]},
    154: {"name": "ivysaur_alt", "types": ["grass", "poison"]},
    155: {"name": "venusaur", "types": ["grass", "poison"]},
    165: {"name": "rattata", "types": ["normal"]},
    166: {"name": "raticate", "types": ["normal"]},
    176: {"name": "charmander", "types": ["fire"]},
    177: {"name": "squirtle", "types": ["water"]},
    178: {"name": "charmeleon", "types": ["fire"]},
    179: {"name": "wartortle", "types": ["water"]},
    180: {"name": "charizard", "types": ["fire", "flying"]},
}


def move_meta(move_id: int) -> Dict[str, object]:
    move_id = int(move_id)
    info = MOVE_DATA.get(move_id)
    if info is None:
        return {
            "id": move_id,
            "name": f"move_{move_id}",
            "type": "normal",
            "power": 0,
            "pp_max": 0,
        }
    return {
        "id": move_id,
        "name": str(info.get("name", f"move_{move_id}")),
        "type": str(info.get("type", "normal")),
        "power": int(info.get("power", 0)),
        "pp_max": int(info.get("pp_max", 0)),
    }


def species_meta(species_id: int) -> Dict[str, object]:
    species_id = int(species_id)
    info = SPECIES_DATA.get(species_id)
    if info is None:
        return {
            "id": species_id,
            "name": f"species_{species_id}",
            "types": ["normal"],
        }
    types = info.get("types", ["normal"])
    if not isinstance(types, list) or not types:
        types = ["normal"]
    return {
        "id": species_id,
        "name": str(info.get("name", f"species_{species_id}")),
        "types": [str(t) for t in types[:2]],
    }


def effectiveness(move_type: str, enemy_types: List[str]) -> float:
    attacker = str(move_type).lower().strip()
    if attacker not in GEN1_TYPES:
        return 1.0
    if not enemy_types:
        return 1.0

    mult = 1.0
    chart = TYPE_CHART.get(attacker, {})
    for defender in enemy_types[:2]:
        d = str(defender).lower().strip()
        mult *= float(chart.get(d, 1.0))
    return float(mult)

