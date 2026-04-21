from types import SimpleNamespace

import torch

from src.tokenizer import _estimate_stat, assemble_observation, extract_obs
from src.vocab import build_default_vocab


class _MockMon:
    def __init__(
        self,
        ident: str,
        species: str,
        *,
        stats: dict[str, int] | None = None,
        moves: list[str] | None = None,
        item: str | None = None,
        ability: str | None = None,
        tera_type: str | None = None,
        effects: list[str] | None = None,
        possible_abilities: list[str] | None = None,
    ):
        self._ident = ident
        self.species = species
        self.current_hp_fraction = 1.0
        self.status = None
        self.boosts = {}
        self.level = 80
        self.base_stats = {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90}
        self.stats = stats
        self.moves = {
            move: SimpleNamespace(
                id=move,
                current_pp=16,
                max_pp=16,
                accuracy=100,
                base_power=80,
                category="special",
                priority=0,
                type="Electric",
            )
            for move in (moves or [])
        }
        self.item = item
        self.ability = ability
        self.possible_abilities = list(possible_abilities or [])
        self.tera_type = tera_type
        self.is_terastallized = False
        self.active = False
        self.fainted = False
        self.status_counter = 0
        self.type_1 = "Electric"
        self.type_2 = None
        self.effects = effects or []
        self.weight = 6.0
        self.height = 0.4

    def identifier(self, role: str):
        _ = role
        return self._ident


class _MockBattle:
    def __init__(self, active_stats: dict[str, int]):
        self.player_role = "p1"
        self.opponent_role = "p2"
        self.turn = 2
        self.force_switch = False
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.can_tera = True
        self.used_tera = False
        self.opponent_used_tera = False
        self.available_moves = []
        self._current_observation = SimpleNamespace(events=[])

        self.active_pokemon = _MockMon(
            "p1: active",
            "Pikachu",
            stats=active_stats,
            moves=["thunderbolt", "protect"],
            item="Light Ball",
            ability="Static",
            tera_type="Electric",
        )
        self.active_pokemon.active = True
        self.opponent_active_pokemon = _MockMon("p2: active", "Garchomp", moves=["earthquake"])
        self.opponent_active_pokemon.active = True

        self.team = {
            "p1: active": self.active_pokemon,
            "p1: bench1": _MockMon(
                "p1: bench1",
                "Raichu",
                moves=["voltswitch", "grassknot"],
                item="Choice Scarf",
                ability="Lightning Rod",
                tera_type="Electric",
            ),
        }
        self.opponent_team = {
            "p2: active": self.opponent_active_pokemon,
        }


def _entry(
    *,
    species: str = "pikachu",
    effects: list[str] | None = None,
    item: str = "lightball",
    ability: str = "static",
    types: list[str] | None = None,
    tera_type: str = "electric",
    moves: list[str] | None = None,
) -> dict:
    move_ids = list(moves or ["thunderbolt"])
    summaries = [
        {
            "id": move_id,
            "acc_int": 100,
            "pwr_int": 80,
            "pp_int": 16,
            "category": "special",
            "priority": 0,
            "type": "electric",
        }
        for move_id in move_ids
    ]
    while len(summaries) < 4:
        summaries.append(
            {
                "id": "unk",
                "acc_int": 0,
                "pwr_int": 0,
                "pp_int": 0,
                "category": "status",
                "priority": 0,
                "type": "unk",
            }
        )

    return {
        "species": species,
        "hp_int": 100,
        "status": "none",
        "boosts": {stat: 0 for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]},
        "stats_int": [211, 120, 110, 130, 115, 190],
        "level_int": 80,
        "weight_int": 4,
        "height_int": 4,
        "active": True,
        "fainted": False,
        "terastallized": False,
        "revealed": True,
        "status_counter": 0,
        "types": list(types or ["electric"]),
        "tera_type": tera_type,
        "effects": list(effects or []),
        "item": item,
        "ability": ability,
        "possible_abilities": ["lightningrod"],
        "moves": summaries,
    }


def _effect_start(meta: dict, slot_idx: int) -> int:
    body_base = int(meta["offsets"]["pokemon_body"][0]) + slot_idx * int(meta["dim_pokemon_body"])
    return body_base + 101 + int(meta["body_flags_dim"]) + (2 * int(meta["vocab_type"]))


def _side_start(meta: dict, *, opponent: bool = False) -> int:
    global_start = int(meta["offsets"]["global_scalars"][0])
    base = global_start + 3 + int(meta["vocab_weather"]) + 10
    if opponent:
        base += int(meta["vocab_side_condition"])
    return base


def test_gen9_randbats_vocab_curates_schema_dimensions():
    curated = build_default_vocab(gen=9, battle_format="gen9randombattle")
    full = build_default_vocab(gen=9, battle_format="gen9ou")

    curated_meta = curated.schema_meta()
    full_meta = full.schema_meta()

    assert curated_meta["dim_single_move_scalars"] == 19 + int(curated_meta["vocab_type"])
    assert curated_meta["dim_move_scalars"] == 4 * int(curated_meta["dim_single_move_scalars"])
    assert curated_meta["n_ability_slots"] == 4
    assert curated_meta["dim_global_scalars"] == 3 + int(curated_meta["vocab_weather"]) + 10 + (2 * int(curated_meta["vocab_side_condition"]))
    assert curated_meta["dim_transition_scalars"] == 10

    expected_curated_obs_dim = (
        12 * int(curated_meta["dim_pokemon_body"])
        + (12 * 2)
        + (12 * int(curated_meta["n_ability_slots"]))
        + (12 * 4)
        + (12 * int(curated_meta["dim_move_scalars"]))
        + int(curated_meta["dim_global_scalars"])
        + 2
        + int(curated_meta["dim_transition_scalars"])
        + 14
    )
    assert curated_meta["obs_dim"] == expected_curated_obs_dim

    assert full_meta["obs_dim"] > curated_meta["obs_dim"]
    assert full_meta["dim_pokemon_body"] > curated_meta["dim_pokemon_body"]
    assert full_meta["dim_global_scalars"] > curated_meta["dim_global_scalars"]

    assert "substitute" in curated.categories["pokemon.effect"]
    assert "protosynthesis" in curated.categories["pokemon.effect"]
    assert "dynamax" not in curated.categories["pokemon.effect"]
    assert "gravity" in curated.categories["global.field"]
    assert "trickroom" not in curated.categories["global.field"]
    assert "stealthrock" in curated.categories["global.side_condition"]
    assert "gmaxsteelsurge" not in curated.categories["global.side_condition"]


def test_assemble_observation_only_activates_curated_randbats_channels():
    curated = build_default_vocab(gen=9, battle_format="gen9randombattle")
    full = build_default_vocab(gen=9, battle_format="gen9ou")

    obs = {
        "field": {
            "turn": 5,
            "used_tera": False,
            "opponent_used_tera": False,
            "weather": "raindance",
            "weather_duration": 3.0,
            "side_conditions": {"spikes": 1.0, "gmaxsteelsurge": 1.0},
            "opponent_side_conditions": {"stickyweb": 1.0},
        },
        "my_active": _entry(effects=["substitute", "dynamax"]),
        "my_bench": [],
        "opp_active": _entry(
            species="garchomp",
            item="unk",
            ability="roughskin",
            types=["dragon", "ground"],
        ),
        "opp_bench": [],
        "transitions": {"self_move": "thunderbolt", "opp_move": "earthquake", "flags": {}},
    }
    curated_tensor = assemble_observation(obs, curated)
    full_tensor = assemble_observation(obs, full)

    curated_meta = curated.schema_meta()
    full_meta = full.schema_meta()

    curated_effect_start = _effect_start(curated_meta, 0)
    full_effect_start = _effect_start(full_meta, 0)

    substitute_cur = curated.encode("pokemon.effect", "substitute")
    dynamax_cur = curated.encode("pokemon.effect", "dynamax")
    substitute_full = full.encode("pokemon.effect", "substitute")
    dynamax_full = full.encode("pokemon.effect", "dynamax")

    assert substitute_cur > 0
    assert dynamax_cur == 0
    assert curated_tensor[curated_effect_start + substitute_cur].item() == 1.0
    assert full_tensor[full_effect_start + dynamax_full].item() == 1.0

    global_start = int(curated_meta["offsets"]["global_scalars"][0])
    weather_start = global_start + 3
    rain_idx = curated.encode("global.weather", "raindance")
    assert rain_idx > 0
    assert curated_tensor[weather_start + rain_idx].item() == 1.0

    duration_start = weather_start + int(curated_meta["vocab_weather"])
    weather_duration_slice = curated_tensor[duration_start : duration_start + 10]
    assert torch.isclose(weather_duration_slice.sum(), torch.tensor(1.0), atol=1e-6)

    spikes_cur = curated.encode("global.side_condition", "spikes")
    gmax_cur = curated.encode("global.side_condition", "gmaxsteelsurge")
    gmax_full = full.encode("global.side_condition", "gmaxsteelsurge")
    assert spikes_cur > 0
    assert gmax_cur == 0
    assert curated_tensor[_side_start(curated_meta) + spikes_cur].item() > 0.0
    assert curated_tensor[_side_start(curated_meta, opponent=True) + curated.encode("global.side_condition", "stickyweb")].item() > 0.0
    assert full_tensor[_side_start(full_meta) + gmax_full].item() > 0.0

    move_scalar_start = int(curated_meta["offsets"]["move_scalars"][0])
    electric_type_idx = curated.encode("pokemon.type", "electric")
    assert curated_tensor[move_scalar_start + 0].item() == 100.0
    assert curated_tensor[move_scalar_start + 1].item() == 80.0
    assert curated_tensor[move_scalar_start + 2].item() == 16.0
    assert curated_tensor[move_scalar_start + 4].item() == 1.0
    assert curated_tensor[move_scalar_start + 12].item() == 1.0
    assert curated_tensor[move_scalar_start + 19 + electric_type_idx].item() == 1.0


def test_extract_obs_uses_estimated_visible_self_stats():
    exact_stats = {"hp": 281, "atk": 149, "def": 121, "spa": 177, "spd": 132, "spe": 211}
    battle = _MockBattle(active_stats=exact_stats)

    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )

    expected_stats = [
        _estimate_stat(battle.active_pokemon.base_stats, battle.active_pokemon.level, list(battle.active_pokemon.moves.keys()), stat)
        for stat in ("hp", "atk", "def", "spa", "spd", "spe")
    ]

    assert obs["my_active"]["stats_int"] == expected_stats
    assert obs["my_active"]["stats_int"] != [281, 149, 121, 177, 132, 211]
    assert obs["my_active"]["item"] == "lightball"
    assert obs["my_active"]["ability"] == "static"


def test_extract_obs_hidden_info_does_not_guess_item_or_ability():
    battle = _MockBattle(active_stats={"hp": 281, "atk": 149, "def": 121, "spa": 177, "spd": 132, "spe": 211})
    battle.opponent_active_pokemon = _MockMon(
        "p2: active",
        "Garchomp",
        possible_abilities=["Rough Skin", "Sand Veil"],
    )
    battle.opponent_active_pokemon.active = True
    battle.opponent_team = {"p2: active": battle.opponent_active_pokemon}

    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )

    opp_active = obs["opp_active"]
    assert opp_active["item"] == "unk"
    assert opp_active["ability"] == "unk"
    assert opp_active["possible_abilities"] == ["roughskin", "sandveil"]
    assert opp_active["tera_type"] == "unk"
    assert [move["id"] for move in opp_active["moves"]] == ["unk", "unk", "unk", "unk"]


def test_extract_obs_keeps_possible_ability_list_for_self():
    battle = _MockBattle(active_stats={"hp": 281, "atk": 149, "def": 121, "spa": 177, "spd": 132, "spe": 211})
    battle.active_pokemon.possible_abilities = ["Static", "Lightning Rod"]

    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )

    assert obs["my_active"]["ability"] == "static"
    assert obs["my_active"]["possible_abilities"] == ["static", "lightningrod"]
