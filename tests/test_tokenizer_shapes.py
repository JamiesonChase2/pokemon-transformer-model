from types import SimpleNamespace

import torch

from src.tokenizer import assemble_observation, extract_obs, transition_summary_from_events
from src.vocab import build_default_vocab


class _MockMon:
    def __init__(
        self,
        ident: str,
        species: str,
        *,
        hp: float = 1.0,
        tera_type: str | None = None,
        moves: list[str] | None = None,
        item: str | None = None,
        ability: str | None = None,
    ):
        self._ident = ident
        self.species = species
        self.current_hp_fraction = hp
        self.status = None
        self.boosts = {}
        self.is_terastallized = False
        self.tera_type = tera_type
        self.item = item
        self.ability = ability
        self.level = 80
        self.base_stats = {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90}
        self.moves = {move: SimpleNamespace(id=move) for move in (moves or [])}

    def identifier(self, role: str):
        _ = role
        return self._ident


class _MockCurrentObservation:
    def __init__(self, events):
        self.events = list(events)


class _MockBattle:
    def __init__(self):
        self.player_role = "p1"
        self.opponent_role = "p2"
        self.turn = 3
        self.force_switch = False
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.can_tera = True
        self.used_tera = False
        self.opponent_used_tera = False

        self.active_pokemon = _MockMon(
            "p1: active",
            "Pikachu",
            tera_type="Electric",
            moves=["thunderbolt", "protect"],
            item="Light Ball",
            ability="Static",
        )
        self.opponent_active_pokemon = _MockMon("p2: active", "Garchomp", moves=["earthquake"])

        self.team = {
            "p1: active": self.active_pokemon,
            "p1: bench1": _MockMon(
                "p1: bench1",
                "Raichu",
                tera_type="Electric",
                moves=["voltswitch", "grassknot"],
                item="Choice Scarf",
                ability="Lightning Rod",
            ),
        }
        self.opponent_team = {
            "p2: active": self.opponent_active_pokemon,
            "p2: bench1": _MockMon("p2: bench1", "RotomWash", moves=["hydropump", "voltswitch"]),
        }
        self.observations = {
            1: _MockCurrentObservation(
                [
                    ["", "move", "p1a: Pikachu", "Protect", "p2a: Garchomp"],
                    ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Pikachu"],
                ]
            ),
            2: _MockCurrentObservation(
                [
                    ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Garchomp"],
                    ["", "-supereffective", "p1a: Pikachu", "p2a: Garchomp"],
                    ["", "-terastallize", "p2a: Garchomp", "Ground"],
                ]
            ),
        }

        self.available_moves = []
        self._current_observation = _MockCurrentObservation(
            [
                ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Garchomp"],
                ["", "switch", "p2a: Rotom", "Garchomp, L80", "100/100"],
            ]
        )
        self.last_request = {
            "side": {
                "pokemon": [
                    {
                        "ident": "p1: active",
                        "item": "Light Ball",
                        "ability": "Static",
                        "moves": ["thunderbolt", "protect", "fakeout", "volttackle"],
                    },
                    {
                        "ident": "p1: bench1",
                        "item": "Choice Scarf",
                        "ability": "Lightning Rod",
                        "moves": ["voltswitch", "grassknot", "focusblast", "nastyplot"],
                    },
                ]
            }
        }


def test_extract_obs_preserves_self_and_opponent_known_information():
    battle = _MockBattle()
    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )

    assert obs["field"]["used_tera"] is False
    assert obs["my_active"]["tera_type"] == "electric"
    assert obs["my_active"]["item"] == "lightball"
    assert obs["my_active"]["ability"] == "static"


def test_assemble_observation_emits_flat_schema_and_embeds_legal_mask():
    vocab = build_default_vocab(gen=9)
    battle = _MockBattle()
    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )
    legal_mask = torch.zeros((14,), dtype=torch.bool)
    legal_mask[0] = True
    legal_mask[9] = True

    obs_tensor = assemble_observation(obs, vocab, legal_mask=legal_mask)
    meta = vocab.schema_meta()
    offsets = meta["offsets"]

    assert obs_tensor.shape == (int(meta["obs_dim"]),)
    assert torch.equal(obs_tensor[offsets["action_mask"][0] : offsets["action_mask"][1]], legal_mask.float())

    pokemon_ids = obs_tensor[offsets["pokemon_ids"][0] : offsets["pokemon_ids"][1]].reshape(12, 2).long()
    move_ids = obs_tensor[offsets["move_ids"][0] : offsets["move_ids"][1]].reshape(12, 4).long()
    assert vocab.decode("pokemon.species", int(pokemon_ids[0, 0].item())) == "pikachu"
    assert vocab.decode("pokemon.item", int(pokemon_ids[0, 1].item())) == "lightball"
    assert [vocab.decode("move.id", int(value.item())) for value in move_ids[0]] == [
        "thunderbolt",
        "protect",
        "unk",
        "unk",
    ]
    assert [vocab.decode("move.id", int(value.item())) for value in move_ids[6]] == [
        "earthquake",
        "unk",
        "unk",
        "unk",
    ]
    assert [vocab.decode("move.id", int(value.item())) for value in move_ids[7]][:2] == ["hydropump", "voltswitch"]


def test_transition_summary_from_events_tracks_last_turn_state():
    summary = transition_summary_from_events(
        [
            ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Garchomp"],
            ["", "-supereffective", "p1a: Pikachu", "p2a: Garchomp"],
            ["", "-crit", "p1a: Pikachu", "p2a: Garchomp"],
            ["", "switch", "p1a: Pikachu", "Raichu, L80", "100/100"],
            ["", "-terastallize", "p2a: Garchomp", "Ground"],
            ["", "faint", "p1a: Pikachu"],
            ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Pikachu"],
        ],
        player_role="p1",
        opponent_role="p2",
    )

    assert summary["self_move"] == "thunderbolt"
    assert summary["opp_move"] == "earthquake"
    assert summary["flags"]["self_moved_first"] == 1.0
    assert summary["flags"]["self_supereffective"] == 1.0
    assert summary["flags"]["self_crit"] == 1.0
    assert "self_switched" not in summary["flags"]
    assert "self_fainted" not in summary["flags"]


def test_extract_obs_keeps_possible_abilities_including_known_slot():
    battle = _MockBattle()
    battle.active_pokemon.possible_abilities = ["Static", "Lightning Rod"]

    obs = extract_obs(
        battle,
        self_slot_order=list(battle.team.keys()),
        opponent_seen_order=list(battle.opponent_team.keys()),
    )

    assert obs["my_active"]["ability"] == "static"
    assert obs["my_active"]["possible_abilities"] == ["static", "lightningrod"]
