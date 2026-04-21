from dataclasses import dataclass

import torch

from src.bot import history_tokens_from_events
from src.policy import ACTION_DIM, SWITCH_ACTION_OFFSET, TERA_ACTION_OFFSET, action_index_to_order, legal_mask, order_to_action_index


class _MockMon:
    def __init__(self, ident: str):
        self._ident = ident

    def identifier(self, role: str):
        _ = role
        return self._ident

    def __repr__(self) -> str:
        return self._ident


class _MockBattle:
    def __init__(self):
        self.player_role = "p1"
        self.can_tera = True
        self.available_moves = ["m1", "m2"]
        self.available_switches = [_MockMon("p1: mon2"), _MockMon("p1: mon5")]
        self.team = {
            "p1: mon1": object(),
            "p1: mon2": object(),
            "p1: mon3": object(),
            "p1: mon4": object(),
            "p1: mon5": object(),
            "p1: mon6": object(),
        }


class _MockPlayer:
    def create_order(self, order, **kwargs):
        return {"order": order, **kwargs}


@dataclass(frozen=True)
class _MockOrder:
    message: str


class _MessagePlayer:
    def create_order(self, order, **kwargs):
        if kwargs.get("terastallize"):
            return _MockOrder(f"move:{order}:tera")
        if isinstance(order, str) and order.startswith("m"):
            return _MockOrder(f"move:{order}")
        if hasattr(order, "identifier"):
            return _MockOrder(f"switch:{order.identifier('p1')}")
        return _MockOrder(f"switch:{order}")


def test_legal_mask_respects_moves_tera_and_switch_slots():
    battle = _MockBattle()
    slots = list(battle.team.keys())
    mask = legal_mask(battle, team_slot_ids=slots)

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (ACTION_DIM,)

    # two legal moves in slots 0 and 1
    assert bool(mask[0])
    assert bool(mask[1])
    assert not bool(mask[2])
    assert not bool(mask[3])

    # two legal tera-move variants in slots 4 and 5
    assert bool(mask[TERA_ACTION_OFFSET + 0])
    assert bool(mask[TERA_ACTION_OFFSET + 1])
    assert not bool(mask[TERA_ACTION_OFFSET + 2])
    assert not bool(mask[TERA_ACTION_OFFSET + 3])

    # legal switches only for slot2 and slot5 in this setup
    assert not bool(mask[SWITCH_ACTION_OFFSET + 0])
    assert bool(mask[SWITCH_ACTION_OFFSET + 1])
    assert not bool(mask[SWITCH_ACTION_OFFSET + 2])
    assert not bool(mask[SWITCH_ACTION_OFFSET + 3])
    assert bool(mask[SWITCH_ACTION_OFFSET + 4])
    assert not bool(mask[SWITCH_ACTION_OFFSET + 5])


def test_action_index_to_order_sets_terastallize_flag():
    battle = _MockBattle()
    player = _MockPlayer()

    order = action_index_to_order(player, battle, TERA_ACTION_OFFSET + 1, team_slot_ids=list(battle.team.keys()))

    assert order == {"order": "m2", "terastallize": True}


def test_order_to_action_index_recovers_move_and_switch_slots():
    battle = _MockBattle()
    player = _MessagePlayer()
    slots = list(battle.team.keys())
    mask = legal_mask(battle, team_slot_ids=slots)

    move_order = _MockOrder("move:m2")
    tera_order = _MockOrder("move:m1:tera")

    assert order_to_action_index(player, battle, move_order, team_slot_ids=slots, legal_mask_tensor=mask) == 1
    assert order_to_action_index(player, battle, tera_order, team_slot_ids=slots, legal_mask_tensor=mask) == TERA_ACTION_OFFSET

    target_switch = player.create_order(battle.available_switches[1])
    assert order_to_action_index(player, battle, target_switch, team_slot_ids=slots, legal_mask_tensor=mask) == (
        SWITCH_ACTION_OFFSET + 4
    )


def test_history_tokens_from_events_tracks_recent_turn_events():
    events = [
        ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Garchomp"],
        ["", "switch", "p2a: Rotom", "Garchomp, L80", "100/100"],
        ["", "-terastallize", "p2a: Garchomp", "Ground"],
        ["", "faint", "p1a: Pikachu"],
    ]

    tokens = history_tokens_from_events(events, player_role="p1", opponent_role="p2")

    assert tokens == [
        "HISTORY:self_move:thunderbolt",
        "HISTORY:opp_switch:garchomp",
        "HISTORY:opp_tera:ground",
        "HISTORY:self_faint",
    ]
