"""Action mapping and legal action masking for poke-env singles."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

MOVE_ACTIONS = 4
TERA_MOVE_ACTIONS = 4
SWITCH_ACTIONS = 6
TERA_ACTION_OFFSET = MOVE_ACTIONS
SWITCH_ACTION_OFFSET = MOVE_ACTIONS + TERA_MOVE_ACTIONS
ACTION_DIM = SWITCH_ACTION_OFFSET + SWITCH_ACTIONS


def team_slot_ids_from_battle(battle: Any) -> List[str]:
    """Return stable team-slot identifiers for the current player."""

    team = getattr(battle, "team", {}) or {}
    return list(team.keys())[:SWITCH_ACTIONS]


def legal_mask(battle: Any, team_slot_ids: Optional[Sequence[str]] = None) -> torch.BoolTensor:
    """Build legal-action mask with layout [4 moves, 4 tera-moves, 6 switches]."""

    mask = torch.zeros((ACTION_DIM,), dtype=torch.bool)

    available_moves = list(getattr(battle, "available_moves", []) or [])
    for idx in range(min(MOVE_ACTIONS, len(available_moves))):
        mask[idx] = True
    if bool(getattr(battle, "can_tera", False)):
        for idx in range(min(TERA_MOVE_ACTIONS, len(available_moves))):
            mask[TERA_ACTION_OFFSET + idx] = True

    available_switches = list(getattr(battle, "available_switches", []) or [])
    available_ids = set()
    player_role = getattr(battle, "player_role", "p1")
    for mon in available_switches:
        try:
            available_ids.add(mon.identifier(player_role))
        except Exception:
            available_ids.add(getattr(mon, "species", None))

    slots = list(team_slot_ids) if team_slot_ids is not None else team_slot_ids_from_battle(battle)
    for slot_idx in range(SWITCH_ACTIONS):
        if slot_idx >= len(slots):
            continue
        if slots[slot_idx] in available_ids:
            mask[SWITCH_ACTION_OFFSET + slot_idx] = True

    return mask


def _masked_logits(policy_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = policy_logits.clone()
    masked[~mask] = -1e9
    return masked


def select_action(
    policy_logits: torch.Tensor,
    legal_mask_tensor: torch.Tensor,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample/select an action from masked logits."""

    if policy_logits.ndim != 1:
        raise ValueError(f"Expected policy_logits rank 1, got shape {tuple(policy_logits.shape)}")
    if legal_mask_tensor.ndim != 1 or legal_mask_tensor.shape[0] != ACTION_DIM:
        raise ValueError(f"legal_mask_tensor must be shape [{ACTION_DIM}]")

    if not torch.any(legal_mask_tensor):
        return int(torch.argmax(policy_logits).item())

    masked_logits = _masked_logits(policy_logits, legal_mask_tensor)

    if temperature <= 0:
        return int(torch.argmax(masked_logits).item())

    scaled = masked_logits / max(1e-6, float(temperature))
    probs = torch.softmax(scaled, dim=-1)

    # Softmax with -inf masking may produce tiny numerical junk on illegal entries.
    probs = probs * legal_mask_tensor.float()
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return int(torch.argmax(masked_logits).item())
    probs = probs / probs_sum

    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item())


def action_index_to_order(
    player: Any,
    battle: Any,
    action_index: int,
    *,
    team_slot_ids: Optional[Sequence[str]] = None,
):
    """Map action index to a poke-env order object."""

    available_moves = list(getattr(battle, "available_moves", []) or [])
    available_switches = list(getattr(battle, "available_switches", []) or [])

    if 0 <= action_index < MOVE_ACTIONS:
        if action_index < len(available_moves):
            return player.create_order(available_moves[action_index])
        return None

    if TERA_ACTION_OFFSET <= action_index < SWITCH_ACTION_OFFSET:
        move_slot = action_index - TERA_ACTION_OFFSET
        if move_slot < len(available_moves) and bool(getattr(battle, "can_tera", False)):
            return player.create_order(available_moves[move_slot], terastallize=True)
        return None

    if SWITCH_ACTION_OFFSET <= action_index < ACTION_DIM:
        switch_slot = action_index - SWITCH_ACTION_OFFSET
        slots = list(team_slot_ids) if team_slot_ids is not None else team_slot_ids_from_battle(battle)
        if switch_slot >= len(slots):
            return None

        target_slot_id = slots[switch_slot]
        player_role = getattr(battle, "player_role", "p1")

        for mon in available_switches:
            try:
                mon_id = mon.identifier(player_role)
            except Exception:
                mon_id = getattr(mon, "species", None)
            if mon_id == target_slot_id:
                return player.create_order(mon)

        return None

    return None


def order_to_action_index(
    player: Any,
    battle: Any,
    order: Any,
    *,
    team_slot_ids: Optional[Sequence[str]] = None,
    legal_mask_tensor: Optional[torch.Tensor] = None,
) -> Optional[int]:
    """Map a poke-env order object back to the discrete action index."""

    if order is None:
        return None

    target_message = getattr(order, "message", None)
    if target_message is None:
        try:
            target_message = str(order)
        except Exception:
            return None

    mask = legal_mask_tensor if legal_mask_tensor is not None else legal_mask(battle, team_slot_ids=team_slot_ids)
    if mask.ndim != 1 or mask.shape[0] != ACTION_DIM:
        raise ValueError(f"legal_mask_tensor must be shape [{ACTION_DIM}]")

    legal_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    for tensor_idx in legal_indices:
        action_idx = int(tensor_idx.item())
        candidate = action_index_to_order(player, battle, action_idx, team_slot_ids=team_slot_ids)
        if candidate is None:
            continue
        candidate_message = getattr(candidate, "message", None)
        if candidate_message is None:
            candidate_message = str(candidate)
        if candidate_message == target_message:
            return action_idx

    return None
