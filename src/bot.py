"""Transformer-driven poke-env player with trajectory collection."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from poke_env.data.normalize import to_id_str
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.ps_client import PSClient

from .policy import action_index_to_order, legal_mask, order_to_action_index, select_action
from .tokenizer import assemble_observation, extract_obs
from .vocab import ObservationVocabulary

HISTORY_WINDOW = 2


def _install_ps_client_message_guard() -> None:
    """Ignore blank/malformed websocket payloads that older poke-env versions mishandle."""

    if bool(getattr(PSClient, "_pokeenv_transformer_message_guard", False)):
        return

    original_handle_message = PSClient._handle_message

    async def _guarded_handle_message(self: PSClient, message: str):
        if not str(message).strip():
            return None

        split_messages = [m.split("|") for m in str(message).split("\n")]
        if not split_messages or not split_messages[0]:
            return None

        first = split_messages[0]
        first_entry = str(first[0]) if first else ""
        if len(first) < 2 and not first_entry.startswith(">battle"):
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.debug("Ignoring malformed showdown payload: %r", message)
            return None

        return await original_handle_message(self, message)

    PSClient._handle_message = _guarded_handle_message
    PSClient._pokeenv_transformer_message_guard = True


_install_ps_client_message_guard()


def compute_step_reward(
    *,
    current_self_fainted: int,
    next_self_fainted: int,
    current_opp_fainted: int,
    next_opp_fainted: int,
    use_faint_reward: bool,
    faint_self: float,
    faint_opp: float,
) -> float:
    if not bool(use_faint_reward):
        return 0.0

    self_delta = max(0, int(next_self_fainted) - int(current_self_fainted))
    opp_delta = max(0, int(next_opp_fainted) - int(current_opp_fainted))
    return float((float(self_delta) * float(faint_self)) + (float(opp_delta) * float(faint_opp)))


def compute_gae_targets(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
    terminal_value: float = 0.0,
    clip_return: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")

    advantages: List[float] = [0.0] * len(rewards)
    returns: List[float] = [0.0] * len(rewards)
    gae = 0.0
    next_values = list(values[1:]) + [float(terminal_value)]

    for idx in range(len(rewards) - 1, -1, -1):
        delta = float(rewards[idx]) + float(gamma) * float(next_values[idx]) - float(values[idx])
        gae = delta + float(gamma) * float(gae_lambda) * gae
        advantages[idx] = float(gae)
        ret = float(values[idx]) + float(gae)
        if clip_return is not None:
            ret = max(-float(clip_return), min(float(clip_return), ret))
        returns[idx] = float(ret)

    return returns, advantages


def _event_side(event_target: object, *, player_role: str, opponent_role: str) -> Optional[str]:
    text = str(event_target or "")
    if text.startswith(player_role):
        return "self"
    if text.startswith(opponent_role):
        return "opp"
    return None


def _species_from_switch_details(details: object) -> str:
    text = str(details or "")
    species = text.split(",", 1)[0].strip()
    return to_id_str(species) if species else "unk"


def history_tokens_from_events(
    events: Iterable[Sequence[object]],
    *,
    player_role: str,
    opponent_role: str,
) -> List[str]:
    tokens: List[str] = []

    for raw_event in events:
        if len(raw_event) < 2:
            continue

        kind = str(raw_event[1])
        token: Optional[str] = None

        if kind == "move" and len(raw_event) >= 4:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            if side is not None:
                token = f"HISTORY:{side}_move:{to_id_str(raw_event[3])}"
        elif kind in {"switch", "drag"} and len(raw_event) >= 4:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            if side is not None:
                token = f"HISTORY:{side}_switch:{_species_from_switch_details(raw_event[3])}"
        elif kind == "faint" and len(raw_event) >= 3:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            if side is not None:
                token = f"HISTORY:{side}_faint"
        elif kind == "-terastallize" and len(raw_event) >= 4:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            if side is not None:
                token = f"HISTORY:{side}_tera:{to_id_str(raw_event[3])}"
        elif kind == "-status" and len(raw_event) >= 4:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            status_id = to_id_str(raw_event[3])
            if side is not None and status_id:
                token = f"HISTORY:{side}_status:{status_id}"
        elif kind == "cant" and len(raw_event) >= 3:
            side = _event_side(raw_event[2], player_role=player_role, opponent_role=opponent_role)
            if side is not None:
                token = f"HISTORY:{side}_cant"

        if token is not None:
            tokens.append(token)

    return tokens


class TransformerPlayer(Player):
    """poke-env Player that selects actions from a Transformer policy/value model."""

    def __init__(
        self,
        *args: Any,
        model: torch.nn.Module,
        vocab: ObservationVocabulary,
        max_len: int | None = None,
        device: str = "cpu",
        temperature: float = 1.0,
        collect_trajectories: bool = True,
        reward_terminal_win: float = 1.0,
        reward_terminal_loss: float = -1.0,
        reward_use_faint: bool = True,
        reward_faint_self: float = -0.1,
        reward_faint_opp: float = 0.1,
        reward_discount: float = 0.9999,
        reward_gae_lambda: float = 0.75,
        reward_target_clip: float = 1.6,
        max_turns_before_forfeit: Optional[int] = 500,
        seed: Optional[int] = None,
        inference_client: Any | None = None,
        inference_model_key: str = "current",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.vocab = vocab
        self.max_len = (int(max_len) if max_len is not None else None)
        self.device = device
        self.temperature = float(temperature)
        self.collect_trajectories = bool(collect_trajectories)
        self.reward_terminal_win = float(reward_terminal_win)
        self.reward_terminal_loss = float(reward_terminal_loss)
        self.reward_use_faint = bool(reward_use_faint)
        self.reward_faint_self = float(reward_faint_self)
        self.reward_faint_opp = float(reward_faint_opp)
        self.reward_discount = float(reward_discount)
        self.reward_gae_lambda = float(reward_gae_lambda)
        self.reward_target_clip = float(reward_target_clip)
        self.max_turns_before_forfeit = (
            int(max_turns_before_forfeit) if max_turns_before_forfeit is not None else None
        )
        self.inference_client = inference_client
        self.inference_model_key = str(inference_model_key)

        self._torch_rng = torch.Generator()
        if seed is not None:
            self._torch_rng.manual_seed(int(seed))

        self._pending_steps: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._completed_episodes: List[List[Dict[str, Any]]] = []
        self._self_slot_order: Dict[str, List[str]] = {}
        self._opponent_seen_order: Dict[str, List[str]] = {}
        self._history_tokens: Dict[str, Deque[str]] = {}
        self._history_completed_turn: Dict[str, int] = {}
        self._history_live_turn: Dict[str, int] = {}
        self._history_live_offset: Dict[str, int] = {}

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(temperature)

    def build_model_input(self, battle: Any) -> Dict[str, Any]:
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))

        self._update_seen_orders(battle)
        self._update_turn_history(battle)

        self_slots = self._self_slot_order.get(battle_tag, [])
        opp_seen = self._opponent_seen_order.get(battle_tag, [])
        history = list(self._history_tokens.get(battle_tag, ()))

        obs = extract_obs(
            battle,
            perspective="self",
            self_slot_order=self_slots,
            opponent_seen_order=opp_seen,
        )
        legal = legal_mask(battle, team_slot_ids=self_slots)
        obs_tensor = assemble_observation(obs, self.vocab, legal_mask=legal)

        return {
            "battle_tag": battle_tag,
            "self_slots": self_slots,
            "opponent_seen": opp_seen,
            "history": history,
            "obs": obs,
            "obs_tensor": obs_tensor,
            "legal_mask": legal,
        }

    def choose_move(self, battle: Any):  # type: ignore[override]
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))

        if self.max_turns_before_forfeit is not None:
            turn = int(getattr(battle, "turn", 0) or 0)
            if turn > self.max_turns_before_forfeit:
                return ForfeitBattleOrder()

        model_input = self.build_model_input(battle)
        self_slots = list(model_input["self_slots"])
        obs_tensor = model_input["obs_tensor"]
        legal = model_input["legal_mask"]

        if self.inference_client is not None:
            policy_logits, value_pred = self.inference_client.infer(
                obs_tensor=obs_tensor,
                model_key=self.inference_model_key,
            )
        else:
            self.model.eval()
            with torch.no_grad():
                policy_logits, value = self.model(obs_tensor.to(self.device))
            policy_logits = policy_logits.detach().cpu().float()
            value_pred = float(value.detach().cpu().float().squeeze().item())
        masked_logits = policy_logits.masked_fill(~legal, -1e9)
        pure_log_probs = torch.log_softmax(masked_logits, dim=-1)

        action_idx = select_action(
            policy_logits,
            legal,
            temperature=self.temperature,
            generator=self._torch_rng,
        )
        action_log_prob = float(pure_log_probs[action_idx].item()) if bool(legal[action_idx]) else 0.0

        order = action_index_to_order(self, battle, action_idx, team_slot_ids=self_slots)
        if order is None:
            legal_indices = torch.nonzero(legal, as_tuple=False).reshape(-1)
            if legal_indices.numel() > 0:
                fallback_idx = int(legal_indices[0].item())
                order = action_index_to_order(self, battle, fallback_idx, team_slot_ids=self_slots)
                action_idx = fallback_idx
                action_log_prob = float(pure_log_probs[action_idx].item())

        if order is None:
            order = self.choose_random_move(battle)

        if self.collect_trajectories:
            self_fainted_before, opp_fainted_before = self._battle_fainted_counts(battle)
            self._pending_steps[battle_tag].append(
                {
                    "obs": obs_tensor.cpu(),
                    "legal_mask": legal.cpu(),
                    "action_index": int(action_idx),
                    "old_log_prob": float(action_log_prob),
                    "old_value": float(value_pred),
                    "self_fainted_before": int(self_fainted_before),
                    "opp_fainted_before": int(opp_fainted_before),
                }
            )

        return order

    def _battle_finished_callback(self, battle: Any):
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))

        outcome_z = self.reward_terminal_win if bool(getattr(battle, "won", False)) else self.reward_terminal_loss

        steps = self._pending_steps.pop(battle_tag, [])
        if steps:
            final_self_fainted, final_opp_fainted = self._battle_fainted_counts(battle)
            self_fainted = [int(step.get("self_fainted_before", 0)) for step in steps]
            next_self_fainted = self_fainted[1:] + [int(final_self_fainted)]
            opp_fainted = [int(step.get("opp_fainted_before", 0)) for step in steps]
            next_opp_fainted = opp_fainted[1:] + [int(final_opp_fainted)]

            rewards: List[float] = []
            for current_self, next_self, current_opp, next_opp in zip(
                self_fainted,
                next_self_fainted,
                opp_fainted,
                next_opp_fainted,
            ):
                rewards.append(
                    compute_step_reward(
                        current_self_fainted=current_self,
                        next_self_fainted=next_self,
                        current_opp_fainted=current_opp,
                        next_opp_fainted=next_opp,
                        use_faint_reward=self.reward_use_faint,
                        faint_self=self.reward_faint_self,
                        faint_opp=self.reward_faint_opp,
                    )
                )
            if rewards:
                rewards[-1] = float(rewards[-1] + outcome_z)

            old_values = [float(step.get("old_value", 0.0)) for step in steps]
            targets, advantages = compute_gae_targets(
                rewards,
                old_values,
                gamma=self.reward_discount,
                gae_lambda=self.reward_gae_lambda,
                terminal_value=0.0,
                clip_return=self.reward_target_clip,
            )

            episode = []
            for step, shaped_reward, target, advantage in zip(steps, rewards, targets, advantages):
                row = dict(step)
                row.pop("self_fainted_before", None)
                row.pop("opp_fainted_before", None)
                row["outcome_z"] = float(target)
                row["advantage"] = float(advantage)
                row["terminal_z"] = float(outcome_z)
                row["shaped_reward"] = float(shaped_reward)
                episode.append(row)
            self._completed_episodes.append(episode)

        self._cleanup_battle_state(battle_tag)

    def _cleanup_battle_state(self, battle_tag: str) -> None:
        self._self_slot_order.pop(battle_tag, None)
        self._opponent_seen_order.pop(battle_tag, None)
        self._history_tokens.pop(battle_tag, None)
        self._history_completed_turn.pop(battle_tag, None)
        self._history_live_turn.pop(battle_tag, None)
        self._history_live_offset.pop(battle_tag, None)

    def pop_completed_episodes(self) -> List[List[Dict[str, Any]]]:
        episodes = self._completed_episodes
        self._completed_episodes = []
        return episodes

    def _update_seen_orders(self, battle: Any) -> None:
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))

        if battle_tag not in self._self_slot_order:
            team = getattr(battle, "team", {}) or {}
            self._self_slot_order[battle_tag] = list(team.keys())[:6]

        seen = self._opponent_seen_order.setdefault(battle_tag, [])

        active = getattr(battle, "opponent_active_pokemon", None)
        if active is not None:
            active_id = self._try_identifier(active, getattr(battle, "opponent_role", "p2"))
            if active_id and active_id not in seen:
                seen.append(active_id)

        opponent_team = getattr(battle, "opponent_team", {}) or {}
        for opp_id in opponent_team.keys():
            if opp_id not in seen:
                seen.append(opp_id)

    def _update_turn_history(self, battle: Any) -> None:
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))
        current_turn = int(getattr(battle, "turn", 0) or 0)
        completed_turn = self._history_completed_turn.get(battle_tag, 0)
        live_turn = self._history_live_turn.get(battle_tag, current_turn)
        live_offset = self._history_live_offset.get(battle_tag, 0)

        observations = getattr(battle, "observations", {}) or {}
        for turn_no in range(completed_turn + 1, current_turn):
            turn_obs = observations.get(turn_no)
            events = list(getattr(turn_obs, "events", []) or [])
            start_idx = live_offset if turn_no == live_turn else 0
            self._append_history_tokens(
                battle_tag,
                history_tokens_from_events(
                    events[start_idx:],
                    player_role=getattr(battle, "player_role", "p1"),
                    opponent_role=getattr(battle, "opponent_role", "p2"),
                ),
            )
            completed_turn = turn_no

        current_obs = getattr(battle, "_current_observation", None)
        current_events = list(getattr(current_obs, "events", []) or [])
        if current_turn != live_turn:
            live_turn = current_turn
            live_offset = 0

        if len(current_events) > live_offset:
            self._append_history_tokens(
                battle_tag,
                history_tokens_from_events(
                    current_events[live_offset:],
                    player_role=getattr(battle, "player_role", "p1"),
                    opponent_role=getattr(battle, "opponent_role", "p2"),
                ),
            )
            live_offset = len(current_events)

        self._history_completed_turn[battle_tag] = completed_turn
        self._history_live_turn[battle_tag] = live_turn
        self._history_live_offset[battle_tag] = live_offset

    def _append_history_tokens(self, battle_tag: str, new_tokens: Iterable[str]) -> None:
        history = self._history_tokens.setdefault(battle_tag, deque(maxlen=HISTORY_WINDOW))
        for token in new_tokens:
            if history and history[-1] == token:
                continue
            history.append(token)

    @staticmethod
    def _try_identifier(pokemon: Any, role: str) -> Optional[str]:
        try:
            return pokemon.identifier(role)
        except Exception:
            species = getattr(pokemon, "species", None)
            return str(species) if species is not None else None

    def _battle_fainted_counts(self, battle: Any) -> Tuple[int, int]:
        self_team = getattr(battle, "team", {}) or {}
        opp_team = getattr(battle, "opponent_team", {}) or {}
        return self._side_fainted_count(self_team), self._side_fainted_count(opp_team)

    @staticmethod
    def _side_fainted_count(team: Mapping[str, Any]) -> int:
        return sum(bool(getattr(mon, "fainted", False)) for mon in team.values())


class SimpleHeuristicImitationPlayer(TransformerPlayer, SimpleHeuristicsPlayer):
    """Teacher player that records structured observations and heuristic actions."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["collect_trajectories"] = False
        super().__init__(*args, **kwargs)
        self._completed_samples: List[Dict[str, Any]] = []
        self._skipped_samples = 0

    def choose_move(self, battle: Any):  # type: ignore[override]
        if self.max_turns_before_forfeit is not None:
            turn = int(getattr(battle, "turn", 0) or 0)
            if turn > self.max_turns_before_forfeit:
                return ForfeitBattleOrder()

        model_input = self.build_model_input(battle)
        self_slots = list(model_input["self_slots"])
        obs_tensor = model_input["obs_tensor"]
        legal = model_input["legal_mask"]

        order = SimpleHeuristicsPlayer.choose_move(self, battle)
        action_idx = order_to_action_index(
            self,
            battle,
            order,
            team_slot_ids=self_slots,
            legal_mask_tensor=legal,
        )
        if action_idx is None:
            self._skipped_samples += 1
        else:
            self._completed_samples.append(
                {
                    "obs": obs_tensor.cpu(),
                    "legal_mask": legal.cpu(),
                    "action_index": int(action_idx),
                }
            )
        return order

    def _battle_finished_callback(self, battle: Any):
        battle_tag = str(getattr(battle, "battle_tag", "unknown"))
        self._cleanup_battle_state(battle_tag)

    def pop_completed_samples(self) -> List[Dict[str, Any]]:
        samples = self._completed_samples
        self._completed_samples = []
        return samples

    def pop_skipped_samples(self) -> int:
        skipped = int(self._skipped_samples)
        self._skipped_samples = 0
        return skipped
