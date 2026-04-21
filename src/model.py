"""Structured-observation Transformer policy/value model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .policy import ACTION_DIM


@dataclass
class TransformerConfig:
    """Configuration for the structured-observation Transformer."""

    obs_dim: int
    obs_meta: Dict[str, Any]
    d_model: int = 1024
    nhead: int = 8
    num_layers: int = 2
    ff_dim: int = 4096
    dropout: float = 0.0
    use_twohot_value: bool = True
    v_min: float = -1.6
    v_max: float = 1.6
    v_bins: int = 51


class ObservationUnpacker(nn.Module):
    """Slice a flat observation vector back into structured tensors."""

    def __init__(self, meta: Mapping[str, Any]):
        super().__init__()
        self.meta = dict(meta)
        self.offsets = dict(meta["offsets"])
        self.n_pokemon_slots = int(meta["n_pokemon_slots"])
        self.n_move_slots = int(meta["n_move_slots"])
        self.n_ability_slots = int(meta["n_ability_slots"])
        self.dim_pokemon_body = int(meta["dim_pokemon_body"])
        self.dim_single_move = int(meta["dim_single_move_scalars"])
        self.dim_transition_scalars = int(meta["dim_transition_scalars"])

    def forward(self, obs_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        chunks = {
            name: obs_flat[:, start:end]
            for name, (start, end) in self.offsets.items()
        }
        chunks["pokemon_body"] = chunks["pokemon_body"].reshape(-1, self.n_pokemon_slots, self.dim_pokemon_body)
        chunks["pokemon_ids"] = chunks["pokemon_ids"].reshape(-1, self.n_pokemon_slots, 2).long()
        chunks["ability_ids"] = chunks["ability_ids"].reshape(-1, self.n_pokemon_slots, self.n_ability_slots).long()
        chunks["move_ids"] = chunks["move_ids"].reshape(-1, self.n_pokemon_slots, self.n_move_slots).long()
        chunks["move_scalars"] = chunks["move_scalars"].reshape(
            -1,
            self.n_pokemon_slots,
            self.n_move_slots,
            self.dim_single_move,
        )
        chunks["transition_move_ids"] = chunks["transition_move_ids"].reshape(-1, 2).long()
        chunks["transition_scalars"] = chunks["transition_scalars"].reshape(-1, self.dim_transition_scalars)
        return chunks


class TransformerPolicyValueNet(nn.Module):
    """ps-ppo-style entity-token Transformer with decision-token readout."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.meta = dict(config.obs_meta)
        self.feature_map = dict(self.meta["feature_map"])
        self.n_pokemon_slots = int(self.meta["n_pokemon_slots"])
        self.n_move_slots = int(self.meta["n_move_slots"])
        self.n_ability_slots = int(self.meta["n_ability_slots"])
        self.total_tokens = 2 + 1 + self.n_pokemon_slots

        self.emb_dims = {
            "pokemon": 96,
            "item": 96,
            "ability": 96,
            "move": 96,
        }
        self.bank_dims = {
            "val_100": 64,
            "stat": 64,
            "power": 64,
        }
        self.bank_ranges = {
            "val_100": 101,
            "stat": 800,
            "power": 251,
        }
        self.move_vec_dim = 128
        self.ability_vec_dim = 128
        self.ff_expansion = max(1.0, float(config.ff_dim) / float(max(1, config.d_model)))

        self.unpacker = ObservationUnpacker(self.meta)

        self.pokemon_id_emb = nn.Embedding(int(self.meta["vocab_pokemon"]), self.emb_dims["pokemon"])
        self.item_emb = nn.Embedding(int(self.meta["vocab_item"]), self.emb_dims["item"])
        self.ability_emb = nn.Embedding(int(self.meta["vocab_ability"]), self.emb_dims["ability"])
        self.move_emb = nn.Embedding(int(self.meta["vocab_move"]), self.emb_dims["move"])

        self.val_100_emb = nn.Embedding(self.bank_ranges["val_100"], self.bank_dims["val_100"])
        self.stat_emb = nn.Embedding(self.bank_ranges["stat"], self.bank_dims["stat"])
        self.power_emb = nn.Embedding(self.bank_ranges["power"], self.bank_dims["power"])

        move_in_dim = self._calc_move_in_dim()
        ability_in_dim = self.emb_dims["ability"] * self.n_ability_slots
        pokemon_in_dim = self._calc_pokemon_in_dim()
        field_in_dim = (
            self.bank_dims["val_100"]
            + (int(self.meta["dim_global_scalars"]) - 1)
            + (self.emb_dims["move"] * 2)
            + int(self.meta["dim_transition_scalars"])
        )

        self.move_net = self._build_subnet(move_in_dim, self.move_vec_dim)
        self.ability_net = self._build_subnet(ability_in_dim, self.ability_vec_dim)
        self.pokemon_net = self._build_subnet(pokemon_in_dim, int(config.d_model))
        self.field_net = self._build_subnet(field_in_dim, int(config.d_model))

        self.actor_token = nn.Parameter(torch.randn(1, 1, int(config.d_model)))
        self.critic_token = nn.Parameter(torch.randn(1, 1, int(config.d_model)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.d_model),
            nhead=int(config.nhead),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.num_layers))

        self.readout_mha = nn.MultiheadAttention(
            int(config.d_model),
            int(config.nhead),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.readout_norm_attn = nn.LayerNorm(int(config.d_model))
        self.readout_norm_ff = nn.LayerNorm(int(config.d_model))
        self.readout_net = nn.Sequential(
            nn.Linear(int(config.d_model), int(config.ff_dim)),
            nn.GELU(),
            nn.Linear(int(config.ff_dim), int(config.d_model)),
        )

        self.policy_head = nn.Linear(int(config.d_model), ACTION_DIM)
        self.value_head = nn.Linear(int(config.d_model), int(config.v_bins))

        self.register_buffer("attn_mask", self._build_attention_mask(), persistent=False)
        self.register_buffer(
            "value_support",
            torch.linspace(float(config.v_min), float(config.v_max), int(config.v_bins)),
            persistent=False,
        )

        self._reset_parameters()

    def _calc_move_in_dim(self) -> int:
        move_map = self.feature_map["move"]
        onehot_raw = move_map["onehots_raw"]
        type_raw = move_map["type_raw"]
        onehot_len = int(type_raw[1]) - int(onehot_raw[0])
        return (
            self.emb_dims["move"]
            + (self.bank_dims["val_100"] * 2)
            + self.bank_dims["power"]
            + onehot_len
        )

    def _calc_pokemon_in_dim(self) -> int:
        body_map = self.feature_map["body"]
        raw_body_slice_len = (
            int(body_map["boosts_raw"][1])
            - int(body_map["boosts_raw"][0])
            + (int(self.meta["dim_pokemon_body"]) - int(body_map["flags_raw"][0]))
        )
        return (
            self.emb_dims["pokemon"]
            + self.emb_dims["item"]
            + (self.bank_dims["val_100"] * 2)
            + (self.bank_dims["stat"] * 8)
            + self.ability_vec_dim
            + (self.n_move_slots * self.move_vec_dim)
            + raw_body_slice_len
        )

    def _build_subnet(self, in_dim: int, out_dim: int) -> nn.Sequential:
        hidden_dim = int(out_dim * self.ff_expansion)
        return nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def _build_attention_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.total_tokens, self.total_tokens), dtype=torch.float32)
        mask[2:, 0:2] = float("-inf")
        mask[0, 1] = float("-inf")
        mask[1, 0] = float("-inf")
        return mask

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.actor_token, mean=0.0, std=0.02)
        nn.init.normal_(self.critic_token, mean=0.0, std=0.02)

    def _clamp_ids(self, values: torch.Tensor, *, max_id: int) -> torch.Tensor:
        return values.long().clamp_(0, max(0, int(max_id) - 1))

    def forward(
        self,
        obs_flat: torch.Tensor,
        *,
        return_value_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        single = obs_flat.dim() == 1
        if single:
            obs_flat = obs_flat.unsqueeze(0)

        if obs_flat.dim() != 2:
            raise ValueError(f"Expected obs_flat rank 2, got shape {tuple(obs_flat.shape)}")
        if obs_flat.shape[-1] != int(self.config.obs_dim):
            raise ValueError(
                f"Observation width {int(obs_flat.shape[-1])} does not match configured obs_dim {int(self.config.obs_dim)}"
            )

        obs = self.unpacker(obs_flat.float())
        batch_size = obs_flat.shape[0]

        body_map = self.feature_map["body"]
        move_map = self.feature_map["move"]
        global_map = self.feature_map["global"]

        move_ids = self._clamp_ids(obs["move_ids"], max_id=int(self.meta["vocab_move"]))
        ability_ids = self._clamp_ids(obs["ability_ids"], max_id=int(self.meta["vocab_ability"]))
        pokemon_ids = self._clamp_ids(obs["pokemon_ids"], max_id=int(self.meta["vocab_pokemon"]))
        pokemon_item_ids = self._clamp_ids(obs["pokemon_ids"][:, :, 1], max_id=int(self.meta["vocab_item"]))
        transition_move_ids = self._clamp_ids(obs["transition_move_ids"], max_id=int(self.meta["vocab_move"]))

        move_scalars = obs["move_scalars"].float()
        acc_idx = int(move_map["acc_int"])
        pwr_idx = int(move_map["pwr_int"])
        pp_idx = int(move_map["pp_int"])
        onehots_start = int(move_map["onehots_raw"][0])
        onehots_end = int(move_map["type_raw"][1])

        m_combined = torch.cat(
            [
                self.move_emb(move_ids),
                self.val_100_emb(move_scalars[..., acc_idx].long().clamp_(0, 100)),
                self.power_emb(move_scalars[..., pwr_idx].long().clamp_(0, 250)),
                self.val_100_emb(move_scalars[..., pp_idx].long().clamp_(0, 100)),
                move_scalars[..., onehots_start:onehots_end],
            ],
            dim=-1,
        )

        move_vecs = self.move_net(
            m_combined.reshape(batch_size * self.n_pokemon_slots * self.n_move_slots, -1)
        ).reshape(batch_size, self.n_pokemon_slots, -1)

        ability_vecs = self.ability_net(
            self.ability_emb(ability_ids).reshape(batch_size * self.n_pokemon_slots, -1)
        ).reshape(batch_size, self.n_pokemon_slots, -1)

        pokemon_body = obs["pokemon_body"].float()
        hp_index = int(body_map["hp_int"])
        stats_start, stats_end = body_map["stats_int"]
        level_index = int(body_map["level_int"])
        weight_index = int(body_map["weight_int"])

        pokemon_inputs = torch.cat(
            [
                self.pokemon_id_emb(pokemon_ids[:, :, 0]),
                self.item_emb(pokemon_item_ids),
                self.val_100_emb(pokemon_body[:, :, hp_index].long().clamp_(0, 100)),
                self.stat_emb(pokemon_body[:, :, int(stats_start):int(stats_end)].long().clamp_(0, 799)).flatten(2),
                self.val_100_emb(pokemon_body[:, :, level_index].long().clamp_(0, 100)),
                self.stat_emb(
                    pokemon_body[:, :, weight_index:weight_index + 2].long().clamp_(0, 799)
                ).flatten(2),
                ability_vecs,
                move_vecs,
                pokemon_body[:, :, int(body_map["boosts_raw"][0]):int(body_map["boosts_raw"][1])],
                pokemon_body[:, :, int(body_map["flags_raw"][0]):],
            ],
            dim=-1,
        )
        pokemon_tokens = self.pokemon_net(
            pokemon_inputs.reshape(batch_size * self.n_pokemon_slots, -1)
        ).reshape(batch_size, self.n_pokemon_slots, int(self.config.d_model))

        turn_idx = int(global_map["turn_int"])
        remainder_start = int(global_map["remainder_raw"][0])
        turn_emb = self.val_100_emb((obs["global_scalars"][:, turn_idx] * 100.0).long().clamp_(0, 100))
        field_in = torch.cat(
            [
                turn_emb,
                obs["global_scalars"][:, remainder_start:].float(),
                self.move_emb(transition_move_ids).reshape(batch_size, -1),
                obs["transition_scalars"].float(),
            ],
            dim=-1,
        )
        field_token = self.field_net(field_in).unsqueeze(1)

        sequence = torch.cat(
            [
                self.actor_token.expand(batch_size, -1, -1),
                self.critic_token.expand(batch_size, -1, -1),
                field_token,
                pokemon_tokens,
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, mask=self.attn_mask)

        q = self.readout_norm_attn(encoded[:, 0:2, :])
        kv = self.readout_norm_attn(encoded)
        attended, _ = self.readout_mha(query=q, key=kv, value=kv, attn_mask=self.attn_mask[0:2, :])
        q_out = encoded[:, 0:2, :] + attended
        q_out = q_out + self.readout_net(self.readout_norm_ff(q_out))

        policy_logits = self.policy_head(q_out[:, 0, :])
        value_logits = self.value_head(q_out[:, 1, :])
        value = (torch.softmax(value_logits, dim=-1) * self.value_support).sum(dim=-1, keepdim=True)

        if single:
            if return_value_logits:
                return policy_logits[0], value[0], value_logits[0]
            return policy_logits[0], value[0]
        if return_value_logits:
            return policy_logits, value, value_logits
        return policy_logits, value
