"""Checkpoint helpers for structured-observation Transformer training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .model import TransformerConfig, TransformerPolicyValueNet
from .policy import ACTION_DIM
from .vocab import ObservationVocabulary

ENCODER_TYPE = "structured_obs_transformer_v6_psppo_schema_type20_global_align_twohot_value"


def build_model_from_vocab(
    vocab: ObservationVocabulary,
    *,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
    use_twohot_value: bool,
    v_min: float,
    v_max: float,
    v_bins: int,
    device: str,
) -> TransformerPolicyValueNet:
    meta = vocab.schema_meta()
    cfg = TransformerConfig(
        obs_dim=int(meta["obs_dim"]),
        obs_meta=meta,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        use_twohot_value=bool(use_twohot_value),
        v_min=float(v_min),
        v_max=float(v_max),
        v_bins=int(v_bins),
    )
    return TransformerPolicyValueNet(cfg).to(device)


def save_checkpoint(
    *,
    path: str | Path,
    model: TransformerPolicyValueNet,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    iteration: int,
    vocab_path: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "encoder_type": ENCODER_TYPE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "iteration": int(iteration),
        "obs_dim": int(model.config.obs_dim),
        "vocab_path": str(vocab_path),
        "model_config": {
            "obs_dim": int(model.config.obs_dim),
            "obs_meta": dict(model.config.obs_meta),
            "d_model": int(model.config.d_model),
            "nhead": int(model.config.nhead),
            "num_layers": int(model.config.num_layers),
            "ff_dim": int(model.config.ff_dim),
            "dropout": float(model.config.dropout),
            "use_twohot_value": bool(model.config.use_twohot_value),
            "v_min": float(model.config.v_min),
            "v_max": float(model.config.v_max),
            "v_bins": int(model.config.v_bins),
            "action_dim": int(ACTION_DIM),
        },
        "extra": dict(extra or {}),
    }
    torch.save(payload, target)


def save_checkpoint_payload(*, path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), target)


def load_checkpoint(path: str | Path, device: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=device)
    return payload


def model_from_checkpoint_payload(payload: Mapping[str, Any], device: str) -> TransformerPolicyValueNet:
    encoder_type = payload.get("encoder_type")
    if encoder_type != ENCODER_TYPE:
        raise ValueError(
            "Checkpoint encoder_type is incompatible with the current structured observation model. "
            "Start from a fresh checkpoint directory after the ps-ppo-style refactor."
        )

    cfg_raw = payload["model_config"]
    action_dim = int(cfg_raw.get("action_dim", 0))
    if action_dim != ACTION_DIM:
        raise ValueError(
            f"Checkpoint action_dim={action_dim} is incompatible with current code action_dim={ACTION_DIM}. "
            "Start a fresh checkpoint directory."
        )
    if "obs_meta" not in cfg_raw:
        raise ValueError("Checkpoint is missing structured observation metadata.")

    cfg = TransformerConfig(
        obs_dim=int(cfg_raw["obs_dim"]),
        obs_meta=dict(cfg_raw["obs_meta"]),
        d_model=int(cfg_raw["d_model"]),
        nhead=int(cfg_raw["nhead"]),
        num_layers=int(cfg_raw["num_layers"]),
        ff_dim=int(cfg_raw["ff_dim"]),
        dropout=float(cfg_raw["dropout"]),
        use_twohot_value=bool(cfg_raw.get("use_twohot_value", True)),
        v_min=float(cfg_raw.get("v_min", -1.6)),
        v_max=float(cfg_raw.get("v_max", 1.6)),
        v_bins=int(cfg_raw.get("v_bins", 51)),
    )
    model = TransformerPolicyValueNet(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model
