"""Training utilities for policy/value transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .replay import ReplayBuffer


@dataclass
class TrainConfig:
    steps: int = 100
    batch_size: int = 4096
    epochs: int = 3
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    amp: bool = True
    ppo_clip_epsilon: float = 0.2
    ppo_value_clip: float = 0.0
    target_kl: float = 0.02
    target_kl_factor: float = 1.5
    min_steps_before_early_stop: int = 10
    normalize_advantages: bool = True
    policy_temperature: float = 1.0
    use_twohot_value: bool = True
    v_min: float = -1.6
    v_max: float = 1.6
    v_bins: int = 51
    lr_warmup_steps: int = 1000
    lr_hold_steps: int = 20000
    lr_total_steps: int = 500000
    lr_backbone_mult: float = 1.0
    lr_pi_mult: float = 1.0
    lr_v_mult: float = 1.0
    imitation_label_smoothing: float = 0.1


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    lr_backbone_mult: float = 1.0,
    lr_pi_mult: float = 1.0,
    lr_v_mult: float = 1.0,
) -> torch.optim.Optimizer:
    trunk_decay = []
    trunk_no_decay = []
    subnet_decay = []
    subnet_no_decay = []
    pi_decay = []
    pi_no_decay = []
    v_decay = []
    v_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = ("_emb" in name) or ("norm" in name) or name.endswith(".bias") or (param.ndim == 1)

        if (
            "policy_context" in name
            or "policy_head" in name
        ):
            (pi_no_decay if no_decay else pi_decay).append(param)
        elif "value_head" in name or "critic_token" in name:
            (v_no_decay if no_decay else v_decay).append(param)
        elif "encoder" in name or "actor_token" in name:
            (trunk_no_decay if no_decay else trunk_decay).append(param)
        else:
            (subnet_no_decay if no_decay else subnet_decay).append(param)

    base_lr = float(lr)
    lr_back = base_lr * float(lr_backbone_mult)
    lr_pi = base_lr * float(lr_pi_mult)
    lr_v = base_lr * float(lr_v_mult)
    wd_val = float(weight_decay)

    param_groups = [
        {"params": trunk_decay, "lr": lr_back, "weight_decay": wd_val, "name": "trunk_wd"},
        {"params": trunk_no_decay, "lr": lr_back, "weight_decay": 0.0, "name": "trunk_stable"},
        {"params": subnet_decay, "lr": lr_back, "weight_decay": wd_val, "name": "subnets_wd"},
        {"params": subnet_no_decay, "lr": lr_back, "weight_decay": 0.0, "name": "subnets_stable"},
        {"params": pi_decay, "lr": lr_pi, "weight_decay": wd_val, "name": "pi_wd"},
        {"params": pi_no_decay, "lr": lr_pi, "weight_decay": 0.0, "name": "pi_stable"},
        {"params": v_decay, "lr": lr_v, "weight_decay": wd_val, "name": "v_wd"},
        {"params": v_no_decay, "lr": lr_v, "weight_decay": 0.0, "name": "v_stable"},
    ]
    return torch.optim.AdamW([group for group in param_groups if group["params"]], eps=1e-5)


def lr_schedule_factor(
    step: int,
    *,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
) -> float:
    if warmup_steps > 0 and step < int(warmup_steps):
        return float(step + 1) / float(warmup_steps)
    if step < int(warmup_steps) + int(hold_steps):
        return 1.0

    anneal_start = int(warmup_steps) + int(hold_steps)
    progress = min(1.0, max(0.0, float(step - anneal_start) / float(max(1, int(total_steps) - anneal_start))))
    return 1.0 / ((8.0 * progress + 1.0) ** 1.5)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_schedule_factor(
            int(step),
            warmup_steps=int(warmup_steps),
            hold_steps=int(hold_steps),
            total_steps=int(total_steps),
        ),
    )


def _zero_metrics() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "policy_entropy": 0.0,
        "grad_norm": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "accuracy": 0.0,
        "steps": 0.0,
    }


def twohot_targets(x: torch.Tensor, *, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    clipped = torch.clamp(x.float(), min=float(v_min), max=float(v_max))
    scale = float(v_bins - 1) / max(1e-8, float(v_max - v_min))
    positions = (clipped - float(v_min)) * scale
    idx0 = torch.floor(positions).long()
    idx1 = torch.clamp(idx0 + 1, max=int(v_bins) - 1)
    weight1 = (positions - idx0.float()).clamp(min=0.0, max=1.0)
    weight0 = 1.0 - weight1

    targets = torch.zeros((clipped.shape[0], int(v_bins)), device=clipped.device, dtype=torch.float32)
    targets.scatter_add_(1, idx0.unsqueeze(-1), weight0.unsqueeze(-1))
    targets.scatter_add_(1, idx1.unsqueeze(-1), weight1.unsqueeze(-1))
    return targets


def dist_value_loss(value_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    return -(target_dist * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()


def masked_fill_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def _did_optimizer_step_with_scaler(scaler: torch.amp.GradScaler, optimizer: torch.optim.Optimizer) -> bool:
    prev_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= prev_scale


def _materialize_rollout_source(
    rollout_or_replay: ReplayBuffer | Mapping[str, torch.Tensor],
    *,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    if isinstance(rollout_or_replay, ReplayBuffer):
        return rollout_or_replay.materialize(device=device)
    return {
        key: value.to(device)
        for key, value in rollout_or_replay.items()
    }


def _ppo_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    config: TrainConfig,
) -> Dict[str, torch.Tensor]:
    obs = batch["obs"].float()
    legal_mask = batch["legal_mask"].bool()
    actions = batch["action_index"].long()
    returns = batch["outcome_z"].float()
    advantages = batch["advantage"].float()
    old_log_prob = batch["old_log_prob"].float()
    old_value = batch["old_value"].float()

    policy_logits, value, value_logits = model(obs, return_value_logits=True)
    value = value.squeeze(-1)

    masked_logits = policy_logits.masked_fill(~legal_mask, masked_fill_value(policy_logits))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    action_log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = torch.exp(action_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - float(config.ppo_clip_epsilon),
        1.0 + float(config.ppo_clip_epsilon),
    )
    ppo_unclipped = ratio * advantages
    ppo_clipped = clipped_ratio * advantages
    policy_loss = -torch.min(ppo_unclipped, ppo_clipped).mean()

    value_pred = value
    if config.use_twohot_value:
        target_dist = twohot_targets(
            returns,
            v_min=float(config.v_min),
            v_max=float(config.v_max),
            v_bins=int(config.v_bins),
        )
        value_loss = dist_value_loss(value_logits, target_dist)
    elif config.ppo_value_clip > 0:
        value_pred_clipped = old_value + torch.clamp(
            value_pred - old_value,
            -float(config.ppo_value_clip),
            float(config.ppo_value_clip),
        )
        value_loss_unclipped = (value_pred - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = 0.5 * F.mse_loss(value_pred, returns)

    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    with torch.no_grad():
        approx_kl = (old_log_prob - action_log_prob).mean()
        clip_frac = (torch.abs(ratio - 1.0) > float(config.ppo_clip_epsilon)).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }


def _imitation_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    config: TrainConfig,
) -> Dict[str, torch.Tensor]:
    obs = batch["obs"].float()
    legal_mask = batch["legal_mask"].bool()
    actions = batch["action_index"].long()

    policy_logits, _ = model(obs)
    masked_logits = policy_logits.masked_fill(~legal_mask, masked_fill_value(policy_logits))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    smoothing = float(max(0.0, config.imitation_label_smoothing))
    if smoothing <= 0.0:
        loss = F.nll_loss(log_probs, actions)
    else:
        legal_counts = legal_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(log_probs.dtype)
        target_dist = legal_mask.to(log_probs.dtype) * (smoothing / legal_counts)
        target_dist.scatter_add_(
            1,
            actions.unsqueeze(-1),
            torch.full((actions.shape[0], 1), 1.0 - smoothing, device=log_probs.device, dtype=log_probs.dtype),
        )
        loss = -(target_dist * log_probs).sum(dim=-1).mean()

    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    predictions = masked_logits.argmax(dim=-1)
    accuracy = (predictions == actions).float().mean()
    zero = loss.detach().new_zeros(())

    return {
        "loss": loss,
        "policy_loss": loss,
        "value_loss": zero,
        "entropy": entropy,
        "approx_kl": zero,
        "clip_frac": zero,
        "accuracy": accuracy,
    }


def train_on_replay(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    device: str,
    config: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    progress_interval: int = 0,
    progress_prefix: str = "",
) -> Dict[str, float]:
    """Run gradient updates from replay samples."""

    if config.steps < 1:
        return _zero_metrics()

    model.train()

    use_cuda_amp = bool(config.amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    avg_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_entropy = 0.0
    avg_grad = 0.0
    avg_kl = 0.0
    avg_clip_frac = 0.0
    completed_steps = 0

    for step_idx in range(config.steps):
        batch = replay.sample_batch(config.batch_size, device=device)
        advantages = batch["advantage"].float()

        if config.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            batch["advantage"] = (advantages - adv_mean) / (adv_std + 1e-8)

        optimizer.zero_grad(set_to_none=True)

        autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=use_cuda_amp):
            loss_terms = _ppo_loss(model, batch, config=config)
            loss = loss_terms["loss"]
            policy_loss = loss_terms["policy_loss"]
            value_loss = loss_terms["value_loss"]
            entropy = loss_terms["entropy"]
            approx_kl = loss_terms["approx_kl"]
            clip_frac = loss_terms["clip_frac"]

        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer_stepped = _did_optimizer_step_with_scaler(scaler, optimizer)
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_stepped = True
        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        avg_loss += float(loss.detach().cpu())
        avg_policy_loss += float(policy_loss.detach().cpu())
        avg_value_loss += float(value_loss.detach().cpu())
        avg_entropy += float(entropy.detach().cpu())
        avg_grad += float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
        avg_kl += float(approx_kl.detach().cpu())
        avg_clip_frac += float(clip_frac.detach().cpu())
        completed_steps += 1

        if progress_interval > 0 and ((step_idx + 1) % progress_interval == 0 or (step_idx + 1) == config.steps):
            print(
                f"{progress_prefix}train_step={step_idx + 1}/{config.steps} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"policy={float(policy_loss.detach().cpu()):.4f} "
                f"value={float(value_loss.detach().cpu()):.4f} "
                f"entropy={float(entropy.detach().cpu()):.4f} "
                f"kl={float(approx_kl.detach().cpu()):.4f} "
                f"clip_frac={float(clip_frac.detach().cpu()):.3f}",
                flush=True,
            )

        if (
            float(config.target_kl) > 0.0
            and completed_steps >= int(config.min_steps_before_early_stop)
            and (avg_kl / float(completed_steps)) > (float(config.target_kl) * float(config.target_kl_factor))
        ):
            print(
                f"{progress_prefix}early_stop=train_step={completed_steps}/{config.steps} "
                f"avg_kl={avg_kl / float(completed_steps):.4f} "
                f"target_kl={float(config.target_kl):.4f}",
                flush=True,
            )
            break

    denom = float(max(1, completed_steps))
    return {
        "loss": avg_loss / denom,
        "policy_loss": avg_policy_loss / denom,
        "value_loss": avg_value_loss / denom,
        "policy_entropy": avg_entropy / denom,
        "grad_norm": avg_grad / denom,
        "approx_kl": avg_kl / denom,
        "clip_frac": avg_clip_frac / denom,
        "accuracy": 0.0,
        "steps": float(completed_steps),
    }


def train_on_frozen_rollout(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    device: str,
    config: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    progress_interval: int = 0,
    progress_prefix: str = "",
) -> Dict[str, float]:
    """Run PPO epochs over one frozen rollout dataset without replacement."""

    if config.steps < 1 or config.epochs < 1:
        return _zero_metrics()

    rollout = _materialize_rollout_source(replay, device="cpu")
    num_samples = int(rollout["obs"].shape[0])
    if num_samples < 1:
        return _zero_metrics()

    if config.normalize_advantages:
        advantages = rollout["advantage"].float()
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        rollout["advantage"] = (advantages - adv_mean) / (adv_std + 1e-8)

    model.train()
    use_cuda_amp = bool(config.amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    avg_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_entropy = 0.0
    avg_grad = 0.0
    avg_kl = 0.0
    avg_clip_frac = 0.0
    completed_steps = 0

    optimizer.zero_grad(set_to_none=True)
    should_stop = False

    for epoch_idx in range(int(config.epochs)):
        permutation = torch.randperm(num_samples)
        epoch_kl_total = 0.0
        epoch_steps = 0
        accum_batches = 0
        accum_loss = 0.0
        accum_policy_loss = 0.0
        accum_value_loss = 0.0
        accum_entropy = 0.0
        accum_kl = 0.0
        accum_clip_frac = 0.0

        microbatch_ranges = range(0, num_samples, int(config.batch_size))
        for microbatch_start in microbatch_ranges:
            microbatch_end = min(num_samples, microbatch_start + int(config.batch_size))
            mb_indices = permutation[microbatch_start:microbatch_end]
            batch = {
                key: value[mb_indices].to(device)
                for key, value in rollout.items()
            }

            with torch.autocast(device_type=autocast_device, enabled=use_cuda_amp):
                loss_terms = _ppo_loss(model, batch, config=config)
                scaled_loss = loss_terms["loss"] / max(1, int(config.grad_accum_steps))

            if use_cuda_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_batches += 1
            accum_loss += float(loss_terms["loss"].detach().cpu())
            accum_policy_loss += float(loss_terms["policy_loss"].detach().cpu())
            accum_value_loss += float(loss_terms["value_loss"].detach().cpu())
            accum_entropy += float(loss_terms["entropy"].detach().cpu())
            accum_kl += float(loss_terms["approx_kl"].detach().cpu())
            accum_clip_frac += float(loss_terms["clip_frac"].detach().cpu())

            is_last_microbatch = microbatch_end >= num_samples
            should_step = accum_batches >= int(config.grad_accum_steps) or is_last_microbatch
            if not should_step:
                continue

            if use_cuda_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer_stepped = _did_optimizer_step_with_scaler(scaler, optimizer)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            denom = float(max(1, accum_batches))
            step_loss = accum_loss / denom
            step_policy_loss = accum_policy_loss / denom
            step_value_loss = accum_value_loss / denom
            step_entropy = accum_entropy / denom
            step_kl = accum_kl / denom
            step_clip_frac = accum_clip_frac / denom

            avg_loss += step_loss
            avg_policy_loss += step_policy_loss
            avg_value_loss += step_value_loss
            avg_entropy += step_entropy
            avg_grad += float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            avg_kl += step_kl
            avg_clip_frac += step_clip_frac
            completed_steps += 1
            epoch_kl_total += step_kl
            epoch_steps += 1

            if progress_interval > 0 and (
                (completed_steps % progress_interval) == 0 or completed_steps == int(config.steps)
            ):
                print(
                    f"{progress_prefix}train_step={completed_steps}/{config.steps} "
                    f"loss={step_loss:.4f} "
                    f"policy={step_policy_loss:.4f} "
                    f"value={step_value_loss:.4f} "
                    f"entropy={step_entropy:.4f} "
                    f"kl={step_kl:.4f} "
                    f"clip_frac={step_clip_frac:.3f}",
                    flush=True,
                )

            accum_batches = 0
            accum_loss = 0.0
            accum_policy_loss = 0.0
            accum_value_loss = 0.0
            accum_entropy = 0.0
            accum_kl = 0.0
            accum_clip_frac = 0.0

            if (
                float(config.target_kl) > 0.0
                and completed_steps >= int(config.min_steps_before_early_stop)
                and (epoch_kl_total / float(max(1, epoch_steps)))
                > (float(config.target_kl) * float(config.target_kl_factor))
            ):
                print(
                    f"{progress_prefix}early_stop=train_step={completed_steps}/{config.steps} "
                    f"epoch_kl={epoch_kl_total / float(max(1, epoch_steps)):.4f} "
                    f"target_kl={float(config.target_kl):.4f}",
                    flush=True,
                )
                should_stop = True
                break

            if completed_steps >= int(config.steps):
                should_stop = True
                break

        if should_stop:
            break

    denom = float(max(1, completed_steps))
    return {
        "loss": avg_loss / denom,
        "policy_loss": avg_policy_loss / denom,
        "value_loss": avg_value_loss / denom,
        "policy_entropy": avg_entropy / denom,
        "grad_norm": avg_grad / denom,
        "approx_kl": avg_kl / denom,
        "clip_frac": avg_clip_frac / denom,
        "accuracy": 0.0,
        "steps": float(completed_steps),
    }


def train_on_imitation_rollout(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer | Mapping[str, torch.Tensor],
    *,
    device: str,
    config: TrainConfig,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    progress_interval: int = 0,
    progress_prefix: str = "",
) -> Dict[str, float]:
    """Run supervised imitation epochs over one frozen demonstration dataset."""

    if config.steps < 1 or config.epochs < 1:
        return _zero_metrics()

    rollout = _materialize_rollout_source(replay, device="cpu")
    num_samples = int(rollout["obs"].shape[0])
    if num_samples < 1:
        return _zero_metrics()

    model.train()
    use_cuda_amp = bool(config.amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    avg_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_entropy = 0.0
    avg_grad = 0.0
    avg_kl = 0.0
    avg_clip_frac = 0.0
    avg_accuracy = 0.0
    completed_steps = 0

    optimizer.zero_grad(set_to_none=True)
    should_stop = False

    for _epoch_idx in range(int(config.epochs)):
        permutation = torch.randperm(num_samples)
        accum_batches = 0
        accum_loss = 0.0
        accum_policy_loss = 0.0
        accum_value_loss = 0.0
        accum_entropy = 0.0
        accum_accuracy = 0.0

        microbatch_ranges = range(0, num_samples, int(config.batch_size))
        for microbatch_start in microbatch_ranges:
            microbatch_end = min(num_samples, microbatch_start + int(config.batch_size))
            mb_indices = permutation[microbatch_start:microbatch_end]
            batch = {
                key: value[mb_indices].to(device)
                for key, value in rollout.items()
            }

            with torch.autocast(device_type=autocast_device, enabled=use_cuda_amp):
                loss_terms = _imitation_loss(model, batch, config=config)
                scaled_loss = loss_terms["loss"] / max(1, int(config.grad_accum_steps))

            if use_cuda_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_batches += 1
            accum_loss += float(loss_terms["loss"].detach().cpu())
            accum_policy_loss += float(loss_terms["policy_loss"].detach().cpu())
            accum_value_loss += float(loss_terms["value_loss"].detach().cpu())
            accum_entropy += float(loss_terms["entropy"].detach().cpu())
            accum_accuracy += float(loss_terms["accuracy"].detach().cpu())

            is_last_microbatch = microbatch_end >= num_samples
            should_step = accum_batches >= int(config.grad_accum_steps) or is_last_microbatch
            if not should_step:
                continue

            if use_cuda_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer_stepped = _did_optimizer_step_with_scaler(scaler, optimizer)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            denom = float(max(1, accum_batches))
            step_loss = accum_loss / denom
            step_policy_loss = accum_policy_loss / denom
            step_value_loss = accum_value_loss / denom
            step_entropy = accum_entropy / denom
            step_accuracy = accum_accuracy / denom

            avg_loss += step_loss
            avg_policy_loss += step_policy_loss
            avg_value_loss += step_value_loss
            avg_entropy += step_entropy
            avg_grad += float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            avg_accuracy += step_accuracy
            completed_steps += 1

            if progress_interval > 0 and (
                (completed_steps % progress_interval) == 0 or completed_steps == int(config.steps)
            ):
                print(
                    f"{progress_prefix}train_step={completed_steps}/{config.steps} "
                    f"loss={step_loss:.4f} "
                    f"policy={step_policy_loss:.4f} "
                    f"value={step_value_loss:.4f} "
                    f"entropy={step_entropy:.4f} "
                    f"acc={step_accuracy:.3f}",
                    flush=True,
                )

            accum_batches = 0
            accum_loss = 0.0
            accum_policy_loss = 0.0
            accum_value_loss = 0.0
            accum_entropy = 0.0
            accum_accuracy = 0.0

            if completed_steps >= int(config.steps):
                should_stop = True
                break

        if should_stop:
            break

    denom = float(max(1, completed_steps))
    return {
        "loss": avg_loss / denom,
        "policy_loss": avg_policy_loss / denom,
        "value_loss": avg_value_loss / denom,
        "policy_entropy": avg_entropy / denom,
        "grad_norm": avg_grad / denom,
        "approx_kl": avg_kl / denom,
        "clip_frac": avg_clip_frac / denom,
        "accuracy": avg_accuracy / denom,
        "steps": float(completed_steps),
    }


@torch.no_grad()
def evaluate_imitation_rollout(
    model: torch.nn.Module,
    replay: ReplayBuffer | Mapping[str, torch.Tensor],
    *,
    device: str,
    batch_size: int,
    config: TrainConfig,
) -> Dict[str, float]:
    """Evaluate imitation accuracy on a frozen demonstration dataset without updates."""

    rollout = _materialize_rollout_source(replay, device="cpu")
    num_samples = int(rollout["obs"].shape[0])
    if num_samples < 1:
        return _zero_metrics()

    model.eval()

    avg_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_entropy = 0.0
    avg_accuracy = 0.0
    completed_steps = 0

    for microbatch_start in range(0, num_samples, int(batch_size)):
        microbatch_end = min(num_samples, microbatch_start + int(batch_size))
        batch = {
            key: value[microbatch_start:microbatch_end].to(device)
            for key, value in rollout.items()
        }
        loss_terms = _imitation_loss(model, batch, config=config)

        avg_loss += float(loss_terms["loss"].detach().cpu())
        avg_policy_loss += float(loss_terms["policy_loss"].detach().cpu())
        avg_value_loss += float(loss_terms["value_loss"].detach().cpu())
        avg_entropy += float(loss_terms["entropy"].detach().cpu())
        avg_accuracy += float(loss_terms["accuracy"].detach().cpu())
        completed_steps += 1

    denom = float(max(1, completed_steps))
    return {
        "loss": avg_loss / denom,
        "policy_loss": avg_policy_loss / denom,
        "value_loss": avg_value_loss / denom,
        "policy_entropy": avg_entropy / denom,
        "grad_norm": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "accuracy": avg_accuracy / denom,
        "steps": float(completed_steps),
    }
