"""Ray-actor self-play training loop with ps-ppo-style rollout workers."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from scripts.train_selfplay import (
    _build_server_configurations,
    _check_server,
    _load_or_init,
    _run_selfplay_phase,
    _split_counts,
    annealed_temperature,
    optimizer_steps_for_epochs,
)
from src.checkpoint import ENCODER_TYPE, load_checkpoint, model_from_checkpoint_payload, save_checkpoint_payload
from src.policy import ACTION_DIM
from src.device import resolve_device
from src.eval import EvalConfig, evaluate_models
from src.train import TrainConfig, build_optimizer, build_scheduler, train_on_frozen_rollout
from src.vocab import ObservationVocabulary

try:
    import ray
except ModuleNotFoundError:
    ray = None


ROLLOUT_KEYS = (
    "obs",
    "legal_mask",
    "action_index",
    "outcome_z",
    "old_log_prob",
    "old_value",
    "advantage",
)
FINAL_EVAL_PROMOTE_THRESHOLD = 0.60
WORKER_LOG_LEVELS = {
    "quiet": 0,
    "summary": 1,
    "chunk": 2,
    "pair": 3,
}


def _worker_log_enabled(log_level: str, category: str) -> bool:
    current = WORKER_LOG_LEVELS.get(str(log_level), WORKER_LOG_LEVELS["summary"])
    required = WORKER_LOG_LEVELS.get(str(category), WORKER_LOG_LEVELS["summary"])
    return current >= required


def build_parser() -> argparse.ArgumentParser:
    from scripts.train_selfplay import build_parser as build_base_parser

    parser = build_base_parser()
    parser.description = "Train a Transformer poke-env bot with Ray rollout workers."
    parser.set_defaults(promote_threshold=FINAL_EVAL_PROMOTE_THRESHOLD)
    parser.add_argument(
        "--worker-log-level",
        default="summary",
        choices=tuple(WORKER_LOG_LEVELS.keys()),
        help="Rollout worker log verbosity: quiet, summary, chunk, or pair.",
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help="Number of Ray rollout worker actors to run in parallel.",
    )
    parser.add_argument(
        "--worker-cpus",
        type=float,
        default=1.0,
        help="CPU resources to reserve per rollout worker actor.",
    )
    parser.add_argument(
        "--worker-gpus",
        type=float,
        default=0.0,
        help="GPU fraction to reserve per rollout worker actor.",
    )
    parser.add_argument(
        "--worker-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used inside rollout workers. 'auto' picks cuda only when worker-gpus > 0.",
    )
    parser.add_argument(
        "--learner-cpus",
        type=float,
        default=1.0,
        help="CPU resources to reserve for the learner actor.",
    )
    parser.add_argument(
        "--learner-gpus",
        type=float,
        default=0.0,
        help="GPU resources to reserve for the learner actor. 0 enables automatic allocation when --device=cuda.",
    )
    parser.add_argument(
        "--ray-address",
        default="",
        help="Existing Ray cluster address. Empty string starts a local Ray runtime.",
    )
    parser.add_argument(
        "--ray-namespace",
        default="pokeenv_transformer",
        help="Ray namespace for rollout worker actors.",
    )
    parser.add_argument(
        "--inference-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used by the centralized inference actor. 'auto' follows --device.",
    )
    parser.add_argument(
        "--inference-cpus",
        type=float,
        default=1.0,
        help="CPU resources to reserve for the centralized inference actor.",
    )
    parser.add_argument(
        "--inference-gpus",
        type=float,
        default=0.0,
        help="GPU resources to reserve for the centralized inference actor.",
    )
    parser.add_argument(
        "--inference-batch-wait-ms",
        type=float,
        default=2.0,
        help="How long the centralized inference actor waits to batch rollout requests.",
    )
    parser.add_argument(
        "--inference-max-batch-size",
        type=int,
        default=1024,
        help="Maximum rollout requests per centralized inference batch.",
    )
    return parser


def _model_payload(model: torch.nn.Module, *, vocab_path: str, iteration: int, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    config = model.config
    return {
        "encoder_type": ENCODER_TYPE,
        "model_state_dict": state_dict,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "iteration": int(iteration),
        "obs_dim": int(config.obs_dim),
        "vocab_path": str(vocab_path),
        "model_config": {
            "obs_dim": int(config.obs_dim),
            "obs_meta": dict(config.obs_meta),
            "d_model": int(config.d_model),
            "nhead": int(config.nhead),
            "num_layers": int(config.num_layers),
            "ff_dim": int(config.ff_dim),
            "dropout": float(config.dropout),
            "use_twohot_value": bool(config.use_twohot_value),
            "v_min": float(config.v_min),
            "v_max": float(config.v_max),
            "v_bins": int(config.v_bins),
            "action_dim": int(ACTION_DIM),
        },
        "extra": dict(extra or {}),
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    vocab_path: str,
    iteration: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    config = model.config
    return {
        "encoder_type": ENCODER_TYPE,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "iteration": int(iteration),
        "obs_dim": int(config.obs_dim),
        "vocab_path": str(vocab_path),
        "model_config": {
            "obs_dim": int(config.obs_dim),
            "obs_meta": dict(config.obs_meta),
            "d_model": int(config.d_model),
            "nhead": int(config.nhead),
            "num_layers": int(config.num_layers),
            "ff_dim": int(config.ff_dim),
            "dropout": float(config.dropout),
            "use_twohot_value": bool(config.use_twohot_value),
            "v_min": float(config.v_min),
            "v_max": float(config.v_max),
            "v_bins": int(config.v_bins),
            "action_dim": int(ACTION_DIM),
        },
        "extra": dict(extra or {}),
    }


def pack_episodes(
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    *,
    obs_dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    rows = [row for episode in episodes for row in episode]
    if not rows:
        raise ValueError("Cannot pack an empty episode list.")

    obs = torch.stack([row["obs"].detach().cpu().to(dtype=obs_dtype) for row in rows], dim=0)
    legal_mask = torch.stack([row["legal_mask"].detach().cpu().bool() for row in rows], dim=0)
    action_index = torch.tensor([int(row["action_index"]) for row in rows], dtype=torch.long)
    outcome_z = torch.tensor([float(row["outcome_z"]) for row in rows], dtype=torch.float32)
    old_log_prob = torch.tensor([float(row.get("old_log_prob", 0.0)) for row in rows], dtype=torch.float32)
    old_value = torch.tensor([float(row.get("old_value", 0.0)) for row in rows], dtype=torch.float32)
    advantage = torch.tensor([float(row.get("advantage", row.get("outcome_z", 0.0))) for row in rows], dtype=torch.float32)

    return {
        "obs": obs,
        "legal_mask": legal_mask,
        "action_index": action_index,
        "outcome_z": outcome_z,
        "old_log_prob": old_log_prob,
        "old_value": old_value,
        "advantage": advantage,
    }


def concat_rollouts(chunks: Iterable[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    parts = [chunk for chunk in chunks if int(chunk["obs"].shape[0]) > 0]
    if not parts:
        raise ValueError("No rollout chunks contained any samples.")
    return {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in ROLLOUT_KEYS
    }


def rollout_size(rollout: Mapping[str, torch.Tensor]) -> int:
    return int(rollout["obs"].shape[0])


def summarize_worker_stats(results: Sequence[Tuple[Mapping[str, torch.Tensor], Mapping[str, int], Mapping[str, int]]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    wlt = {"wins": 0, "losses": 0, "ties": 0}
    meta = {"steps": 0, "battles": 0, "episodes": 0}
    for _rollout, stats, info in results:
        wlt["wins"] += int(stats["wins"])
        wlt["losses"] += int(stats["losses"])
        wlt["ties"] += int(stats["ties"])
        meta["steps"] += int(info["steps"])
        meta["battles"] += int(info["battles"])
        meta["episodes"] += int(info["episodes"])
    return wlt, meta


def _resolve_worker_device(args: argparse.Namespace) -> str:
    if str(args.worker_device) == "auto":
        if str(args.device).startswith("cuda") and float(args.worker_gpus) > 0.0:
            return "cuda"
        return "cpu"
    return str(args.worker_device)


def _resolve_inference_device(args: argparse.Namespace) -> str:
    if str(args.inference_device) == "auto":
        return str(args.device)
    return str(args.inference_device)


def _resolve_learner_gpus(args: argparse.Namespace, *, inference_device: str) -> float:
    requested = float(args.learner_gpus)
    if requested > 0.0:
        return requested
    if not str(args.device).startswith("cuda"):
        return 0.0
    if str(inference_device).startswith("cuda"):
        inference_requested = float(args.inference_gpus)
        if inference_requested > 0.0:
            return max(0.0, 1.0 - inference_requested)
        return 0.75
    return 1.0


def _resolve_inference_gpus(args: argparse.Namespace, *, inference_device: str) -> float:
    requested = float(args.inference_gpus)
    if requested > 0.0:
        return requested
    if not str(inference_device).startswith("cuda"):
        return 0.0
    if str(args.device).startswith("cuda"):
        learner_requested = float(args.learner_gpus)
        if learner_requested > 0.0:
            return max(0.0, 1.0 - learner_requested)
        return 0.25
    return 1.0


class RayInferenceClient:
    def __init__(self, *, ray_module: Any, actor_handle: Any, model_key: str, version: int):
        self._ray = ray_module
        self._actor = actor_handle
        self._model_key = str(model_key)
        self._version = int(version)

    def infer(self, *, obs_tensor: torch.Tensor, model_key: str | None = None) -> tuple[torch.Tensor, float]:
        key = str(model_key or self._model_key)
        obs = obs_tensor.detach().cpu().to(dtype=torch.float16)
        policy_logits, value = self._ray.get(self._actor.infer.remote(self._version, key, obs))
        policy_logits = policy_logits.detach().cpu().float().reshape(-1)
        value_pred = float(value.detach().cpu().float().reshape(-1)[0].item())
        return policy_logits, value_pred


def _build_inference_actor(ray_module):
    @ray_module.remote(max_restarts=0, max_concurrency=2048)
    class CentralInferenceActor:
        def __init__(
            self,
            *,
            device: str,
            batch_wait_ms: float,
            max_batch_size: int,
            max_versions: int = 4,
        ):
            self.device = str(device)
            self.batch_wait_s = max(0.0, float(batch_wait_ms)) / 1000.0
            self.max_batch_size = max(1, int(max_batch_size))
            self.max_versions = max(1, int(max_versions))
            self._models_by_version: dict[int, dict[str, torch.nn.Module]] = {}
            self._version_order: list[int] = []
            self._pending: list[tuple[int, str, torch.Tensor, asyncio.Future]] = []
            self._flush_task: asyncio.Task | None = None

        def _ensure_models(self, snapshot: Mapping[str, Any]) -> None:
            version = int(snapshot["version"])
            if version in self._models_by_version:
                return

            current_payload = snapshot["current"]
            best_payload = snapshot["best"]
            self._models_by_version[version] = {
                "current": model_from_checkpoint_payload(current_payload, self.device),
                "best": model_from_checkpoint_payload(best_payload, self.device),
            }
            self._version_order.append(version)
            while len(self._version_order) > self.max_versions:
                stale_version = self._version_order.pop(0)
                self._models_by_version.pop(stale_version, None)

        def set_snapshot(self, snapshot: Mapping[str, Any]) -> int:
            self._ensure_models(snapshot)
            return int(snapshot["version"])

        async def _drain_pending(self) -> None:
            await asyncio.sleep(self.batch_wait_s)
            try:
                while self._pending:
                    batch = self._pending[: self.max_batch_size]
                    self._pending = self._pending[self.max_batch_size :]
                    grouped: dict[tuple[int, str], list[tuple[torch.Tensor, asyncio.Future]]] = {}
                    for version, model_key, obs, fut in batch:
                        grouped.setdefault((version, model_key), []).append((obs, fut))

                    for (version, model_key), entries in grouped.items():
                        model = self._models_by_version.get(version, {}).get(model_key)
                        if model is None:
                            error = RuntimeError(f"Inference snapshot missing version={version} model_key={model_key}")
                            for _obs, fut in entries:
                                if not fut.done():
                                    fut.set_exception(error)
                            continue

                        obs_batch = torch.stack([obs for obs, _fut in entries], dim=0).to(self.device).float()
                        with torch.no_grad():
                            policy_logits, values = model(obs_batch)
                        policy_logits = policy_logits.detach().cpu().float()
                        values = values.detach().cpu().float().reshape(-1)
                        for row_idx, (_obs, fut) in enumerate(entries):
                            if not fut.done():
                                fut.set_result((policy_logits[row_idx], values[row_idx : row_idx + 1]))
            finally:
                self._flush_task = None
                if self._pending:
                    self._flush_task = asyncio.create_task(self._drain_pending())

        async def infer(self, version: int, model_key: str, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if int(version) not in self._models_by_version:
                raise RuntimeError(f"Inference actor missing snapshot version={int(version)}. Call set_snapshot first.")
            if obs.dim() != 1:
                obs = obs.reshape(-1)
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._pending.append((int(version), str(model_key), obs.detach().cpu(), fut))
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(self._drain_pending())
            return await fut

    return CentralInferenceActor


def _build_weight_store_actor(ray_module):
    @ray_module.remote(max_restarts=0)
    class WeightStoreActor:
        def __init__(self, snapshot: Mapping[str, Any]):
            self._snapshot = dict(snapshot)

        def get_snapshot(self) -> Mapping[str, Any]:
            return self._snapshot

        def set_snapshot(self, snapshot: Mapping[str, Any]) -> int:
            self._snapshot = dict(snapshot)
            return int(self._snapshot["version"])

    return WeightStoreActor


def _build_learner_actor(ray_module):
    @ray_module.remote(max_restarts=0)
    class LearnerActor:
        def __init__(
            self,
            *,
            current_payload: Mapping[str, Any],
            best_payload: Mapping[str, Any],
            device: str,
            train_defaults: Mapping[str, Any],
        ):
            self.device = str(device)
            self.vocab_path = str(current_payload["vocab_path"])
            self.current_model = model_from_checkpoint_payload(current_payload, self.device)
            self.best_model = model_from_checkpoint_payload(best_payload, self.device)
            self.train_defaults = dict(train_defaults)

            self.optimizer = build_optimizer(
                self.current_model,
                lr=float(self.train_defaults["lr"]),
                weight_decay=float(self.train_defaults["weight_decay"]),
                lr_backbone_mult=float(self.train_defaults["lr_backbone_mult"]),
                lr_pi_mult=float(self.train_defaults["lr_pi_mult"]),
                lr_v_mult=float(self.train_defaults["lr_v_mult"]),
            )
            optimizer_state = current_payload.get("optimizer_state_dict")
            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
                for state in self.optimizer.state.values():
                    for key, value in list(state.items()):
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)

            self.scheduler = build_scheduler(
                self.optimizer,
                warmup_steps=int(self.train_defaults["lr_warmup_steps"]),
                hold_steps=int(self.train_defaults["lr_hold_steps"]),
                total_steps=int(self.train_defaults["lr_total_steps"]),
            )
            scheduler_state = current_payload.get("scheduler_state_dict")
            if scheduler_state:
                self.scheduler.load_state_dict(scheduler_state)

        def snapshot(self, *, version: int, total_env_steps: int) -> Dict[str, Any]:
            current_lr = 0.0
            if self.optimizer.param_groups:
                current_lr = float(self.optimizer.param_groups[0]["lr"])
            extra = {"total_env_steps": int(total_env_steps), "current_lr": float(current_lr)}
            return {
                "version": int(version),
                "current_lr": float(current_lr),
                "current": _model_payload(
                    self.current_model,
                    vocab_path=self.vocab_path,
                    iteration=int(version),
                    extra=extra,
                ),
                "best": _model_payload(
                    self.best_model,
                    vocab_path=self.vocab_path,
                    iteration=int(version),
                    extra=extra,
                ),
            }

        def export_checkpoints(
            self,
            *,
            iteration: int,
            total_env_steps: int,
            current_extra: Optional[Mapping[str, Any]] = None,
            best_extra: Optional[Mapping[str, Any]] = None,
        ) -> Dict[str, Any]:
            return {
                "current": _checkpoint_payload(
                    self.current_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    vocab_path=self.vocab_path,
                    iteration=int(iteration),
                    extra=current_extra or {"total_env_steps": int(total_env_steps)},
                ),
                "best": _checkpoint_payload(
                    self.best_model,
                    optimizer=None,
                    scheduler=None,
                    vocab_path=self.vocab_path,
                    iteration=int(iteration),
                    extra=best_extra or {"total_env_steps": int(total_env_steps)},
                ),
            }

        def train_rollout(
            self,
            rollout: Mapping[str, torch.Tensor],
            *,
            iteration: int,
            snapshot_version: int,
            total_env_steps: int,
            steps: int,
            progress_interval: int,
            progress_prefix: str,
        ) -> Dict[str, Any]:
            config = TrainConfig(**self.train_defaults, steps=int(steps))
            metrics = train_on_frozen_rollout(
                self.current_model,
                self.optimizer,
                rollout,
                device=self.device,
                config=config,
                scheduler=self.scheduler,
                progress_interval=int(progress_interval),
                progress_prefix=str(progress_prefix),
            )
            return {
                "metrics": metrics,
                "snapshot": self.snapshot(version=int(snapshot_version), total_env_steps=int(total_env_steps)),
                "current_checkpoint": _checkpoint_payload(
                    self.current_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    vocab_path=self.vocab_path,
                    iteration=int(iteration),
                    extra={"total_env_steps": int(total_env_steps), "train_metrics": dict(metrics)},
                ),
            }

        def promote_current_to_best(
            self,
            *,
            iteration: int,
            snapshot_version: int,
            total_env_steps: int,
            eval_stats: Mapping[str, Any],
        ) -> Dict[str, Any]:
            self.best_model.load_state_dict(self.current_model.state_dict())
            return {
                "snapshot": self.snapshot(version=int(snapshot_version), total_env_steps=int(total_env_steps)),
                "best_checkpoint": _checkpoint_payload(
                    self.best_model,
                    optimizer=None,
                    scheduler=None,
                    vocab_path=self.vocab_path,
                    iteration=int(iteration),
                    extra={"promoted_from": int(iteration), "eval": dict(eval_stats)},
                ),
            }

    return LearnerActor


def _build_rollout_worker_actor(ray_module):
    from src.checkpoint import model_from_checkpoint_payload

    @ray_module.remote(max_restarts=0)
    class RolloutWorkerActor:
        def __init__(
            self,
            *,
            worker_id: int,
            server_configuration: Any,
            battle_format: str,
            vocab_path: str,
            device: str,
            max_concurrent_battles: int,
            parallel_pairs: int,
            reward_terminal_win: float,
            reward_terminal_loss: float,
            reward_use_faint: bool,
            reward_faint_self: float,
            reward_faint_opp: float,
            reward_discount: float,
            reward_gae_lambda: float,
            reward_target_clip: float,
            max_turns_before_forfeit: int | None,
            worker_log_level: str,
            weight_store: Any,
            inference_actor: Any,
        ):
            self.worker_id = int(worker_id)
            self.server_configuration = server_configuration
            self.battle_format = str(battle_format)
            self.vocab = ObservationVocabulary.load_json(vocab_path)
            self.device = str(device)
            self.max_concurrent_battles = int(max_concurrent_battles)
            self.parallel_pairs = int(parallel_pairs)
            self.reward_terminal_win = float(reward_terminal_win)
            self.reward_terminal_loss = float(reward_terminal_loss)
            self.reward_use_faint = bool(reward_use_faint)
            self.reward_faint_self = float(reward_faint_self)
            self.reward_faint_opp = float(reward_faint_opp)
            self.reward_discount = float(reward_discount)
            self.reward_gae_lambda = float(reward_gae_lambda)
            self.reward_target_clip = float(reward_target_clip)
            self.max_turns_before_forfeit = max_turns_before_forfeit
            self.worker_log_level = str(worker_log_level)
            self.weight_store = weight_store
            self.inference_actor = inference_actor

            self._loaded_version: Optional[int] = None
            self._current_model: Optional[torch.nn.Module] = None
            self._best_model: Optional[torch.nn.Module] = None

        def _ensure_models(self, snapshot: Mapping[str, Any]) -> None:
            version = int(snapshot["version"])
            if self._loaded_version == version and self._current_model is not None and self._best_model is not None:
                return

            current_payload = snapshot["current"]
            best_payload = snapshot["best"]

            if self._current_model is None:
                self._current_model = model_from_checkpoint_payload(current_payload, self.device)
            else:
                self._current_model.load_state_dict(current_payload["model_state_dict"])
                self._current_model.to(self.device)

            if self._best_model is None:
                self._best_model = model_from_checkpoint_payload(best_payload, self.device)
            else:
                self._best_model.load_state_dict(best_payload["model_state_dict"])
                self._best_model.to(self.device)

            self._loaded_version = version

        def collect(self, request: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int]]:
            snapshot = ray_module.get(self.weight_store.get_snapshot.remote())
            self._ensure_models(snapshot)

            assert self._current_model is not None
            assert self._best_model is not None

            target_steps = max(0, int(request.get("target_steps", 0)))
            fixed_battles = max(0, int(request.get("fixed_battles", 0)))
            battle_chunk = max(1, int(request.get("battle_chunk", 1)))
            temperature = float(request.get("temperature", 0.0))
            seed = int(request.get("seed", 0))
            snapshot_version = int(snapshot["version"])
            server_url = getattr(self.server_configuration, "websocket_url", "<unknown>")

            if _worker_log_enabled(self.worker_log_level, "summary"):
                print(
                    f"[worker {self.worker_id}] collect start version={snapshot_version} "
                    f"target_steps={target_steps} fixed_battles={fixed_battles} "
                    f"battle_chunk={battle_chunk} pairs={self.parallel_pairs} "
                    f"max_concurrent={self.max_concurrent_battles} server={server_url}",
                    flush=True,
                )

            episodes = []
            stats = {"wins": 0, "losses": 0, "ties": 0}
            collected_steps = 0
            collected_battles = 0
            current_inference_client = RayInferenceClient(
                ray_module=ray_module,
                actor_handle=self.inference_actor,
                model_key="current",
                version=snapshot_version,
            )
            best_inference_client = RayInferenceClient(
                ray_module=ray_module,
                actor_handle=self.inference_actor,
                model_key="best",
                version=snapshot_version,
            )

            async def _collect_chunk(chunk_battles: int, chunk_seed: int):
                return await _run_selfplay_phase(
                    current_model=self._current_model,
                    best_model=self._best_model,
                    vocab=self.vocab,
                    n_battles=chunk_battles,
                    battle_format=self.battle_format,
                    server_configuration=self.server_configuration,
                    device=self.device,
                    temperature=temperature,
                    reward_terminal_win=self.reward_terminal_win,
                    reward_terminal_loss=self.reward_terminal_loss,
                    reward_use_faint=self.reward_use_faint,
                    reward_faint_self=self.reward_faint_self,
                    reward_faint_opp=self.reward_faint_opp,
                    reward_discount=self.reward_discount,
                    reward_gae_lambda=self.reward_gae_lambda,
                    reward_target_clip=self.reward_target_clip,
                        max_turns_before_forfeit=self.max_turns_before_forfeit,
                        max_concurrent_battles=self.max_concurrent_battles,
                        parallel_pairs=self.parallel_pairs,
                        seed=chunk_seed,
                        account_prefix=f"w{self.worker_id}_",
                        current_inference_client=current_inference_client,
                        best_inference_client=best_inference_client,
                        log_level=self.worker_log_level,
                    )

            while True:
                if target_steps > 0:
                    next_battles = battle_chunk
                else:
                    remaining_battles = max(0, fixed_battles - collected_battles)
                    if remaining_battles <= 0:
                        break
                    next_battles = min(battle_chunk, remaining_battles)

                if _worker_log_enabled(self.worker_log_level, "chunk"):
                    print(
                        f"[worker {self.worker_id}] chunk start battles={next_battles} "
                        f"collected_steps={collected_steps} collected_battles={collected_battles}",
                        flush=True,
                    )
                next_episodes, next_stats = asyncio.run(
                    _collect_chunk(next_battles, seed + collected_battles + self.worker_id * 100_000)
                )
                batch_steps = sum(len(episode) for episode in next_episodes)
                if batch_steps <= 0:
                    raise RuntimeError(f"Worker {self.worker_id} collected zero steps from a rollout chunk.")

                episodes.extend(next_episodes)
                collected_steps += int(batch_steps)
                collected_battles += int(next_battles)
                stats["wins"] += int(next_stats["wins"])
                stats["losses"] += int(next_stats["losses"])
                stats["ties"] += int(next_stats["ties"])
                if _worker_log_enabled(self.worker_log_level, "chunk"):
                    print(
                        f"[worker {self.worker_id}] chunk done batch_steps={batch_steps} "
                        f"total_steps={collected_steps} total_battles={collected_battles} "
                        f"stats={stats}",
                        flush=True,
                    )

                if target_steps > 0:
                    if collected_steps >= target_steps:
                        break
                else:
                    if collected_battles >= fixed_battles:
                        break

            if _worker_log_enabled(self.worker_log_level, "summary"):
                print(
                    f"[worker {self.worker_id}] collect done version={snapshot_version} "
                    f"steps={collected_steps} battles={collected_battles} episodes={len(episodes)} "
                    f"stats={stats}",
                    flush=True,
                )

            packed = pack_episodes(episodes, obs_dtype=torch.float16)
            info = {
                "steps": int(collected_steps),
                "battles": int(collected_battles),
                "episodes": int(len(episodes)),
            }
            return packed, stats, info

    return RolloutWorkerActor


def main() -> int:
    args = build_parser().parse_args()
    if ray is None:
        raise SystemExit(
            "Ray is not installed. Install it first, for example:\n"
            "  pip install ray\n"
            "Then rerun train_selfplay_ray."
        )
    if int(args.rollout_workers) < 1:
        raise SystemExit("--rollout-workers must be >= 1")
    if float(args.worker_cpus) <= 0:
        raise SystemExit("--worker-cpus must be > 0")
    if float(args.worker_gpus) < 0:
        raise SystemExit("--worker-gpus must be >= 0")
    if float(args.learner_cpus) <= 0:
        raise SystemExit("--learner-cpus must be > 0")
    if float(args.learner_gpus) < 0:
        raise SystemExit("--learner-gpus must be >= 0")
    if float(args.inference_cpus) <= 0:
        raise SystemExit("--inference-cpus must be > 0")
    if float(args.inference_gpus) < 0:
        raise SystemExit("--inference-gpus must be >= 0")
    if int(args.inference_max_batch_size) < 1:
        raise SystemExit("--inference-max-batch-size must be >= 1")
    if int(args.max_turns_before_forfeit) < 0:
        args.max_turns_before_forfeit = None
    if int(args.max_concurrent_battles) < 1:
        raise SystemExit("--max-concurrent-battles must be >= 1")
    if int(args.selfplay_pairs) < 1:
        raise SystemExit("--selfplay-pairs must be >= 1")
    if int(args.eval_pairs) < 1:
        raise SystemExit("--eval-pairs must be >= 1")

    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}", flush=True)
    else:
        print(f"[device] using={args.device}", flush=True)
    if int(args.max_len) > 0:
        print("[config] --max-len is deprecated and ignored with the structured observation encoder.", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    final_promote_threshold = max(FINAL_EVAL_PROMOTE_THRESHOLD, float(args.promote_threshold))
    print(
        f"[config] eval_every_iterations={int(args.eval_every_iterations)} "
        f"promote_threshold={final_promote_threshold:.3f}",
        flush=True,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    current_ckpt = ckpt_dir / "current.pt"
    best_ckpt = ckpt_dir / "best.pt"
    vocab_json = ckpt_dir / "vocab.json"

    vocab, current_model, best_model, optimizer, scheduler, start_iter, total_env_steps = _load_or_init(
        args,
        current_ckpt=current_ckpt,
        best_ckpt=best_ckpt,
        vocab_json=vocab_json,
    )
    del current_model, best_model, optimizer, scheduler

    current_payload = load_checkpoint(current_ckpt, "cpu")
    best_payload = load_checkpoint(best_ckpt, "cpu")

    server_configurations = _build_server_configurations(args)
    needs_server = args.selfplay_battles > 0 or int(args.selfplay_steps) > 0 or args.eval_battles > 0
    if needs_server:
        for server_configuration in server_configurations:
            ok, err = asyncio.run(_check_server(server_configuration.websocket_url))
            if not ok:
                raise SystemExit(
                    "Could not connect to Showdown websocket at "
                    f"{server_configuration.websocket_url}. Error: {err}\n"
                    "Start a local server (example):\n"
                    "  node pokemon-showdown start --no-security 8000\n"
                    "Then rerun training."
                )

    ray.init(
        address=(args.ray_address or None),
        ignore_reinit_error=True,
        namespace=str(args.ray_namespace),
        log_to_driver=True,
    )

    worker_device = _resolve_worker_device(args)
    inference_device = _resolve_inference_device(args)
    learner_gpus = _resolve_learner_gpus(args, inference_device=inference_device)
    inference_gpus = _resolve_inference_gpus(args, inference_device=inference_device)
    if (learner_gpus + inference_gpus) > 1.000001:
        raise SystemExit(
            f"Requested learner/inference GPU fractions exceed one GPU: learner={learner_gpus:.3f} "
            f"inference={inference_gpus:.3f}. Lower --learner-gpus or --inference-gpus."
        )
    inference_actor = None
    weight_store = None
    learner = None
    workers = []
    CentralInferenceActor = _build_inference_actor(ray)
    WeightStoreActor = _build_weight_store_actor(ray)
    LearnerActor = _build_learner_actor(ray)
    learner_train_defaults = {
        "batch_size": int(args.batch_size),
        "epochs": int(args.train_epochs),
        "grad_accum_steps": int(args.grad_accum_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "value_coef": float(args.value_coef),
        "entropy_coef": float(args.entropy_coef),
        "max_grad_norm": float(args.max_grad_norm),
        "amp": True,
        "ppo_clip_epsilon": float(args.ppo_clip_epsilon),
        "ppo_value_clip": float(args.ppo_value_clip),
        "target_kl": float(args.target_kl),
        "target_kl_factor": float(args.target_kl_factor),
        "min_steps_before_early_stop": int(args.min_steps_before_early_stop),
        "normalize_advantages": not bool(args.no_advantage_norm),
        "policy_temperature": float(args.selfplay_temperature),
        "use_twohot_value": not bool(args.no_twohot_value),
        "v_min": float(args.v_min),
        "v_max": float(args.v_max),
        "v_bins": int(args.v_bins),
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_hold_steps": int(args.lr_hold_steps),
        "lr_total_steps": int(args.lr_total_steps),
        "lr_backbone_mult": float(args.lr_backbone_mult),
        "lr_pi_mult": float(args.lr_pi_mult),
        "lr_v_mult": float(args.lr_v_mult),
    }
    learner = LearnerActor.options(
        num_cpus=float(args.learner_cpus),
        num_gpus=float(learner_gpus),
    ).remote(
        current_payload=current_payload,
        best_payload=best_payload,
        device=str(args.device),
        train_defaults=learner_train_defaults,
    )
    snapshot_version = max(1, (int(start_iter) - 1) * 2 + 1)
    initial_snapshot = ray.get(learner.snapshot.remote(version=int(snapshot_version), total_env_steps=int(total_env_steps)))
    current_snapshot = dict(initial_snapshot)
    weight_store = WeightStoreActor.options(num_cpus=0.1).remote(initial_snapshot)
    inference_actor = CentralInferenceActor.options(
        num_cpus=float(args.inference_cpus),
        num_gpus=float(inference_gpus),
    ).remote(
        device=inference_device,
        batch_wait_ms=float(args.inference_batch_wait_ms),
        max_batch_size=int(args.inference_max_batch_size),
    )
    ray.get(inference_actor.set_snapshot.remote(initial_snapshot))
    RolloutWorkerActor = _build_rollout_worker_actor(ray)
    workers = [
        RolloutWorkerActor.options(
            num_cpus=float(args.worker_cpus),
            num_gpus=float(args.worker_gpus),
        ).remote(
            worker_id=idx,
            server_configuration=server_configurations[idx % len(server_configurations)],
            battle_format=args.battle_format,
            vocab_path=str(vocab_json),
            device=worker_device,
            max_concurrent_battles=int(args.max_concurrent_battles),
            parallel_pairs=int(args.selfplay_pairs),
            reward_terminal_win=float(args.reward_terminal_win),
            reward_terminal_loss=float(args.reward_terminal_loss),
            reward_use_faint=not bool(args.no_faint_reward),
            reward_faint_self=float(args.reward_faint_self),
            reward_faint_opp=float(args.reward_faint_opp),
            reward_discount=float(args.reward_discount),
            reward_gae_lambda=float(args.gae_lambda),
            reward_target_clip=float(args.reward_target_clip),
            max_turns_before_forfeit=args.max_turns_before_forfeit,
            worker_log_level=str(args.worker_log_level),
            weight_store=weight_store,
            inference_actor=inference_actor,
        )
        for idx in range(int(args.rollout_workers))
    ]

    try:
        def _launch_rollout_jobs(iteration_index: int, current_steps: int):
            current_temperature = annealed_temperature(
                start=float(args.selfplay_temperature),
                end=float(args.selfplay_temperature_end),
                total_steps=int(args.selfplay_temperature_total_steps),
                current_steps=int(current_steps),
            )
            step_target = max(0, int(args.selfplay_steps))
            battle_chunk = max(1, int(args.selfplay_battles))
            if step_target > 0:
                step_splits = _split_counts(step_target, len(workers))
                battle_splits = [0 for _ in workers]
            else:
                step_splits = [0 for _ in workers]
                battle_splits = _split_counts(int(args.selfplay_battles), len(workers))

            result_refs = [
                worker.collect.remote(
                    {
                        "target_steps": int(step_splits[idx]),
                        "fixed_battles": int(battle_splits[idx]),
                        "battle_chunk": int(battle_chunk),
                        "temperature": float(current_temperature),
                        "seed": int(args.seed + iteration_index * 10_000 + idx * 1_000),
                    },
                )
                for idx, worker in enumerate(workers)
                if int(step_splits[idx]) > 0 or int(battle_splits[idx]) > 0
            ]
            if not result_refs:
                raise RuntimeError("No rollout workers were assigned any battles or steps.")
            return result_refs, float(current_temperature)

        end_iter = start_iter + max(0, args.iterations)
        pending_rollout_refs = None
        pending_rollout_iteration: int | None = None

        for iteration in range(start_iter, end_iter):
            iter_start = time.perf_counter()
            is_final_iteration = iteration == (end_iter - 1)
            should_eval = (
                int(args.eval_battles) > 0
                and (
                    is_final_iteration
                    or ((iteration - start_iter + 1) % max(1, int(args.eval_every_iterations)) == 0)
                )
            )
            current_lr = float(current_snapshot.get("current_lr", float(args.lr)))
            print(
                f"[iter {iteration}] start selfplay_battles={args.selfplay_battles} "
                f"selfplay_steps={int(args.selfplay_steps)} train_steps={args.train_steps} "
                f"train_epochs={int(args.train_epochs)} eval_battles={args.eval_battles} "
                f"total_env_steps={total_env_steps} "
                f"lr={current_lr:.6g} "
                f"reward_terminal=({args.reward_terminal_win:+.3f},{args.reward_terminal_loss:+.3f}) "
                f"reward_faint={'off' if bool(args.no_faint_reward) else f'({args.reward_faint_self:+.3f},{args.reward_faint_opp:+.3f})'} "
                f"max_turns_before_forfeit={args.max_turns_before_forfeit} "
                f"max_concurrent_battles={int(args.max_concurrent_battles)} "
                f"pairs_per_worker={int(args.selfplay_pairs)} rollout_workers={int(args.rollout_workers)} "
                f"eval_pairs={int(args.eval_pairs)} servers={len(server_configurations)} "
                f"worker_device={worker_device} worker_gpus={float(args.worker_gpus):.3f} "
                f"learner_device={str(args.device)} learner_gpus={float(learner_gpus):.3f} "
                f"inference_device={inference_device} inference_gpus={float(inference_gpus):.3f}",
                flush=True,
            )

            if pending_rollout_refs is None:
                print(f"[iter {iteration}] self-play start", flush=True)
                pending_rollout_refs, _ = _launch_rollout_jobs(iteration, total_env_steps)
                pending_rollout_iteration = iteration
            if pending_rollout_iteration != iteration:
                raise RuntimeError(
                    f"Pending rollout iteration mismatch: expected {iteration}, got {pending_rollout_iteration}"
                )

            phase_t0 = time.perf_counter()
            worker_results = ray.get(pending_rollout_refs)
            rollout = concat_rollouts([packed for packed, _stats, _info in worker_results])
            selfplay_stats, rollout_meta = summarize_worker_stats(worker_results)
            if int(rollout_meta["steps"]) <= 0:
                raise RuntimeError("Ray rollout workers returned zero decision steps.")

            total_env_steps += int(rollout_meta["steps"])
            print(
                f"[iter {iteration}] self-play done battles={int(rollout_meta['battles'])} "
                f"steps={int(rollout_meta['steps'])} episodes={int(rollout_meta['episodes'])} "
                f"replay={rollout_size(rollout)} elapsed={time.perf_counter() - phase_t0:.1f}s",
                flush=True,
            )

            next_rollout_refs = None
            next_rollout_iteration: int | None = None
            if iteration + 1 < end_iter:
                print(f"[iter {iteration + 1}] self-play start", flush=True)
                next_rollout_refs, _ = _launch_rollout_jobs(iteration + 1, total_env_steps)
                next_rollout_iteration = iteration + 1

            rollout_replay_size = rollout_size(rollout)
            rollout_step_budget = optimizer_steps_for_epochs(
                rollout_replay_size,
                batch_size=int(args.batch_size),
                grad_accum_steps=int(args.grad_accum_steps),
                epochs=int(args.train_epochs),
            )
            effective_train_steps = rollout_step_budget if int(args.train_steps) <= 0 else min(
                int(args.train_steps),
                rollout_step_budget,
            )

            train_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "policy_entropy": 0.0,
                "grad_norm": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "steps": 0.0,
            }
            current_snapshot = ray.get(learner.snapshot.remote(version=int(snapshot_version), total_env_steps=int(total_env_steps)))
            if effective_train_steps > 0:
                phase_t0 = time.perf_counter()
                requested_steps_text = str(int(args.train_steps)) if int(args.train_steps) > 0 else "auto"
                print(
                    f"[iter {iteration}] train start steps={effective_train_steps} "
                    f"requested_steps={requested_steps_text} batch={args.batch_size} replay={rollout_replay_size} "
                    f"target_kl={float(args.target_kl):.3f} epochs={int(args.train_epochs)} "
                    f"grad_accum={int(args.grad_accum_steps)} effective_batch={int(args.batch_size) * int(args.grad_accum_steps)}",
                    flush=True,
                )
                train_result = ray.get(
                    learner.train_rollout.remote(
                        rollout,
                        iteration=int(iteration),
                        snapshot_version=int(snapshot_version + 1),
                        total_env_steps=int(total_env_steps),
                        steps=int(effective_train_steps),
                        progress_interval=max(0, int(args.train_log_interval)),
                        progress_prefix=f"[iter {iteration}] ",
                    )
                )
                train_metrics = dict(train_result["metrics"])
                current_snapshot = dict(train_result["snapshot"])
                snapshot_version = int(current_snapshot["version"])
                ray.get(weight_store.set_snapshot.remote(current_snapshot))
                ray.get(inference_actor.set_snapshot.remote(current_snapshot))
                print(f"[iter {iteration}] train done elapsed={time.perf_counter() - phase_t0:.1f}s", flush=True)

            eval_stats = {
                "wins": 0.0,
                "losses": 0.0,
                "ties": 0.0,
                "win_rate": 0.0,
                "score_rate": 0.0,
                "ci_low": 0.0,
                "ci_high": 1.0,
                "n_battles": 0.0,
            }
            promoted = False
            if should_eval:
                phase_t0 = time.perf_counter()
                print(f"[iter {iteration}] eval start battles={args.eval_battles}", flush=True)
                current_eval_model = model_from_checkpoint_payload(current_snapshot["current"], args.device)
                best_eval_model = model_from_checkpoint_payload(current_snapshot["best"], args.device)
                current_eval_client = RayInferenceClient(
                    ray_module=ray,
                    actor_handle=inference_actor,
                    model_key="current",
                    version=int(current_snapshot["version"]),
                )
                best_eval_client = RayInferenceClient(
                    ray_module=ray,
                    actor_handle=inference_actor,
                    model_key="best",
                    version=int(current_snapshot["version"]),
                )
                eval_stats = asyncio.run(
                    evaluate_models(
                        current_model=current_eval_model,
                        best_model=best_eval_model,
                        vocab=vocab,
                        server_configuration=server_configurations,
                        config=EvalConfig(
                            n_battles=args.eval_battles,
                            battle_format=args.battle_format,
                            temperature=args.eval_temperature,
                            seed=args.seed + iteration,
                            max_turns_before_forfeit=args.max_turns_before_forfeit,
                            max_concurrent_battles=args.max_concurrent_battles,
                            parallel_pairs=args.eval_pairs,
                        ),
                        device=args.device,
                        current_inference_client=current_eval_client,
                        best_inference_client=best_eval_client,
                        account_prefix=f"eval{iteration}_",
                    )
                )
                print(
                    f"[iter {iteration}] eval done elapsed={time.perf_counter() - phase_t0:.1f}s "
                    f"win_rate={eval_stats['win_rate']:.3f}",
                    flush=True,
                )
                if eval_stats["win_rate"] >= final_promote_threshold:
                    promoted = True
                    promote_result = ray.get(
                        learner.promote_current_to_best.remote(
                            iteration=int(iteration),
                            snapshot_version=int(snapshot_version + 1),
                            total_env_steps=int(total_env_steps),
                            eval_stats=eval_stats,
                        )
                    )
                    current_snapshot = dict(promote_result["snapshot"])
                    snapshot_version = int(current_snapshot["version"])
                    ray.get(weight_store.set_snapshot.remote(current_snapshot))
                    ray.get(inference_actor.set_snapshot.remote(current_snapshot))
                    print(f"[iter {iteration}] promoted current -> best", flush=True)
            elif int(args.eval_battles) > 0:
                print(
                    f"[iter {iteration}] eval skip next_interval={int(args.eval_every_iterations)}",
                    flush=True,
                )

            checkpoint_payloads = ray.get(
                learner.export_checkpoints.remote(
                    iteration=int(iteration),
                    total_env_steps=int(total_env_steps),
                    current_extra={
                        "replay_size": int(rollout_replay_size),
                        "total_env_steps": int(total_env_steps),
                        "selfplay": dict(selfplay_stats),
                        "train_metrics": dict(train_metrics),
                        "ray_rollout_workers": int(args.rollout_workers),
                        "ray_pairs_per_worker": int(args.selfplay_pairs),
                    },
                    best_extra={"promoted_from": int(iteration), "eval": eval_stats} if promoted else None,
                )
            )
            save_checkpoint_payload(path=current_ckpt, payload=checkpoint_payloads["current"])
            if promoted:
                save_checkpoint_payload(path=best_ckpt, payload=checkpoint_payloads["best"])

            print(
                f"iter={iteration} "
                f"replay={rollout_replay_size} "
                f"selfplay_wlt={selfplay_stats['wins']}/{selfplay_stats['losses']}/{selfplay_stats['ties']} "
                f"loss={train_metrics['loss']:.4f} "
                f"policy_loss={train_metrics['policy_loss']:.4f} "
                f"value_loss={train_metrics['value_loss']:.4f} "
                f"entropy={train_metrics['policy_entropy']:.4f} "
                f"kl={train_metrics['approx_kl']:.4f} "
                f"clip_frac={train_metrics['clip_frac']:.3f} "
                f"eval_win_rate={eval_stats['win_rate']:.3f} "
                f"eval_ci=[{eval_stats['ci_low']:.3f},{eval_stats['ci_high']:.3f}] "
                f"promoted={promoted} "
                f"iter_elapsed={time.perf_counter() - iter_start:.1f}s"
            )

            pending_rollout_refs = next_rollout_refs
            pending_rollout_iteration = next_rollout_iteration
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        if weight_store is not None:
            ray.kill(weight_store, no_restart=True)
        if learner is not None:
            ray.kill(learner, no_restart=True)
        if inference_actor is not None:
            ray.kill(inference_actor, no_restart=True)
        ray.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
