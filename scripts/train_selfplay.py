"""Self-play league training loop for the Transformer poke-env bot."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import websockets
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.bot import TransformerPlayer
from src.checkpoint import (
    build_model_from_vocab,
    load_checkpoint,
    model_from_checkpoint_payload,
    save_checkpoint,
)
from src.device import resolve_device
from src.eval import EvalConfig, evaluate_models
from src.policy import ACTION_DIM, MOVE_ACTIONS
from src.replay import ReplayBuffer, ReplayConfig
from src.train import (
    TrainConfig,
    build_optimizer,
    build_scheduler,
    train_on_frozen_rollout,
    train_on_replay,
)
from src.vocab import ObservationVocabulary, build_default_vocab, missing_required_feature_tokens

SELFPLAY_LOG_LEVELS = {
    "quiet": 0,
    "summary": 1,
    "chunk": 2,
    "pair": 3,
}


def cap_train_steps_by_reuse(
    requested_steps: int,
    *,
    replay_size: int,
    batch_size: int,
    max_reuse_ratio: float,
) -> int:
    if requested_steps <= 0 or replay_size <= 0 or batch_size <= 0:
        return 0
    if max_reuse_ratio <= 0:
        return int(requested_steps)
    capped = max(1, int((float(replay_size) * float(max_reuse_ratio)) // int(batch_size)))
    return min(int(requested_steps), capped)


def optimizer_steps_for_epochs(
    replay_size: int,
    *,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
) -> int:
    if replay_size <= 0 or batch_size <= 0 or grad_accum_steps <= 0 or epochs <= 0:
        return 0
    microbatches_per_epoch = math.ceil(float(replay_size) / float(batch_size))
    optimizer_steps_per_epoch = math.ceil(float(microbatches_per_epoch) / float(grad_accum_steps))
    return int(optimizer_steps_per_epoch * int(epochs))


def annealed_temperature(
    *,
    start: float,
    end: float,
    total_steps: int,
    current_steps: int,
) -> float:
    if total_steps <= 0 or current_steps >= total_steps:
        return float(end)
    frac = max(0.0, min(1.0, float(current_steps) / float(total_steps)))
    return float(start + frac * (float(end) - float(start)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer poke-env bot with league-style self-play.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--selfplay-battles",
        type=int,
        default=8,
        help="Battles per self-play chunk, or total battles if --selfplay-steps is unset.",
    )
    parser.add_argument(
        "--selfplay-steps",
        type=int,
        default=32768,
        help="Collect at least this many decision steps per iteration. Overrides fixed-battle collection when > 0.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="Optimizer steps per iteration. 0 means use the full frozen-rollout epoch budget.",
    )
    parser.add_argument("--train-epochs", type=int, default=3, help="PPO epochs over each frozen rollout.")
    parser.add_argument("--eval-battles", type=int, default=20, help="E battles for promotion eval.")
    parser.add_argument(
        "--eval-every-iterations",
        type=int,
        default=1,
        help="Run promotion eval every N iterations, and always on the final iteration.",
    )
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument(
        "--off-policy-replay",
        action="store_true",
        help="Keep replay shards across iterations. Default behavior is PPO-style on-policy batches per iteration.",
    )
    parser.add_argument(
        "--train-log-interval",
        type=int,
        default=20,
        help="Print training progress every N gradient steps (0 disables step-level logs).",
    )
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=1,
        help="Maximum number of simultaneous battles per poke-env player during self-play and eval.",
    )
    parser.add_argument(
        "--selfplay-pairs",
        type=int,
        default=1,
        help="Number of concurrent current-vs-best player pairs to use during rollout collection.",
    )
    parser.add_argument(
        "--eval-pairs",
        type=int,
        default=1,
        help="Number of concurrent player pairs to use during evaluation.",
    )

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation microbatches per optimizer step for frozen-rollout training.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-hold-steps", type=int, default=20000)
    parser.add_argument("--lr-total-steps", type=int, default=500000)
    parser.add_argument("--lr-backbone-mult", type=float, default=1.0)
    parser.add_argument("--lr-pi-mult", type=float, default=1.0)
    parser.add_argument("--lr-v-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-value-clip", type=float, default=0.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Early-stop training updates in an iteration if mean PPO KL exceeds this threshold.",
    )
    parser.add_argument(
        "--target-kl-factor",
        type=float,
        default=1.5,
        help="Allow PPO early stop once mean KL exceeds target_kl multiplied by this factor.",
    )
    parser.add_argument(
        "--min-steps-before-early-stop",
        type=int,
        default=10,
        help="Minimum completed optimizer steps before KL-based early stop can trigger.",
    )
    parser.add_argument(
        "--max-reuse-ratio",
        type=float,
        default=0.50,
        help="Cap optimizer steps per iteration so sampled replay rows stay below this reuse ratio.",
    )
    parser.add_argument(
        "--no-advantage-norm",
        action="store_true",
        help="Disable per-batch advantage normalization.",
    )

    parser.add_argument("--selfplay-temperature", type=float, default=1.0)
    parser.add_argument("--selfplay-temperature-end", type=float, default=0.9)
    parser.add_argument("--selfplay-temperature-total-steps", type=int, default=500000)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--reward-terminal-win", type=float, default=1.0)
    parser.add_argument("--reward-terminal-loss", type=float, default=-1.0)
    parser.add_argument("--reward-faint-self", type=float, default=-0.1)
    parser.add_argument("--reward-faint-opp", type=float, default=0.1)
    parser.add_argument(
        "--no-faint-reward",
        action="store_true",
        help="Disable per-faint shaping and use only terminal win/loss rewards.",
    )
    parser.add_argument(
        "--reward-discount",
        type=float,
        default=0.9999,
        help="Discount applied when building per-step value targets from shaped rewards.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.75,
        help="GAE lambda used to build policy advantages and value targets from episode rewards.",
    )
    parser.add_argument(
        "--reward-target-clip",
        type=float,
        default=1.6,
        help="Clip bound for per-step value targets after adding shaping and terminal outcome.",
    )
    parser.add_argument("--no-twohot-value", action="store_true")
    parser.add_argument("--v-min", type=float, default=-1.6)
    parser.add_argument("--v-max", type=float, default=1.6)
    parser.add_argument("--v-bins", type=int, default=51)
    parser.add_argument(
        "--max-turns-before-forfeit",
        type=int,
        default=500,
        help="Auto-forfeit if battle turn exceeds this threshold. Use -1 to disable.",
    )

    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="Deprecated. Sequence length is fixed by the structured observation schema and this flag is ignored.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")

    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--replay-shard-size", type=int, default=2048)
    parser.add_argument("--replay-max-shards", type=int, default=256)
    parser.add_argument("--bootstrap-random-samples", type=int, default=0)

    parser.add_argument("--showdown-root", default=None)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--server-ws",
        default=os.environ.get("SHOWDOWN_WS_URL", "ws://localhost:8000/showdown/websocket"),
    )
    parser.add_argument(
        "--server-auth",
        default=os.environ.get("SHOWDOWN_AUTH_URL", "http://localhost:8000/action.php?"),
    )
    parser.add_argument(
        "--server-ws-list",
        default="",
        help="Comma-separated websocket URLs for multiple Showdown servers. Overrides --server-ws when set.",
    )
    parser.add_argument(
        "--server-auth-list",
        default="",
        help="Comma-separated auth URLs for multiple Showdown servers. Used with --server-ws-list.",
    )

    return parser


async def _check_server(ws_url: str) -> Tuple[bool, str]:
    try:
        async with websockets.connect(ws_url, open_timeout=2.0, ping_interval=20.0, ping_timeout=20.0):
            return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _build_server_configurations(args: argparse.Namespace) -> list[ServerConfiguration]:
    ws_list = [part.strip() for part in str(getattr(args, "server_ws_list", "")).split(",") if part.strip()]
    auth_list = [part.strip() for part in str(getattr(args, "server_auth_list", "")).split(",") if part.strip()]
    if not ws_list:
        return [
            ServerConfiguration(
                websocket_url=args.server_ws,
                authentication_url=args.server_auth,
            )
        ]
    if auth_list and len(auth_list) not in (1, len(ws_list)):
        raise SystemExit("--server-auth-list must have either 1 entry or the same number of entries as --server-ws-list")
    if not auth_list:
        raise SystemExit("--server-auth-list is required when --server-ws-list is set")
    if len(auth_list) == 1 and len(ws_list) > 1:
        auth_list = auth_list * len(ws_list)
    return [
        ServerConfiguration(websocket_url=ws_url, authentication_url=auth_url)
        for ws_url, auth_url in zip(ws_list, auth_list)
    ]


def _summarize_results(player: TransformerPlayer) -> Dict[str, int]:
    wins = 0
    losses = 0
    ties = 0
    for battle in player.battles.values():
        if bool(getattr(battle, "won", False)):
            wins += 1
        elif bool(getattr(battle, "lost", False)):
            losses += 1
        elif bool(getattr(battle, "finished", False)):
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


async def _close_player(player: TransformerPlayer) -> None:
    try:
        await player.ps_client.stop_listening()
    except Exception:
        pass


def _split_counts(total: int, parts: int) -> list[int]:
    total = max(0, int(total))
    parts = max(1, int(parts))
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _selfplay_log_enabled(log_level: str, category: str) -> bool:
    current = SELFPLAY_LOG_LEVELS.get(str(log_level), SELFPLAY_LOG_LEVELS["pair"])
    required = SELFPLAY_LOG_LEVELS.get(str(category), SELFPLAY_LOG_LEVELS["pair"])
    return current >= required


async def _run_selfplay_pair(
    *,
    current_model: torch.nn.Module,
    best_model: torch.nn.Module,
    vocab: ObservationVocabulary,
    n_battles: int,
    battle_format: str,
    server_configuration: ServerConfiguration,
    device: str,
    temperature: float,
    reward_terminal_win: float,
    reward_terminal_loss: float,
    reward_use_faint: bool,
    reward_faint_self: float,
    reward_faint_opp: float,
    reward_discount: float,
    reward_gae_lambda: float,
    reward_target_clip: float,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
    pair_index: int,
    account_prefix: str = "",
    current_inference_client: Any | None = None,
    best_inference_client: Any | None = None,
    log_level: str = "pair",
) -> Tuple[list[list[dict[str, Any]]], Dict[str, int]]:
    if int(n_battles) <= 0:
        return [], {"wins": 0, "losses": 0, "ties": 0}

    current_account = AccountConfiguration.generate(f"{account_prefix}current{pair_index}", rand=True)
    best_account = AccountConfiguration.generate(f"{account_prefix}best{pair_index}", rand=True)
    server_url = str(getattr(server_configuration, "websocket_url", "<unknown>"))
    if _selfplay_log_enabled(log_level, "pair"):
        print(
            f"[pair {account_prefix}{pair_index}] init server={server_url} battles={int(n_battles)} "
            f"max_concurrent={max(1, int(max_concurrent_battles))} current={current_account.username} "
            f"best={best_account.username}",
            flush=True,
        )

    current_player = TransformerPlayer(
        account_configuration=current_account,
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=current_model,
        vocab=vocab,
        device=device,
        temperature=temperature,
        collect_trajectories=True,
        reward_terminal_win=reward_terminal_win,
        reward_terminal_loss=reward_terminal_loss,
        reward_use_faint=reward_use_faint,
        reward_faint_self=reward_faint_self,
        reward_faint_opp=reward_faint_opp,
        reward_discount=reward_discount,
        reward_gae_lambda=reward_gae_lambda,
        reward_target_clip=reward_target_clip,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=seed,
        start_listening=True,
        inference_client=current_inference_client,
        inference_model_key="current",
    )
    best_player = TransformerPlayer(
        account_configuration=best_account,
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=best_model,
        vocab=vocab,
        device=device,
        temperature=0.0,
        collect_trajectories=False,
        reward_terminal_win=reward_terminal_win,
        reward_terminal_loss=reward_terminal_loss,
        reward_use_faint=reward_use_faint,
        reward_faint_self=reward_faint_self,
        reward_faint_opp=reward_faint_opp,
        reward_discount=reward_discount,
        reward_gae_lambda=reward_gae_lambda,
        reward_target_clip=reward_target_clip,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=seed + 1,
        start_listening=True,
        inference_client=best_inference_client,
        inference_model_key="best",
    )

    try:
        if _selfplay_log_enabled(log_level, "pair"):
            print(
                f"[pair {account_prefix}{pair_index}] battle_against start server={server_url} "
                f"battles={int(n_battles)}",
                flush=True,
            )
        await current_player.battle_against(best_player, n_battles=int(n_battles))
        if _selfplay_log_enabled(log_level, "pair"):
            print(
                f"[pair {account_prefix}{pair_index}] battle_against done server={server_url} "
                f"completed={len(current_player.battles)}",
                flush=True,
            )
        episodes = current_player.pop_completed_episodes()
        stats = _summarize_results(current_player)
        return episodes, stats
    finally:
        if _selfplay_log_enabled(log_level, "pair"):
            print(f"[pair {account_prefix}{pair_index}] closing players server={server_url}", flush=True)
        await _close_player(current_player)
        await _close_player(best_player)


async def _run_selfplay_phase(
    *,
    current_model: torch.nn.Module,
    best_model: torch.nn.Module,
    vocab: ObservationVocabulary,
    n_battles: int,
    battle_format: str,
    server_configuration: ServerConfiguration | Sequence[ServerConfiguration],
    device: str,
    temperature: float,
    reward_terminal_win: float,
    reward_terminal_loss: float,
    reward_use_faint: bool,
    reward_faint_self: float,
    reward_faint_opp: float,
    reward_discount: float,
    reward_gae_lambda: float,
    reward_target_clip: float,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    parallel_pairs: int,
    seed: int,
    account_prefix: str = "",
    current_inference_client: Any | None = None,
    best_inference_client: Any | None = None,
    log_level: str = "pair",
) -> Tuple[list[list[dict[str, Any]]], Dict[str, int]]:
    server_configurations = (
        list(server_configuration)
        if isinstance(server_configuration, Sequence) and not isinstance(server_configuration, ServerConfiguration)
        else [server_configuration]  # type: ignore[list-item]
    )
    if not server_configurations:
        raise ValueError("At least one server configuration is required for self-play.")
    pair_count = max(1, min(int(parallel_pairs), int(max(1, n_battles))))
    pair_battles = _split_counts(n_battles, pair_count)
    results = await asyncio.gather(
        *[
            _run_selfplay_pair(
                current_model=current_model,
                best_model=best_model,
                vocab=vocab,
                n_battles=count,
                battle_format=battle_format,
                server_configuration=server_configurations[idx % len(server_configurations)],
                device=device,
                temperature=temperature,
                reward_terminal_win=reward_terminal_win,
                reward_terminal_loss=reward_terminal_loss,
                reward_use_faint=reward_use_faint,
                reward_faint_self=reward_faint_self,
                reward_faint_opp=reward_faint_opp,
                reward_discount=reward_discount,
                reward_gae_lambda=reward_gae_lambda,
                reward_target_clip=reward_target_clip,
                max_turns_before_forfeit=max_turns_before_forfeit,
                max_concurrent_battles=max_concurrent_battles,
                seed=seed + (2 * idx),
                pair_index=idx,
                account_prefix=account_prefix,
                current_inference_client=current_inference_client,
                best_inference_client=best_inference_client,
                log_level=log_level,
            )
            for idx, count in enumerate(pair_battles)
            if count > 0
        ]
    )
    episodes: list[list[dict[str, Any]]] = []
    stats = {"wins": 0, "losses": 0, "ties": 0}
    for pair_episodes, pair_stats in results:
        episodes.extend(pair_episodes)
        stats["wins"] += int(pair_stats["wins"])
        stats["losses"] += int(pair_stats["losses"])
        stats["ties"] += int(pair_stats["ties"])
    return episodes, stats


def _seed_replay_with_random_samples(
    replay: ReplayBuffer,
    *,
    num_samples: int,
    vocab: ObservationVocabulary,
    seed: int,
) -> None:
    if num_samples <= 0:
        return

    rng = torch.Generator().manual_seed(seed)
    meta = vocab.schema_meta()
    obs_dim = int(meta["obs_dim"])
    action_start, action_end = meta["offsets"]["action_mask"]
    for _ in range(num_samples):
        obs = torch.zeros((obs_dim,), dtype=torch.float32)
        legal_mask = torch.zeros((ACTION_DIM,), dtype=torch.bool)
        legal_mask[:MOVE_ACTIONS] = True
        obs[action_start:action_end] = legal_mask.float()
        action_index = int(torch.randint(low=0, high=MOVE_ACTIONS, size=(1,), generator=rng).item())
        outcome = float(-1.0 if torch.randint(low=0, high=2, size=(1,), generator=rng).item() == 0 else 1.0)

        replay.add_sample(
            obs=obs,
            legal_mask=legal_mask,
            action_index=action_index,
            outcome_z=outcome,
            old_log_prob=math.log(1.0 / MOVE_ACTIONS),
            old_value=0.0,
            advantage=outcome,
        )
    replay.flush()


def _load_or_init(
    args: argparse.Namespace,
    *,
    current_ckpt: Path,
    best_ckpt: Path,
    vocab_json: Path,
) -> Tuple[
    ObservationVocabulary,
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
    int,
]:
    if args.resume and current_ckpt.exists() and vocab_json.exists():
        vocab = ObservationVocabulary.load_json(vocab_json)
        missing_tokens = missing_required_feature_tokens(vocab)
        if missing_tokens:
            raise ValueError(
                "Checkpoint vocab is missing required observation tokens "
                f"({', '.join(missing_tokens[:4])}). Start a fresh checkpoint directory."
            )
        current_payload = load_checkpoint(current_ckpt, args.device)
        current_model = model_from_checkpoint_payload(current_payload, args.device)
        vocab_obs_dim = int(vocab.schema_meta()["obs_dim"])
        saved_obs_dim = int(current_payload.get("obs_dim", current_model.config.obs_dim))
        if saved_obs_dim != vocab_obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={saved_obs_dim} does not match current vocab obs_dim={vocab_obs_dim}. "
                "Start a fresh checkpoint directory."
            )

        optimizer = build_optimizer(
            current_model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_backbone_mult=args.lr_backbone_mult,
            lr_pi_mult=args.lr_pi_mult,
            lr_v_mult=args.lr_v_mult,
        )
        opt_state = current_payload.get("optimizer_state_dict")
        optimizer_state_loaded = False
        if opt_state:
            try:
                optimizer.load_state_dict(opt_state)
                optimizer_state_loaded = True
            except ValueError as exc:
                print(
                    "[resume] optimizer state incompatible with current param-group layout; "
                    f"starting with fresh optimizer. detail={exc}",
                    flush=True,
                )
        scheduler = build_scheduler(
            optimizer,
            warmup_steps=args.lr_warmup_steps,
            hold_steps=args.lr_hold_steps,
            total_steps=args.lr_total_steps,
        )
        scheduler_state = current_payload.get("scheduler_state_dict")
        if scheduler_state and optimizer_state_loaded:
            try:
                scheduler.load_state_dict(scheduler_state)
            except ValueError as exc:
                print(
                    "[resume] scheduler state incompatible with current optimizer layout; "
                    f"starting with fresh scheduler. detail={exc}",
                    flush=True,
                )

        start_iter = int(current_payload.get("iteration", 0)) + 1
        total_env_steps = int((current_payload.get("extra") or {}).get("total_env_steps", 0))

        if best_ckpt.exists():
            best_payload = load_checkpoint(best_ckpt, args.device)
            best_model = model_from_checkpoint_payload(best_payload, args.device)
        else:
            best_model = model_from_checkpoint_payload(current_payload, args.device)

        return vocab, current_model, best_model, optimizer, scheduler, start_iter, total_env_steps

    vocab = build_default_vocab(gen=9, showdown_root=args.showdown_root, battle_format=args.battle_format)
    vocab.save_json(vocab_json)

    current_model = build_model_from_vocab(
        vocab,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_twohot_value=not bool(args.no_twohot_value),
        v_min=args.v_min,
        v_max=args.v_max,
        v_bins=args.v_bins,
        device=args.device,
    )
    best_model = build_model_from_vocab(
        vocab,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_twohot_value=not bool(args.no_twohot_value),
        v_min=args.v_min,
        v_max=args.v_max,
        v_bins=args.v_bins,
        device=args.device,
    )
    best_model.load_state_dict(current_model.state_dict())

    optimizer = build_optimizer(
        current_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_backbone_mult=args.lr_backbone_mult,
        lr_pi_mult=args.lr_pi_mult,
        lr_v_mult=args.lr_v_mult,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.lr_warmup_steps,
        hold_steps=args.lr_hold_steps,
        total_steps=args.lr_total_steps,
    )

    save_checkpoint(
        path=current_ckpt,
        model=current_model,
        optimizer=optimizer,
        scheduler=scheduler,
        iteration=0,
        vocab_path=str(vocab_json),
        extra={"bootstrap": True},
    )
    save_checkpoint(
        path=best_ckpt,
        model=best_model,
        optimizer=None,
        scheduler=None,
        iteration=0,
        vocab_path=str(vocab_json),
        extra={"bootstrap": True},
    )

    return vocab, current_model, best_model, optimizer, scheduler, 1, 0


def main() -> int:
    args = build_parser().parse_args()
    if int(args.max_turns_before_forfeit) < 0:
        args.max_turns_before_forfeit = None
    if int(args.max_concurrent_battles) < 1:
        raise SystemExit("--max-concurrent-battles must be >= 1")
    if int(args.selfplay_pairs) < 1:
        raise SystemExit("--selfplay-pairs must be >= 1")
    if int(args.eval_pairs) < 1:
        raise SystemExit("--eval-pairs must be >= 1")
    if int(args.eval_every_iterations) < 1:
        raise SystemExit("--eval-every-iterations must be >= 1")
    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}", flush=True)
    else:
        print(f"[device] using={args.device}", flush=True)
    if int(args.max_len) > 0:
        print("[config] --max-len is deprecated and ignored with the structured observation encoder.", flush=True)
    if args.device == "mps" and int(args.batch_size) > 16:
        print(
            f"[mps] batch_size={int(args.batch_size)} may exceed Apple Silicon memory limits; "
            "reduce to 8 or 16 if you hit OOM.",
            flush=True,
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_ckpt = ckpt_dir / "current.pt"
    best_ckpt = ckpt_dir / "best.pt"
    vocab_json = ckpt_dir / "vocab.json"

    replay_dir = ckpt_dir / "replay"
    replay = ReplayBuffer(
        replay_dir,
        config=ReplayConfig(
            shard_size=args.replay_shard_size,
            max_shards=args.replay_max_shards,
            seed=args.seed,
        ),
    )

    vocab, current_model, best_model, optimizer, scheduler, start_iter, total_env_steps = _load_or_init(
        args,
        current_ckpt=current_ckpt,
        best_ckpt=best_ckpt,
        vocab_json=vocab_json,
    )

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
                    "  node pokemon-showdown start --no-security --port 8000\n"
                    "Then rerun training."
                )

    if len(replay) == 0 and args.bootstrap_random_samples > 0:
        _seed_replay_with_random_samples(
            replay,
            num_samples=args.bootstrap_random_samples,
            vocab=vocab,
            seed=args.seed,
        )

    end_iter = start_iter + max(0, args.iterations)
    for iteration in range(start_iter, end_iter):
        iter_start = time.perf_counter()
        is_final_iteration = iteration == (end_iter - 1)
        should_eval = bool(
            int(args.eval_battles) > 0
            and (
                (int(iteration) % int(args.eval_every_iterations) == 0)
                or is_final_iteration
            )
        )
        print(
            f"[iter {iteration}] start selfplay_battles={args.selfplay_battles} "
            f"selfplay_steps={int(args.selfplay_steps)} train_steps={args.train_steps} "
            f"train_epochs={int(args.train_epochs)} eval_battles={args.eval_battles} "
            f"eval_every={int(args.eval_every_iterations)} "
            f"total_env_steps={total_env_steps} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"reward_terminal=({args.reward_terminal_win:+.3f},{args.reward_terminal_loss:+.3f}) "
            f"reward_faint={'off' if bool(args.no_faint_reward) else f'({args.reward_faint_self:+.3f},{args.reward_faint_opp:+.3f})'} "
            f"max_turns_before_forfeit={args.max_turns_before_forfeit} "
            f"max_concurrent_battles={int(args.max_concurrent_battles)} "
            f"selfplay_pairs={int(args.selfplay_pairs)} eval_pairs={int(args.eval_pairs)} "
            f"servers={len(server_configurations)}",
            flush=True,
        )
        selfplay_stats = {"wins": 0, "losses": 0, "ties": 0}

        if args.selfplay_battles > 0 or int(args.selfplay_steps) > 0:
            phase_t0 = time.perf_counter()
            print(f"[iter {iteration}] self-play start", flush=True)
            current_temperature = annealed_temperature(
                start=float(args.selfplay_temperature),
                end=float(args.selfplay_temperature_end),
                total_steps=int(args.selfplay_temperature_total_steps),
                current_steps=int(total_env_steps),
            )
            episodes = []
            collected_steps = 0
            collected_battles = 0
            chunk_battles = max(1, int(args.selfplay_battles))
            target_steps = max(0, int(args.selfplay_steps))

            while True:
                next_episodes, next_stats = asyncio.run(
                    _run_selfplay_phase(
                        current_model=current_model,
                        best_model=best_model,
                        vocab=vocab,
                        n_battles=chunk_battles,
                        battle_format=args.battle_format,
                        server_configuration=server_configurations,
                        device=args.device,
                        temperature=current_temperature,
                        reward_terminal_win=args.reward_terminal_win,
                        reward_terminal_loss=args.reward_terminal_loss,
                        reward_use_faint=not bool(args.no_faint_reward),
                        reward_faint_self=args.reward_faint_self,
                        reward_faint_opp=args.reward_faint_opp,
                        reward_discount=args.reward_discount,
                        reward_gae_lambda=args.gae_lambda,
                        reward_target_clip=args.reward_target_clip,
                        max_turns_before_forfeit=args.max_turns_before_forfeit,
                        max_concurrent_battles=args.max_concurrent_battles,
                        parallel_pairs=args.selfplay_pairs,
                        seed=args.seed + iteration + collected_battles,
                    )
                )
                batch_steps = sum(len(episode) for episode in next_episodes)
                if batch_steps <= 0:
                    raise RuntimeError("Self-play chunk produced no decision steps; aborting collection.")
                episodes.extend(next_episodes)
                collected_steps += batch_steps
                collected_battles += chunk_battles
                selfplay_stats["wins"] += int(next_stats["wins"])
                selfplay_stats["losses"] += int(next_stats["losses"])
                selfplay_stats["ties"] += int(next_stats["ties"])

                print(
                    f"[iter {iteration}] self-play progress battles={collected_battles} "
                    f"steps={collected_steps}"
                    + (f"/{target_steps}" if target_steps > 0 else f"/chunk_target={chunk_battles}")
                    + f" episodes={len(episodes)} temp={current_temperature:.3f}",
                    flush=True,
                )

                if target_steps > 0:
                    if collected_steps >= target_steps:
                        break
                else:
                    break

            if not bool(args.off_policy_replay):
                replay.clear()
            replay.add_episodes(episodes)
            replay.flush()
            total_env_steps += collected_steps
            print(
                f"[iter {iteration}] self-play done battles={collected_battles} "
                f"steps={collected_steps} episodes={len(episodes)} replay={len(replay)} "
                f"elapsed={time.perf_counter() - phase_t0:.1f}s",
                flush=True,
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

        if len(replay) > 0 and (int(args.train_steps) > 0 or not bool(args.off_policy_replay)):
            use_frozen_rollout = not bool(args.off_policy_replay)
            if use_frozen_rollout:
                rollout_step_budget = optimizer_steps_for_epochs(
                    len(replay),
                    batch_size=int(args.batch_size),
                    grad_accum_steps=int(args.grad_accum_steps),
                    epochs=int(args.train_epochs),
                )
                effective_train_steps = rollout_step_budget if int(args.train_steps) <= 0 else min(
                    int(args.train_steps),
                    rollout_step_budget,
                )
                if int(args.train_steps) > 0 and effective_train_steps < int(args.train_steps):
                    print(
                        f"[iter {iteration}] epoch_budget requested_steps={int(args.train_steps)} "
                        f"effective_steps={effective_train_steps} replay={len(replay)} "
                        f"epochs={int(args.train_epochs)} batch={int(args.batch_size)} "
                        f"grad_accum={int(args.grad_accum_steps)}",
                        flush=True,
                    )
            else:
                effective_train_steps = cap_train_steps_by_reuse(
                    int(args.train_steps),
                    replay_size=len(replay),
                    batch_size=int(args.batch_size),
                    max_reuse_ratio=float(args.max_reuse_ratio),
                )
                if effective_train_steps < int(args.train_steps):
                    print(
                        f"[iter {iteration}] reuse_cap requested_steps={int(args.train_steps)} "
                        f"effective_steps={effective_train_steps} replay={len(replay)} "
                        f"batch={int(args.batch_size)} max_reuse_ratio={float(args.max_reuse_ratio):.2f}",
                        flush=True,
                    )
            phase_t0 = time.perf_counter()
            requested_steps_text = str(int(args.train_steps)) if int(args.train_steps) > 0 else "auto"
            print(
                f"[iter {iteration}] train start steps={effective_train_steps} "
                f"requested_steps={requested_steps_text} batch={args.batch_size} replay={len(replay)} "
                f"target_kl={float(args.target_kl):.3f} "
                f"epochs={int(args.train_epochs)} grad_accum={int(args.grad_accum_steps)} "
                f"effective_batch={int(args.batch_size) * int(args.grad_accum_steps)}",
                flush=True,
            )
            train_fn = train_on_frozen_rollout if use_frozen_rollout else train_on_replay
            train_metrics = train_fn(
                current_model,
                optimizer,
                replay,
                device=args.device,
                config=TrainConfig(
                    steps=effective_train_steps,
                    batch_size=args.batch_size,
                    epochs=args.train_epochs if use_frozen_rollout else 1,
                    grad_accum_steps=args.grad_accum_steps if use_frozen_rollout else 1,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    max_grad_norm=args.max_grad_norm,
                    amp=True,
                    ppo_clip_epsilon=args.ppo_clip_epsilon,
                    ppo_value_clip=args.ppo_value_clip,
                    target_kl=args.target_kl,
                    target_kl_factor=args.target_kl_factor,
                    min_steps_before_early_stop=args.min_steps_before_early_stop,
                    normalize_advantages=not bool(args.no_advantage_norm),
                    policy_temperature=args.selfplay_temperature,
                    use_twohot_value=not bool(args.no_twohot_value),
                    v_min=args.v_min,
                    v_max=args.v_max,
                    v_bins=args.v_bins,
                    lr_warmup_steps=args.lr_warmup_steps,
                    lr_hold_steps=args.lr_hold_steps,
                    lr_total_steps=args.lr_total_steps,
                    lr_backbone_mult=args.lr_backbone_mult,
                    lr_pi_mult=args.lr_pi_mult,
                    lr_v_mult=args.lr_v_mult,
                ),
                scheduler=scheduler,
                progress_interval=max(0, int(args.train_log_interval)),
                progress_prefix=f"[iter {iteration}] ",
            )
            print(
                f"[iter {iteration}] train done elapsed={time.perf_counter() - phase_t0:.1f}s",
                flush=True,
            )

        save_checkpoint(
            path=current_ckpt,
            model=current_model,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=iteration,
            vocab_path=str(vocab_json),
            extra={
                "replay_size": len(replay),
                "total_env_steps": int(total_env_steps),
                "selfplay": selfplay_stats,
                "train_metrics": train_metrics,
            },
        )

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
            eval_stats = asyncio.run(
                evaluate_models(
                    current_model=current_model,
                    best_model=best_model,
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
                )
            )
            print(
                f"[iter {iteration}] eval done elapsed={time.perf_counter() - phase_t0:.1f}s "
                f"win_rate={eval_stats['win_rate']:.3f}",
                flush=True,
            )

            if eval_stats["win_rate"] >= args.promote_threshold:
                best_model.load_state_dict(current_model.state_dict())
                promoted = True
                save_checkpoint(
                    path=best_ckpt,
                    model=best_model,
                    optimizer=None,
                    scheduler=None,
                    iteration=iteration,
                    vocab_path=str(vocab_json),
                    extra={"promoted_from": int(iteration), "eval": eval_stats},
                )
                print(f"[iter {iteration}] promoted current -> best", flush=True)
        elif int(args.eval_battles) > 0:
            print(
                f"[iter {iteration}] eval skip next_interval={int(args.eval_every_iterations)}",
                flush=True,
            )

        print(
            f"iter={iteration} "
            f"replay={len(replay)} "
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

    # Always ensure best checkpoint exists.
    if not best_ckpt.exists():
        shutil.copy2(current_ckpt, best_ckpt)

    replay.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
