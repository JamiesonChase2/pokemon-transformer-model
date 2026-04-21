"""Train a Transformer bot against poke-env's max-damage baseline."""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from scripts.eval_maxdamage import MaxDamagePlayer
from scripts.train_selfplay import (
    _check_server,
    _close_player,
    _load_or_init,
    _seed_replay_with_random_samples,
    annealed_temperature,
    cap_train_steps_by_reuse,
    optimizer_steps_for_epochs,
)
from src.bot import TransformerPlayer
from src.checkpoint import load_checkpoint, save_checkpoint
from src.device import resolve_device
from src.replay import ReplayBuffer, ReplayConfig
from src.train import TrainConfig, train_on_frozen_rollout, train_on_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer bot against the max-damage baseline.")
    parser.add_argument("--checkpoint-dir", default="checkpoints_maxdamage")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--rollout-battles",
        type=int,
        default=8,
        help="Battles per rollout chunk, or total battles if --rollout-steps is unset.",
    )
    parser.add_argument(
        "--rollout-steps",
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
    parser.add_argument("--eval-battles", type=int, default=20, help="Evaluation battles versus max-damage.")
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

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--max-reuse-ratio", type=float, default=0.50)
    parser.add_argument("--no-advantage-norm", action="store_true")

    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-temperature-end", type=float, default=0.9)
    parser.add_argument("--rollout-temperature-total-steps", type=int, default=500000)
    parser.add_argument("--eval-temperature", type=float, default=0.0)

    parser.add_argument("--reward-terminal-win", type=float, default=1.0)
    parser.add_argument("--reward-terminal-loss", type=float, default=-1.0)
    parser.add_argument("--reward-faint-self", type=float, default=-0.1)
    parser.add_argument("--reward-faint-opp", type=float, default=0.1)
    parser.add_argument("--no-faint-reward", action="store_true")
    parser.add_argument("--reward-discount", type=float, default=0.9999)
    parser.add_argument("--gae-lambda", type=float, default=0.75)
    parser.add_argument("--reward-target-clip", type=float, default=1.6)

    parser.add_argument("--no-twohot-value", action="store_true")
    parser.add_argument("--v-min", type=float, default=-1.6)
    parser.add_argument("--v-max", type=float, default=1.6)
    parser.add_argument("--v-bins", type=int, default=51)
    parser.add_argument("--max-turns-before-forfeit", type=int, default=500)
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=1,
        help="Maximum simultaneous battles per player instance during rollout and eval.",
    )

    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--max-len", type=int, default=0)
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
    parser.add_argument("--server-ws", default="ws://localhost:8000/showdown/websocket")
    parser.add_argument("--server-auth", default="http://localhost:8000/action.php?")
    return parser


def _result_counts(player: TransformerPlayer) -> Tuple[int, int, int]:
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
    return wins, losses, ties


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    import math

    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _build_model_player(
    *,
    username: str,
    model: torch.nn.Module,
    vocab: Any,
    battle_format: str,
    server_configuration: ServerConfiguration,
    device: str,
    temperature: float,
    collect_trajectories: bool,
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
) -> TransformerPlayer:
    return TransformerPlayer(
        account_configuration=AccountConfiguration.generate(username, rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=model,
        vocab=vocab,
        device=device,
        temperature=temperature,
        collect_trajectories=collect_trajectories,
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
    )


async def _run_rollout_phase(
    *,
    current_model: torch.nn.Module,
    vocab: Any,
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
) -> Tuple[list[list[dict[str, Any]]], Dict[str, int]]:
    first_half = n_battles // 2
    second_half = n_battles - first_half
    episodes: list[list[dict[str, Any]]] = []
    stats = {"wins": 0, "losses": 0, "ties": 0}

    if first_half > 0:
        model_p1 = _build_model_player(
            username="currmaxp1",
            model=current_model,
            vocab=vocab,
            battle_format=battle_format,
            server_configuration=server_configuration,
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
            max_concurrent_battles=max_concurrent_battles,
            seed=seed,
        )
        max_p2 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("maxdmgp2", rand=True),
            battle_format=battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max(1, int(max_concurrent_battles)),
            start_listening=True,
        )
        try:
            await model_p1.battle_against(max_p2, n_battles=first_half)
            episodes.extend(model_p1.pop_completed_episodes())
            w, l, t = _result_counts(model_p1)
            stats["wins"] += w
            stats["losses"] += l
            stats["ties"] += t
        finally:
            await _close_player(model_p1)
            await _close_player(max_p2)

    if second_half > 0:
        max_p1 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("maxdmgp1", rand=True),
            battle_format=battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max(1, int(max_concurrent_battles)),
            start_listening=True,
        )
        model_p2 = _build_model_player(
            username="currmaxp2",
            model=current_model,
            vocab=vocab,
            battle_format=battle_format,
            server_configuration=server_configuration,
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
            max_concurrent_battles=max_concurrent_battles,
            seed=seed + 1,
        )
        try:
            await max_p1.battle_against(model_p2, n_battles=second_half)
            episodes.extend(model_p2.pop_completed_episodes())
            w, l, t = _result_counts(model_p2)
            stats["wins"] += w
            stats["losses"] += l
            stats["ties"] += t
        finally:
            await _close_player(max_p1)
            await _close_player(model_p2)

    return episodes, stats


async def evaluate_vs_maxdamage(
    *,
    model: torch.nn.Module,
    vocab: Any,
    n_battles: int,
    battle_format: str,
    server_configuration: ServerConfiguration,
    device: str,
    temperature: float,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
) -> Dict[str, float]:
    first_half = n_battles // 2
    second_half = n_battles - first_half
    wins = 0
    losses = 0
    ties = 0

    if first_half > 0:
        model_p1 = _build_model_player(
            username="evalmaxp1",
            model=model,
            vocab=vocab,
            battle_format=battle_format,
            server_configuration=server_configuration,
            device=device,
            temperature=temperature,
            collect_trajectories=False,
            reward_terminal_win=1.0,
            reward_terminal_loss=-1.0,
            reward_use_faint=True,
            reward_faint_self=-0.1,
            reward_faint_opp=0.1,
            reward_discount=0.9999,
            reward_gae_lambda=0.75,
            reward_target_clip=1.6,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max_concurrent_battles,
            seed=seed,
        )
        max_p2 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("evalmaxopp2", rand=True),
            battle_format=battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max(1, int(max_concurrent_battles)),
            start_listening=True,
        )
        try:
            await model_p1.battle_against(max_p2, n_battles=first_half)
            w, l, t = _result_counts(model_p1)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(model_p1)
            await _close_player(max_p2)

    if second_half > 0:
        max_p1 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("evalmaxopp1", rand=True),
            battle_format=battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max(1, int(max_concurrent_battles)),
            start_listening=True,
        )
        model_p2 = _build_model_player(
            username="evalmaxp2",
            model=model,
            vocab=vocab,
            battle_format=battle_format,
            server_configuration=server_configuration,
            device=device,
            temperature=temperature,
            collect_trajectories=False,
            reward_terminal_win=1.0,
            reward_terminal_loss=-1.0,
            reward_use_faint=True,
            reward_faint_self=-0.1,
            reward_faint_opp=0.1,
            reward_discount=0.9999,
            reward_gae_lambda=0.75,
            reward_target_clip=1.6,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max_concurrent_battles,
            seed=seed + 1,
        )
        try:
            await max_p1.battle_against(model_p2, n_battles=second_half)
            w, l, t = _result_counts(model_p2)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(max_p1)
            await _close_player(model_p2)

    decisive = wins + losses
    win_rate = wins / decisive if decisive > 0 else 0.0
    score_rate = (wins + 0.5 * ties) / max(1, wins + losses + ties)
    ci_low, ci_high = _wilson_interval(wins, decisive)
    return {
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "win_rate": float(win_rate),
        "score_rate": float(score_rate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_battles": float(wins + losses + ties),
    }


def main() -> int:
    args = build_parser().parse_args()
    if int(args.max_turns_before_forfeit) < 0:
        args.max_turns_before_forfeit = None
    if int(args.max_concurrent_battles) < 1:
        raise SystemExit("--max-concurrent-battles must be >= 1")
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

    best_eval_win_rate = -1.0
    if best_ckpt.exists():
        try:
            best_payload = load_checkpoint(best_ckpt, args.device)
            best_eval_win_rate = float((best_payload.get("extra") or {}).get("best_eval_win_rate", -1.0))
            if best_eval_win_rate < 0:
                best_eval = (best_payload.get("extra") or {}).get("eval", {})
                best_eval_win_rate = float(best_eval.get("win_rate", -1.0))
        except Exception:
            best_eval_win_rate = -1.0

    server_configuration = ServerConfiguration(
        websocket_url=args.server_ws,
        authentication_url=args.server_auth,
    )

    needs_server = args.rollout_battles > 0 or int(args.rollout_steps) > 0 or args.eval_battles > 0
    if needs_server:
        ok, err = asyncio.run(_check_server(args.server_ws))
        if not ok:
            raise SystemExit(
                "Could not connect to Showdown websocket at "
                f"{args.server_ws}. Error: {err}\n"
                "Start a local server (example):\n"
                "  node pokemon-showdown start --no-security --port 8000\n"
                "Then rerun training."
            )

    if len(replay) == 0 and args.bootstrap_random_samples > 0:
        _seed_replay_with_random_samples(replay, num_samples=args.bootstrap_random_samples, vocab=vocab, seed=args.seed)

    for iteration in range(start_iter, start_iter + max(0, args.iterations)):
        iter_start = time.perf_counter()
        print(
            f"[iter {iteration}] start rollout_battles={args.rollout_battles} "
            f"rollout_steps={int(args.rollout_steps)} train_steps={args.train_steps} "
            f"train_epochs={int(args.train_epochs)} eval_battles={args.eval_battles} "
            f"max_concurrent_battles={int(args.max_concurrent_battles)} "
            f"total_env_steps={total_env_steps} lr={optimizer.param_groups[0]['lr']:.6g} "
            f"best_eval_win_rate={best_eval_win_rate:.3f}",
            flush=True,
        )
        rollout_stats = {"wins": 0, "losses": 0, "ties": 0}

        if args.rollout_battles > 0 or int(args.rollout_steps) > 0:
            phase_t0 = time.perf_counter()
            print(f"[iter {iteration}] rollout start opponent=maxdamage", flush=True)
            current_temperature = annealed_temperature(
                start=float(args.rollout_temperature),
                end=float(args.rollout_temperature_end),
                total_steps=int(args.rollout_temperature_total_steps),
                current_steps=int(total_env_steps),
            )
            episodes = []
            collected_steps = 0
            collected_battles = 0
            chunk_battles = max(1, int(args.rollout_battles))
            target_steps = max(0, int(args.rollout_steps))

            while True:
                next_episodes, next_stats = asyncio.run(
                    _run_rollout_phase(
                        current_model=current_model,
                        vocab=vocab,
                        n_battles=chunk_battles,
                        battle_format=args.battle_format,
                        server_configuration=server_configuration,
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
                        seed=args.seed + iteration + collected_battles,
                    )
                )
                batch_steps = sum(len(episode) for episode in next_episodes)
                if batch_steps <= 0:
                    raise RuntimeError("Rollout chunk produced no decision steps; aborting collection.")

                episodes.extend(next_episodes)
                collected_steps += batch_steps
                collected_battles += chunk_battles
                rollout_stats["wins"] += int(next_stats["wins"])
                rollout_stats["losses"] += int(next_stats["losses"])
                rollout_stats["ties"] += int(next_stats["ties"])

                print(
                    f"[iter {iteration}] rollout progress battles={collected_battles} "
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
                f"[iter {iteration}] rollout done battles={collected_battles} "
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
            else:
                effective_train_steps = cap_train_steps_by_reuse(
                    int(args.train_steps),
                    replay_size=len(replay),
                    batch_size=int(args.batch_size),
                    max_reuse_ratio=float(args.max_reuse_ratio),
                )

            phase_t0 = time.perf_counter()
            requested_steps_text = str(int(args.train_steps)) if int(args.train_steps) > 0 else "auto"
            print(
                f"[iter {iteration}] train start steps={effective_train_steps} "
                f"requested_steps={requested_steps_text} batch={args.batch_size} replay={len(replay)} "
                f"target_kl={float(args.target_kl):.3f} epochs={int(args.train_epochs)} "
                f"grad_accum={int(args.grad_accum_steps)} "
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
                    normalize_advantages=not bool(args.no_advantage_norm),
                    policy_temperature=args.rollout_temperature,
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
            print(f"[iter {iteration}] train done elapsed={time.perf_counter() - phase_t0:.1f}s", flush=True)

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
                "rollout": rollout_stats,
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

        if args.eval_battles > 0:
            phase_t0 = time.perf_counter()
            print(f"[iter {iteration}] eval start battles={args.eval_battles} opponent=maxdamage", flush=True)
            eval_stats = asyncio.run(
                evaluate_vs_maxdamage(
                    model=current_model,
                    vocab=vocab,
                    n_battles=int(args.eval_battles),
                    battle_format=args.battle_format,
                    server_configuration=server_configuration,
                    device=args.device,
                    temperature=float(args.eval_temperature),
                    max_turns_before_forfeit=args.max_turns_before_forfeit,
                    max_concurrent_battles=args.max_concurrent_battles,
                    seed=args.seed + iteration,
                )
            )
            print(
                f"[iter {iteration}] eval done elapsed={time.perf_counter() - phase_t0:.1f}s "
                f"win_rate={eval_stats['win_rate']:.3f}",
                flush=True,
            )

            if eval_stats["win_rate"] >= best_eval_win_rate:
                best_eval_win_rate = float(eval_stats["win_rate"])
                best_model.load_state_dict(current_model.state_dict())
                promoted = True
                save_checkpoint(
                    path=best_ckpt,
                    model=best_model,
                    optimizer=None,
                    scheduler=None,
                    iteration=iteration,
                    vocab_path=str(vocab_json),
                    extra={
                        "promoted_from": int(iteration),
                        "best_eval_win_rate": float(best_eval_win_rate),
                        "eval": eval_stats,
                    },
                )
                print(f"[iter {iteration}] promoted current -> best", flush=True)

        print(
            f"iter={iteration} replay={len(replay)} "
            f"rollout_wlt={rollout_stats['wins']}/{rollout_stats['losses']}/{rollout_stats['ties']} "
            f"loss={train_metrics['loss']:.4f} policy_loss={train_metrics['policy_loss']:.4f} "
            f"value_loss={train_metrics['value_loss']:.4f} entropy={train_metrics['policy_entropy']:.4f} "
            f"kl={train_metrics['approx_kl']:.4f} clip_frac={train_metrics['clip_frac']:.3f} "
            f"eval_win_rate={eval_stats['win_rate']:.3f} "
            f"eval_ci=[{eval_stats['ci_low']:.3f},{eval_stats['ci_high']:.3f}] "
            f"best_eval_win_rate={best_eval_win_rate:.3f} promoted={promoted} "
            f"iter_elapsed={time.perf_counter() - iter_start:.1f}s"
        )

    replay.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
