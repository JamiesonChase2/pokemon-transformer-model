"""Behavior-cloning trainer using poke-env's SimpleHeuristicsPlayer."""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from scripts.train_selfplay import (
    _build_server_configurations,
    _check_server,
    _close_player,
    _load_or_init,
    optimizer_steps_for_epochs,
)
from src.bot import SimpleHeuristicImitationPlayer, TransformerPlayer
from src.checkpoint import load_checkpoint, save_checkpoint
from src.device import resolve_device
from src.replay import ReplayBuffer, ReplayConfig
from src.train import TrainConfig, train_on_imitation_rollout
from src.vocab import ObservationVocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior-clone SimpleHeuristicsPlayer into the Transformer policy.")
    parser.add_argument("--checkpoint-dir", default="checkpoints_imitation")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--demo-battles", type=int, default=50, help="Battles per demonstration chunk.")
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=32768,
        help="Collect at least this many labeled decisions per iteration. Use 0 for a single chunk.",
    )
    parser.add_argument(
        "--clear-replay-each-iter",
        action="store_true",
        help="Discard older demonstrations before collecting the next iteration.",
    )
    parser.add_argument(
        "--collect-demos-once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect demonstrations only until replay is non-empty, then keep training on the same frozen dataset.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="Optimizer steps per iteration. 0 means use the full frozen-dataset epoch budget.",
    )
    parser.add_argument("--train-epochs", type=int, default=3, help="Supervised epochs over each fresh update batch.")
    parser.add_argument(
        "--train-log-interval",
        type=int,
        default=20,
        help="Print training progress every N optimizer steps (0 disables step-level logs).",
    )

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-hold-steps", type=int, default=20000)
    parser.add_argument("--lr-total-steps", type=int, default=500000)
    parser.add_argument("--lr-backbone-mult", type=float, default=1.0)
    parser.add_argument("--lr-pi-mult", type=float, default=1.0)
    parser.add_argument("--lr-v-mult", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Teacher-label smoothing. ps-ppo imitation used 0.1.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Held-out fraction of demonstrations used for BC validation and gating.",
    )
    parser.add_argument(
        "--fixed-validation-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the same train/validation split each iteration so BC accuracy stays directly comparable.",
    )
    parser.add_argument(
        "--gate-train-accuracy",
        type=float,
        default=0.995,
        help="Minimum training accuracy required to pass the BC gate.",
    )
    parser.add_argument(
        "--gate-val-accuracy",
        type=float,
        default=0.99,
        help="Minimum validation accuracy required to pass the BC gate.",
    )
    parser.add_argument(
        "--stop-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop early once the BC gate is satisfied.",
    )
    parser.add_argument(
        "--require-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if the BC gate was not satisfied by the end of the run.",
    )

    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-twohot-value", action="store_true")
    parser.add_argument("--v-min", type=float, default=-1.6)
    parser.add_argument("--v-max", type=float, default=1.6)
    parser.add_argument("--v-bins", type=int, default=51)
    parser.add_argument("--max-turns-before-forfeit", type=int, default=300)
    parser.add_argument("--max-concurrent-battles", type=int, default=8)

    parser.add_argument("--replay-shard-size", type=int, default=2048)
    parser.add_argument("--replay-max-shards", type=int, default=256)
    parser.add_argument("--showdown-root", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--server-ws", default="ws://localhost:8000/showdown/websocket")
    parser.add_argument("--server-auth", default="http://localhost:8000/action.php?")
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
    parser.add_argument("--eval-battles", type=int, default=100, help="Battles for periodic eval vs SimpleHeuristics.")
    parser.add_argument("--eval-every-updates", type=int, default=25, help="Run eval every N BC updates.")
    parser.add_argument("--eval-pairs", type=int, default=1, help="Concurrent eval pairs to run across servers.")
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="Greedy-style eval temperature for the model.")
    return parser


def _build_teacher_player(
    *,
    username: str,
    model: torch.nn.Module | None,
    vocab: Any,
    battle_format: str,
    server_configuration: ServerConfiguration,
    device: str,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
) -> SimpleHeuristicImitationPlayer:
    teacher_model = model if model is not None else torch.nn.Identity()
    return SimpleHeuristicImitationPlayer(
        account_configuration=AccountConfiguration.generate(username, rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=teacher_model,
        vocab=vocab,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=seed,
        start_listening=True,
    )


async def _collect_demo_chunk(
    *,
    model: torch.nn.Module | None,
    vocab: Any,
    n_battles: int,
    battle_format: str,
    server_configuration: ServerConfiguration,
    device: str,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
    username_prefix: str = "bct",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    player_one = _build_teacher_player(
        username=f"{username_prefix}a",
        model=model,
        vocab=vocab,
        battle_format=battle_format,
        server_configuration=server_configuration,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max_concurrent_battles,
        seed=seed,
    )
    player_two = _build_teacher_player(
        username=f"{username_prefix}b",
        model=model,
        vocab=vocab,
        battle_format=battle_format,
        server_configuration=server_configuration,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max_concurrent_battles,
        seed=seed + 1,
    )
    try:
        await player_one.battle_against(player_two, n_battles=int(n_battles))
        samples = player_one.pop_completed_samples() + player_two.pop_completed_samples()
        skipped = player_one.pop_skipped_samples() + player_two.pop_skipped_samples()
        return samples, {
            "battles": int(n_battles),
            "samples": len(samples),
            "skipped": int(skipped),
        }
    finally:
        await _close_player(player_one)
        await _close_player(player_two)


def _distribute_battles(total_battles: int, n_servers: int) -> List[int]:
    if total_battles <= 0 or n_servers <= 0:
        return []
    base = total_battles // n_servers
    remainder = total_battles % n_servers
    counts = [base + (1 if idx < remainder else 0) for idx in range(n_servers)]
    return [count for count in counts if count > 0]


def _collect_demo_chunk_parallel(
    *,
    vocab: ObservationVocabulary,
    total_battles: int,
    battle_format: str,
    server_configurations: List[ServerConfiguration],
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
    executor: ProcessPoolExecutor | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    if not server_configurations:
        raise RuntimeError("No server configurations available for BC demo collection.")
    if executor is None:
        raise RuntimeError("BC multi-server collection requires a process executor.")

    battle_counts = _distribute_battles(max(1, int(total_battles)), len(server_configurations))
    active_configs = server_configurations[: len(battle_counts)]
    vocab_state = _vocab_payload(vocab)
    futures = [
        (server_idx, n_battles, server_configuration, executor.submit(
            _collect_demo_chunk_worker,
            vocab_payload=vocab_state,
            n_battles=n_battles,
            battle_format=battle_format,
            server_ws=server_configuration.websocket_url,
            server_auth=server_configuration.authentication_url,
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max_concurrent_battles,
            seed=seed + (server_idx * 9973),
            username_prefix=f"bc{server_idx:02d}",
        ))
        for server_idx, (server_configuration, n_battles) in enumerate(zip(active_configs, battle_counts))
    ]

    payload_parts: List[Dict[str, torch.Tensor]] = []
    total_stats = {"battles": 0, "samples": 0, "skipped": 0}
    for _server_idx, _n_battles, _server_configuration, future in futures:
        payload, stats = future.result()
        if int(payload["action_index"].numel()) > 0:
            payload_parts.append(payload)
        total_stats["battles"] += int(stats.get("battles", 0))
        total_stats["samples"] += int(stats.get("samples", 0))
        total_stats["skipped"] += int(stats.get("skipped", 0))
    if payload_parts:
        merged_payload = {
            "obs": torch.cat([part["obs"] for part in payload_parts], dim=0),
            "legal_mask": torch.cat([part["legal_mask"] for part in payload_parts], dim=0),
            "action_index": torch.cat([part["action_index"] for part in payload_parts], dim=0),
        }
    else:
        merged_payload = _pack_demo_samples([])
    return merged_payload, total_stats


async def _check_servers_parallel(
    server_configurations: List[ServerConfiguration],
) -> List[Tuple[bool, str]]:
    return await asyncio.gather(*[_check_server(cfg.websocket_url) for cfg in server_configurations])


def _vocab_payload(vocab: ObservationVocabulary) -> Dict[str, Any]:
    return {
        "categories": {key: list(values) for key, values in vocab.categories.items()},
        "schema_version": int(vocab.schema_version),
    }


def _rebuild_vocab(payload: Dict[str, Any]) -> ObservationVocabulary:
    return ObservationVocabulary(
        categories={str(key): [str(value) for value in values] for key, values in payload["categories"].items()},
        schema_version=int(payload.get("schema_version", 0)),
    )


def _pack_demo_samples(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not samples:
        return {
            "obs": torch.empty((0,), dtype=torch.float32),
            "legal_mask": torch.empty((0,), dtype=torch.bool),
            "action_index": torch.empty((0,), dtype=torch.long),
        }
    return {
        "obs": torch.stack([sample["obs"] for sample in samples], dim=0),
        "legal_mask": torch.stack([sample["legal_mask"] for sample in samples], dim=0),
        "action_index": torch.tensor([int(sample["action_index"]) for sample in samples], dtype=torch.long),
    }


def _collect_demo_chunk_worker(
    *,
    vocab_payload: Dict[str, Any],
    n_battles: int,
    battle_format: str,
    server_ws: str,
    server_auth: str,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    seed: int,
    username_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    vocab = _rebuild_vocab(vocab_payload)
    server_configuration = ServerConfiguration(
        websocket_url=str(server_ws),
        authentication_url=str(server_auth),
    )
    samples, stats = asyncio.run(
        _collect_demo_chunk(
            model=None,
            vocab=vocab,
            n_battles=int(n_battles),
            battle_format=battle_format,
            server_configuration=server_configuration,
            device="cpu",
            max_turns_before_forfeit=max_turns_before_forfeit,
            max_concurrent_battles=max_concurrent_battles,
            seed=seed,
            username_prefix=username_prefix,
        )
    )
    return _pack_demo_samples(samples), stats


def _split_replay_for_gate(
    replay: ReplayBuffer,
    *,
    validation_fraction: float,
    seed: int,
    workspace_root: Path | None = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor] | None, int, int]:
    _ = workspace_root
    materialized = replay.materialize(device="cpu")
    num_samples = int(materialized["obs"].shape[0])

    if num_samples < 2 or float(validation_fraction) <= 0.0:
        return materialized, None, num_samples, 0

    val_count = int(round(num_samples * float(validation_fraction)))
    val_count = max(1, min(num_samples - 1, val_count))

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator)
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]

    train_data = {
        key: value[train_indices].clone()
        for key, value in materialized.items()
    }
    val_data = {
        key: value[val_indices].clone()
        for key, value in materialized.items()
    }
    return train_data, val_data, int(train_indices.numel()), int(val_indices.numel())


class _TurnCapMixin:
    def __init__(self, *args, max_turns_before_forfeit: int | None = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns_before_forfeit = (
            int(max_turns_before_forfeit) if max_turns_before_forfeit is not None else None
        )

    def _maybe_forfeit(self, battle):
        if self.max_turns_before_forfeit is None:
            return None
        turn = int(getattr(battle, "turn", 0) or 0)
        if turn > self.max_turns_before_forfeit:
            return ForfeitBattleOrder()
        return None


class SimpleHeuristicEvalPlayer(_TurnCapMixin, SimpleHeuristicsPlayer):
    def choose_move(self, battle):  # type: ignore[override]
        forfeit = self._maybe_forfeit(battle)
        if forfeit is not None:
            return forfeit
        return super().choose_move(battle)


def _result_counts(player) -> tuple[int, int, int]:
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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _split_counts(total: int, parts: int) -> list[int]:
    total = max(0, int(total))
    parts = max(1, int(parts))
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


async def _run_eval_pair_vs_simple(
    *,
    model: torch.nn.Module,
    vocab: ObservationVocabulary,
    server_configuration: ServerConfiguration,
    battle_format: str,
    temperature: float,
    device: str,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    n_battles: int,
    pair_index: int,
    seed: int | None,
    model_on_left: bool,
    account_prefix: str = "",
) -> tuple[int, int, int]:
    if int(n_battles) <= 0:
        return 0, 0, 0

    model_player = TransformerPlayer(
        account_configuration=AccountConfiguration.generate(f"{account_prefix}modeleval{pair_index}", rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=model,
        vocab=vocab,
        temperature=float(temperature),
        collect_trajectories=False,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=None if seed is None else int(seed) + (2 * pair_index),
        start_listening=True,
    )
    heuristic_player = SimpleHeuristicEvalPlayer(
        account_configuration=AccountConfiguration.generate(f"{account_prefix}heurieval{pair_index}", rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        start_listening=True,
    )

    left_player = model_player if model_on_left else heuristic_player
    right_player = heuristic_player if model_on_left else model_player
    try:
        await left_player.battle_against(right_player, n_battles=int(n_battles))
        if model_on_left:
            return _result_counts(model_player)
        wins, losses, ties = _result_counts(model_player)
        return wins, losses, ties
    finally:
        await _close_player(model_player)
        await _close_player(heuristic_player)


async def evaluate_vs_simple_heuristic(
    *,
    model: torch.nn.Module,
    vocab: ObservationVocabulary,
    server_configurations: Sequence[ServerConfiguration],
    n_battles: int,
    battle_format: str,
    temperature: float,
    device: str,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    parallel_pairs: int,
    seed: int | None,
    account_prefix: str = "",
) -> Dict[str, float]:
    total_battles = max(0, int(n_battles))
    if total_battles <= 0:
        return {
            "wins": 0.0,
            "losses": 0.0,
            "ties": 0.0,
            "win_rate": 0.0,
            "score_rate": 0.0,
            "ci_low": 0.0,
            "ci_high": 1.0,
            "n_battles": 0.0,
        }
    if not server_configurations:
        raise ValueError("At least one healthy server is required for BC eval.")

    first_half = total_battles // 2
    second_half = total_battles - first_half
    wins = 0
    losses = 0
    ties = 0

    async def _run_half(count: int, *, model_on_left: bool, seed_offset: int) -> tuple[int, int, int]:
        if count <= 0:
            return 0, 0, 0
        pair_count = max(1, min(int(parallel_pairs), len(server_configurations), count))
        pair_battles = _split_counts(count, pair_count)
        results = await asyncio.gather(
            *[
                _run_eval_pair_vs_simple(
                    model=model,
                    vocab=vocab,
                    server_configuration=server_configurations[idx % len(server_configurations)],
                    battle_format=battle_format,
                    temperature=temperature,
                    device=device,
                    max_turns_before_forfeit=max_turns_before_forfeit,
                    max_concurrent_battles=max_concurrent_battles,
                    n_battles=battle_count,
                    pair_index=seed_offset + idx,
                    seed=None if seed is None else int(seed) + seed_offset,
                    model_on_left=model_on_left,
                    account_prefix=account_prefix,
                )
                for idx, battle_count in enumerate(pair_battles)
                if battle_count > 0
            ]
        )
        out_wins = 0
        out_losses = 0
        out_ties = 0
        for w, l, t in results:
            out_wins += int(w)
            out_losses += int(l)
            out_ties += int(t)
        return out_wins, out_losses, out_ties

    left_wins, left_losses, left_ties = await _run_half(first_half, model_on_left=True, seed_offset=0)
    wins += left_wins
    losses += left_losses
    ties += left_ties

    right_wins, right_losses, right_ties = await _run_half(second_half, model_on_left=False, seed_offset=10_000)
    wins += right_wins
    losses += right_losses
    ties += right_ties

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
    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if int(args.max_turns_before_forfeit) < 0:
        args.max_turns_before_forfeit = None
    if int(args.max_concurrent_battles) < 1:
        raise SystemExit("--max-concurrent-battles must be >= 1")
    if int(args.eval_pairs) < 1:
        raise SystemExit("--eval-pairs must be >= 1")
    if bool(args.clear_replay_each_iter) and bool(args.collect_demos_once):
        raise SystemExit("--clear-replay-each-iter and --collect-demos-once are mutually exclusive")
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}", flush=True)
    else:
        print(f"[device] using={args.device}", flush=True)

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
            best_extra = best_payload.get("extra") or {}
            best_eval_win_rate = float(best_extra.get("best_eval_win_rate", (best_extra.get("eval") or {}).get("win_rate", -1.0)))
        except Exception:
            best_eval_win_rate = -1.0

    server_configurations = _build_server_configurations(args)
    server_checks = asyncio.run(_check_servers_parallel(server_configurations))
    healthy_server_configurations: List[ServerConfiguration] = []
    unhealthy_messages: List[str] = []
    for cfg, (ok, err) in zip(server_configurations, server_checks):
        if ok:
            healthy_server_configurations.append(cfg)
        else:
            unhealthy_messages.append(f"{cfg.websocket_url} ({err})")
    if not healthy_server_configurations:
        raise SystemExit(
            "Could not connect to any Showdown websocket.\n"
            + "\n".join(f"  - {message}" for message in unhealthy_messages)
            + "\nStart a local server (example):\n"
            + "  node pokemon-showdown start --no-security --port 8000\n"
            + "Then rerun training."
        )
    if unhealthy_messages:
        print(
            "[server] skipping unreachable servers: " + "; ".join(unhealthy_messages),
            flush=True,
        )

    print(
        f"[config] bc_mode=ps_ppo_style updates={int(args.iterations)} "
        f"steps_per_update={int(args.demo_steps)} train_epochs={int(args.train_epochs)} "
        f"label_smoothing={float(args.label_smoothing):.3f} "
        f"lr_v_mult={float(args.lr_v_mult):.3f} eval_every_updates={int(args.eval_every_updates)} "
        f"eval_battles={int(args.eval_battles)}",
        flush=True,
    )
    print(
        "[config] validation/gating flags are ignored in ps-ppo-style BC; model selection uses periodic heuristic eval.",
        flush=True,
    )

    executor_workers = max(1, len(healthy_server_configurations))
    with ProcessPoolExecutor(
        max_workers=executor_workers,
        mp_context=get_context("spawn"),
    ) as demo_executor:
        for update_idx in range(start_iter, start_iter + max(0, args.iterations)):
            update_start = time.perf_counter()
            replay.clear()
            print(
                f"[upd {update_idx}] start demo_battles={args.demo_battles} "
                f"steps_per_update={int(args.demo_steps)} train_steps={args.train_steps} "
                f"train_epochs={int(args.train_epochs)} max_concurrent_battles={int(args.max_concurrent_battles)} "
                f"servers={len(healthy_server_configurations)} workers={executor_workers} "
                f"total_env_steps={total_env_steps} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"best_eval_win_rate={best_eval_win_rate:.3f} "
                f"label_smoothing={float(args.label_smoothing):.3f}",
                flush=True,
            )

            collect_t0 = time.perf_counter()
            collected_steps = 0
            collected_battles = 0
            skipped_samples = 0
            chunk_battles = max(1, int(args.demo_battles))
            target_steps = max(0, int(args.demo_steps))

            print(f"[upd {update_idx}] demo start opponent=simple_heuristic", flush=True)
            while True:
                payload, stats = _collect_demo_chunk_parallel(
                    vocab=vocab,
                    total_battles=chunk_battles,
                    battle_format=args.battle_format,
                    server_configurations=healthy_server_configurations,
                    max_turns_before_forfeit=args.max_turns_before_forfeit,
                    max_concurrent_battles=args.max_concurrent_battles,
                    seed=args.seed + update_idx + collected_battles,
                    executor=demo_executor,
                )
                batch_steps = int(payload["action_index"].shape[0])
                if batch_steps <= 0:
                    raise RuntimeError("Demonstration chunk produced no labeled samples; aborting collection.")

                for sample_idx in range(batch_steps):
                    replay.add_sample(
                        obs=payload["obs"][sample_idx],
                        legal_mask=payload["legal_mask"][sample_idx],
                        action_index=int(payload["action_index"][sample_idx].item()),
                        outcome_z=0.0,
                        old_log_prob=0.0,
                        old_value=0.0,
                        advantage=0.0,
                    )
                replay.flush()

                collected_steps += batch_steps
                collected_battles += int(stats["battles"])
                skipped_samples += int(stats["skipped"])
                print(
                    f"[upd {update_idx}] demo progress battles={collected_battles} "
                    f"steps={collected_steps}"
                    + (f"/{target_steps}" if target_steps > 0 else f"/chunk_target={chunk_battles}")
                    + f" skipped={skipped_samples}",
                    flush=True,
                )

                if target_steps > 0:
                    if collected_steps >= target_steps:
                        break
                else:
                    break

            total_env_steps += collected_steps
            train_samples = len(replay)
            print(
                f"[upd {update_idx}] demo done battles={collected_battles} "
                f"steps={collected_steps} replay={train_samples} skipped={skipped_samples} "
                f"elapsed={time.perf_counter() - collect_t0:.1f}s",
                flush=True,
            )

            effective_train_steps = optimizer_steps_for_epochs(
                train_samples,
                batch_size=int(args.batch_size),
                grad_accum_steps=int(args.grad_accum_steps),
                epochs=int(args.train_epochs),
            )
            if int(args.train_steps) > 0:
                effective_train_steps = min(int(args.train_steps), int(effective_train_steps))

            train_t0 = time.perf_counter()
            print(
                f"[upd {update_idx}] train start steps={effective_train_steps} "
                f"requested_steps={'auto' if int(args.train_steps) <= 0 else int(args.train_steps)} "
                f"batch={args.batch_size} replay={train_samples} epochs={int(args.train_epochs)} "
                f"grad_accum={int(args.grad_accum_steps)} "
                f"effective_batch={int(args.batch_size) * int(args.grad_accum_steps)}",
                flush=True,
            )
            train_metrics = train_on_imitation_rollout(
                current_model,
                optimizer,
                replay,
                device=args.device,
                config=TrainConfig(
                    steps=int(effective_train_steps),
                    batch_size=int(args.batch_size),
                    epochs=int(args.train_epochs),
                    grad_accum_steps=int(args.grad_accum_steps),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    max_grad_norm=float(args.max_grad_norm),
                    amp=True,
                    use_twohot_value=not bool(args.no_twohot_value),
                    v_min=float(args.v_min),
                    v_max=float(args.v_max),
                    v_bins=int(args.v_bins),
                    lr_warmup_steps=int(args.lr_warmup_steps),
                    lr_hold_steps=int(args.lr_hold_steps),
                    lr_total_steps=int(args.lr_total_steps),
                    lr_backbone_mult=float(args.lr_backbone_mult),
                    lr_pi_mult=float(args.lr_pi_mult),
                    lr_v_mult=float(args.lr_v_mult),
                    imitation_label_smoothing=float(args.label_smoothing),
                ),
                scheduler=scheduler,
                progress_interval=max(0, int(args.train_log_interval)),
                progress_prefix=f"[upd {update_idx}] ",
            )
            print(f"[upd {update_idx}] train done elapsed={time.perf_counter() - train_t0:.1f}s", flush=True)

            eval_stats: Dict[str, float] | None = None
            promoted = False
            should_eval = int(args.eval_battles) > 0 and int(args.eval_every_updates) > 0 and ((int(update_idx) + 1) % int(args.eval_every_updates) == 0)
            if should_eval:
                eval_t0 = time.perf_counter()
                print(
                    f"[upd {update_idx}] eval start battles={int(args.eval_battles)} opponent=simple_heuristic",
                    flush=True,
                )
                eval_stats = asyncio.run(
                    evaluate_vs_simple_heuristic(
                        model=current_model,
                        vocab=vocab,
                        server_configurations=healthy_server_configurations,
                        n_battles=int(args.eval_battles),
                        battle_format=args.battle_format,
                        temperature=float(args.eval_temperature),
                        device=args.device,
                        max_turns_before_forfeit=args.max_turns_before_forfeit,
                        max_concurrent_battles=int(args.max_concurrent_battles),
                        parallel_pairs=int(args.eval_pairs),
                        seed=int(args.seed) + int(update_idx),
                        account_prefix=f"bcupd{update_idx}_",
                    )
                )
                print(
                    f"[upd {update_idx}] eval done elapsed={time.perf_counter() - eval_t0:.1f}s "
                    f"win_rate={eval_stats['win_rate']:.3f} "
                    f"score_rate={eval_stats['score_rate']:.3f} "
                    f"ci95=[{eval_stats['ci_low']:.3f},{eval_stats['ci_high']:.3f}]",
                    flush=True,
                )
                if float(eval_stats["win_rate"]) >= float(best_eval_win_rate):
                    best_eval_win_rate = float(eval_stats["win_rate"])
                    best_model.load_state_dict(current_model.state_dict())
                    promoted = True
                    save_checkpoint(
                        path=best_ckpt,
                        model=best_model,
                        optimizer=None,
                        scheduler=None,
                        iteration=update_idx,
                        vocab_path=str(vocab_json),
                        extra={
                            "best_eval_win_rate": float(best_eval_win_rate),
                            "eval": dict(eval_stats),
                            "train_metrics": train_metrics,
                        },
                    )
                    print(f"[upd {update_idx}] promoted current -> best", flush=True)

            save_checkpoint(
                path=current_ckpt,
                model=current_model,
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=update_idx,
                vocab_path=str(vocab_json),
                extra={
                    "replay_size": int(train_samples),
                    "total_env_steps": int(total_env_steps),
                    "demo_steps": int(collected_steps),
                    "skipped_samples": int(skipped_samples),
                    "train_metrics": train_metrics,
                    "eval": dict(eval_stats) if eval_stats is not None else None,
                    "best_eval_win_rate": float(best_eval_win_rate),
                },
            )

            print(
                f"upd={update_idx} replay={train_samples} demo_steps={collected_steps} "
                f"skipped={skipped_samples} loss={train_metrics['loss']:.4f} "
                f"policy_loss={train_metrics['policy_loss']:.4f} "
                f"value_loss={train_metrics['value_loss']:.4f} "
                f"entropy={train_metrics['policy_entropy']:.4f} "
                f"train_accuracy={float(train_metrics.get('accuracy', 0.0)):.3f} "
                f"eval_win_rate={float(eval_stats['win_rate']) if eval_stats is not None else float('nan'):.3f} "
                f"best_eval_win_rate={best_eval_win_rate:.3f} promoted={promoted} "
                f"upd_elapsed={time.perf_counter() - update_start:.1f}s",
                flush=True,
            )

            replay.clear()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
