"""Benchmark rollout throughput across worker/server configurations.

The primary metric is decision-step throughput. The script also reports an
approximate turn latency assuming two policy decisions per full battle turn.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from scripts.train_selfplay import _build_server_configurations, _check_server, _split_counts
from scripts.train_selfplay_ray import (
    _build_inference_actor,
    _build_weight_store_actor,
    _build_rollout_worker_actor,
    _resolve_inference_device,
    _resolve_worker_device,
    ray,
    summarize_worker_stats,
)
from src.checkpoint import ENCODER_TYPE, load_checkpoint
from src.device import resolve_device


@dataclass(frozen=True)
class BenchmarkConfig:
    rollout_workers: int
    server_count: int
    selfplay_pairs: int
    max_concurrent_battles: int


@dataclass(frozen=True)
class BenchmarkMetrics:
    rollout_workers: int
    server_count: int
    selfplay_pairs: int
    max_concurrent_battles: int
    repeat_index: int
    elapsed_s: float
    steps: int
    battles: int
    episodes: int
    wins: int
    losses: int
    ties: int
    steps_per_sec: float
    battles_per_sec: float
    ms_per_step: float
    approx_turn_ms: float
    avg_steps_per_battle: float


def _positive_int_grid(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"Grid values must be >= 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one positive integer value is required.")
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_config_grid(
    *,
    rollout_workers: Sequence[int],
    server_counts: Sequence[int],
    selfplay_pairs: Sequence[int],
    max_concurrent_battles: Sequence[int],
    match_workers_to_servers: bool,
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    if match_workers_to_servers:
        for server_count in server_counts:
            worker_count = int(server_count)
            for pair_count in selfplay_pairs:
                for concurrent_count in max_concurrent_battles:
                    configs.append(
                        BenchmarkConfig(
                            rollout_workers=worker_count,
                            server_count=int(server_count),
                            selfplay_pairs=int(pair_count),
                            max_concurrent_battles=int(concurrent_count),
                        )
                    )
        return configs

    for worker_count in rollout_workers:
        for server_count in server_counts:
            for pair_count in selfplay_pairs:
                for concurrent_count in max_concurrent_battles:
                    configs.append(
                        BenchmarkConfig(
                            rollout_workers=int(worker_count),
                            server_count=int(server_count),
                            selfplay_pairs=int(pair_count),
                            max_concurrent_battles=int(concurrent_count),
                        )
                    )
    return configs


def _derive_metrics(
    *,
    config: BenchmarkConfig,
    repeat_index: int,
    elapsed_s: float,
    steps: int,
    battles: int,
    episodes: int,
    wins: int,
    losses: int,
    ties: int,
) -> BenchmarkMetrics:
    if elapsed_s <= 0.0:
        raise ValueError("elapsed_s must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    steps_per_sec = float(steps) / float(elapsed_s)
    battles_per_sec = float(battles) / float(elapsed_s) if battles > 0 else 0.0
    ms_per_step = 1000.0 * float(elapsed_s) / float(steps)
    return BenchmarkMetrics(
        rollout_workers=int(config.rollout_workers),
        server_count=int(config.server_count),
        selfplay_pairs=int(config.selfplay_pairs),
        max_concurrent_battles=int(config.max_concurrent_battles),
        repeat_index=int(repeat_index),
        elapsed_s=float(elapsed_s),
        steps=int(steps),
        battles=int(battles),
        episodes=int(episodes),
        wins=int(wins),
        losses=int(losses),
        ties=int(ties),
        steps_per_sec=steps_per_sec,
        battles_per_sec=battles_per_sec,
        ms_per_step=ms_per_step,
        approx_turn_ms=ms_per_step * 2.0,
        avg_steps_per_battle=(float(steps) / float(battles)) if battles > 0 else 0.0,
    )


def _create_snapshot(
    *,
    current_payload: Mapping[str, Any],
    best_payload: Mapping[str, Any],
    version: int,
) -> dict[str, Any]:
    def _strip_optimizer(payload: Mapping[str, Any]) -> dict[str, Any]:
        stripped = dict(payload)
        stripped["encoder_type"] = stripped.get("encoder_type", ENCODER_TYPE)
        stripped["optimizer_state_dict"] = None
        stripped["scheduler_state_dict"] = None
        return stripped

    return {
        "version": int(version),
        "current_lr": 0.0,
        "current": _strip_optimizer(current_payload),
        "best": _strip_optimizer(best_payload),
    }


def _resolve_benchmark_inference_gpus(args: argparse.Namespace, *, inference_device: str) -> float:
    requested = float(args.inference_gpus)
    if inference_device != "cuda":
        return 0.0
    if requested > 0.0:
        return requested
    return 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark rollout throughput for Ray self-play configurations.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint directory containing current.pt, best.pt, and vocab.json.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device used by the benchmarked model snapshot. 'auto' resolves to the best available device.",
    )
    parser.add_argument(
        "--battle-format",
        default="gen9randombattle",
        help="Battle format used during rollout benchmarking.",
    )
    parser.add_argument(
        "--rollout-workers-grid",
        default="1,2,4",
        help="Comma-separated rollout worker counts to test when worker/server matching is disabled.",
    )
    parser.add_argument(
        "--server-counts",
        default="4,2,1",
        help="Comma-separated counts of showdown servers to use from the configured server list.",
    )
    parser.add_argument(
        "--match-workers-to-servers",
        action="store_true",
        help="Match rollout worker count to each requested server count instead of using the rollout worker grid.",
    )
    parser.add_argument(
        "--allow-worker-server-mismatch",
        action="store_true",
        help="Disable worker/server matching and use the full rollout worker x server count grid.",
    )
    parser.add_argument(
        "--selfplay-pairs-grid",
        default="1",
        help="Comma-separated self-play pair counts per worker to test.",
    )
    parser.add_argument(
        "--max-concurrent-grid",
        default="2,4,8",
        help="Comma-separated max_concurrent_battles values to test.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=4096,
        help="Total rollout decision steps to collect per benchmark run.",
    )
    parser.add_argument(
        "--battle-chunk",
        type=int,
        default=25,
        help="Chunk size passed into each worker collect request.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Measured repeats per configuration.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Unmeasured warmup runs per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for rollout requests.",
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
        help="GPU resources to reserve per rollout worker actor.",
    )
    parser.add_argument(
        "--worker-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used inside rollout workers. 'auto' picks cuda only when worker-gpus > 0.",
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
        help="How long the inference actor waits to batch rollout requests.",
    )
    parser.add_argument(
        "--inference-max-batch-size",
        type=int,
        default=1024,
        help="Maximum rollout requests per centralized inference batch.",
    )
    parser.add_argument(
        "--showdown-server",
        default="localhost",
        help="Fallback showdown hostname when --server-ws-list is not provided.",
    )
    parser.add_argument(
        "--showdown-port",
        type=int,
        default=8000,
        help="Fallback showdown port when --server-ws-list is not provided.",
    )
    parser.add_argument(
        "--showdown-auth-url",
        default="http://localhost:8000/action.php?",
        help="Fallback showdown auth URL when --server-auth-list is not provided.",
    )
    parser.add_argument(
        "--server-ws-list",
        default="",
        help="Comma-separated websocket URLs for multiple showdown servers.",
    )
    parser.add_argument(
        "--server-auth-list",
        default="",
        help="Comma-separated auth URLs matching --server-ws-list.",
    )
    parser.add_argument(
        "--max-turns-before-forfeit",
        type=int,
        default=200,
        help="Auto-forfeit threshold for benchmark battles. Negative disables auto-forfeit.",
    )
    parser.add_argument(
        "--ray-address",
        default="",
        help="Existing Ray cluster address. Empty string starts a local Ray runtime.",
    )
    parser.add_argument(
        "--ray-namespace",
        default="pokeenv_transformer_bench",
        help="Ray namespace for benchmark actors.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional CSV path for per-run benchmark metrics.",
    )
    return parser


def _format_server_label(server_configurations: Sequence[Any]) -> str:
    return ",".join(getattr(cfg, "websocket_url", "<unknown>") for cfg in server_configurations)


def _print_summary(rows: Sequence[BenchmarkMetrics]) -> None:
    if not rows:
        return
    grouped: dict[tuple[int, int, int, int], list[BenchmarkMetrics]] = {}
    for row in rows:
        key = (row.rollout_workers, row.server_count, row.selfplay_pairs, row.max_concurrent_battles)
        grouped.setdefault(key, []).append(row)

    print("\n[summary] mean throughput by configuration", flush=True)
    print(
        "workers servers pairs max_conc repeats steps/sec battles/sec ms/step approx_turn_ms avg_steps/battle",
        flush=True,
    )
    ranked = []
    for key, entries in grouped.items():
        ranked.append(
            (
                statistics.fmean(entry.steps_per_sec for entry in entries),
                key,
                entries,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    for mean_steps_per_sec, key, entries in ranked:
        workers, servers, pairs, max_concurrent = key
        print(
            f"{workers:>7} {servers:>7} {pairs:>5} {max_concurrent:>8} {len(entries):>7} "
            f"{mean_steps_per_sec:>9.1f} "
            f"{statistics.fmean(entry.battles_per_sec for entry in entries):>10.2f} "
            f"{statistics.fmean(entry.ms_per_step for entry in entries):>8.3f} "
            f"{statistics.fmean(entry.approx_turn_ms for entry in entries):>14.3f} "
            f"{statistics.fmean(entry.avg_steps_per_battle for entry in entries):>16.2f}",
            flush=True,
        )


def _write_csv(rows: Sequence[BenchmarkMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(BenchmarkMetrics(1, 1, 1, 1, 0, 0.0, 1, 1, 1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = build_parser().parse_args()
    if ray is None:
        raise SystemExit("Ray is not installed. Install it first with `pip install ray`.")
    if int(args.target_steps) < 1:
        raise SystemExit("--target-steps must be >= 1")
    if int(args.battle_chunk) < 1:
        raise SystemExit("--battle-chunk must be >= 1")
    if int(args.repeats) < 1:
        raise SystemExit("--repeats must be >= 1")
    if int(args.warmup_runs) < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if float(args.worker_cpus) <= 0:
        raise SystemExit("--worker-cpus must be > 0")
    if float(args.inference_cpus) <= 0:
        raise SystemExit("--inference-cpus must be > 0")
    if float(args.worker_gpus) < 0 or float(args.inference_gpus) < 0:
        raise SystemExit("--worker-gpus and --inference-gpus must be >= 0")
    if int(args.inference_max_batch_size) < 1:
        raise SystemExit("--inference-max-batch-size must be >= 1")
    if int(args.max_turns_before_forfeit) < 0:
        args.max_turns_before_forfeit = None

    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}", flush=True)
    else:
        print(f"[device] using={args.device}", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    current_ckpt = checkpoint_dir / "current.pt"
    best_ckpt = checkpoint_dir / "best.pt"
    vocab_json = checkpoint_dir / "vocab.json"
    if not current_ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {current_ckpt}")
    if not best_ckpt.exists():
        print(f"[config] missing {best_ckpt}; falling back to current.pt for both sides", flush=True)
    if not vocab_json.exists():
        raise SystemExit(f"Missing vocab file: {vocab_json}")

    current_payload = load_checkpoint(current_ckpt, "cpu")
    best_payload = load_checkpoint(best_ckpt if best_ckpt.exists() else current_ckpt, "cpu")
    snapshot = _create_snapshot(current_payload=current_payload, best_payload=best_payload, version=1)

    server_configurations = _build_server_configurations(args)
    max_requested_servers = max(_positive_int_grid(args.server_counts))
    if max_requested_servers > len(server_configurations):
        raise SystemExit(
            f"Requested up to {max_requested_servers} servers but only {len(server_configurations)} are configured."
        )
    for server_configuration in server_configurations[:max_requested_servers]:
        ok, err = asyncio.run(_check_server(server_configuration.websocket_url))
        if not ok:
            raise SystemExit(
                "Could not connect to Showdown websocket at "
                f"{server_configuration.websocket_url}. Error: {err}"
            )

    match_workers_to_servers = bool(args.match_workers_to_servers or not args.allow_worker_server_mismatch)
    server_counts = _positive_int_grid(args.server_counts)
    rollout_workers = _positive_int_grid(args.rollout_workers_grid)
    configs = _build_config_grid(
        rollout_workers=rollout_workers,
        server_counts=server_counts,
        selfplay_pairs=_positive_int_grid(args.selfplay_pairs_grid),
        max_concurrent_battles=_positive_int_grid(args.max_concurrent_grid),
        match_workers_to_servers=match_workers_to_servers,
    )

    inference_device = _resolve_inference_device(args)
    worker_device = _resolve_worker_device(args)
    inference_gpus = _resolve_benchmark_inference_gpus(args, inference_device=inference_device)

    all_rows: list[BenchmarkMetrics] = []

    for config in configs:
        weight_store = None
        inference_actor = None
        workers = []
        ray.init(
            address=(args.ray_address or None),
            ignore_reinit_error=True,
            namespace=str(args.ray_namespace),
            log_to_driver=True,
        )
        CentralInferenceActor = _build_inference_actor(ray)
        WeightStoreActor = _build_weight_store_actor(ray)
        RolloutWorkerActor = _build_rollout_worker_actor(ray)
        weight_store = WeightStoreActor.options(num_cpus=0.1).remote(snapshot)
        inference_actor = CentralInferenceActor.options(
            num_cpus=float(args.inference_cpus),
            num_gpus=float(inference_gpus),
        ).remote(
            device=inference_device,
            batch_wait_ms=float(args.inference_batch_wait_ms),
            max_batch_size=int(args.inference_max_batch_size),
        )
        ray.get(inference_actor.set_snapshot.remote(snapshot))
        try:
            active_servers = server_configurations[: int(config.server_count)]
            label = (
                f"workers={config.rollout_workers} servers={config.server_count} "
                f"pairs={config.selfplay_pairs} max_conc={config.max_concurrent_battles}"
            )
            print(f"\n[config] {label}", flush=True)
            print(f"[servers] {_format_server_label(active_servers)}", flush=True)

            workers = [
                RolloutWorkerActor.options(
                    num_cpus=float(args.worker_cpus),
                    num_gpus=float(args.worker_gpus),
                ).remote(
                    worker_id=idx,
                    server_configuration=active_servers[idx % len(active_servers)],
                    battle_format=str(args.battle_format),
                    vocab_path=str(vocab_json),
                    device=worker_device,
                    max_concurrent_battles=int(config.max_concurrent_battles),
                    parallel_pairs=int(config.selfplay_pairs),
                    reward_terminal_win=1.0,
                    reward_terminal_loss=-1.0,
                    reward_use_faint=True,
                    reward_faint_self=-0.1,
                    reward_faint_opp=0.1,
                    reward_discount=0.99,
                    reward_gae_lambda=0.95,
                    reward_target_clip=1.0,
                    max_turns_before_forfeit=args.max_turns_before_forfeit,
                    weight_store=weight_store,
                    inference_actor=inference_actor,
                )
                for idx in range(int(config.rollout_workers))
            ]
            total_runs = int(args.warmup_runs) + int(args.repeats)
            for run_index in range(total_runs):
                is_warmup = run_index < int(args.warmup_runs)
                request_seed = int(args.seed + (run_index * 10_000) + (config.rollout_workers * 100) + (config.server_count * 10))
                step_splits = _split_counts(int(args.target_steps), len(workers))
                refs = [
                    worker.collect.remote(
                        {
                            "target_steps": int(step_splits[idx]),
                            "fixed_battles": 0,
                            "battle_chunk": int(args.battle_chunk),
                            "temperature": 0.0,
                            "seed": int(request_seed + idx),
                        }
                    )
                    for idx, worker in enumerate(workers)
                    if int(step_splits[idx]) > 0
                ]
                t0 = time.perf_counter()
                worker_results = ray.get(refs)
                elapsed_s = time.perf_counter() - t0
                stats, meta = summarize_worker_stats(worker_results)
                metrics = _derive_metrics(
                    config=config,
                    repeat_index=run_index - int(args.warmup_runs),
                    elapsed_s=elapsed_s,
                    steps=int(meta["steps"]),
                    battles=int(meta["battles"]),
                    episodes=int(meta["episodes"]),
                    wins=int(stats["wins"]),
                    losses=int(stats["losses"]),
                    ties=int(stats["ties"]),
                )
                if is_warmup:
                    print(
                        f"[warmup] elapsed={metrics.elapsed_s:.2f}s steps={metrics.steps} "
                        f"steps/sec={metrics.steps_per_sec:.1f} ms/step={metrics.ms_per_step:.3f}",
                        flush=True,
                    )
                else:
                    all_rows.append(metrics)
                    print(
                        f"[repeat {metrics.repeat_index}] elapsed={metrics.elapsed_s:.2f}s "
                        f"steps={metrics.steps} battles={metrics.battles} "
                        f"steps/sec={metrics.steps_per_sec:.1f} battles/sec={metrics.battles_per_sec:.2f} "
                        f"ms/step={metrics.ms_per_step:.3f} approx_turn_ms={metrics.approx_turn_ms:.3f} "
                        f"avg_steps/battle={metrics.avg_steps_per_battle:.2f}",
                        flush=True,
                    )
                del worker_results
        finally:
            for worker in workers:
                ray.kill(worker, no_restart=True)
            if inference_actor is not None:
                ray.kill(inference_actor, no_restart=True)
            if weight_store is not None:
                ray.kill(weight_store, no_restart=True)
            ray.shutdown()
            time.sleep(0.5)

    _print_summary(all_rows)
    if str(args.output_csv).strip():
        output_csv = Path(args.output_csv)
        _write_csv(all_rows, output_csv)
        print(f"[output] wrote CSV to {output_csv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
