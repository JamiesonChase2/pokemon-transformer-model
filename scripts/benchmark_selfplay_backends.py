"""Benchmark self-play collection throughput for Ray vs process pool backends.

This script compares rollout collection speed for identical self-play settings:
- same checkpoint snapshot (current vs best),
- same number of workers,
- same battle chunking,
- same max concurrent battles and pair count.

Metrics are reported per backend (steps/sec, battles/sec, latency) and
summarized side-by-side.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import multiprocessing as mp
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from poke_env.ps_client.server_configuration import ServerConfiguration

from scripts.train_selfplay import _build_server_configurations, _check_server, _run_selfplay_phase, _split_counts
from scripts.train_selfplay_ray import (
    _build_inference_actor,
    _build_rollout_worker_actor,
    _build_weight_store_actor,
    _resolve_inference_device,
    _resolve_worker_device,
    concat_rollouts,
    pack_episodes,
    ray,
    summarize_worker_stats,
)
from src.checkpoint import ENCODER_TYPE, load_checkpoint, model_from_checkpoint_payload
from src.device import resolve_device
from src.vocab import ObservationVocabulary

REWARD_TERMINAL_WIN = 1.0
REWARD_TERMINAL_LOSS = -1.0
REWARD_FAINT_SELF = -0.1
REWARD_FAINT_OPP = 0.1
REWARD_DISCOUNT = 0.99
REWARD_GAE_LAMBDA = 0.95
REWARD_TARGET_CLIP = 1.0

_PROCESS_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class BenchmarkMetrics:
    backend: str
    repeat_index: int
    elapsed_s: float
    steps: int
    battles: int
    episodes: int
    replay_size: int
    wins: int
    losses: int
    ties: int
    steps_per_sec: float
    battles_per_sec: float
    ms_per_step: float
    approx_turn_ms: float
    avg_steps_per_battle: float


def _derive_metrics(
    *,
    backend: str,
    repeat_index: int,
    elapsed_s: float,
    steps: int,
    battles: int,
    episodes: int,
    replay_size: int,
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
        backend=str(backend),
        repeat_index=int(repeat_index),
        elapsed_s=float(elapsed_s),
        steps=int(steps),
        battles=int(battles),
        episodes=int(episodes),
        replay_size=int(replay_size),
        wins=int(wins),
        losses=int(losses),
        ties=int(ties),
        steps_per_sec=float(steps_per_sec),
        battles_per_sec=float(battles_per_sec),
        ms_per_step=float(ms_per_step),
        approx_turn_ms=float(ms_per_step * 2.0),
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


def _load_process_state(
    *,
    current_ckpt: str,
    best_ckpt: str,
    vocab_json: str,
    device: str,
) -> tuple[ObservationVocabulary, torch.nn.Module, torch.nn.Module]:
    cache_key = (str(current_ckpt), str(best_ckpt), str(vocab_json), str(device))
    if _PROCESS_CACHE.get("key") == cache_key:
        return _PROCESS_CACHE["vocab"], _PROCESS_CACHE["current_model"], _PROCESS_CACHE["best_model"]

    vocab = ObservationVocabulary.load_json(vocab_json)
    current_payload = load_checkpoint(current_ckpt, "cpu")
    best_payload = load_checkpoint(best_ckpt if Path(best_ckpt).exists() else current_ckpt, "cpu")
    current_model = model_from_checkpoint_payload(current_payload, device)
    best_model = model_from_checkpoint_payload(best_payload, device)
    _PROCESS_CACHE["key"] = cache_key
    _PROCESS_CACHE["vocab"] = vocab
    _PROCESS_CACHE["current_model"] = current_model
    _PROCESS_CACHE["best_model"] = best_model
    return vocab, current_model, best_model


def _process_collect_worker(request: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, int]]:
    vocab, current_model, best_model = _load_process_state(
        current_ckpt=str(request["current_ckpt"]),
        best_ckpt=str(request["best_ckpt"]),
        vocab_json=str(request["vocab_json"]),
        device=str(request["device"]),
    )
    server_configuration = ServerConfiguration(
        websocket_url=str(request["server_ws"]),
        authentication_url=str(request["server_auth"]),
    )
    target_steps = max(0, int(request.get("target_steps", 0)))
    fixed_battles = max(0, int(request.get("fixed_battles", 0)))
    battle_chunk = max(1, int(request.get("battle_chunk", 1)))
    temperature = float(request.get("temperature", 0.0))
    seed = int(request.get("seed", 0))
    worker_id = int(request.get("worker_id", 0))

    episodes: list[list[dict[str, Any]]] = []
    stats = {"wins": 0, "losses": 0, "ties": 0}
    collected_steps = 0
    collected_battles = 0
    while True:
        if target_steps > 0:
            next_battles = battle_chunk
        else:
            remaining_battles = max(0, fixed_battles - collected_battles)
            if remaining_battles <= 0:
                break
            next_battles = min(battle_chunk, remaining_battles)

        chunk_episodes, chunk_stats = asyncio.run(
            _run_selfplay_phase(
                current_model=current_model,
                best_model=best_model,
                vocab=vocab,
                n_battles=next_battles,
                battle_format=str(request["battle_format"]),
                server_configuration=server_configuration,
                device=str(request["device"]),
                temperature=temperature,
                reward_terminal_win=REWARD_TERMINAL_WIN,
                reward_terminal_loss=REWARD_TERMINAL_LOSS,
                reward_use_faint=True,
                reward_faint_self=REWARD_FAINT_SELF,
                reward_faint_opp=REWARD_FAINT_OPP,
                reward_discount=REWARD_DISCOUNT,
                reward_gae_lambda=REWARD_GAE_LAMBDA,
                reward_target_clip=REWARD_TARGET_CLIP,
                max_turns_before_forfeit=request.get("max_turns_before_forfeit"),
                max_concurrent_battles=int(request["max_concurrent_battles"]),
                parallel_pairs=int(request["parallel_pairs"]),
                seed=seed + collected_battles + worker_id * 100_000,
                account_prefix=str(request.get("account_prefix", f"pp{worker_id}_")),
                log_level=str(request.get("log_level", "quiet")),
            )
        )
        batch_steps = sum(len(episode) for episode in chunk_episodes)
        if batch_steps <= 0:
            raise RuntimeError(f"Process worker {worker_id} collected zero steps from a rollout chunk.")

        episodes.extend(chunk_episodes)
        collected_steps += int(batch_steps)
        collected_battles += int(next_battles)
        stats["wins"] += int(chunk_stats["wins"])
        stats["losses"] += int(chunk_stats["losses"])
        stats["ties"] += int(chunk_stats["ties"])

        if target_steps > 0:
            if collected_steps >= target_steps:
                break
        else:
            if collected_battles >= fixed_battles:
                break

    packed = pack_episodes(episodes, obs_dtype=torch.float16)
    info = {
        "steps": int(collected_steps),
        "battles": int(collected_battles),
        "episodes": int(len(episodes)),
    }
    return packed, stats, info


def _format_server_label(server_configurations: Sequence[ServerConfiguration]) -> str:
    return ",".join(getattr(cfg, "websocket_url", "<unknown>") for cfg in server_configurations)


def _compute_splits(
    *,
    target_steps: int,
    total_battles: int,
    worker_count: int,
) -> tuple[list[int], list[int]]:
    if int(target_steps) > 0:
        return _split_counts(int(target_steps), int(worker_count)), [0 for _ in range(int(worker_count))]
    return [0 for _ in range(int(worker_count))], _split_counts(int(total_battles), int(worker_count))


def _run_process_backend(
    args: argparse.Namespace,
    *,
    active_servers: Sequence[ServerConfiguration],
    current_ckpt: Path,
    best_ckpt: Path,
    vocab_json: Path,
) -> list[BenchmarkMetrics]:
    process_workers = int(args.process_workers) if int(args.process_workers) > 0 else int(args.workers)
    if process_workers < 1:
        raise SystemExit("process backend requires at least one worker.")
    process_device = str(args.process_device)
    if process_device == "auto":
        process_device = "cpu"
    else:
        process_device = resolve_device(process_device)

    print(
        f"\n[backend=process] workers={process_workers} device={process_device} "
        f"pairs={int(args.selfplay_pairs)} max_conc={int(args.max_concurrent_battles)} "
        f"battle_chunk={int(args.battle_chunk)} target_steps={int(args.target_steps)} total_battles={int(args.total_battles)}",
        flush=True,
    )
    print(f"[backend=process] servers={_format_server_label(active_servers)}", flush=True)

    context = mp.get_context(str(args.process_start_method))
    rows: list[BenchmarkMetrics] = []
    total_runs = int(args.warmup_runs) + int(args.repeats)
    with ProcessPoolExecutor(max_workers=process_workers, mp_context=context) as executor:
        for run_index in range(total_runs):
            is_warmup = run_index < int(args.warmup_runs)
            repeat_index = run_index - int(args.warmup_runs)
            step_splits, battle_splits = _compute_splits(
                target_steps=int(args.target_steps),
                total_battles=int(args.total_battles),
                worker_count=process_workers,
            )
            t0 = time.perf_counter()
            futures = []
            for idx in range(process_workers):
                target_steps = int(step_splits[idx])
                fixed_battles = int(battle_splits[idx])
                if target_steps <= 0 and fixed_battles <= 0:
                    continue
                server_configuration = active_servers[idx % len(active_servers)]
                request = {
                    "worker_id": int(idx),
                    "current_ckpt": str(current_ckpt),
                    "best_ckpt": str(best_ckpt),
                    "vocab_json": str(vocab_json),
                    "device": str(process_device),
                    "server_ws": str(server_configuration.websocket_url),
                    "server_auth": str(server_configuration.authentication_url),
                    "battle_format": str(args.battle_format),
                    "target_steps": int(target_steps),
                    "fixed_battles": int(fixed_battles),
                    "battle_chunk": int(args.battle_chunk),
                    "temperature": float(args.temperature),
                    "seed": int(args.seed + run_index * 10_000 + idx * 1_000),
                    "parallel_pairs": int(args.selfplay_pairs),
                    "max_concurrent_battles": int(args.max_concurrent_battles),
                    "max_turns_before_forfeit": args.max_turns_before_forfeit,
                    "account_prefix": f"pp_r{run_index}_w{idx}_",
                    "log_level": str(args.worker_log_level),
                }
                futures.append(executor.submit(_process_collect_worker, request))
            if not futures:
                raise RuntimeError("No process workers were assigned any steps or battles.")

            worker_results = []
            pending = set(futures)
            last_progress_t = time.perf_counter()
            while pending:
                done, pending = wait(pending, timeout=15.0, return_when=FIRST_COMPLETED)
                for future in done:
                    worker_results.append(future.result())
                now = time.perf_counter()
                if done or (now - last_progress_t) >= 15.0:
                    print(
                        f"[process progress] run={run_index} completed_workers={len(worker_results)}/{len(futures)}",
                        flush=True,
                    )
                    last_progress_t = now
            rollout = concat_rollouts([packed for packed, _stats, _info in worker_results])
            stats, meta = summarize_worker_stats(worker_results)
            elapsed_s = time.perf_counter() - t0
            metrics = _derive_metrics(
                backend="process",
                repeat_index=int(repeat_index),
                elapsed_s=elapsed_s,
                steps=int(meta["steps"]),
                battles=int(meta["battles"]),
                episodes=int(meta["episodes"]),
                replay_size=int(rollout["obs"].shape[0]),
                wins=int(stats["wins"]),
                losses=int(stats["losses"]),
                ties=int(stats["ties"]),
            )
            if is_warmup:
                print(
                    f"[process warmup] elapsed={metrics.elapsed_s:.2f}s steps={metrics.steps} "
                    f"battles={metrics.battles} steps/sec={metrics.steps_per_sec:.1f}",
                    flush=True,
                )
            else:
                rows.append(metrics)
                print(
                    f"[process repeat {metrics.repeat_index}] elapsed={metrics.elapsed_s:.2f}s "
                    f"steps={metrics.steps} battles={metrics.battles} "
                    f"steps/sec={metrics.steps_per_sec:.1f} battles/sec={metrics.battles_per_sec:.2f} "
                    f"ms/step={metrics.ms_per_step:.3f} avg_steps/battle={metrics.avg_steps_per_battle:.2f}",
                    flush=True,
                )
            del worker_results
            del rollout
    return rows


def _run_ray_backend(
    args: argparse.Namespace,
    *,
    active_servers: Sequence[ServerConfiguration],
    snapshot: Mapping[str, Any],
    vocab_json: Path,
) -> list[BenchmarkMetrics]:
    if ray is None:
        raise SystemExit("Ray is not installed. Install it first with `pip install ray`.")
    if float(args.worker_cpus) <= 0:
        raise SystemExit("--worker-cpus must be > 0 for ray backend.")
    if float(args.inference_cpus) <= 0:
        raise SystemExit("--inference-cpus must be > 0 for ray backend.")
    if float(args.worker_gpus) < 0 or float(args.inference_gpus) < 0:
        raise SystemExit("--worker-gpus and --inference-gpus must be >= 0 for ray backend.")
    if int(args.inference_max_batch_size) < 1:
        raise SystemExit("--inference-max-batch-size must be >= 1 for ray backend.")

    worker_count = int(args.workers)
    if worker_count < 1:
        raise SystemExit("ray backend requires at least one worker.")

    inference_device = _resolve_inference_device(args)
    worker_device = _resolve_worker_device(args)
    inference_gpus = _resolve_benchmark_inference_gpus(args, inference_device=inference_device)

    print(
        f"\n[backend=ray] workers={worker_count} worker_device={worker_device} "
        f"inference_device={inference_device} inference_gpus={inference_gpus:.3f} "
        f"pairs={int(args.selfplay_pairs)} max_conc={int(args.max_concurrent_battles)} "
        f"battle_chunk={int(args.battle_chunk)} target_steps={int(args.target_steps)} total_battles={int(args.total_battles)}",
        flush=True,
    )
    print(f"[backend=ray] servers={_format_server_label(active_servers)}", flush=True)

    rows: list[BenchmarkMetrics] = []
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
    weight_store = WeightStoreActor.options(num_cpus=0.1).remote(dict(snapshot))
    inference_actor = CentralInferenceActor.options(
        num_cpus=float(args.inference_cpus),
        num_gpus=float(inference_gpus),
    ).remote(
        device=inference_device,
        batch_wait_ms=float(args.inference_batch_wait_ms),
        max_batch_size=int(args.inference_max_batch_size),
    )
    ray.get(inference_actor.set_snapshot.remote(dict(snapshot)))
    try:
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
                max_concurrent_battles=int(args.max_concurrent_battles),
                parallel_pairs=int(args.selfplay_pairs),
                reward_terminal_win=REWARD_TERMINAL_WIN,
                reward_terminal_loss=REWARD_TERMINAL_LOSS,
                reward_use_faint=True,
                reward_faint_self=REWARD_FAINT_SELF,
                reward_faint_opp=REWARD_FAINT_OPP,
                reward_discount=REWARD_DISCOUNT,
                reward_gae_lambda=REWARD_GAE_LAMBDA,
                reward_target_clip=REWARD_TARGET_CLIP,
                max_turns_before_forfeit=args.max_turns_before_forfeit,
                worker_log_level=str(args.worker_log_level),
                weight_store=weight_store,
                inference_actor=inference_actor,
            )
            for idx in range(worker_count)
        ]

        total_runs = int(args.warmup_runs) + int(args.repeats)
        for run_index in range(total_runs):
            is_warmup = run_index < int(args.warmup_runs)
            repeat_index = run_index - int(args.warmup_runs)
            step_splits, battle_splits = _compute_splits(
                target_steps=int(args.target_steps),
                total_battles=int(args.total_battles),
                worker_count=worker_count,
            )
            t0 = time.perf_counter()
            refs = [
                worker.collect.remote(
                    {
                        "target_steps": int(step_splits[idx]),
                        "fixed_battles": int(battle_splits[idx]),
                        "battle_chunk": int(args.battle_chunk),
                        "temperature": float(args.temperature),
                        "seed": int(args.seed + run_index * 10_000 + idx * 1_000),
                    }
                )
                for idx, worker in enumerate(workers)
                if int(step_splits[idx]) > 0 or int(battle_splits[idx]) > 0
            ]
            if not refs:
                raise RuntimeError("No ray workers were assigned any steps or battles.")
            worker_results = ray.get(refs)
            rollout = concat_rollouts([packed for packed, _stats, _info in worker_results])
            stats, meta = summarize_worker_stats(worker_results)
            elapsed_s = time.perf_counter() - t0
            metrics = _derive_metrics(
                backend="ray",
                repeat_index=int(repeat_index),
                elapsed_s=elapsed_s,
                steps=int(meta["steps"]),
                battles=int(meta["battles"]),
                episodes=int(meta["episodes"]),
                replay_size=int(rollout["obs"].shape[0]),
                wins=int(stats["wins"]),
                losses=int(stats["losses"]),
                ties=int(stats["ties"]),
            )
            if is_warmup:
                print(
                    f"[ray warmup] elapsed={metrics.elapsed_s:.2f}s steps={metrics.steps} "
                    f"battles={metrics.battles} steps/sec={metrics.steps_per_sec:.1f}",
                    flush=True,
                )
            else:
                rows.append(metrics)
                print(
                    f"[ray repeat {metrics.repeat_index}] elapsed={metrics.elapsed_s:.2f}s "
                    f"steps={metrics.steps} battles={metrics.battles} "
                    f"steps/sec={metrics.steps_per_sec:.1f} battles/sec={metrics.battles_per_sec:.2f} "
                    f"ms/step={metrics.ms_per_step:.3f} avg_steps/battle={metrics.avg_steps_per_battle:.2f}",
                    flush=True,
                )
            del worker_results
            del rollout
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        if inference_actor is not None:
            ray.kill(inference_actor, no_restart=True)
        if weight_store is not None:
            ray.kill(weight_store, no_restart=True)
        ray.shutdown()
        time.sleep(0.5)

    return rows


def _print_summary(rows: Sequence[BenchmarkMetrics]) -> None:
    if not rows:
        return
    grouped: dict[str, list[BenchmarkMetrics]] = {}
    for row in rows:
        grouped.setdefault(str(row.backend), []).append(row)

    print("\n[summary] per-backend throughput", flush=True)
    print("backend repeats mean_steps/sec stdev mean_battles/sec mean_ms/step mean_steps/battle", flush=True)
    for backend in sorted(grouped):
        entries = grouped[backend]
        mean_steps = statistics.fmean(item.steps_per_sec for item in entries)
        std_steps = statistics.pstdev(item.steps_per_sec for item in entries) if len(entries) > 1 else 0.0
        mean_battles = statistics.fmean(item.battles_per_sec for item in entries)
        mean_ms = statistics.fmean(item.ms_per_step for item in entries)
        mean_steps_per_battle = statistics.fmean(item.avg_steps_per_battle for item in entries)
        print(
            f"{backend:>7} {len(entries):>7} {mean_steps:>14.1f} {std_steps:>5.1f} "
            f"{mean_battles:>16.2f} {mean_ms:>12.3f} {mean_steps_per_battle:>17.2f}",
            flush=True,
        )

    if "ray" in grouped and "process" in grouped:
        ray_mean = statistics.fmean(item.steps_per_sec for item in grouped["ray"])
        process_mean = statistics.fmean(item.steps_per_sec for item in grouped["process"])
        if process_mean > 0:
            print(
                f"[summary] speedup(ray/process)={ray_mean / process_mean:.3f}x "
                f"speedup(process/ray)={process_mean / ray_mean if ray_mean > 0 else 0.0:.3f}x",
                flush=True,
            )


def _write_csv(rows: Sequence[BenchmarkMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(BenchmarkMetrics("", 0, 1.0, 1, 1, 1, 1, 0, 0, 0, 1.0, 1.0, 1.0, 2.0, 1.0)).keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark self-play collection throughput for ray and process backends."
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with current.pt, best.pt, vocab.json.")
    parser.add_argument(
        "--backend",
        default="both",
        choices=("both", "ray", "process"),
        help="Benchmark backend selection.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument(
        "--target-steps",
        type=int,
        default=8192,
        help="Total target rollout steps across all workers. Set 0 to use --total-battles mode.",
    )
    parser.add_argument(
        "--total-battles",
        type=int,
        default=0,
        help="Total battles across all workers when --target-steps=0.",
    )
    parser.add_argument("--battle-chunk", type=int, default=8, help="Chunk battles per worker collect loop.")
    parser.add_argument("--workers", type=int, default=8, help="Worker count for both backends.")
    parser.add_argument("--process-workers", type=int, default=0, help="Process worker override. 0 uses --workers.")
    parser.add_argument("--selfplay-pairs", type=int, default=1)
    parser.add_argument("--max-concurrent-battles", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns-before-forfeit", type=int, default=200)
    parser.add_argument("--server-count", type=int, default=0, help="0 uses all configured servers.")
    parser.add_argument(
        "--worker-log-level",
        default="quiet",
        choices=("quiet", "summary", "chunk", "pair"),
        help="Worker log level for both backends.",
    )

    parser.add_argument(
        "--server-ws",
        default="ws://localhost:8000/showdown/websocket",
        help="Fallback websocket URL when --server-ws-list is not provided.",
    )
    parser.add_argument(
        "--server-auth",
        default="http://localhost:8000/action.php?",
        help="Fallback auth URL when --server-ws-list is not provided.",
    )
    parser.add_argument("--server-ws-list", default="")
    parser.add_argument("--server-auth-list", default="")

    parser.add_argument("--ray-address", default="", help="Existing Ray cluster address. Empty starts local Ray.")
    parser.add_argument("--ray-namespace", default="pokeenv_transformer_backend_bench")
    parser.add_argument("--worker-cpus", type=float, default=1.0)
    parser.add_argument("--worker-gpus", type=float, default=0.0)
    parser.add_argument("--worker-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--inference-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--inference-cpus", type=float, default=1.0)
    parser.add_argument("--inference-gpus", type=float, default=0.0)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--inference-max-batch-size", type=int, default=1024)

    parser.add_argument(
        "--process-device",
        default="cpu",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Model device for process workers. 'auto' resolves to cpu.",
    )
    parser.add_argument(
        "--process-start-method",
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Multiprocessing start method for process backend.",
    )
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.target_steps) < 0:
        raise SystemExit("--target-steps must be >= 0")
    if int(args.total_battles) < 0:
        raise SystemExit("--total-battles must be >= 0")
    if int(args.target_steps) == 0 and int(args.total_battles) == 0:
        raise SystemExit("Set either --target-steps > 0 or --total-battles > 0.")
    if int(args.battle_chunk) < 1:
        raise SystemExit("--battle-chunk must be >= 1")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if int(args.repeats) < 1:
        raise SystemExit("--repeats must be >= 1")
    if int(args.warmup_runs) < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if int(args.selfplay_pairs) < 1:
        raise SystemExit("--selfplay-pairs must be >= 1")
    if int(args.max_concurrent_battles) < 1:
        raise SystemExit("--max-concurrent-battles must be >= 1")
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
        print(f"[config] missing {best_ckpt}; falling back to current.pt for best side", flush=True)
    if not vocab_json.exists():
        raise SystemExit(f"Missing vocab file: {vocab_json}")

    current_payload = load_checkpoint(current_ckpt, "cpu")
    best_payload = load_checkpoint(best_ckpt if best_ckpt.exists() else current_ckpt, "cpu")
    snapshot = _create_snapshot(current_payload=current_payload, best_payload=best_payload, version=1)

    server_configurations = _build_server_configurations(args)
    if not server_configurations:
        raise SystemExit("No server configuration available.")
    requested_server_count = int(args.server_count) if int(args.server_count) > 0 else len(server_configurations)
    if requested_server_count > len(server_configurations):
        raise SystemExit(
            f"Requested --server-count={requested_server_count}, but only {len(server_configurations)} servers are configured."
        )
    active_servers = server_configurations[:requested_server_count]
    for server_configuration in active_servers:
        ok, err = asyncio.run(_check_server(server_configuration.websocket_url))
        if not ok:
            raise SystemExit(
                "Could not connect to Showdown websocket at "
                f"{server_configuration.websocket_url}. Error: {err}"
            )

    backends: list[str]
    if str(args.backend) == "both":
        backends = ["process", "ray"]
    else:
        backends = [str(args.backend)]

    all_rows: list[BenchmarkMetrics] = []
    for backend in backends:
        if backend == "process":
            all_rows.extend(
                _run_process_backend(
                    args,
                    active_servers=active_servers,
                    current_ckpt=current_ckpt,
                    best_ckpt=best_ckpt if best_ckpt.exists() else current_ckpt,
                    vocab_json=vocab_json,
                )
            )
        elif backend == "ray":
            all_rows.extend(
                _run_ray_backend(
                    args,
                    active_servers=active_servers,
                    snapshot=snapshot,
                    vocab_json=vocab_json,
                )
            )
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    _print_summary(all_rows)
    if str(args.output_csv).strip():
        output_csv = Path(args.output_csv)
        _write_csv(all_rows, output_csv)
        print(f"[output] wrote CSV to {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
