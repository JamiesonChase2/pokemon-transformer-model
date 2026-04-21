"""Benchmark local model inference latency for structured observation checkpoints."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Iterable, List

import torch

from src.checkpoint import load_checkpoint, model_from_checkpoint_payload
from src.device import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark checkpoint inference latency.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt).")
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Optional replay directory with shard_*.pt files. Defaults to <checkpoint_dir>/replay if it exists.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32,128",
        help="Comma-separated batch sizes to benchmark.",
    )
    parser.add_argument("--sample-count", type=int, default=2048, help="Number of replay observations to load.")
    parser.add_argument("--warmup-iters", type=int, default=20, help="Warmup iterations per batch size.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations per batch size.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _parse_batch_sizes(spec: str) -> List[int]:
    values: List[int] = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < 1:
            raise ValueError("batch sizes must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("At least one batch size is required.")
    return values


def _default_replay_dir(checkpoint_path: Path) -> Path | None:
    candidate = checkpoint_path.parent / "replay"
    return candidate if candidate.exists() else None


def _load_replay_observations(replay_dir: Path, sample_count: int, seed: int) -> torch.Tensor:
    shard_paths = sorted(replay_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No replay shards found in {replay_dir}")

    rng = random.Random(seed)
    rng.shuffle(shard_paths)

    parts: List[torch.Tensor] = []
    total = 0
    for shard_path in shard_paths:
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(shard_path, map_location="cpu")
        obs = payload["obs"].float().cpu()
        parts.append(obs)
        total += int(obs.shape[0])
        if total >= sample_count:
            break

    obs_all = torch.cat(parts, dim=0)
    if int(obs_all.shape[0]) > sample_count:
        obs_all = obs_all[:sample_count]
    return obs_all


def _build_synthetic_observations(obs_dim: int, sample_count: int) -> torch.Tensor:
    return torch.zeros((sample_count, obs_dim), dtype=torch.float32)


def _take_batch(obs_bank: torch.Tensor, batch_size: int, offset: int) -> torch.Tensor:
    count = int(obs_bank.shape[0])
    if count >= batch_size:
        start = offset % max(1, count - batch_size + 1)
        return obs_bank[start : start + batch_size].clone()
    repeats = math.ceil(batch_size / max(1, count))
    return obs_bank.repeat((repeats, 1))[:batch_size].clone()


def _benchmark_batch(
    model: torch.nn.Module,
    obs_bank: torch.Tensor,
    *,
    batch_size: int,
    warmup_iters: int,
    iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()

    with torch.inference_mode():
        for idx in range(warmup_iters):
            batch_cpu = _take_batch(obs_bank, batch_size, idx)
            batch_device = batch_cpu.to(device)
            _ = model(batch_device)
        _synchronize(device)

        start = time.perf_counter()
        for idx in range(iters):
            batch_cpu = _take_batch(obs_bank, batch_size, idx)
            batch_device = batch_cpu.to(device)
            _ = model(batch_device)
        _synchronize(device)
        end = time.perf_counter()

        batch_device = _take_batch(obs_bank, batch_size, 0).to(device)
        for _ in range(warmup_iters):
            _ = model(batch_device)
        _synchronize(device)

        model_only_start = time.perf_counter()
        for _ in range(iters):
            _ = model(batch_device)
        _synchronize(device)
        model_only_end = time.perf_counter()

    end_to_end_s = (end - start) / max(1, iters)
    model_only_s = (model_only_end - model_only_start) / max(1, iters)
    return {
        "batch_size": float(batch_size),
        "end_to_end_ms": end_to_end_s * 1000.0,
        "model_only_ms": model_only_s * 1000.0,
        "samples_per_second": float(batch_size) / max(end_to_end_s, 1e-12),
        "moves_per_second_model_only": float(batch_size) / max(model_only_s, 1e-12),
    }


def main() -> int:
    args = build_parser().parse_args()
    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}")
    else:
        print(f"[device] using={args.device}")

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = load_checkpoint(checkpoint_path, args.device)
    model = model_from_checkpoint_payload(payload, args.device)

    replay_dir = None
    if args.replay_dir:
        replay_dir = Path(args.replay_dir).expanduser().resolve()
    else:
        replay_dir = _default_replay_dir(checkpoint_path)

    if replay_dir is not None and replay_dir.exists():
        obs_bank = _load_replay_observations(replay_dir, int(args.sample_count), int(args.seed))
        source_label = str(replay_dir)
    else:
        obs_dim = int(payload["model_config"]["obs_dim"])
        obs_bank = _build_synthetic_observations(obs_dim, int(args.sample_count))
        source_label = "synthetic_zeros"

    print(
        f"[bench] checkpoint={checkpoint_path} source={source_label} "
        f"obs_rows={int(obs_bank.shape[0])} obs_dim={int(obs_bank.shape[1])}"
    )

    results: List[dict[str, float]] = []
    for batch_size in batch_sizes:
        metrics = _benchmark_batch(
            model,
            obs_bank,
            batch_size=int(batch_size),
            warmup_iters=int(args.warmup_iters),
            iters=int(args.iters),
            device=str(args.device),
        )
        results.append(metrics)
        print(
            f"[batch {int(metrics['batch_size'])}] "
            f"end_to_end_ms={metrics['end_to_end_ms']:.3f} "
            f"model_only_ms={metrics['model_only_ms']:.3f} "
            f"samples_per_sec={metrics['samples_per_second']:.1f} "
            f"model_only_samples_per_sec={metrics['moves_per_second_model_only']:.1f}"
        )

    if results:
        b1 = next((row for row in results if int(row["batch_size"]) == 1), None)
        if b1 is not None:
            print(
                f"[summary] single_decision_latency_ms={b1['end_to_end_ms']:.3f} "
                f"(includes host->device copy)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
