"""Disk-backed replay buffer with .pt shards."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class ReplayConfig:
    shard_size: int = 2048
    max_shards: int = 256
    seed: Optional[int] = None


class ReplayBuffer:
    """Simple shard-based replay buffer to avoid unbounded RAM usage."""

    def __init__(self, root_dir: str | Path, config: ReplayConfig | None = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or ReplayConfig()
        self._rng = random.Random(self.config.seed)

        self._meta_path = self.root_dir / "meta.json"
        self._pending: List[Dict[str, Any]] = []

        self._next_shard_id = 0
        self._num_samples = 0

        self._load_meta()
        self._refresh_shards()

    def _load_meta(self) -> None:
        if not self._meta_path.exists():
            return
        payload = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._next_shard_id = int(payload.get("next_shard_id", 0))
        self._num_samples = int(payload.get("num_samples", 0))

    def _save_meta(self) -> None:
        payload = {
            "next_shard_id": self._next_shard_id,
            "num_samples": self._num_samples,
        }
        self._meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _refresh_shards(self) -> None:
        self._shards = sorted(self.root_dir.glob("shard_*.pt"))

    def __len__(self) -> int:
        return self._num_samples + len(self._pending)

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    def add_sample(
        self,
        *,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        action_index: int,
        outcome_z: float,
        old_log_prob: float = 0.0,
        old_value: float = 0.0,
        advantage: float = 0.0,
    ) -> None:
        self._pending.append(
            {
                "obs": obs.detach().cpu().float(),
                "legal_mask": legal_mask.detach().cpu().bool(),
                "action_index": int(action_index),
                "outcome_z": float(outcome_z),
                "old_log_prob": float(old_log_prob),
                "old_value": float(old_value),
                "advantage": float(advantage),
            }
        )

        if len(self._pending) >= self.config.shard_size:
            self.flush()

    def add_episode(self, steps: Iterable[Dict[str, Any]]) -> None:
        for step in steps:
            self.add_sample(
                obs=step["obs"],
                legal_mask=step["legal_mask"],
                action_index=int(step["action_index"]),
                outcome_z=float(step["outcome_z"]),
                old_log_prob=float(step.get("old_log_prob", 0.0)),
                old_value=float(step.get("old_value", 0.0)),
                advantage=float(step.get("advantage", step.get("outcome_z", 0.0))),
            )

    def add_episodes(self, episodes: Iterable[Sequence[Dict[str, Any]]]) -> None:
        for episode in episodes:
            self.add_episode(episode)

    def clear(self) -> None:
        """Remove all buffered samples and on-disk shards."""

        self._pending.clear()
        for shard in list(getattr(self, "_shards", [])):
            shard.unlink(missing_ok=True)
        self._refresh_shards()
        self._next_shard_id = 0
        self._num_samples = 0
        self._save_meta()

    def flush(self) -> None:
        if not self._pending:
            return

        shard_path = self.root_dir / f"shard_{self._next_shard_id:06d}.pt"
        self._next_shard_id += 1

        payload = {
            "obs": torch.stack([row["obs"] for row in self._pending], dim=0),
            "legal_mask": torch.stack([row["legal_mask"] for row in self._pending], dim=0),
            "action_index": torch.tensor([row["action_index"] for row in self._pending], dtype=torch.long),
            "outcome_z": torch.tensor([row["outcome_z"] for row in self._pending], dtype=torch.float32),
            "old_log_prob": torch.tensor([row["old_log_prob"] for row in self._pending], dtype=torch.float32),
            "old_value": torch.tensor([row["old_value"] for row in self._pending], dtype=torch.float32),
            "advantage": torch.tensor([row["advantage"] for row in self._pending], dtype=torch.float32),
            "num_samples": len(self._pending),
        }
        torch.save(payload, shard_path)

        self._num_samples += len(self._pending)
        self._pending.clear()

        self._refresh_shards()
        self._prune_old_shards()
        self._save_meta()

    def _prune_old_shards(self) -> None:
        while len(self._shards) > self.config.max_shards:
            oldest = self._shards.pop(0)
            try:
                data = torch.load(oldest, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(oldest, map_location="cpu")
            removed = int(data.get("num_samples", data["obs"].shape[0]))
            self._num_samples = max(0, self._num_samples - removed)
            oldest.unlink(missing_ok=True)

    def _load_random_shard(self) -> Dict[str, torch.Tensor]:
        if not self._shards:
            raise RuntimeError("Replay buffer has no on-disk shards.")
        shard = self._rng.choice(self._shards)
        try:
            payload = torch.load(shard, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(shard, map_location="cpu")
        return payload

    @staticmethod
    def _ensure_optional_fields(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        n = int(payload["obs"].shape[0])
        if "old_log_prob" not in payload:
            payload["old_log_prob"] = torch.zeros((n,), dtype=torch.float32)
        if "old_value" not in payload:
            payload["old_value"] = torch.zeros((n,), dtype=torch.float32)
        if "advantage" not in payload:
            payload["advantage"] = payload["outcome_z"] - payload["old_value"]
        return payload

    def materialize(self, *, device: str = "cpu") -> Dict[str, torch.Tensor]:
        parts: List[Dict[str, torch.Tensor]] = []

        for shard in self._shards:
            try:
                payload = torch.load(shard, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(shard, map_location="cpu")
            parts.append(self._ensure_optional_fields(payload))

        if self._pending:
            pending_payload = {
                "obs": torch.stack([row["obs"] for row in self._pending], dim=0),
                "legal_mask": torch.stack([row["legal_mask"] for row in self._pending], dim=0),
                "action_index": torch.tensor([row["action_index"] for row in self._pending], dtype=torch.long),
                "outcome_z": torch.tensor([row["outcome_z"] for row in self._pending], dtype=torch.float32),
                "old_log_prob": torch.tensor(
                    [row.get("old_log_prob", 0.0) for row in self._pending],
                    dtype=torch.float32,
                ),
                "old_value": torch.tensor(
                    [row.get("old_value", 0.0) for row in self._pending],
                    dtype=torch.float32,
                ),
                "advantage": torch.tensor(
                    [
                        row.get("advantage", row.get("outcome_z", 0.0) - row.get("old_value", 0.0))
                        for row in self._pending
                    ],
                    dtype=torch.float32,
                ),
            }
            parts.append(pending_payload)

        if not parts:
            raise RuntimeError("Replay buffer is empty.")

        keys = ("obs", "legal_mask", "action_index", "outcome_z", "old_log_prob", "old_value", "advantage")
        return {
            key: torch.cat([part[key] for part in parts], dim=0).to(device)
            for key in keys
        }

    def sample_batch(self, batch_size: int, *, device: str = "cpu") -> Dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self._shards:
            payload = self._ensure_optional_fields(self._load_random_shard())
            n = payload["obs"].shape[0]
            indices = torch.randint(0, n, size=(batch_size,))

            batch = {
                "obs": payload["obs"][indices].to(device),
                "legal_mask": payload["legal_mask"][indices].to(device),
                "action_index": payload["action_index"][indices].to(device),
                "outcome_z": payload["outcome_z"][indices].to(device),
                "old_log_prob": payload["old_log_prob"][indices].to(device),
                "old_value": payload["old_value"][indices].to(device),
                "advantage": payload["advantage"][indices].to(device),
            }
            return batch

        if self._pending:
            n = len(self._pending)
            indices = [self._rng.randrange(n) for _ in range(batch_size)]
            return {
                "obs": torch.stack([self._pending[i]["obs"] for i in indices], dim=0).to(device),
                "legal_mask": torch.stack([self._pending[i]["legal_mask"] for i in indices], dim=0).to(device),
                "action_index": torch.tensor([self._pending[i]["action_index"] for i in indices], dtype=torch.long, device=device),
                "outcome_z": torch.tensor([self._pending[i]["outcome_z"] for i in indices], dtype=torch.float32, device=device),
                "old_log_prob": torch.tensor(
                    [self._pending[i].get("old_log_prob", 0.0) for i in indices],
                    dtype=torch.float32,
                    device=device,
                ),
                "old_value": torch.tensor(
                    [self._pending[i].get("old_value", 0.0) for i in indices],
                    dtype=torch.float32,
                    device=device,
                ),
                "advantage": torch.tensor(
                    [
                        self._pending[i].get(
                            "advantage",
                            self._pending[i].get("outcome_z", 0.0) - self._pending[i].get("old_value", 0.0),
                        )
                        for i in indices
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
            }

        raise RuntimeError("Replay buffer is empty.")
