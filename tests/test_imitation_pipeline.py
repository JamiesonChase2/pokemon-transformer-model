from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.train_imitation import _split_replay_for_gate
from src.replay import ReplayBuffer, ReplayConfig


def test_split_replay_for_gate_keeps_train_val_subsets_in_memory():
    with TemporaryDirectory() as tmpdir:
        replay = ReplayBuffer(
            Path(tmpdir) / "replay",
            config=ReplayConfig(shard_size=2, max_shards=8, seed=7),
        )
        for idx in range(6):
            replay.add_sample(
                obs=torch.full((4,), float(idx), dtype=torch.float32),
                legal_mask=torch.ones((14,), dtype=torch.bool),
                action_index=idx % 4,
                outcome_z=0.0,
                old_log_prob=0.0,
                old_value=0.0,
                advantage=0.0,
            )
        replay.flush()

        workspace_root = Path(tmpdir) / "split"
        train_data, val_data, train_samples, val_samples = _split_replay_for_gate(
            replay,
            validation_fraction=0.25,
            seed=11,
            workspace_root=workspace_root,
        )

        assert train_samples == 4
        assert val_samples == 2
        assert train_data["obs"].shape[0] == train_samples
        assert val_data is not None
        assert val_data["obs"].shape[0] == val_samples
        assert not (workspace_root / "train").exists()
        assert not (workspace_root / "val").exists()
