from __future__ import annotations

import torch

from scripts.train_selfplay_ray import concat_rollouts, pack_episodes, rollout_size


def test_pack_episodes_uses_fp16_obs_and_expected_shapes() -> None:
    episodes = [
        [
            {
                "obs": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                "legal_mask": torch.tensor([True, False], dtype=torch.bool),
                "action_index": 0,
                "outcome_z": 1.0,
                "old_log_prob": -0.1,
                "old_value": 0.2,
                "advantage": 0.8,
            },
            {
                "obs": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
                "legal_mask": torch.tensor([False, True], dtype=torch.bool),
                "action_index": 1,
                "outcome_z": -1.0,
                "old_log_prob": -0.3,
                "old_value": -0.2,
                "advantage": -0.8,
            },
        ]
    ]

    rollout = pack_episodes(episodes)
    assert rollout["obs"].dtype == torch.float16
    assert tuple(rollout["obs"].shape) == (2, 3)
    assert tuple(rollout["legal_mask"].shape) == (2, 2)
    assert tuple(rollout["action_index"].shape) == (2,)
    assert rollout_size(rollout) == 2


def test_concat_rollouts_combines_chunks() -> None:
    first = {
        "obs": torch.ones((2, 4), dtype=torch.float16),
        "legal_mask": torch.ones((2, 3), dtype=torch.bool),
        "action_index": torch.tensor([0, 1], dtype=torch.long),
        "outcome_z": torch.tensor([1.0, -1.0], dtype=torch.float32),
        "old_log_prob": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "old_value": torch.tensor([0.3, 0.4], dtype=torch.float32),
        "advantage": torch.tensor([0.5, 0.6], dtype=torch.float32),
    }
    second = {
        "obs": torch.zeros((1, 4), dtype=torch.float16),
        "legal_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_index": torch.tensor([2], dtype=torch.long),
        "outcome_z": torch.tensor([0.0], dtype=torch.float32),
        "old_log_prob": torch.tensor([0.7], dtype=torch.float32),
        "old_value": torch.tensor([0.8], dtype=torch.float32),
        "advantage": torch.tensor([0.9], dtype=torch.float32),
    }

    rollout = concat_rollouts([first, second])
    assert rollout_size(rollout) == 3
    assert tuple(rollout["obs"].shape) == (3, 4)
    assert rollout["action_index"].tolist() == [0, 1, 2]
