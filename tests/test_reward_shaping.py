import torch
import torch.nn as nn

from src.bot import compute_gae_targets, compute_step_reward
from src.train import _imitation_loss, TrainConfig, build_optimizer, dist_value_loss, lr_schedule_factor, masked_fill_value, twohot_targets
from scripts.train_selfplay import annealed_temperature, cap_train_steps_by_reuse, optimizer_steps_for_epochs


def test_compute_step_reward_matches_ps_ppo_faint_shaping():
    reward = compute_step_reward(
        current_self_fainted=1,
        next_self_fainted=2,
        current_opp_fainted=2,
        next_opp_fainted=4,
        use_faint_reward=True,
        faint_self=-0.1,
        faint_opp=0.1,
    )

    expected = (-0.1 * 1) + (0.1 * 2)
    assert reward == expected


def test_compute_step_reward_can_disable_faint_shaping():
    reward = compute_step_reward(
        current_self_fainted=0,
        next_self_fainted=1,
        current_opp_fainted=0,
        next_opp_fainted=1,
        use_faint_reward=False,
        faint_self=-0.1,
        faint_opp=0.1,
    )

    assert reward == 0.0


def test_compute_gae_targets_matches_hand_computed_values():
    returns, advantages = compute_gae_targets(
        rewards=[0.5, -0.2],
        values=[0.1, -0.1],
        gamma=0.99,
        gae_lambda=0.95,
        terminal_value=0.0,
        clip_return=1.0,
    )

    delta_1 = -0.2 + (0.99 * 0.0) - (-0.1)
    advantage_1 = delta_1
    return_1 = -0.1 + advantage_1

    delta_0 = 0.5 + (0.99 * -0.1) - 0.1
    advantage_0 = delta_0 + (0.99 * 0.95 * advantage_1)
    return_0 = 0.1 + advantage_0

    assert advantages[1] == advantage_1
    assert returns[1] == return_1
    assert advantages[0] == advantage_0
    assert returns[0] == return_0


def test_cap_train_steps_by_reuse_limits_optimizer_reuse():
    capped = cap_train_steps_by_reuse(
        requested_steps=480,
        replay_size=1000,
        batch_size=12,
        max_reuse_ratio=0.5,
    )

    assert capped == 41


def test_cap_train_steps_by_reuse_preserves_steps_when_disabled():
    uncapped = cap_train_steps_by_reuse(
        requested_steps=480,
        replay_size=1000,
        batch_size=12,
        max_reuse_ratio=0.0,
    )

    assert uncapped == 480


def test_optimizer_steps_for_epochs_accounts_for_accumulation():
    steps = optimizer_steps_for_epochs(
        replay_size=100,
        batch_size=12,
        grad_accum_steps=4,
        epochs=3,
    )

    assert steps == 9


def test_twohot_targets_sum_to_one():
    targets = twohot_targets(torch.tensor([-1.6, 0.0, 1.6]), v_min=-1.6, v_max=1.6, v_bins=51)
    assert targets.shape == (3, 51)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(3))
    assert float(dist_value_loss(torch.zeros((3, 51)), targets).item()) > 0.0


def test_annealed_temperature_interpolates_linearly():
    assert annealed_temperature(start=1.0, end=0.9, total_steps=500_000, current_steps=250_000) == 0.95


def test_masked_fill_value_is_safe_for_float16_logits():
    logits = torch.zeros((2, 4), dtype=torch.float16)
    legal_mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    masked = logits.masked_fill(~legal_mask, masked_fill_value(logits))
    assert torch.isfinite(masked).all()


class _DummyImitationNet(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, obs: torch.Tensor):
        batch = obs.shape[0]
        return self._logits[:batch], torch.zeros((batch, 1), dtype=self._logits.dtype, device=self._logits.device)


def test_imitation_loss_with_label_smoothing_respects_legal_mask():
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    batch = {
        "obs": torch.zeros((1, 3), dtype=torch.float32),
        "legal_mask": torch.tensor([[True, False, True, False]], dtype=torch.bool),
        "action_index": torch.tensor([0], dtype=torch.long),
    }
    metrics = _imitation_loss(
        _DummyImitationNet(logits),
        batch,
        config=TrainConfig(imitation_label_smoothing=0.1),
    )
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["policy_loss"])
    assert float(metrics["accuracy"].item()) == 1.0


def test_lr_schedule_factor_matches_ps_ppo_shape():
    assert lr_schedule_factor(0, warmup_steps=1000, hold_steps=20000, total_steps=500000) == 0.001
    assert lr_schedule_factor(999, warmup_steps=1000, hold_steps=20000, total_steps=500000) == 1.0
    assert lr_schedule_factor(1000, warmup_steps=1000, hold_steps=20000, total_steps=500000) == 1.0
    assert lr_schedule_factor(20999, warmup_steps=1000, hold_steps=20000, total_steps=500000) == 1.0
    assert abs(lr_schedule_factor(500000, warmup_steps=1000, hold_steps=20000, total_steps=500000) - (1.0 / 27.0)) < 1e-12


class _DummyPolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.critic_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.encoder = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.move_net = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        self.field_net = nn.Linear(8, 8)
        self.policy_head = nn.Linear(8, 14)
        self.value_head = nn.Linear(8, 51)


def test_build_optimizer_uses_ps_ppo_style_param_groups():
    model = _DummyPolicyValueNet()
    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=1e-2,
        lr_backbone_mult=0.5,
        lr_pi_mult=2.0,
        lr_v_mult=3.0,
    )

    group_map = {group["name"]: group for group in optimizer.param_groups}
    assert abs(group_map["trunk_wd"]["lr"] - 5e-5) < 1e-12
    assert abs(group_map["subnets_wd"]["lr"] - 5e-5) < 1e-12
    assert abs(group_map["pi_wd"]["lr"] - 2e-4) < 1e-12
    assert abs(group_map["v_wd"]["lr"] - 3e-4) < 1e-12
    assert group_map["trunk_stable"]["weight_decay"] == 0.0
    assert group_map["pi_stable"]["weight_decay"] == 0.0
