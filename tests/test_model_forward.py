import torch
import torch.nn as nn

from src.model import TransformerConfig, TransformerPolicyValueNet
from src.policy import ACTION_DIM
from src.vocab import build_default_vocab


def _build_test_model() -> tuple[TransformerPolicyValueNet, TransformerConfig, dict]:
    vocab = build_default_vocab(gen=9)
    meta = vocab.schema_meta()
    cfg = TransformerConfig(
        obs_dim=int(meta["obs_dim"]),
        obs_meta=meta,
        d_model=64,
        nhead=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.0,
    )
    return TransformerPolicyValueNet(cfg), cfg, meta


def test_model_forward_shapes_batch_and_single():
    model, cfg, _ = _build_test_model()

    obs = torch.zeros((3, cfg.obs_dim), dtype=torch.float32)
    policy_logits, value = model(obs)
    assert policy_logits.shape == (3, ACTION_DIM)
    assert value.shape == (3, 1)

    single_policy, single_value = model(obs[0])
    assert single_policy.shape == (ACTION_DIM,)
    assert single_value.shape == (1,)


def test_model_forward_can_return_distributional_value_logits():
    model, cfg, _ = _build_test_model()

    obs = torch.zeros((2, cfg.obs_dim), dtype=torch.float32)
    policy_logits, value, value_logits = model(obs, return_value_logits=True)
    assert policy_logits.shape == (2, ACTION_DIM)
    assert value.shape == (2, 1)
    assert value_logits.shape == (2, int(cfg.v_bins))


def test_model_policy_head_does_not_condition_on_action_mask():
    torch.manual_seed(0)
    model, cfg, meta = _build_test_model()

    obs_a = torch.zeros((1, cfg.obs_dim), dtype=torch.float32)
    obs_b = obs_a.clone()
    mask_start, mask_end = meta["offsets"]["action_mask"]
    obs_a[:, mask_start:mask_end] = 0.0
    obs_b[:, mask_start:mask_end] = 1.0

    policy_a, value_a = model(obs_a)
    policy_b, value_b = model(obs_b)

    assert torch.allclose(policy_a, policy_b)
    assert torch.allclose(value_a, value_b)


def test_move_subnet_input_matches_psppo_feature_factorization():
    model, _, meta = _build_test_model()

    first_linear = next(module for module in model.move_net if isinstance(module, nn.Linear))
    move_map = meta["feature_map"]["move"]
    expected_in_features = (
        96  # move id embedding
        + (64 * 2)  # accuracy/pp value-bank embeddings
        + 64  # power bank embedding
        + (int(move_map["type_raw"][1]) - int(move_map["onehots_raw"][0]))  # category+priority+type one-hots
    )

    assert first_linear.in_features == expected_in_features
