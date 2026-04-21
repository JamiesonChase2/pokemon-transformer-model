from __future__ import annotations

import pytest

from scripts.benchmark_rollout import (
    BenchmarkConfig,
    _build_config_grid,
    _derive_metrics,
    _positive_int_grid,
)


def test_positive_int_grid_parses_and_dedupes() -> None:
    assert _positive_int_grid("1, 2,2, 4") == [1, 2, 4]


@pytest.mark.parametrize("raw", ["", "0", "-1", "1,0"])
def test_positive_int_grid_rejects_non_positive_values(raw: str) -> None:
    with pytest.raises(ValueError):
        _positive_int_grid(raw)


def test_build_config_grid_cartesian_product() -> None:
    configs = _build_config_grid(
        rollout_workers=[1, 2],
        server_counts=[1],
        selfplay_pairs=[1, 3],
        max_concurrent_battles=[2, 4],
        match_workers_to_servers=False,
    )
    assert configs == [
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=1, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=1, max_concurrent_battles=4),
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=3, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=3, max_concurrent_battles=4),
        BenchmarkConfig(rollout_workers=2, server_count=1, selfplay_pairs=1, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=2, server_count=1, selfplay_pairs=1, max_concurrent_battles=4),
        BenchmarkConfig(rollout_workers=2, server_count=1, selfplay_pairs=3, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=2, server_count=1, selfplay_pairs=3, max_concurrent_battles=4),
    ]


def test_build_config_grid_matches_workers_to_servers_in_requested_order() -> None:
    configs = _build_config_grid(
        rollout_workers=[99],
        server_counts=[4, 2, 1],
        selfplay_pairs=[1],
        max_concurrent_battles=[2, 4],
        match_workers_to_servers=True,
    )
    assert configs == [
        BenchmarkConfig(rollout_workers=4, server_count=4, selfplay_pairs=1, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=4, server_count=4, selfplay_pairs=1, max_concurrent_battles=4),
        BenchmarkConfig(rollout_workers=2, server_count=2, selfplay_pairs=1, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=2, server_count=2, selfplay_pairs=1, max_concurrent_battles=4),
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=1, max_concurrent_battles=2),
        BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=1, max_concurrent_battles=4),
    ]


def test_derive_metrics_computes_expected_throughput() -> None:
    config = BenchmarkConfig(rollout_workers=4, server_count=2, selfplay_pairs=1, max_concurrent_battles=8)
    metrics = _derive_metrics(
        config=config,
        repeat_index=0,
        elapsed_s=2.0,
        steps=1000,
        battles=50,
        episodes=50,
        wins=20,
        losses=30,
        ties=0,
    )
    assert metrics.steps_per_sec == 500.0
    assert metrics.battles_per_sec == 25.0
    assert metrics.ms_per_step == 2.0
    assert metrics.approx_turn_ms == 4.0
    assert metrics.avg_steps_per_battle == 20.0


def test_derive_metrics_rejects_invalid_inputs() -> None:
    config = BenchmarkConfig(rollout_workers=1, server_count=1, selfplay_pairs=1, max_concurrent_battles=1)
    with pytest.raises(ValueError):
        _derive_metrics(
            config=config,
            repeat_index=0,
            elapsed_s=0.0,
            steps=10,
            battles=1,
            episodes=1,
            wins=1,
            losses=0,
            ties=0,
        )
    with pytest.raises(ValueError):
        _derive_metrics(
            config=config,
            repeat_index=0,
            elapsed_s=1.0,
            steps=0,
            battles=1,
            episodes=1,
            wins=1,
            losses=0,
            ties=0,
        )
