"""Head-to-head evaluation utilities for Transformer players."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from .bot import TransformerPlayer
from .vocab import ObservationVocabulary


@dataclass
class EvalConfig:
    n_battles: int = 20
    battle_format: str = "gen9randombattle"
    temperature: float = 0.0
    seed: int | None = None
    max_turns_before_forfeit: int | None = 500
    max_concurrent_battles: int = 1
    parallel_pairs: int = 1


def _result_counts(player: TransformerPlayer) -> Tuple[int, int, int]:
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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


async def _close_player(player: TransformerPlayer) -> None:
    try:
        await player.ps_client.stop_listening()
    except Exception:
        pass


def _split_counts(total: int, parts: int) -> list[int]:
    total = max(0, int(total))
    parts = max(1, int(parts))
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


async def _run_eval_pair(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
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
    account_prefix: str = "",
    inference_client_a: Any | None = None,
    inference_client_b: Any | None = None,
    inference_model_key_a: str = "current",
    inference_model_key_b: str = "best",
) -> Tuple[int, int, int]:
    if int(n_battles) <= 0:
        return 0, 0, 0

    a_player = TransformerPlayer(
        account_configuration=AccountConfiguration.generate(f"{account_prefix}evala{pair_index}", rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=model_a,
        vocab=vocab,
        temperature=temperature,
        collect_trajectories=False,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=None if seed is None else int(seed) + (2 * pair_index),
        start_listening=True,
        inference_client=inference_client_a,
        inference_model_key=str(inference_model_key_a),
    )
    b_player = TransformerPlayer(
        account_configuration=AccountConfiguration.generate(f"{account_prefix}evalb{pair_index}", rand=True),
        battle_format=battle_format,
        server_configuration=server_configuration,
        model=model_b,
        vocab=vocab,
        temperature=temperature,
        collect_trajectories=False,
        device=device,
        max_turns_before_forfeit=max_turns_before_forfeit,
        max_concurrent_battles=max(1, int(max_concurrent_battles)),
        seed=None if seed is None else int(seed) + (2 * pair_index) + 1,
        start_listening=True,
        inference_client=inference_client_b,
        inference_model_key=str(inference_model_key_b),
    )

    try:
        await a_player.battle_against(b_player, n_battles=int(n_battles))
        return _result_counts(a_player)
    finally:
        await _close_player(a_player)
        await _close_player(b_player)


async def evaluate_models(
    *,
    current_model: torch.nn.Module,
    best_model: torch.nn.Module,
    vocab: ObservationVocabulary,
    server_configuration: ServerConfiguration | Sequence[ServerConfiguration],
    config: EvalConfig,
    device: str = "cpu",
    current_inference_client: Any | None = None,
    best_inference_client: Any | None = None,
    account_prefix: str = "",
) -> Dict[str, float]:
    """Evaluate current model against best model with side swapping."""

    n_battles = max(0, int(config.n_battles))
    if n_battles == 0:
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

    first_half = n_battles // 2
    second_half = n_battles - first_half
    server_configurations = (
        list(server_configuration)
        if isinstance(server_configuration, Sequence) and not isinstance(server_configuration, ServerConfiguration)
        else [server_configuration]  # type: ignore[list-item]
    )
    if not server_configurations:
        raise ValueError("At least one server configuration is required for evaluation.")

    wins = 0
    losses = 0
    ties = 0

    if first_half > 0:
        pair_count = max(1, min(int(config.parallel_pairs), int(first_half)))
        pair_battles = _split_counts(first_half, pair_count)
        results = await asyncio.gather(
            *[
                _run_eval_pair(
                    model_a=current_model,
                    model_b=best_model,
                    vocab=vocab,
                    server_configuration=server_configurations[idx % len(server_configurations)],
                    battle_format=config.battle_format,
                    temperature=config.temperature,
                    device=device,
                    max_turns_before_forfeit=config.max_turns_before_forfeit,
                    max_concurrent_battles=config.max_concurrent_battles,
                    n_battles=count,
                    pair_index=idx,
                    seed=config.seed,
                    account_prefix=account_prefix,
                    inference_client_a=current_inference_client,
                    inference_client_b=best_inference_client,
                    inference_model_key_a="current",
                    inference_model_key_b="best",
                )
                for idx, count in enumerate(pair_battles)
                if count > 0
            ]
        )
        for w, l, t in results:
            wins += int(w)
            losses += int(l)
            ties += int(t)

    if second_half > 0:
        pair_count = max(1, min(int(config.parallel_pairs), int(second_half)))
        pair_battles = _split_counts(second_half, pair_count)
        results = await asyncio.gather(
            *[
                _run_eval_pair(
                    model_a=best_model,
                    model_b=current_model,
                    vocab=vocab,
                    server_configuration=server_configurations[idx % len(server_configurations)],
                    battle_format=config.battle_format,
                    temperature=config.temperature,
                    device=device,
                    max_turns_before_forfeit=config.max_turns_before_forfeit,
                    max_concurrent_battles=config.max_concurrent_battles,
                    n_battles=count,
                    pair_index=pair_count + idx,
                    seed=None if config.seed is None else int(config.seed) + 10_000,
                    account_prefix=account_prefix,
                    inference_client_a=best_inference_client,
                    inference_client_b=current_inference_client,
                    inference_model_key_a="best",
                    inference_model_key_b="current",
                )
                for idx, count in enumerate(pair_battles)
                if count > 0
            ]
        )
        for w, l, t in results:
            wins += int(l)
            losses += int(w)
            ties += int(t)

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
