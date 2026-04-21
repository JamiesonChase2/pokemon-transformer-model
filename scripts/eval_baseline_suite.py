"""Evaluate a checkpoint against baseline bots and baseline-vs-baseline matchups."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from poke_env.player import MaxBasePowerPlayer, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.bot import TransformerPlayer
from src.checkpoint import load_checkpoint, model_from_checkpoint_payload
from src.device import resolve_device
from src.vocab import ObservationVocabulary, missing_required_feature_tokens


class _TurnCapMixin:
    def __init__(self, *args, max_turns_before_forfeit: int | None = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns_before_forfeit = (
            int(max_turns_before_forfeit) if max_turns_before_forfeit is not None else None
        )

    def _maybe_forfeit(self, battle):
        if self.max_turns_before_forfeit is None:
            return None
        turn = int(getattr(battle, "turn", 0) or 0)
        if turn > self.max_turns_before_forfeit:
            return ForfeitBattleOrder()
        return None


class MaxDamagePlayer(_TurnCapMixin, MaxBasePowerPlayer):
    def choose_move(self, battle):  # type: ignore[override]
        forfeit = self._maybe_forfeit(battle)
        if forfeit is not None:
            return forfeit
        return super().choose_move(battle)


class RandomBaselinePlayer(_TurnCapMixin, RandomPlayer):
    def choose_move(self, battle):  # type: ignore[override]
        forfeit = self._maybe_forfeit(battle)
        if forfeit is not None:
            return forfeit
        return super().choose_move(battle)


class SimpleHeuristicBaselinePlayer(_TurnCapMixin, SimpleHeuristicsPlayer):
    def choose_move(self, battle):  # type: ignore[override]
        forfeit = self._maybe_forfeit(battle)
        if forfeit is not None:
            return forfeit
        return super().choose_move(battle)


@dataclass(frozen=True)
class MatchResult:
    left: str
    right: str
    wins: int
    losses: int
    ties: int
    win_rate: float
    score_rate: float
    ci_low: float
    ci_high: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against Random, MaxDamage, and SimpleHeuristics baselines."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--vocab", default=None, help="Optional vocab JSON path. Defaults to checkpoint payload path.")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model action temperature.")
    parser.add_argument("--n-battles", type=int, default=100, help="Battles per unordered matchup.")
    parser.add_argument("--max-concurrent-battles", type=int, default=1)
    parser.add_argument(
        "--max-turns-before-forfeit",
        type=int,
        default=500,
        help="Auto-forfeit if battle turn exceeds this threshold. Use -1 to disable.",
    )
    parser.add_argument(
        "--server-ws",
        default=os.environ.get("SHOWDOWN_WS_URL", "ws://localhost:8000/showdown/websocket"),
    )
    parser.add_argument(
        "--server-auth",
        default=os.environ.get("SHOWDOWN_AUTH_URL", "http://localhost:8000/action.php?"),
    )
    return parser


def _result_counts(player) -> tuple[int, int, int]:
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


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + (z**2 / n)
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


async def _close_player(player) -> None:
    try:
        await player.ps_client.stop_listening()
    except Exception:
        pass


def _make_factory(
    *,
    kind: str,
    battle_format: str,
    server_configuration: ServerConfiguration,
    max_turns_before_forfeit: int | None,
    max_concurrent_battles: int,
    model=None,
    vocab=None,
    device: str = "cpu",
    temperature: float = 0.0,
) -> Callable[[str], object]:
    def _factory(name_prefix: str):
        account = AccountConfiguration.generate(name_prefix, rand=True)
        common = {
            "account_configuration": account,
            "battle_format": battle_format,
            "server_configuration": server_configuration,
            "max_turns_before_forfeit": max_turns_before_forfeit,
            "max_concurrent_battles": int(max_concurrent_battles),
            "start_listening": True,
        }
        if kind == "model":
            return TransformerPlayer(
                model=model,
                vocab=vocab,
                device=device,
                temperature=float(temperature),
                collect_trajectories=False,
                **common,
            )
        if kind == "random":
            return RandomBaselinePlayer(**common)
        if kind == "maxdamage":
            return MaxDamagePlayer(**common)
        if kind == "heuristic":
            return SimpleHeuristicBaselinePlayer(**common)
        raise ValueError(f"Unsupported player kind: {kind}")

    return _factory


async def _run_match(
    *,
    left_name: str,
    right_name: str,
    make_left: Callable[[str], object],
    make_right: Callable[[str], object],
    n_battles: int,
) -> MatchResult:
    first_half = max(0, int(n_battles)) // 2
    second_half = max(0, int(n_battles)) - first_half

    wins = 0
    losses = 0
    ties = 0

    if first_half > 0:
        left_p1 = make_left(f"{left_name[:4]}p1")
        right_p2 = make_right(f"{right_name[:4]}p2")
        try:
            await left_p1.battle_against(right_p2, n_battles=first_half)
            w, l, t = _result_counts(left_p1)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(left_p1)
            await _close_player(right_p2)

    if second_half > 0:
        right_p1 = make_right(f"{right_name[:4]}p1")
        left_p2 = make_left(f"{left_name[:4]}p2")
        try:
            await right_p1.battle_against(left_p2, n_battles=second_half)
            w, l, t = _result_counts(left_p2)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(right_p1)
            await _close_player(left_p2)

    decisive = wins + losses
    win_rate = wins / decisive if decisive > 0 else 0.0
    total = wins + losses + ties
    score_rate = (wins + 0.5 * ties) / max(1, total)
    ci_low, ci_high = _wilson_interval(wins, decisive)
    return MatchResult(
        left=left_name,
        right=right_name,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        score_rate=score_rate,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _print_matrix(names: Iterable[str], results: Iterable[MatchResult]) -> None:
    ordered_names = list(names)
    matrix: Dict[str, Dict[str, str]] = {name: {other: "-" for other in ordered_names} for name in ordered_names}
    for result in results:
        matrix[result.left][result.right] = f"{result.score_rate:.3f}"
        matrix[result.right][result.left] = f"{(1.0 - result.score_rate):.3f}"

    widths = {name: max(len(name), 8) for name in ordered_names}
    row_label_width = max(len("player"), max(len(name) for name in ordered_names))

    header = "player".ljust(row_label_width) + "  " + "  ".join(name.rjust(widths[name]) for name in ordered_names)
    print("\nscore-rate matrix (row player vs column player)")
    print(header)
    for name in ordered_names:
        row = name.ljust(row_label_width) + "  " + "  ".join(matrix[name][other].rjust(widths[other]) for other in ordered_names)
        print(row)


async def _run(args: argparse.Namespace) -> None:
    payload = load_checkpoint(args.checkpoint, args.device)
    vocab_path = args.vocab or payload.get("vocab_path")
    if vocab_path is None:
        raise SystemExit("No vocab path provided and checkpoint does not include one.")

    vocab = ObservationVocabulary.load_json(vocab_path)
    missing_tokens = missing_required_feature_tokens(vocab)
    if missing_tokens:
        raise SystemExit(
            "Checkpoint vocab is missing required observation tokens "
            f"({', '.join(missing_tokens[:4])}). Use a newer checkpoint/vocab."
        )

    model = model_from_checkpoint_payload(payload, args.device)
    model.eval()

    server_configuration = ServerConfiguration(
        websocket_url=args.server_ws,
        authentication_url=args.server_auth,
    )
    max_turns = None if int(args.max_turns_before_forfeit) < 0 else int(args.max_turns_before_forfeit)

    factories = {
        "model": _make_factory(
            kind="model",
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            max_concurrent_battles=int(args.max_concurrent_battles),
            model=model,
            vocab=vocab,
            device=args.device,
            temperature=float(args.temperature),
        ),
        "random": _make_factory(
            kind="random",
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            max_concurrent_battles=int(args.max_concurrent_battles),
        ),
        "maxdamage": _make_factory(
            kind="maxdamage",
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            max_concurrent_battles=int(args.max_concurrent_battles),
        ),
        "heuristic": _make_factory(
            kind="heuristic",
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            max_concurrent_battles=int(args.max_concurrent_battles),
        ),
    }

    ordered_players = ["model", "random", "maxdamage", "heuristic"]
    results: List[MatchResult] = []

    for left_name, right_name in itertools.combinations(ordered_players, 2):
        print(f"[match] {left_name} vs {right_name} battles={int(args.n_battles)}", flush=True)
        result = await _run_match(
            left_name=left_name,
            right_name=right_name,
            make_left=factories[left_name],
            make_right=factories[right_name],
            n_battles=int(args.n_battles),
        )
        results.append(result)
        print(
            f"result {left_name} vs {right_name}: "
            f"wins={result.wins} losses={result.losses} ties={result.ties} "
            f"win_rate={result.win_rate:.3f} score_rate={result.score_rate:.3f} "
            f"ci95=[{result.ci_low:.3f},{result.ci_high:.3f}]",
            flush=True,
        )

    _print_matrix(ordered_players, results)


def main() -> int:
    args = build_parser().parse_args()
    requested_device = str(args.device)
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(f"[device] requested={requested_device} resolved={args.device}")
    else:
        print(f"[device] using={args.device}")
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
