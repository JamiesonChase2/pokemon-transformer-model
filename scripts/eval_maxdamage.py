"""Evaluate a trained checkpoint against poke-env's max-damage baseline."""

from __future__ import annotations

import argparse
import asyncio
import math
import os

import torch
from poke_env.player import MaxBasePowerPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.bot import TransformerPlayer
from src.checkpoint import load_checkpoint, model_from_checkpoint_payload
from src.device import resolve_device
from src.vocab import ObservationVocabulary, missing_required_feature_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against the max-damage baseline.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--vocab", default=None, help="Optional vocab JSON path. Defaults to checkpoint payload path.")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-battles", type=int, default=50)
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


class MaxDamagePlayer(MaxBasePowerPlayer):
    def __init__(self, *args, max_turns_before_forfeit: int | None = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns_before_forfeit = (
            int(max_turns_before_forfeit) if max_turns_before_forfeit is not None else None
        )

    def choose_move(self, battle):  # type: ignore[override]
        if self.max_turns_before_forfeit is not None:
            turn = int(getattr(battle, "turn", 0) or 0)
            if turn > self.max_turns_before_forfeit:
                return ForfeitBattleOrder()
        return super().choose_move(battle)


def _result_counts(player: TransformerPlayer) -> tuple[int, int, int]:
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
    n_battles = max(0, int(args.n_battles))
    first_half = n_battles // 2
    second_half = n_battles - first_half

    wins = 0
    losses = 0
    ties = 0

    if first_half > 0:
        model_p1 = TransformerPlayer(
            account_configuration=AccountConfiguration.generate("modelmaxp1", rand=True),
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            model=model,
            vocab=vocab,
            device=args.device,
            temperature=args.temperature,
            max_turns_before_forfeit=max_turns,
            collect_trajectories=False,
            start_listening=True,
        )
        max_p2 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("maxdmgp2", rand=True),
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            start_listening=True,
        )
        try:
            await model_p1.battle_against(max_p2, n_battles=first_half)
            w, l, t = _result_counts(model_p1)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(model_p1)
            await _close_player(max_p2)

    if second_half > 0:
        max_p1 = MaxDamagePlayer(
            account_configuration=AccountConfiguration.generate("maxdmgp1", rand=True),
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            max_turns_before_forfeit=max_turns,
            start_listening=True,
        )
        model_p2 = TransformerPlayer(
            account_configuration=AccountConfiguration.generate("modelmaxp2", rand=True),
            battle_format=args.battle_format,
            server_configuration=server_configuration,
            model=model,
            vocab=vocab,
            device=args.device,
            temperature=args.temperature,
            max_turns_before_forfeit=max_turns,
            collect_trajectories=False,
            start_listening=True,
        )
        try:
            await max_p1.battle_against(model_p2, n_battles=second_half)
            w, l, t = _result_counts(model_p2)
            wins += w
            losses += l
            ties += t
        finally:
            await _close_player(max_p1)
            await _close_player(model_p2)

    decisive = wins + losses
    win_rate = wins / decisive if decisive > 0 else 0.0
    score_rate = (wins + 0.5 * ties) / max(1, wins + losses + ties)
    ci_low, ci_high = _wilson_interval(wins, decisive)

    print(
        f"results: wins={wins} losses={losses} ties={ties} "
        f"win_rate={win_rate:.3f} score_rate={score_rate:.3f} "
        f"ci95=[{ci_low:.3f},{ci_high:.3f}] n={wins + losses + ties}"
    )


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
