"""Play ladder/challenge games with a trained Transformer checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.bot import TransformerPlayer
from src.checkpoint import load_checkpoint, model_from_checkpoint_payload
from src.device import resolve_device
from src.vocab import ObservationVocabulary, missing_required_feature_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Pokemon Showdown with a Transformer checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--vocab", default=None, help="Optional vocab JSON path. Defaults to checkpoint payload path.")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, or cpu.")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-battles", type=int, default=1)
    parser.add_argument(
        "--max-turns-before-forfeit",
        type=int,
        default=500,
        help="Auto-forfeit if battle turn exceeds this threshold. Use -1 to disable.",
    )
    parser.add_argument("--challenge", default=None, help="Username to challenge. If omitted, ladder is used.")
    parser.add_argument(
        "--server-ws",
        default=os.environ.get("SHOWDOWN_WS_URL", "ws://localhost:8000/showdown/websocket"),
    )
    parser.add_argument(
        "--server-auth",
        default=os.environ.get("SHOWDOWN_AUTH_URL", "http://localhost:8000/action.php?"),
    )
    return parser


async def _close_player(player: TransformerPlayer) -> None:
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

    player = TransformerPlayer(
        account_configuration=AccountConfiguration.generate("play", rand=True),
        battle_format=args.battle_format,
        server_configuration=server_configuration,
        model=model,
        vocab=vocab,
        device=args.device,
        temperature=args.temperature,
        max_turns_before_forfeit=(None if int(args.max_turns_before_forfeit) < 0 else int(args.max_turns_before_forfeit)),
        collect_trajectories=False,
        start_listening=True,
    )

    if args.challenge:
        await player.send_challenges(args.challenge, int(args.n_battles))
    else:
        await player.ladder(int(args.n_battles))

    wins = sum(bool(getattr(b, "won", False)) for b in player.battles.values())
    losses = sum(bool(getattr(b, "lost", False)) for b in player.battles.values())
    ties = sum(bool(getattr(b, "finished", False) and not getattr(b, "won", False) and not getattr(b, "lost", False)) for b in player.battles.values())
    print(f"results: wins={wins} losses={losses} ties={ties}")

    await _close_player(player)


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
