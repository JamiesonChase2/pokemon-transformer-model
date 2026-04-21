"""Start a local Gen 9 random battle and print the exact structured model input."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import torch
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.bot import TransformerPlayer
from src.device import resolve_device
from src.vocab import ObservationVocabulary, build_default_vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a local random battle and print structured model inputs.")
    parser.add_argument("--battle-format", default="gen9randombattle")
    parser.add_argument("--max-prints", type=int, default=1, help="Number of move requests to print before stopping.")
    parser.add_argument("--keep-playing", action="store_true", help="Keep playing after printing instead of forfeiting.")
    parser.add_argument("--print-obs", action="store_true", help="Also print the extracted observation as JSON.")
    parser.add_argument("--showdown-root", default=None, help="Optional local pokemon-showdown path for vocab expansion.")
    parser.add_argument("--server-ws", default=os.environ.get("SHOWDOWN_WS_URL", "ws://localhost:8000/showdown/websocket"))
    parser.add_argument("--server-auth", default=os.environ.get("SHOWDOWN_AUTH_URL", "http://localhost:8000/action.php?"))
    parser.add_argument("--device", default="auto", help="Device label for consistency in output.")
    return parser


class RandomBattlePlayer(Player):
    def choose_move(self, battle):  # type: ignore[override]
        return self.choose_random_move(battle)


def _print_decoded_ids(vocab: ObservationVocabulary, obs_tensor: torch.Tensor) -> None:
    meta = vocab.schema_meta()
    offsets = meta["offsets"]
    n_slots = int(meta["n_pokemon_slots"])
    n_moves = int(meta["n_move_slots"])
    n_abilities = int(meta["n_ability_slots"])

    pokemon_ids = obs_tensor[offsets["pokemon_ids"][0] : offsets["pokemon_ids"][1]].reshape(n_slots, 2).long()
    ability_ids = obs_tensor[offsets["ability_ids"][0] : offsets["ability_ids"][1]].reshape(n_slots, n_abilities).long()
    move_ids = obs_tensor[offsets["move_ids"][0] : offsets["move_ids"][1]].reshape(n_slots, n_moves).long()
    transition_ids = obs_tensor[offsets["transition_move_ids"][0] : offsets["transition_move_ids"][1]].long()

    print("decoded_ids=")
    for slot_idx in range(n_slots):
        species = vocab.decode("pokemon.species", int(pokemon_ids[slot_idx, 0].item()))
        item = vocab.decode("pokemon.item", int(pokemon_ids[slot_idx, 1].item()))
        abilities = [
            vocab.decode("pokemon.ability", int(value.item()))
            for value in ability_ids[slot_idx]
            if int(value.item()) > 0
        ]
        moves = [
            vocab.decode("move.id", int(value.item()))
            for value in move_ids[slot_idx]
            if int(value.item()) > 0
        ]
        print(
            f"  slot={slot_idx:02d} species={species} item={item} "
            f"abilities={abilities or ['unk']} moves={moves or ['unk']}"
        )
    print(
        "  transitions="
        f"{[vocab.decode('move.id', int(value.item())) for value in transition_ids]}"
    )


class DebugTokenPlayer(TransformerPlayer):
    def __init__(self, *args, max_prints: int, print_obs: bool, keep_playing: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_prints = int(max_prints)
        self._print_obs = bool(print_obs)
        self._keep_playing = bool(keep_playing)
        self._num_prints = 0

    def choose_move(self, battle):  # type: ignore[override]
        model_input = self.build_model_input(battle)

        if self._num_prints < self._max_prints:
            self._num_prints += 1
            self._print_model_input(battle, model_input)
            if not self._keep_playing and self._num_prints >= self._max_prints:
                return ForfeitBattleOrder()

        return self.choose_random_move(battle)

    def _print_model_input(self, battle, model_input):
        obs_tensor = model_input["obs_tensor"].cpu()
        legal_mask = model_input["legal_mask"].tolist()
        meta = self.vocab.schema_meta()

        print(f"battle_tag={model_input['battle_tag']} turn={int(getattr(battle, 'turn', 0) or 0)}")
        print(f"obs_dim={int(meta['obs_dim'])}")
        print(f"legal_mask={legal_mask}")
        if self._print_obs:
            print("obs_json=")
            print(json.dumps(model_input["obs"], indent=2, sort_keys=True))
        _print_decoded_ids(self.vocab, obs_tensor)
        print("section_offsets=")
        for name, (start, end) in meta["offsets"].items():
            print(f"  {name}: start={start} end={end} size={end - start}")
        print("flat_obs_rows=")
        for idx, value in enumerate(obs_tensor.tolist()):
            print(f"{idx:05d} value={float(value):.6f}")


async def _close_player(player: Player) -> None:
    try:
        await player.ps_client.stop_listening()
    except Exception:
        pass


async def _run(args: argparse.Namespace) -> None:
    vocab = build_default_vocab(
        gen=9,
        showdown_root=args.showdown_root,
        battle_format=args.battle_format,
    )
    server_configuration = ServerConfiguration(
        websocket_url=args.server_ws,
        authentication_url=args.server_auth,
    )

    debug_player = DebugTokenPlayer(
        account_configuration=AccountConfiguration.generate("tokendbg", rand=True),
        battle_format=args.battle_format,
        server_configuration=server_configuration,
        model=torch.nn.Identity(),
        vocab=vocab,
        device=args.device,
        collect_trajectories=False,
        start_listening=True,
        max_prints=int(args.max_prints),
        print_obs=bool(args.print_obs),
        keep_playing=bool(args.keep_playing),
    )
    random_player = RandomBattlePlayer(
        account_configuration=AccountConfiguration.generate("tokenopp", rand=True),
        battle_format=args.battle_format,
        server_configuration=server_configuration,
        start_listening=True,
    )

    try:
        await debug_player.battle_against(random_player, n_battles=1)
    finally:
        await _close_player(debug_player)
        await _close_player(random_player)


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
