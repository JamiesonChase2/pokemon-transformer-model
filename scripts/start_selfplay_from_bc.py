"""Bootstrap a self-play run from a BC checkpoint, then launch training."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.checkpoint import load_checkpoint, save_checkpoint_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a self-play checkpoint directory from a BC checkpoint and launch threaded or Ray self-play."
    )
    parser.add_argument("--bc-checkpoint-dir", required=True, help="BC checkpoint directory containing current.pt/best.pt/vocab.json.")
    parser.add_argument("--selfplay-checkpoint-dir", required=True, help="Target self-play checkpoint directory.")
    parser.add_argument(
        "--source-checkpoint",
        default="best",
        choices=("best", "current"),
        help="Which BC checkpoint to clone into current.pt/best.pt for self-play bootstrapping.",
    )
    parser.add_argument(
        "--rebootstrap",
        action="store_true",
        help="Overwrite target current.pt/best.pt from the BC checkpoint even if the self-play directory already exists.",
    )
    parser.add_argument(
        "--trainer",
        default="threaded",
        choices=("threaded", "ray"),
        help="Which self-play trainer module to launch after bootstrapping.",
    )
    parser.add_argument(
        "selfplay_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the chosen self-play trainer. Prefix with '--' when needed.",
    )
    return parser


def _strip_remainder_separator(args: Sequence[str]) -> list[str]:
    values = list(args)
    if values and values[0] == "--":
        return values[1:]
    return values


def _validate_passthrough_args(args: Sequence[str]) -> None:
    reserved = {"--checkpoint-dir", "--resume"}
    conflicts = [arg for arg in args if arg in reserved]
    if conflicts:
        raise SystemExit(
            "start_selfplay_from_bc.py manages --checkpoint-dir and --resume itself. "
            f"Remove passthrough args: {', '.join(conflicts)}"
        )


def _prepare_payload(payload: dict, *, source_path: Path, source_kind: str) -> dict:
    prepared = dict(payload)
    prepared["optimizer_state_dict"] = None
    prepared["scheduler_state_dict"] = None
    prepared["iteration"] = 0
    extra = dict(prepared.get("extra") or {})
    extra.update(
        {
            "bootstrap": True,
            "bootstrap_from_bc": str(source_path),
            "bootstrap_source_checkpoint": str(source_kind),
        }
    )
    prepared["extra"] = extra
    return prepared


def _bootstrap_selfplay_dir(
    *,
    bc_dir: Path,
    target_dir: Path,
    source_kind: str,
    force: bool,
) -> None:
    source_ckpt = bc_dir / f"{source_kind}.pt"
    source_vocab = bc_dir / "vocab.json"
    if not source_ckpt.exists():
        raise SystemExit(f"Missing BC checkpoint: {source_ckpt}")
    if not source_vocab.exists():
        raise SystemExit(f"Missing BC vocab: {source_vocab}")

    current_ckpt = target_dir / "current.pt"
    best_ckpt = target_dir / "best.pt"
    target_vocab = target_dir / "vocab.json"
    already_bootstrapped = current_ckpt.exists() and best_ckpt.exists() and target_vocab.exists()
    if already_bootstrapped and not force:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint(source_ckpt, device="cpu")
    boot_payload = _prepare_payload(payload, source_path=source_ckpt, source_kind=source_kind)
    save_checkpoint_payload(path=current_ckpt, payload=boot_payload)
    save_checkpoint_payload(path=best_ckpt, payload=boot_payload)
    shutil.copy2(source_vocab, target_vocab)


def main() -> int:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()
    passthrough_args = _strip_remainder_separator([*args.selfplay_args, *unknown_args])
    _validate_passthrough_args(passthrough_args)

    repo_root = Path(__file__).resolve().parents[1]
    bc_dir = Path(args.bc_checkpoint_dir).expanduser().resolve()
    target_dir = Path(args.selfplay_checkpoint_dir).expanduser().resolve()

    _bootstrap_selfplay_dir(
        bc_dir=bc_dir,
        target_dir=target_dir,
        source_kind=str(args.source_checkpoint),
        force=bool(args.rebootstrap),
    )

    trainer_module = "scripts.train_selfplay" if args.trainer == "threaded" else "scripts.train_selfplay_ray"
    command = [
        sys.executable,
        "-m",
        trainer_module,
        "--resume",
        "--checkpoint-dir",
        str(target_dir),
        *passthrough_args,
    ]
    return subprocess.run(command, cwd=repo_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
