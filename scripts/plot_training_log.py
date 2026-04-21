"""Parse training logs and emit useful summary plots."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUMMARY_RE = re.compile(
    r"^iter=(?P<iteration>\d+)\s+"
    r"replay=(?P<replay>\d+)\s+"
    r"(?:selfplay_wlt|rollout_wlt)=(?P<wins>\d+)/(?P<losses>\d+)/(?P<ties>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"policy_loss=(?P<policy_loss>[-+0-9.eE]+)\s+"
    r"value_loss=(?P<value_loss>[-+0-9.eE]+)\s+"
    r"entropy=(?P<entropy>[-+0-9.eE]+)\s+"
    r"kl=(?P<kl>[-+0-9.eE]+)\s+"
    r"clip_frac=(?P<clip_frac>[-+0-9.eE]+)\s+"
    r"eval_win_rate=(?P<eval_win_rate>[-+0-9.eE]+)\s+"
    r"eval_ci=\[(?P<ci_low>[-+0-9.eE]+),(?P<ci_high>[-+0-9.eE]+)\]\s+"
    r"(?:(?:best_eval_win_rate)=(?P<best_eval_win_rate>[-+0-9.eE]+)\s+)?"
    r"promoted=(?P<promoted>True|False)\s+"
    r"iter_elapsed=(?P<iter_elapsed>[-+0-9.eE]+)s$"
)

TRAIN_STEP_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+"
    r"train_step=(?P<step>\d+)/(?P<step_total>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"policy=(?P<policy>[-+0-9.eE]+)\s+"
    r"value=(?P<value>[-+0-9.eE]+)\s+"
    r"entropy=(?P<entropy>[-+0-9.eE]+)\s+"
    r"kl=(?P<kl>[-+0-9.eE]+)\s+"
    r"clip_frac=(?P<clip_frac>[-+0-9.eE]+)$"
)

TRAIN_START_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+"
    r"train start steps=(?P<steps>\d+)\s+"
    r"requested_steps=(?P<requested_steps>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"replay=(?P<replay>\d+)\s+"
    r"target_kl=(?P<target_kl>[-+0-9.eE]+)\s+"
    r"epochs=(?P<epochs>\d+)\s+"
    r"grad_accum=(?P<grad_accum>\d+)\s+"
    r"effective_batch=(?P<effective_batch>\d+)$"
)

ITER_START_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+start\s+"
    r"selfplay_battles=(?P<selfplay_battles>\d+)\s+"
    r"selfplay_steps=(?P<selfplay_steps>\d+)\s+"
    r"train_steps=(?P<train_steps>\d+)\s+"
    r"train_epochs=(?P<train_epochs>\d+)\s+"
    r"eval_battles=(?P<eval_battles>\d+)\s+"
    r"total_env_steps=(?P<total_env_steps>\d+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+"
    r"reward_terminal=\((?P<terminal_win>[-+0-9.eE]+),(?P<terminal_loss>[-+0-9.eE]+)\)\s+"
    r"reward_faint=(?P<reward_faint>\S+)\s+"
    r"max_turns_before_forfeit=(?P<max_turns>\S+)$"
)

ITER_START_MAXDAMAGE_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+start\s+"
    r"rollout_battles=(?P<selfplay_battles>\d+)\s+"
    r"rollout_steps=(?P<selfplay_steps>\d+)\s+"
    r"train_steps=(?P<train_steps>\d+)\s+"
    r"train_epochs=(?P<train_epochs>\d+)\s+"
    r"eval_battles=(?P<eval_battles>\d+)\s+"
    r"max_concurrent_battles=(?P<max_concurrent_battles>\d+)\s+"
    r"total_env_steps=(?P<total_env_steps>\d+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+"
    r"best_eval_win_rate=(?P<best_eval_win_rate>[-+0-9.eE]+)$"
)

SELFPLAY_DONE_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+"
    r"(?:self-play|rollout) done battles=(?P<battles>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"replay=(?P<replay>\d+)\s+"
    r"elapsed=(?P<elapsed>[-+0-9.eE]+)s$"
)

EVAL_DONE_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+"
    r"eval done elapsed=(?P<elapsed>[-+0-9.eE]+)s\s+"
    r"win_rate=(?P<win_rate>[-+0-9.eE]+)$"
)


@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    replay: int
    wins: int
    losses: int
    ties: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    clip_frac: float
    eval_win_rate: float
    ci_low: float
    ci_high: float
    best_eval_win_rate: Optional[float]
    promoted: bool
    iter_elapsed: float


@dataclass(frozen=True)
class TrainStep:
    iteration: int
    step: int
    step_total: int
    loss: float
    policy: float
    value: float
    entropy: float
    kl: float
    clip_frac: float


@dataclass(frozen=True)
class TrainStart:
    iteration: int
    steps: int
    requested_steps: str
    batch: int
    replay: int
    target_kl: float
    epochs: int
    grad_accum: int
    effective_batch: int


@dataclass(frozen=True)
class IterationStart:
    iteration: int
    selfplay_battles: int
    selfplay_steps: int
    train_steps: int
    train_epochs: int
    eval_battles: int
    total_env_steps: int
    lr: float
    max_concurrent_battles: Optional[int]
    best_eval_win_rate: Optional[float]
    terminal_win: float
    terminal_loss: float
    reward_faint: str
    max_turns: str


@dataclass(frozen=True)
class SelfplayDone:
    iteration: int
    battles: int
    steps: int
    episodes: int
    replay: int
    elapsed: float


@dataclass(frozen=True)
class EvalDone:
    iteration: int
    elapsed: float
    win_rate: float


@dataclass(frozen=True)
class ParsedTrainingLog:
    iteration_starts: List[IterationStart]
    train_starts: List[TrainStart]
    selfplay_done: List[SelfplayDone]
    train_steps: List[TrainStep]
    eval_done: List[EvalDone]
    summaries: List[IterationSummary]


def _parse_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Unexpected bool text: {text}")


def parse_training_log_lines(lines: Iterable[str]) -> ParsedTrainingLog:
    iteration_starts: Dict[int, IterationStart] = {}
    train_starts: Dict[int, TrainStart] = {}
    selfplay_done: Dict[int, SelfplayDone] = {}
    train_steps: Dict[Tuple[int, int, int], TrainStep] = {}
    eval_done: Dict[int, EvalDone] = {}
    summaries: Dict[int, IterationSummary] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = SUMMARY_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            summaries[iteration] = IterationSummary(
                iteration=iteration,
                replay=int(data["replay"]),
                wins=int(data["wins"]),
                losses=int(data["losses"]),
                ties=int(data["ties"]),
                loss=float(data["loss"]),
                policy_loss=float(data["policy_loss"]),
                value_loss=float(data["value_loss"]),
                entropy=float(data["entropy"]),
                kl=float(data["kl"]),
                clip_frac=float(data["clip_frac"]),
                eval_win_rate=float(data["eval_win_rate"]),
                ci_low=float(data["ci_low"]),
                ci_high=float(data["ci_high"]),
                best_eval_win_rate=(
                    float(data["best_eval_win_rate"]) if data.get("best_eval_win_rate") not in (None, "") else None
                ),
                promoted=_parse_bool(data["promoted"]),
                iter_elapsed=float(data["iter_elapsed"]),
            )
            continue

        match = TRAIN_STEP_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            step = int(data["step"])
            step_total = int(data["step_total"])
            train_steps[(iteration, step, step_total)] = TrainStep(
                iteration=iteration,
                step=step,
                step_total=step_total,
                loss=float(data["loss"]),
                policy=float(data["policy"]),
                value=float(data["value"]),
                entropy=float(data["entropy"]),
                kl=float(data["kl"]),
                clip_frac=float(data["clip_frac"]),
            )
            continue

        match = TRAIN_START_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            train_starts[iteration] = TrainStart(
                iteration=iteration,
                steps=int(data["steps"]),
                requested_steps=data["requested_steps"],
                batch=int(data["batch"]),
                replay=int(data["replay"]),
                target_kl=float(data["target_kl"]),
                epochs=int(data["epochs"]),
                grad_accum=int(data["grad_accum"]),
                effective_batch=int(data["effective_batch"]),
            )
            continue

        match = ITER_START_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            iteration_starts[iteration] = IterationStart(
                iteration=iteration,
                selfplay_battles=int(data["selfplay_battles"]),
                selfplay_steps=int(data["selfplay_steps"]),
                train_steps=int(data["train_steps"]),
                train_epochs=int(data["train_epochs"]),
                eval_battles=int(data["eval_battles"]),
                total_env_steps=int(data["total_env_steps"]),
                lr=float(data["lr"]),
                max_concurrent_battles=None,
                best_eval_win_rate=None,
                terminal_win=float(data["terminal_win"]),
                terminal_loss=float(data["terminal_loss"]),
                reward_faint=data["reward_faint"],
                max_turns=data["max_turns"],
            )
            continue

        match = ITER_START_MAXDAMAGE_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            iteration_starts[iteration] = IterationStart(
                iteration=iteration,
                selfplay_battles=int(data["selfplay_battles"]),
                selfplay_steps=int(data["selfplay_steps"]),
                train_steps=int(data["train_steps"]),
                train_epochs=int(data["train_epochs"]),
                eval_battles=int(data["eval_battles"]),
                total_env_steps=int(data["total_env_steps"]),
                lr=float(data["lr"]),
                max_concurrent_battles=int(data["max_concurrent_battles"]),
                best_eval_win_rate=float(data["best_eval_win_rate"]),
                terminal_win=float("nan"),
                terminal_loss=float("nan"),
                reward_faint="n/a",
                max_turns="n/a",
            )
            continue

        match = SELFPLAY_DONE_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            selfplay_done[iteration] = SelfplayDone(
                iteration=iteration,
                battles=int(data["battles"]),
                steps=int(data["steps"]),
                episodes=int(data["episodes"]),
                replay=int(data["replay"]),
                elapsed=float(data["elapsed"]),
            )
            continue

        match = EVAL_DONE_RE.match(line)
        if match:
            data = match.groupdict()
            iteration = int(data["iteration"])
            eval_done[iteration] = EvalDone(
                iteration=iteration,
                elapsed=float(data["elapsed"]),
                win_rate=float(data["win_rate"]),
            )
            continue

    return ParsedTrainingLog(
        iteration_starts=sorted(iteration_starts.values(), key=lambda row: row.iteration),
        train_starts=sorted(train_starts.values(), key=lambda row: row.iteration),
        selfplay_done=sorted(selfplay_done.values(), key=lambda row: row.iteration),
        train_steps=sorted(train_steps.values(), key=lambda row: (row.iteration, row.step)),
        eval_done=sorted(eval_done.values(), key=lambda row: row.iteration),
        summaries=sorted(summaries.values(), key=lambda row: row.iteration),
    )


def parse_training_logs(paths: Sequence[str | Path]) -> ParsedTrainingLog:
    lines: List[str] = []
    for path in paths:
        lines.extend(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
    return parse_training_log_lines(lines)


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_report(parsed: ParsedTrainingLog) -> List[str]:
    lines: List[str] = []
    lines.append(f"iterations parsed: {len(parsed.summaries)}")
    lines.append(f"train-step points parsed: {len(parsed.train_steps)}")

    if parsed.summaries:
        first = parsed.summaries[0]
        last = parsed.summaries[-1]
        lines.append(f"iteration span: {first.iteration} -> {last.iteration}")
        if first.iteration > 1:
            lines.append(
                "warning: parsed log does not start at iteration 1; plots only reflect the portion present in the log file"
            )
        lines.append(
            "latest iteration: "
            f"{last.iteration} eval_win_rate={last.eval_win_rate:.3f} "
            f"rollout_win_rate={last.wins / max(1, last.wins + last.losses):.3f} "
            f"kl={last.kl:.4f} clip_frac={last.clip_frac:.3f}"
        )
        best = max(parsed.summaries, key=lambda row: row.eval_win_rate)
        lines.append(
            "best eval iteration: "
            f"{best.iteration} eval_win_rate={best.eval_win_rate:.3f} "
            f"ci=[{best.ci_low:.3f},{best.ci_high:.3f}] promoted={best.promoted}"
        )
        tail = parsed.summaries[-5:]
        lines.append(
            "last-5 averages: "
            f"eval_win_rate={mean(row.eval_win_rate for row in tail):.3f} "
            f"kl={mean(row.kl for row in tail):.4f} "
            f"clip_frac={mean(row.clip_frac for row in tail):.3f}"
        )
    return lines


def _plot_overview(parsed: ParsedTrainingLog, output_dir: Path, *, title: Optional[str]) -> Path:
    _configure_matplotlib_env(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not parsed.summaries:
        raise ValueError("No iteration summary lines were found in the provided logs.")

    summaries = parsed.summaries
    iterations = [row.iteration for row in summaries]
    rollout_win_rate = [row.wins / max(1, row.wins + row.losses) for row in summaries]
    lrs_by_iter = {row.iteration: row.lr for row in parsed.iteration_starts}
    lr_values = [lrs_by_iter.get(iteration, float("nan")) for iteration in iterations]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(iterations, [row.eval_win_rate for row in summaries], marker="o", label="eval win rate")
    ax.fill_between(
        iterations,
        [row.ci_low for row in summaries],
        [row.ci_high for row in summaries],
        alpha=0.2,
        label="95% CI",
    )
    promoted_iters = [row.iteration for row in summaries if row.promoted]
    promoted_wins = [row.eval_win_rate for row in summaries if row.promoted]
    if promoted_iters:
        ax.scatter(promoted_iters, promoted_wins, marker="*", s=120, label="promoted")
    ax.set_title("Eval Win Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(iterations, rollout_win_rate, marker="o", label="rollout win rate")
    ax.set_title("Rollout Win Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(iterations, [row.kl for row in summaries], marker="o", label="KL")
    ax.plot(iterations, [row.clip_frac for row in summaries], marker="o", label="clip fraction")
    ax.set_title("PPO Stability")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(iterations, [row.loss for row in summaries], marker="o", label="loss")
    ax.plot(iterations, [row.policy_loss for row in summaries], marker="o", label="policy loss")
    ax.plot(iterations, [row.value_loss for row in summaries], marker="o", label="value loss")
    ax.set_title("Losses")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(iterations, [row.entropy for row in summaries], marker="o", label="entropy")
    if any(value == value for value in lr_values):
        ax2 = ax.twinx()
        ax2.plot(iterations, lr_values, color="tab:red", marker="x", label="lr")
        ax2.set_ylabel("Learning rate")
        ax2.legend(loc="lower right")
    ax.set_title("Entropy and LR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    ax.plot(iterations, [row.replay for row in summaries], marker="o", label="replay size")
    ax.plot(iterations, [row.iter_elapsed for row in summaries], marker="o", label="iter elapsed (s)")
    ax.set_title("Replay Size and Iteration Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path = output_dir / "training_overview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_train_steps(parsed: ParsedTrainingLog, output_dir: Path, *, title: Optional[str]) -> Optional[Path]:
    _configure_matplotlib_env(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not parsed.train_steps:
        return None

    steps = parsed.train_steps
    x_values = [row.iteration + (row.step / max(1, row.step_total)) for row in steps]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    if title:
        fig.suptitle(f"{title} — Train-Step Detail", fontsize=16)

    axes[0].plot(x_values, [row.kl for row in steps], marker="o", linewidth=1.5)
    axes[0].set_title("Train-Step KL")
    axes[0].set_xlabel("Iteration.progress")
    axes[0].set_ylabel("KL")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_values, [row.clip_frac for row in steps], marker="o", linewidth=1.5)
    axes[1].set_title("Train-Step Clip Fraction")
    axes[1].set_xlabel("Iteration.progress")
    axes[1].set_ylabel("Clip fraction")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_values, [row.loss for row in steps], marker="o", linewidth=1.5, label="loss")
    axes[2].plot(x_values, [row.entropy for row in steps], marker="o", linewidth=1.5, label="entropy")
    axes[2].set_title("Train-Step Loss and Entropy")
    axes[2].set_xlabel("Iteration.progress")
    axes[2].set_ylabel("Value")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    output_path = output_dir / "train_step_detail.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse pokeenv_transformer training logs and render summary graphs.")
    parser.add_argument(
        "log_paths",
        nargs="*",
        help="One or more raw training log files to parse. Defaults to ./output.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated CSVs and PNGs. Defaults to <first_log_parent>/<first_log_stem>_plots.",
    )
    parser.add_argument("--title", default=None, help="Optional title to place on the generated plots.")
    return parser


def _default_log_paths() -> List[str]:
    return ["output.txt"]


def _configure_matplotlib_env(output_dir: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "pokeenv_training_matplotlib"
    mpl_dir = cache_root / "mplconfig"
    cache_dir = cache_root / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", tempfile.gettempdir())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log_paths = list(args.log_paths) if args.log_paths else _default_log_paths()
    parsed = parse_training_logs(log_paths)

    first_log = Path(log_paths[0])
    output_dir = Path(args.output_dir) if args.output_dir else first_log.parent / f"{first_log.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "iteration_summary.csv", parsed.summaries)
    _write_csv(output_dir / "train_steps.csv", parsed.train_steps)
    _write_csv(output_dir / "iteration_start.csv", parsed.iteration_starts)
    _write_csv(output_dir / "train_start.csv", parsed.train_starts)
    _write_csv(output_dir / "selfplay_done.csv", parsed.selfplay_done)
    _write_csv(output_dir / "eval_done.csv", parsed.eval_done)

    overview_path = _plot_overview(parsed, output_dir, title=args.title)
    train_step_path = _plot_train_steps(parsed, output_dir, title=args.title)

    print(f"wrote: {output_dir / 'iteration_summary.csv'}")
    print(f"wrote: {output_dir / 'train_steps.csv'}")
    print(f"wrote: {overview_path}")
    if train_step_path is not None:
        print(f"wrote: {train_step_path}")

    for line in _build_report(parsed):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
