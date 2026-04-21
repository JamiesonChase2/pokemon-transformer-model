"""Parse behavior-cloning training logs and render summary plots."""

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


ITER_START_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+start\s+"
    r"demo_battles=(?P<demo_battles>\d+)\s+"
    r"demo_steps=(?P<demo_steps>\d+)\s+"
    r"train_steps=(?P<train_steps>\d+)\s+"
    r"train_epochs=(?P<train_epochs>\d+)\s+"
    r"max_concurrent_battles=(?P<max_concurrent_battles>\d+)\s+"
    r"total_env_steps=(?P<total_env_steps>\d+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+"
    r"best_bc_metric=(?P<best_bc_metric>[-+0-9.eE]+)\s+"
    r"gate\(train>=(?P<gate_train>[-+0-9.eE]+),val>=(?P<gate_val>[-+0-9.eE]+)\)$"
)

DEMO_DONE_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+demo done battles=(?P<battles>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+replay=(?P<replay>\d+)\s+skipped=(?P<skipped>\d+)\s+"
    r"elapsed=(?P<elapsed>[-+0-9.eE]+)s$"
)

GATE_SPLIT_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+gate split train_samples=(?P<train_samples>\d+)\s+"
    r"val_samples=(?P<val_samples>\d+)$"
)

TRAIN_START_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+train start steps=(?P<steps>\d+)\s+"
    r"requested_steps=(?P<requested_steps>\S+)\s+batch=(?P<batch>\d+)\s+"
    r"replay=(?P<replay>\d+)\s+epochs=(?P<epochs>\d+)\s+grad_accum=(?P<grad_accum>\d+)\s+"
    r"effective_batch=(?P<effective_batch>\d+)$"
)

TRAIN_STEP_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+train_step=(?P<step>\d+)/(?P<step_total>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+policy=(?P<policy>[-+0-9.eE]+)\s+"
    r"value=(?P<value>[-+0-9.eE]+)\s+entropy=(?P<entropy>[-+0-9.eE]+)\s+acc=(?P<acc>[-+0-9.eE]+)$"
)

VAL_RE = re.compile(
    r"^\[iter\s+(?P<iteration>\d+)\]\s+val loss=(?P<val_loss>[-+0-9.eE]+)\s+"
    r"entropy=(?P<val_entropy>[-+0-9.eE]+)\s+accuracy=(?P<val_accuracy>[-+0-9.eE]+)$"
)

SUMMARY_RE = re.compile(
    r"^iter=(?P<iteration>\d+)\s+replay=(?P<replay>\d+)\s+demo_steps=(?P<demo_steps>\d+)\s+"
    r"skipped=(?P<skipped>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)\s+policy_loss=(?P<policy_loss>[-+0-9.eE]+)\s+"
    r"value_loss=(?P<value_loss>[-+0-9.eE]+)\s+entropy=(?P<entropy>[-+0-9.eE]+)\s+"
    r"train_accuracy=(?P<train_accuracy>[-+0-9.eE]+)\s+val_accuracy=(?P<val_accuracy>[-+0-9.eE]+)\s+"
    r"gate_met=(?P<gate_met>True|False)\s+best_bc_metric=(?P<best_bc_metric>[-+0-9.eE]+)\s+"
    r"promoted=(?P<promoted>True|False)\s+iter_elapsed=(?P<iter_elapsed>[-+0-9.eE]+)s$"
)


@dataclass(frozen=True)
class IterationStart:
    iteration: int
    demo_battles: int
    demo_steps: int
    train_steps: int
    train_epochs: int
    max_concurrent_battles: int
    total_env_steps: int
    lr: float
    best_bc_metric: float
    gate_train: float
    gate_val: float


@dataclass(frozen=True)
class DemoDone:
    iteration: int
    battles: int
    steps: int
    replay: int
    skipped: int
    elapsed: float


@dataclass(frozen=True)
class GateSplit:
    iteration: int
    train_samples: int
    val_samples: int


@dataclass(frozen=True)
class TrainStart:
    iteration: int
    steps: int
    requested_steps: str
    batch: int
    replay: int
    epochs: int
    grad_accum: int
    effective_batch: int


@dataclass(frozen=True)
class TrainStep:
    iteration: int
    step: int
    step_total: int
    loss: float
    policy: float
    value: float
    entropy: float
    acc: float


@dataclass(frozen=True)
class ValMetric:
    iteration: int
    val_loss: float
    val_entropy: float
    val_accuracy: float


@dataclass(frozen=True)
class Summary:
    iteration: int
    replay: int
    demo_steps: int
    skipped: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    train_accuracy: float
    val_accuracy: float
    gate_met: bool
    best_bc_metric: float
    promoted: bool
    iter_elapsed: float


@dataclass(frozen=True)
class ParsedImitationLog:
    iteration_starts: List[IterationStart]
    demo_done: List[DemoDone]
    gate_splits: List[GateSplit]
    train_starts: List[TrainStart]
    train_steps: List[TrainStep]
    val_metrics: List[ValMetric]
    summaries: List[Summary]


def _parse_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Unexpected bool text: {text}")


def parse_imitation_log_lines(lines: Iterable[str]) -> ParsedImitationLog:
    iteration_starts: Dict[int, IterationStart] = {}
    demo_done: Dict[int, DemoDone] = {}
    gate_splits: Dict[int, GateSplit] = {}
    train_starts: Dict[int, TrainStart] = {}
    train_steps: Dict[Tuple[int, int, int], TrainStep] = {}
    val_metrics: Dict[int, ValMetric] = {}
    summaries: Dict[int, Summary] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if match := ITER_START_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            iteration_starts[iteration] = IterationStart(
                iteration=iteration,
                demo_battles=int(data["demo_battles"]),
                demo_steps=int(data["demo_steps"]),
                train_steps=int(data["train_steps"]),
                train_epochs=int(data["train_epochs"]),
                max_concurrent_battles=int(data["max_concurrent_battles"]),
                total_env_steps=int(data["total_env_steps"]),
                lr=float(data["lr"]),
                best_bc_metric=float(data["best_bc_metric"]),
                gate_train=float(data["gate_train"]),
                gate_val=float(data["gate_val"]),
            )
            continue

        if match := DEMO_DONE_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            demo_done[iteration] = DemoDone(
                iteration=iteration,
                battles=int(data["battles"]),
                steps=int(data["steps"]),
                replay=int(data["replay"]),
                skipped=int(data["skipped"]),
                elapsed=float(data["elapsed"]),
            )
            continue

        if match := GATE_SPLIT_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            gate_splits[iteration] = GateSplit(
                iteration=iteration,
                train_samples=int(data["train_samples"]),
                val_samples=int(data["val_samples"]),
            )
            continue

        if match := TRAIN_START_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            train_starts[iteration] = TrainStart(
                iteration=iteration,
                steps=int(data["steps"]),
                requested_steps=data["requested_steps"],
                batch=int(data["batch"]),
                replay=int(data["replay"]),
                epochs=int(data["epochs"]),
                grad_accum=int(data["grad_accum"]),
                effective_batch=int(data["effective_batch"]),
            )
            continue

        if match := TRAIN_STEP_RE.match(line):
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
                acc=float(data["acc"]),
            )
            continue

        if match := VAL_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            val_metrics[iteration] = ValMetric(
                iteration=iteration,
                val_loss=float(data["val_loss"]),
                val_entropy=float(data["val_entropy"]),
                val_accuracy=float(data["val_accuracy"]),
            )
            continue

        if match := SUMMARY_RE.match(line):
            data = match.groupdict()
            iteration = int(data["iteration"])
            summaries[iteration] = Summary(
                iteration=iteration,
                replay=int(data["replay"]),
                demo_steps=int(data["demo_steps"]),
                skipped=int(data["skipped"]),
                loss=float(data["loss"]),
                policy_loss=float(data["policy_loss"]),
                value_loss=float(data["value_loss"]),
                entropy=float(data["entropy"]),
                train_accuracy=float(data["train_accuracy"]),
                val_accuracy=float(data["val_accuracy"]),
                gate_met=_parse_bool(data["gate_met"]),
                best_bc_metric=float(data["best_bc_metric"]),
                promoted=_parse_bool(data["promoted"]),
                iter_elapsed=float(data["iter_elapsed"]),
            )
            continue

    return ParsedImitationLog(
        iteration_starts=sorted(iteration_starts.values(), key=lambda row: row.iteration),
        demo_done=sorted(demo_done.values(), key=lambda row: row.iteration),
        gate_splits=sorted(gate_splits.values(), key=lambda row: row.iteration),
        train_starts=sorted(train_starts.values(), key=lambda row: row.iteration),
        train_steps=sorted(train_steps.values(), key=lambda row: (row.iteration, row.step)),
        val_metrics=sorted(val_metrics.values(), key=lambda row: row.iteration),
        summaries=sorted(summaries.values(), key=lambda row: row.iteration),
    )


def parse_imitation_logs(paths: Sequence[str | Path]) -> ParsedImitationLog:
    lines: List[str] = []
    for path in paths:
        lines.extend(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
    return parse_imitation_log_lines(lines)


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


def _configure_matplotlib_env(output_dir: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "pokeenv_imitation_matplotlib"
    mpl_dir = cache_root / "mplconfig"
    cache_dir = cache_root / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", tempfile.gettempdir())


def _plot_overview(parsed: ParsedImitationLog, output_dir: Path, *, title: Optional[str]) -> Path:
    _configure_matplotlib_env(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not parsed.summaries:
        raise ValueError("No imitation summary lines were found in the provided logs.")

    summaries = parsed.summaries
    starts_by_iter = {row.iteration: row for row in parsed.iteration_starts}
    gate_split_by_iter = {row.iteration: row for row in parsed.gate_splits}
    demo_done_by_iter = {row.iteration: row for row in parsed.demo_done}
    val_by_iter = {row.iteration: row for row in parsed.val_metrics}
    iterations = [row.iteration for row in summaries]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(iterations, [row.train_accuracy for row in summaries], marker="o", label="train accuracy")
    ax.plot(iterations, [row.val_accuracy for row in summaries], marker="o", label="val accuracy")
    gate_train = [starts_by_iter[row.iteration].gate_train for row in summaries if row.iteration in starts_by_iter]
    gate_val = [starts_by_iter[row.iteration].gate_val for row in summaries if row.iteration in starts_by_iter]
    if gate_train:
        ax.axhline(gate_train[0], color="tab:green", linestyle="--", linewidth=1.5, label="train gate")
    if gate_val:
        ax.axhline(gate_val[0], color="tab:red", linestyle="--", linewidth=1.5, label="val gate")
    promoted_iters = [row.iteration for row in summaries if row.promoted]
    promoted_vals = [row.val_accuracy for row in summaries if row.promoted]
    if promoted_iters:
        ax.scatter(promoted_iters, promoted_vals, marker="*", s=120, label="promoted")
    ax.set_title("BC Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(iterations, [demo_done_by_iter[row.iteration].steps for row in summaries], marker="o", label="demo steps")
    ax.plot(iterations, [row.replay for row in summaries], marker="o", label="replay size")
    ax.set_title("Replay Growth")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(iterations, [row.loss for row in summaries], marker="o", label="train loss")
    ax.plot(iterations, [val_by_iter[row.iteration].val_loss for row in summaries if row.iteration in val_by_iter], marker="o", label="val loss")
    ax.set_title("Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(iterations, [row.entropy for row in summaries], marker="o", label="train entropy")
    ax.plot(iterations, [val_by_iter[row.iteration].val_entropy for row in summaries if row.iteration in val_by_iter], marker="o", label="val entropy")
    ax.set_title("Entropy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(iterations, [starts_by_iter[row.iteration].lr for row in summaries], marker="x", label="lr")
    ax.plot(iterations, [row.iter_elapsed for row in summaries], marker="o", label="iter elapsed (s)")
    ax.set_title("Learning Rate and Iteration Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(iterations, [demo_done_by_iter[row.iteration].skipped for row in summaries], marker="o", label="skipped labels")
    ax.plot(iterations, [gate_split_by_iter[row.iteration].val_samples for row in summaries if row.iteration in gate_split_by_iter], marker="o", label="val samples")
    ax.set_title("Skipped Labels and Validation Split")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path = output_dir / "imitation_overview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_train_steps(parsed: ParsedImitationLog, output_dir: Path, *, title: Optional[str]) -> Optional[Path]:
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

    axes[0].plot(x_values, [row.acc for row in steps], marker="o", linewidth=1.5)
    axes[0].set_title("Train-Step Accuracy")
    axes[0].set_xlabel("Iteration.progress")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_values, [row.loss for row in steps], marker="o", linewidth=1.5)
    axes[1].set_title("Train-Step Loss")
    axes[1].set_xlabel("Iteration.progress")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_values, [row.entropy for row in steps], marker="o", linewidth=1.5)
    axes[2].set_title("Train-Step Entropy")
    axes[2].set_xlabel("Iteration.progress")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(alpha=0.3)

    output_path = output_dir / "imitation_train_step_detail.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _build_report(parsed: ParsedImitationLog) -> List[str]:
    lines: List[str] = []
    lines.append(f"iterations parsed: {len(parsed.summaries)}")
    lines.append(f"train-step points parsed: {len(parsed.train_steps)}")
    if parsed.summaries:
        first = parsed.summaries[0]
        last = parsed.summaries[-1]
        lines.append(f"iteration span: {first.iteration} -> {last.iteration}")
        lines.append(
            f"latest iteration: {last.iteration} train_accuracy={last.train_accuracy:.3f} "
            f"val_accuracy={last.val_accuracy:.3f} best_bc_metric={last.best_bc_metric:.3f}"
        )
        best = max(parsed.summaries, key=lambda row: row.val_accuracy)
        lines.append(
            f"best validation iteration: {best.iteration} val_accuracy={best.val_accuracy:.3f} "
            f"train_accuracy={best.train_accuracy:.3f} promoted={best.promoted}"
        )
        tail = parsed.summaries[-5:]
        lines.append(
            f"last-5 averages: train_accuracy={mean(row.train_accuracy for row in tail):.3f} "
            f"val_accuracy={mean(row.val_accuracy for row in tail):.3f} "
            f"loss={mean(row.loss for row in tail):.3f}"
        )
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse BC/imitation training logs and render summary graphs.")
    parser.add_argument(
        "log_paths",
        nargs="*",
        help="One or more BC training log files to parse. Defaults to ./output.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated CSVs and PNGs. Defaults to <first_log_parent>/<first_log_stem>_bc_plots.",
    )
    parser.add_argument("--title", default=None, help="Optional title to place on the generated plots.")
    return parser


def _default_log_paths() -> List[str]:
    return ["output.txt"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    log_paths = list(args.log_paths) if args.log_paths else _default_log_paths()
    parsed = parse_imitation_logs(log_paths)

    first_log = Path(log_paths[0])
    output_dir = Path(args.output_dir) if args.output_dir else first_log.parent / f"{first_log.stem}_bc_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "iteration_summary.csv", parsed.summaries)
    _write_csv(output_dir / "train_steps.csv", parsed.train_steps)
    _write_csv(output_dir / "iteration_start.csv", parsed.iteration_starts)
    _write_csv(output_dir / "train_start.csv", parsed.train_starts)
    _write_csv(output_dir / "demo_done.csv", parsed.demo_done)
    _write_csv(output_dir / "gate_split.csv", parsed.gate_splits)
    _write_csv(output_dir / "val_metrics.csv", parsed.val_metrics)

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
