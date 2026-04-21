"""Parse resumed ps-ppo-style BC logs and render a single HTML dashboard."""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from math import isnan
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FLOAT_RE = r"(?:nan|[-+]?inf|[-+0-9.eE]+)"

UPDATE_START_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+start\s+"
    rf"demo_battles=(?P<demo_battles>\d+)\s+"
    rf"steps_per_update=(?P<steps_per_update>\d+)\s+"
    rf"train_steps=(?P<train_steps>\d+)\s+"
    rf"train_epochs=(?P<train_epochs>\d+)\s+"
    rf"max_concurrent_battles=(?P<max_concurrent_battles>\d+)\s+"
    rf"servers=(?P<servers>\d+)\s+workers=(?P<workers>\d+)\s+"
    rf"total_env_steps=(?P<total_env_steps>\d+)\s+"
    rf"lr=(?P<lr>{FLOAT_RE})\s+"
    rf"best_eval_win_rate=(?P<best_eval_win_rate>{FLOAT_RE})\s+"
    rf"label_smoothing=(?P<label_smoothing>{FLOAT_RE})$"
)

DEMO_DONE_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+demo done battles=(?P<battles>\d+)\s+"
    rf"steps=(?P<steps>\d+)\s+replay=(?P<replay>\d+)\s+skipped=(?P<skipped>\d+)\s+"
    rf"elapsed=(?P<elapsed>{FLOAT_RE})s$"
)

TRAIN_START_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+train start steps=(?P<steps>\d+)\s+"
    rf"requested_steps=(?P<requested_steps>\S+)\s+batch=(?P<batch>\d+)\s+"
    rf"replay=(?P<replay>\d+)\s+epochs=(?P<epochs>\d+)\s+grad_accum=(?P<grad_accum>\d+)\s+"
    rf"effective_batch=(?P<effective_batch>\d+)$"
)

TRAIN_STEP_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+train_step=(?P<step>\d+)/(?P<step_total>\d+)\s+"
    rf"loss=(?P<loss>{FLOAT_RE})\s+policy=(?P<policy>{FLOAT_RE})\s+"
    rf"value=(?P<value>{FLOAT_RE})\s+entropy=(?P<entropy>{FLOAT_RE})\s+acc=(?P<acc>{FLOAT_RE})$"
)

TRAIN_DONE_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+train done elapsed=(?P<elapsed>{FLOAT_RE})s$"
)

EVAL_DONE_RE = re.compile(
    rf"^\[upd\s+(?P<update>\d+)\]\s+eval done elapsed=(?P<elapsed>{FLOAT_RE})s\s+"
    rf"win_rate=(?P<win_rate>{FLOAT_RE})\s+score_rate=(?P<score_rate>{FLOAT_RE})\s+"
    rf"ci95=\[(?P<ci_low>{FLOAT_RE}),(?P<ci_high>{FLOAT_RE})\]$"
)

SUMMARY_RE = re.compile(
    rf"^upd=(?P<update>\d+)\s+replay=(?P<replay>\d+)\s+demo_steps=(?P<demo_steps>\d+)\s+"
    rf"skipped=(?P<skipped>\d+)\s+loss=(?P<loss>{FLOAT_RE})\s+"
    rf"policy_loss=(?P<policy_loss>{FLOAT_RE})\s+value_loss=(?P<value_loss>{FLOAT_RE})\s+"
    rf"entropy=(?P<entropy>{FLOAT_RE})\s+train_accuracy=(?P<train_accuracy>{FLOAT_RE})\s+"
    rf"eval_win_rate=(?P<eval_win_rate>{FLOAT_RE})\s+best_eval_win_rate=(?P<best_eval_win_rate>{FLOAT_RE})\s+"
    rf"promoted=(?P<promoted>True|False)\s+upd_elapsed=(?P<upd_elapsed>{FLOAT_RE})s$"
)


@dataclass(frozen=True)
class UpdateStart:
    update: int
    demo_battles: int
    steps_per_update: int
    train_steps: int
    train_epochs: int
    max_concurrent_battles: int
    servers: int
    workers: int
    total_env_steps: int
    lr: float
    best_eval_win_rate: float
    label_smoothing: float


@dataclass(frozen=True)
class DemoDone:
    update: int
    battles: int
    steps: int
    replay: int
    skipped: int
    elapsed: float


@dataclass(frozen=True)
class TrainStart:
    update: int
    steps: int
    requested_steps: str
    batch: int
    replay: int
    epochs: int
    grad_accum: int
    effective_batch: int


@dataclass(frozen=True)
class TrainStep:
    update: int
    step: int
    step_total: int
    loss: float
    policy: float
    value: float
    entropy: float
    acc: float


@dataclass(frozen=True)
class TrainDone:
    update: int
    elapsed: float


@dataclass(frozen=True)
class EvalDone:
    update: int
    elapsed: float
    win_rate: float
    score_rate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class Summary:
    update: int
    replay: int
    demo_steps: int
    skipped: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    train_accuracy: float
    eval_win_rate: float
    best_eval_win_rate: float
    promoted: bool
    upd_elapsed: float


@dataclass(frozen=True)
class ParsedBcLog:
    update_starts: List[UpdateStart]
    demo_done: List[DemoDone]
    train_starts: List[TrainStart]
    train_steps: List[TrainStep]
    train_done: List[TrainDone]
    eval_done: List[EvalDone]
    summaries: List[Summary]


@dataclass(frozen=True)
class SegmentInfo:
    label: str
    path: str
    first_update: int
    last_update: int
    updates: int


@dataclass(frozen=True)
class RunSummary:
    updates: int
    first_update: int
    last_update: int
    final_train_accuracy: float
    best_train_accuracy: float
    final_eval_win_rate: Optional[float]
    best_eval_win_rate: Optional[float]
    promotions: int
    avg_demo_steps_last5: Optional[float]
    avg_update_elapsed_last5: Optional[float]
    avg_demo_elapsed_last5: Optional[float]
    total_skipped: int
    latest_lr: Optional[float]


def _to_float(text: str) -> float:
    return float(text)


def _to_bool(text: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"Unexpected bool text: {text}")


def parse_bc_log_lines(lines: Iterable[str]) -> ParsedBcLog:
    update_starts: Dict[int, UpdateStart] = {}
    demo_done: Dict[int, DemoDone] = {}
    train_starts: Dict[int, TrainStart] = {}
    train_steps: Dict[Tuple[int, int, int], TrainStep] = {}
    train_done: Dict[int, TrainDone] = {}
    eval_done: Dict[int, EvalDone] = {}
    summaries: Dict[int, Summary] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = UPDATE_START_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            update_starts[update] = UpdateStart(
                update=update,
                demo_battles=int(data["demo_battles"]),
                steps_per_update=int(data["steps_per_update"]),
                train_steps=int(data["train_steps"]),
                train_epochs=int(data["train_epochs"]),
                max_concurrent_battles=int(data["max_concurrent_battles"]),
                servers=int(data["servers"]),
                workers=int(data["workers"]),
                total_env_steps=int(data["total_env_steps"]),
                lr=_to_float(data["lr"]),
                best_eval_win_rate=_to_float(data["best_eval_win_rate"]),
                label_smoothing=_to_float(data["label_smoothing"]),
            )
            continue

        match = DEMO_DONE_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            demo_done[update] = DemoDone(
                update=update,
                battles=int(data["battles"]),
                steps=int(data["steps"]),
                replay=int(data["replay"]),
                skipped=int(data["skipped"]),
                elapsed=_to_float(data["elapsed"]),
            )
            continue

        match = TRAIN_START_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            train_starts[update] = TrainStart(
                update=update,
                steps=int(data["steps"]),
                requested_steps=data["requested_steps"],
                batch=int(data["batch"]),
                replay=int(data["replay"]),
                epochs=int(data["epochs"]),
                grad_accum=int(data["grad_accum"]),
                effective_batch=int(data["effective_batch"]),
            )
            continue

        match = TRAIN_STEP_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            step = int(data["step"])
            step_total = int(data["step_total"])
            train_steps[(update, step, step_total)] = TrainStep(
                update=update,
                step=step,
                step_total=step_total,
                loss=_to_float(data["loss"]),
                policy=_to_float(data["policy"]),
                value=_to_float(data["value"]),
                entropy=_to_float(data["entropy"]),
                acc=_to_float(data["acc"]),
            )
            continue

        match = TRAIN_DONE_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            train_done[update] = TrainDone(update=update, elapsed=_to_float(data["elapsed"]))
            continue

        match = EVAL_DONE_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            eval_done[update] = EvalDone(
                update=update,
                elapsed=_to_float(data["elapsed"]),
                win_rate=_to_float(data["win_rate"]),
                score_rate=_to_float(data["score_rate"]),
                ci_low=_to_float(data["ci_low"]),
                ci_high=_to_float(data["ci_high"]),
            )
            continue

        match = SUMMARY_RE.match(line)
        if match:
            data = match.groupdict()
            update = int(data["update"])
            summaries[update] = Summary(
                update=update,
                replay=int(data["replay"]),
                demo_steps=int(data["demo_steps"]),
                skipped=int(data["skipped"]),
                loss=_to_float(data["loss"]),
                policy_loss=_to_float(data["policy_loss"]),
                value_loss=_to_float(data["value_loss"]),
                entropy=_to_float(data["entropy"]),
                train_accuracy=_to_float(data["train_accuracy"]),
                eval_win_rate=_to_float(data["eval_win_rate"]),
                best_eval_win_rate=_to_float(data["best_eval_win_rate"]),
                promoted=_to_bool(data["promoted"]),
                upd_elapsed=_to_float(data["upd_elapsed"]),
            )
            continue

    return ParsedBcLog(
        update_starts=sorted(update_starts.values(), key=lambda row: row.update),
        demo_done=sorted(demo_done.values(), key=lambda row: row.update),
        train_starts=sorted(train_starts.values(), key=lambda row: row.update),
        train_steps=sorted(train_steps.values(), key=lambda row: (row.update, row.step, row.step_total)),
        train_done=sorted(train_done.values(), key=lambda row: row.update),
        eval_done=sorted(eval_done.values(), key=lambda row: row.update),
        summaries=sorted(summaries.values(), key=lambda row: row.update),
    )


def parse_bc_logs(paths: Sequence[str | Path]) -> ParsedBcLog:
    lines: List[str] = []
    for path in paths:
        lines.extend(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
    return parse_bc_log_lines(lines)


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


def _configure_matplotlib_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "pokeenv_bc_dashboard_matplotlib"
    mpl_dir = cache_root / "mplconfig"
    cache_dir = cache_root / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", tempfile.gettempdir())


def summarize_run(parsed: ParsedBcLog) -> RunSummary:
    if not parsed.summaries:
        raise ValueError("No update summary lines found in the provided logs.")

    summaries = parsed.summaries
    eval_values = [row.win_rate for row in parsed.eval_done]
    last5_summaries = summaries[-5:]
    demo_by_update = {row.update: row for row in parsed.demo_done}
    starts_by_update = {row.update: row for row in parsed.update_starts}

    return RunSummary(
        updates=len(summaries),
        first_update=summaries[0].update,
        last_update=summaries[-1].update,
        final_train_accuracy=summaries[-1].train_accuracy,
        best_train_accuracy=max(row.train_accuracy for row in summaries),
        final_eval_win_rate=(eval_values[-1] if eval_values else None),
        best_eval_win_rate=(max(eval_values) if eval_values else None),
        promotions=sum(1 for row in summaries if row.promoted),
        avg_demo_steps_last5=mean(float(row.demo_steps) for row in last5_summaries),
        avg_update_elapsed_last5=mean(float(row.upd_elapsed) for row in last5_summaries),
        avg_demo_elapsed_last5=(
            mean(demo_by_update[row.update].elapsed for row in last5_summaries if row.update in demo_by_update)
            if any(row.update in demo_by_update for row in last5_summaries)
            else None
        ),
        total_skipped=sum(int(row.skipped) for row in summaries),
        latest_lr=starts_by_update.get(summaries[-1].update).lr if summaries[-1].update in starts_by_update else None,
    )


def _collect_segments(paths: Sequence[Path], labels: Sequence[str]) -> List[SegmentInfo]:
    segments: List[SegmentInfo] = []
    for path, label in zip(paths, labels):
        parsed = parse_bc_logs([path])
        if not parsed.summaries:
            continue
        segments.append(
            SegmentInfo(
                label=label,
                path=str(path),
                first_update=parsed.summaries[0].update,
                last_update=parsed.summaries[-1].update,
                updates=len(parsed.summaries),
            )
        )
    return segments


def _add_segment_markers(ax, segments: Sequence[SegmentInfo]) -> None:
    for idx, segment in enumerate(segments):
        if idx == 0:
            continue
        ax.axvline(segment.first_update, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.8)


def _plot_overview(parsed: ParsedBcLog, output_dir: Path, *, title: Optional[str], segments: Sequence[SegmentInfo]) -> Path:
    _configure_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not parsed.summaries:
        raise ValueError("No BC summary lines were found in the provided logs.")

    summaries = parsed.summaries
    updates = [row.update for row in summaries]
    demo_by_update = {row.update: row for row in parsed.demo_done}

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)

    axes[0, 0].plot(updates, [row.train_accuracy for row in summaries], marker="o", color="tab:blue")
    axes[0, 0].set_title("Train Accuracy")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].grid(alpha=0.3)
    _add_segment_markers(axes[0, 0], segments)

    axes[0, 1].plot(updates, [row.best_eval_win_rate for row in summaries], linestyle="--", color="tab:orange", label="best eval")
    if parsed.eval_done:
        eval_updates = [row.update for row in parsed.eval_done]
        eval_wins = [row.win_rate for row in parsed.eval_done]
        lower = [row.win_rate - row.ci_low for row in parsed.eval_done]
        upper = [row.ci_high - row.win_rate for row in parsed.eval_done]
        axes[0, 1].errorbar(eval_updates, eval_wins, yerr=[lower, upper], fmt="o", color="tab:green", label="eval")
    axes[0, 1].set_title("Eval Win Rate")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Win Rate")
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    _add_segment_markers(axes[0, 1], segments)

    axes[1, 0].plot(updates, [row.loss for row in summaries], marker="o", color="tab:purple")
    axes[1, 0].set_title("Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(alpha=0.3)
    _add_segment_markers(axes[1, 0], segments)

    axes[1, 1].plot(updates, [row.entropy for row in summaries], marker="o", color="tab:brown")
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].grid(alpha=0.3)
    _add_segment_markers(axes[1, 1], segments)

    axes[2, 0].plot(updates, [row.upd_elapsed for row in summaries], marker="o", color="tab:red", label="update")
    demo_updates = [row.update for row in summaries if row.update in demo_by_update]
    demo_elapsed = [demo_by_update[row.update].elapsed for row in summaries if row.update in demo_by_update]
    if demo_updates:
        axes[2, 0].plot(demo_updates, demo_elapsed, linestyle="--", marker="x", color="tab:gray", label="demo")
    axes[2, 0].set_title("Elapsed Time")
    axes[2, 0].set_xlabel("Update")
    axes[2, 0].set_ylabel("Seconds")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend()
    _add_segment_markers(axes[2, 0], segments)

    axes[2, 1].plot(updates, [row.demo_steps for row in summaries], marker="o", color="tab:cyan", label="demo steps")
    axes[2, 1].plot(updates, [row.skipped for row in summaries], linestyle="--", marker="x", color="tab:olive", label="skipped")
    axes[2, 1].set_title("Demo Steps and Skipped Labels")
    axes[2, 1].set_xlabel("Update")
    axes[2, 1].set_ylabel("Count")
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend()
    _add_segment_markers(axes[2, 1], segments)

    output_path = output_dir / "overview.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_train_steps(parsed: ParsedBcLog, output_dir: Path, *, title: Optional[str], segments: Sequence[SegmentInfo]) -> Optional[Path]:
    if not parsed.train_steps:
        return None

    _configure_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [row.update + (row.step / max(1, row.step_total)) for row in parsed.train_steps]
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), constrained_layout=True)
    if title:
        fig.suptitle(f"{title} — Train-Step Detail", fontsize=16)

    axes[0].plot(x_values, [row.acc for row in parsed.train_steps], marker="o", linewidth=1.2)
    axes[0].set_title("Train-Step Accuracy")
    axes[0].set_xlabel("Update.progress")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    _add_segment_markers(axes[0], segments)

    axes[1].plot(x_values, [row.loss for row in parsed.train_steps], marker="o", linewidth=1.2)
    axes[1].set_title("Train-Step Loss")
    axes[1].set_xlabel("Update.progress")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    _add_segment_markers(axes[1], segments)

    axes[2].plot(x_values, [row.entropy for row in parsed.train_steps], marker="o", linewidth=1.2)
    axes[2].set_title("Train-Step Entropy")
    axes[2].set_xlabel("Update.progress")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(alpha=0.3)
    _add_segment_markers(axes[2], segments)

    output_path = output_dir / "train_steps.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _fmt_optional(value: Optional[float], *, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _summary_table(summary: RunSummary) -> str:
    rows = [
        ("Updates", str(summary.updates)),
        ("Update Range", f"{summary.first_update} → {summary.last_update}"),
        ("Final Train Accuracy", f"{summary.final_train_accuracy:.3f}"),
        ("Best Train Accuracy", f"{summary.best_train_accuracy:.3f}"),
        ("Final Eval Win Rate", _fmt_optional(summary.final_eval_win_rate)),
        ("Best Eval Win Rate", _fmt_optional(summary.best_eval_win_rate)),
        ("Promotions", str(summary.promotions)),
        ("Avg Demo Steps (last 5)", _fmt_optional(summary.avg_demo_steps_last5, digits=1)),
        ("Avg Update Time (s)", _fmt_optional(summary.avg_update_elapsed_last5, digits=1)),
        ("Avg Demo Time (s)", _fmt_optional(summary.avg_demo_elapsed_last5, digits=1)),
        ("Total Skipped", str(summary.total_skipped)),
        ("Latest LR", _fmt_optional(summary.latest_lr, digits=6)),
    ]
    body = "".join(f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>" for name, value in rows)
    return f"<table class=\"kv\">{body}</table>"


def _segment_table(segments: Sequence[SegmentInfo]) -> str:
    header = "<tr><th>Segment</th><th>File</th><th>Update Range</th><th>Updates</th></tr>"
    body = "".join(
        "<tr>"
        f"<td>{html.escape(segment.label)}</td>"
        f"<td><code>{html.escape(segment.path)}</code></td>"
        f"<td>{segment.first_update} → {segment.last_update}</td>"
        f"<td>{segment.updates}</td>"
        "</tr>"
        for segment in segments
    )
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


def _build_highlights(summary: RunSummary, segments: Sequence[SegmentInfo]) -> List[str]:
    highlights = [
        f"Parsed {summary.updates} total updates across {len(segments)} log segment(s).",
        f"Final train accuracy is {summary.final_train_accuracy:.3f}.",
        f"Best eval win rate is {_fmt_optional(summary.best_eval_win_rate)}.",
        f"Recent updates average {_fmt_optional(summary.avg_update_elapsed_last5, digits=1)}s.",
    ]
    if len(segments) > 1:
        resumes = ", ".join(f"{segment.label} at update {segment.first_update}" for segment in segments[1:])
        highlights.append(f"Resume markers: {resumes}.")
    return highlights


def _render_dashboard_html(
    *,
    title: str,
    summary: RunSummary,
    segments: Sequence[SegmentInfo],
    overview_image: Path,
    train_step_image: Optional[Path],
) -> str:
    highlights = "".join(f"<li>{html.escape(item)}</li>" for item in _build_highlights(summary, segments))
    plots_html = f'<section><h2>Overview</h2><img src="{html.escape(overview_image.name)}" alt="overview plot" /></section>'
    if train_step_image is not None:
        plots_html += f'<section><h2>Train-Step Detail</h2><img src="{html.escape(train_step_image.name)}" alt="train-step plot" /></section>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --card: #fffdf8;
      --ink: #1f1d19;
      --muted: #6f685e;
      --line: #ddd2c3;
      --accent: #8b5c2e;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #f3ede1 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 10px 28px rgba(63, 48, 28, 0.05);
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .subtle {{
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .kv th {{
      width: 42%;
      color: var(--ink);
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    code {{
      background: #efe6d7;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{html.escape(title)}</h1>
      <div class="subtle">Merged from resumed BC log files in the order provided.</div>
      <ul>{highlights}</ul>
    </section>
    <section class="grid">
      <div>
        <h2>Run Summary</h2>
        {_summary_table(summary)}
      </div>
      <div>
        <h2>Log Segments</h2>
        {_segment_table(segments)}
      </div>
    </section>
    {plots_html}
  </main>
</body>
</html>
"""


def _default_output_dir(paths: Sequence[Path]) -> Path:
    parent = paths[0].parent
    stems = "_to_".join(path.stem for path in paths)
    return parent / f"{stems}_dashboard"


def _normalize_labels(paths: Sequence[Path], labels_arg: Optional[str]) -> List[str]:
    if labels_arg is None or not labels_arg.strip():
        return [path.stem for path in paths]
    labels = [label.strip() for label in labels_arg.split(",") if label.strip()]
    if len(labels) != len(paths):
        raise SystemExit("--labels must contain exactly one comma-separated label per log path")
    return labels


def build_dashboard(
    *,
    log_paths: Sequence[str | Path],
    output_dir: Path,
    labels: Sequence[str],
    title: str,
) -> Path:
    paths = [Path(path) for path in log_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_bc_logs(paths)
    summary = summarize_run(parsed)
    segments = _collect_segments(paths, labels)

    _write_csv(output_dir / "update_start.csv", parsed.update_starts)
    _write_csv(output_dir / "demo_done.csv", parsed.demo_done)
    _write_csv(output_dir / "train_start.csv", parsed.train_starts)
    _write_csv(output_dir / "train_steps.csv", parsed.train_steps)
    _write_csv(output_dir / "train_done.csv", parsed.train_done)
    _write_csv(output_dir / "eval_done.csv", parsed.eval_done)
    _write_csv(output_dir / "summary.csv", parsed.summaries)
    _write_csv(output_dir / "segments.csv", segments)
    _write_csv(output_dir / "run_summary.csv", [summary])

    overview_image = _plot_overview(parsed, output_dir, title=title, segments=segments)
    train_step_image = _plot_train_steps(parsed, output_dir, title=title, segments=segments)

    dashboard_html = _render_dashboard_html(
        title=title,
        summary=summary,
        segments=segments,
        overview_image=overview_image,
        train_step_image=train_step_image,
    )
    dashboard_path = output_dir / "index.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    return dashboard_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse resumed ps-ppo-style BC logs and render one dashboard.")
    parser.add_argument("log_paths", nargs="+", help="Finished BC log files in chronological order.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels for each log segment. Defaults to file stems.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated HTML dashboard, plots, and CSVs.",
    )
    parser.add_argument(
        "--title",
        default="BC Resume Dashboard",
        help="Title shown in the HTML dashboard and plots.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = [Path(path) for path in args.log_paths]
    labels = _normalize_labels(paths, args.labels)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(paths)

    dashboard_path = build_dashboard(
        log_paths=paths,
        output_dir=output_dir,
        labels=labels,
        title=str(args.title),
    )
    print(f"wrote dashboard: {dashboard_path}")
    print(f"open: {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
