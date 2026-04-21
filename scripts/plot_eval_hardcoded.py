"""Hardcoded BC-style eval win-rate plot from x/y vectors with constant CI."""

from __future__ import annotations

from pathlib import Path
import tempfile
import os


# Edit these two vectors directly.
X_VALUES = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
Y_VALUES = [0.59, 0.51, 0.49, 0.56, 0.50, 0.51, 0.48, 0.57, 0.47, 0.52, 0.53, 0.51, 0.51, 0.615]

# Constant symmetric CI half-width.
CI_HALF_WIDTH = 0.05
PROMOTE_THRESHOLD = 0.55

OUTPUT_PATH = Path("runs/selfplay_350/eval_win_rate_hardcoded.png")
TITLE = "Eval Win Rate"


def _configure_matplotlib_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "pokeenv_hardcoded_eval_plot"
    mpl_dir = cache_root / "mplconfig"
    cache_dir = cache_root / "cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", tempfile.gettempdir())


def main() -> int:
    if len(X_VALUES) != len(Y_VALUES):
        raise SystemExit(f"X/Y vector length mismatch: len(X_VALUES)={len(X_VALUES)} len(Y_VALUES)={len(Y_VALUES)}")
    if not X_VALUES:
        raise SystemExit("X_VALUES is empty.")

    _configure_matplotlib_env()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = [int(x) for x in X_VALUES]
    eval_wins = [float(y) for y in Y_VALUES]

    lower = [float(CI_HALF_WIDTH) for _ in eval_wins]
    upper = [float(CI_HALF_WIDTH) for _ in eval_wins]

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.8), constrained_layout=True)
    ax.errorbar(
        updates,
        eval_wins,
        yerr=[lower, upper],
        fmt="o-",
        color="tab:green",
        ecolor="tab:green",
        linewidth=1.8,
        markersize=6.5,
        label="eval",
    )
    star_x = [x for x, y in zip(updates, eval_wins) if float(y) > float(PROMOTE_THRESHOLD)]
    star_y = [y for y in eval_wins if float(y) > float(PROMOTE_THRESHOLD)]
    if star_x:
        ax.scatter(
            star_x,
            star_y,
            marker="*",
            s=150,
            color="tab:orange",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
            label="eval > 55%",
        )
    ax.axhline(float(PROMOTE_THRESHOLD), color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8, label="threshold 55%")
    ax.set_title(TITLE)
    ax.set_xlabel("Update")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
