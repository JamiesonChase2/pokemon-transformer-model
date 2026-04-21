"""Render a PNG diagram of the current observation-to-output model flow."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.vocab import build_default_vocab


@dataclass(frozen=True)
class BoxSpec:
    x: float
    y: float
    w: float
    h: float
    label: str
    fontsize: int = 11
    facecolor: str = "#0f172a"
    edgecolor: str = "#334155"
    textcolor: str = "#e2e8f0"


def _add_box(ax, spec: BoxSpec) -> tuple[float, float, float, float]:
    rect = FancyBboxPatch(
        (spec.x, spec.y),
        spec.w,
        spec.h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=spec.edgecolor,
        facecolor=spec.facecolor,
    )
    ax.add_patch(rect)
    ax.text(
        spec.x + spec.w / 2,
        spec.y + spec.h / 2,
        spec.label,
        ha="center",
        va="center",
        color=spec.textcolor,
        fontsize=spec.fontsize,
        wrap=True,
    )
    return (spec.x, spec.y, spec.w, spec.h)


def _point(bounds: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, y, w, h = bounds
    if side == "left":
        return (x, y + h / 2)
    if side == "right":
        return (x + w, y + h / 2)
    if side == "top":
        return (x + w / 2, y + h)
    if side == "bottom":
        return (x + w / 2, y)
    raise ValueError(f"Unknown side: {side}")


def _add_arrow(ax, start, end, *, color="#64748b", connectionstyle="arc3,rad=0.0") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.6,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def render(output_path: Path) -> None:
    vocab = build_default_vocab(battle_format="gen9randombattle")
    meta = vocab.schema_meta()
    obs_dim = int(meta["obs_dim"])
    body_dim = int(meta["dim_pokemon_body"])
    move_scalars_dim = int(meta["dim_move_scalars"])
    global_dim = int(meta["dim_global_scalars"])
    transition_scalar_dim = int(meta["dim_transition_scalars"])
    ability_slots = int(meta["n_ability_slots"])
    d_model = 512

    fig, ax = plt.subplots(figsize=(18, 10), dpi=180)
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    ax.set_xlim(0, 19.5)
    ax.set_ylim(0, 12)
    ax.axis("off")

    boxes = {}
    boxes["battle"] = _add_box(ax, BoxSpec(0.6, 5.0, 2.5, 1.3, "Battle state\n(gen9 randbats)", fontsize=13, facecolor="#111827"))
    boxes["assembly"] = _add_box(ax, BoxSpec(3.5, 4.45, 3.1, 2.4, "Observation assembly\n- curated randbats schema\n- ps-ppo-compatible feature layout\n- estimated base stats + visible battle state", fontsize=12, facecolor="#172554", edgecolor="#3b82f6"))
    boxes["flat"] = _add_box(ax, BoxSpec(7.2, 5.0, 2.7, 1.35, f"Flat observation\n{obs_dim} dims", fontsize=13, facecolor="#1d4ed8", edgecolor="#60a5fa"))

    boxes["segments"] = _add_box(
        ax,
        BoxSpec(
            6.7,
            8.0,
            3.7,
            2.7,
            f"Segments\npokemon_body: 12 x {body_dim}\npokemon_ids: 12 x 2\nability_ids: 12 x {ability_slots}\nmove_ids: 12 x 4\nmove_scalars: 12 x {move_scalars_dim // 12}\nglobal_scalars: {global_dim}\ntransition: 2 + {transition_scalar_dim}\naction_mask: 14",
            fontsize=10,
            facecolor="#111827",
        ),
    )

    boxes["unpack"] = _add_box(ax, BoxSpec(10.8, 5.0, 2.4, 1.35, "ObservationUnpacker", fontsize=13, facecolor="#111827"))
    boxes["move"] = _add_box(ax, BoxSpec(13.8, 8.2, 2.7, 1.2, "move_net\n(id emb + acc/pwr/pp + tactical one-hots)\n-> 4 x 128 move vecs", fontsize=11, facecolor="#0f766e", edgecolor="#2dd4bf"))
    boxes["ability"] = _add_box(ax, BoxSpec(13.8, 6.35, 2.7, 1.2, "ability_net\n(4 ability id slots)\n-> 1 x 128 ability vec", fontsize=11, facecolor="#0f766e", edgecolor="#2dd4bf"))
    boxes["pokemon"] = _add_box(ax, BoxSpec(13.6, 4.3, 3.1, 1.35, "pokemon_net\n(body + species/item emb\n+ ability vec + move vecs)\n-> 12 Pokémon tokens", fontsize=11, facecolor="#7c2d12", edgecolor="#fb923c"))
    boxes["field"] = _add_box(ax, BoxSpec(13.9, 2.35, 2.5, 1.2, "field_net\n(global + transition)\n-> 1 field token", fontsize=11, facecolor="#7c2d12", edgecolor="#fb923c"))
    boxes["tokens"] = _add_box(ax, BoxSpec(17.3, 4.35, 1.7, 1.25, f"15 tokens\nx {d_model}", fontsize=13, facecolor="#4c1d95", edgecolor="#a78bfa"))
    boxes["encoder"] = _add_box(ax, BoxSpec(17.0, 6.2, 2.2, 1.55, "Transformer encoder\n2 layers\n8 heads\nwidth 512", fontsize=12, facecolor="#581c87", edgecolor="#c084fc"))
    boxes["heads"] = _add_box(ax, BoxSpec(16.9, 8.55, 2.4, 1.65, "Post-encoder MHA readout\n-> policy head (14)\n-> value head (51 bins)", fontsize=11, facecolor="#1f2937", edgecolor="#94a3b8"))
    boxes["mask"] = _add_box(ax, BoxSpec(10.9, 10.2, 2.4, 1.0, "Legal action mask\n(14)", fontsize=11, facecolor="#111827"))
    boxes["output"] = _add_box(ax, BoxSpec(17.0, 0.6, 2.2, 1.1, "Action selection\nmove / tera-move /\nswitch", fontsize=11, facecolor="#065f46", edgecolor="#34d399"))

    _add_arrow(ax, _point(boxes["battle"], "right"), _point(boxes["assembly"], "left"))
    _add_arrow(ax, _point(boxes["assembly"], "right"), _point(boxes["flat"], "left"))
    _add_arrow(ax, _point(boxes["flat"], "top"), _point(boxes["segments"], "bottom"), connectionstyle="arc3,rad=0.0")
    _add_arrow(ax, _point(boxes["flat"], "right"), _point(boxes["unpack"], "left"))
    _add_arrow(ax, _point(boxes["unpack"], "right"), _point(boxes["move"], "left"), connectionstyle="arc3,rad=0.10")
    _add_arrow(ax, _point(boxes["unpack"], "right"), _point(boxes["ability"], "left"), connectionstyle="arc3,rad=0.04")
    _add_arrow(ax, _point(boxes["unpack"], "right"), _point(boxes["pokemon"], "left"))
    _add_arrow(ax, _point(boxes["unpack"], "right"), _point(boxes["field"], "left"), connectionstyle="arc3,rad=-0.08")
    _add_arrow(ax, _point(boxes["move"], "bottom"), _point(boxes["pokemon"], "top"))
    _add_arrow(ax, _point(boxes["ability"], "bottom"), _point(boxes["pokemon"], "top"), connectionstyle="arc3,rad=-0.1")
    _add_arrow(ax, _point(boxes["pokemon"], "right"), _point(boxes["tokens"], "left"))
    _add_arrow(ax, _point(boxes["field"], "right"), _point(boxes["tokens"], "left"), connectionstyle="arc3,rad=0.15")
    _add_arrow(ax, _point(boxes["tokens"], "top"), _point(boxes["encoder"], "bottom"))
    _add_arrow(ax, _point(boxes["encoder"], "top"), _point(boxes["heads"], "bottom"))
    _add_arrow(ax, _point(boxes["heads"], "bottom"), _point(boxes["output"], "top"), connectionstyle="arc3,rad=-0.15")
    _add_arrow(ax, _point(boxes["mask"], "right"), _point(boxes["heads"], "left"), connectionstyle="arc3,rad=-0.10")
    _add_arrow(ax, _point(boxes["flat"], "top"), _point(boxes["mask"], "bottom"), connectionstyle="arc3,rad=0.22")

    ax.text(
        0.6,
        11.35,
        "Observation -> Output Flow",
        color="#f8fafc",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.6,
        10.85,
        "Current fresh BC config: curated gen9 randbats schema, d_model=512, 15-token internal Transformer sequence",
        color="#cbd5e1",
        fontsize=12,
        ha="left",
        va="center",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a PNG diagram of the current observation-to-output flow.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("obs_to_output_flow.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()
    render(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
