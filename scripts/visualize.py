from __future__ import annotations

import sys
from pathlib import Path

from torchinfo import summary
from torchview import draw_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import TransformerConfig, TransformerPolicyValueNet
from src.vocab import build_default_vocab


def main() -> None:
    vocab = build_default_vocab(gen=9, battle_format="gen9randombattle")
    meta = vocab.schema_meta()

    model = TransformerPolicyValueNet(
        TransformerConfig(
            obs_dim=int(meta["obs_dim"]),
            obs_meta=meta,
            d_model=1024,
            nhead=8,
            num_layers=2,
            ff_dim=4096,
            dropout=0.0,
        )
    )

    summary(model, input_size=(1, int(meta["obs_dim"])))
    graph = draw_graph(model, input_size=(1, int(meta["obs_dim"])), expand_nested=True)
    output_base = REPO_ROOT / "model_architecture"
    try:
        graph.visual_graph.render(str(output_base), format="png")
        print(f"Wrote architecture image: {output_base}.png")
    except Exception as exc:
        dot_path = output_base.with_suffix(".dot")
        dot_path.write_text(graph.visual_graph.source, encoding="utf-8")
        print(f"Could not render PNG ({exc}).")
        print(f"Wrote DOT graph instead: {dot_path}")
        print("Install Graphviz binary and rerun: brew install graphviz")


if __name__ == "__main__":
    main()
