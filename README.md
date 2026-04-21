# pokeenv_transformer

`pokeenv_transformer` is a Gen 9 Random Battles research/training project built on top of `poke-env`. It packages the full loop for a Pokemon Showdown bot: structured observation extraction, Transformer policy/value modeling, replay storage, behavior cloning, league-style self-play, evaluation against baselines, and log/benchmark tooling.

## What You Built

You built a local training stack for a partially observed battle bot rather than just a single model file. The main pieces are:

- A deterministic tokenizer that converts battle state into fixed-layout tensors with hidden-information masking.
- A Transformer policy/value network that predicts both a 14-action policy and a value target.
- A disk-backed replay pipeline so long self-play or imitation runs do not need to stay entirely in RAM.
- Two training paths:
  - behavior cloning from `SimpleHeuristicsPlayer`
  - league-style self-play where `current` plays `best`, trains on collected rollouts, then gets promoted if it clears an eval threshold
- Utility scripts for evaluation, plotting, dashboards, interactive play, and backend benchmarking.

## Observation Schema And Model

For the default `gen9randombattle` vocabulary, each battle state is serialized into a flat `5534`-dimensional observation vector. That vector is split into:

- `pokemon_body`: `12 x 286 = 3432` dims
- `pokemon_ids`: `12 x 2 = 24` dims
- `ability_ids`: `12 x 4 = 48` dims
- `move_ids`: `12 x 4 = 48` dims
- `move_scalars`: `12 x (4 x 40) = 1920` dims
- `global_scalars`: `36` dims
- `transition_move_ids`: `2` dims
- `transition_scalars`: `10` dims
- `action_mask`: `14` dims

The schema represents both teams as `12` Pokemon slots total, with up to `4` moves and `4` ability candidates per slot. It also includes global battle state, the last transition summary, and a legal-action mask over the fixed `14`-action policy space:

- `4` move actions
- `4` tera-move actions
- `6` switch actions

The Transformer does not attend over the flat vector directly. It first repacks the observation into a `15`-token internal sequence:

- `1` actor token
- `1` critic token
- `1` field/global token
- `12` Pokemon tokens

Each Pokemon token is built from learned embeddings plus scalar banks:

- species embedding
- item embedding
- embedded HP, level, stats, and weight buckets
- pooled ability representation
- pooled per-move representation
- raw boost, flag, type, effect, and status features

Default training runs use this model layout:

- `d_model=1024`
- `nhead=8`
- `num_layers=2`
- `ff_dim=4096`
- `dropout=0.0`
- policy head over `14` actions
- value head with `51` bins over `[-1.6, 1.6]`

At those defaults, the model has about `55.5M` trainable parameters for the curated Gen 9 Random Battles schema.

## Repo Layout

- `src/tokenizer.py`: Builds structured observations, preserves revealed information, and folds in Gen 9 random battles schema details.
- `src/model.py`: Defines the Transformer encoder plus policy/value heads.
- `src/policy.py`: Handles action masking and policy selection over the 14-action space.
- `src/replay.py`: Stores replay samples as disk-backed `.pt` shards.
- `src/train.py`: Implements PPO-style and imitation-learning update logic.
- `src/bot.py`: Adapts the model to `poke-env` players.
- `src/checkpoint.py`: Saves and restores checkpoints, vocab, and model config.
- `scripts/train_selfplay.py`: Main league-training entry point.
- `scripts/train_imitation.py`: Behavior-cloning trainer against `SimpleHeuristicsPlayer`.
- `scripts/eval_baseline_suite.py`: Evaluates checkpoints versus Random, MaxDamage, and SimpleHeuristics baselines.
- `scripts/plot_training_log.py`, `scripts/plot_imitation_log.py`, `scripts/dashboard_imitation_runs.py`: Parse logs into charts and dashboards.
- `scripts/benchmark_selfplay_backends.py`: Compares rollout throughput across self-play execution backends.
- `tests/`: Unit coverage for tokenizer shape rules, model forward passes, reward shaping, replay helpers, baseline parsing, and rollout utilities.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run A Local Showdown Server

From a local `pokemon-showdown` checkout:

```bash
node pokemon-showdown start --no-security --port 8000
```

Optional environment variables:

```bash
export SHOWDOWN_WS_URL=ws://localhost:8000/showdown/websocket
export SHOWDOWN_AUTH_URL=http://localhost:8000/action.php?
export SHOWDOWN_ROOT=/path/to/pokemon-showdown
```

`SHOWDOWN_ROOT` lets the vocab/tokenizer pull Showdown data such as Gen 9 item metadata.

## Common Workflows

Smoke-test a self-play run:

```bash
python -m scripts.train_selfplay \
  --iterations 1 \
  --selfplay-battles 2 \
  --train-steps 2 \
  --eval-battles 2 \
  --batch-size 8 \
  --checkpoint-dir checkpoints_smoke
```

Warm-start with behavior cloning:

```bash
python -m scripts.train_imitation \
  --iterations 1 \
  --demo-battles 4 \
  --train-steps 2 \
  --eval-battles 4 \
  --checkpoint-dir checkpoints_imitation_smoke
```

Evaluate a checkpoint against baseline bots:

```bash
python -m scripts.eval_baseline_suite \
  --checkpoint checkpoints_smoke/current.pt \
  --n-battles 20
```

Play with a trained checkpoint:

```bash
python -m scripts.play \
  --checkpoint checkpoints_smoke/current.pt \
  --n-battles 1 \
  --battle-format gen9randombattle
```

Plot training logs:

```bash
python -m scripts.plot_training_log path/to/output.txt
python -m scripts.plot_imitation_log path/to/output.txt
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix avoids unrelated third-party pytest plugin crashes that can happen in some global Python or Conda installs.

## GitLab Push Notes

This repository now ignores generated Python caches, replay shards, checkpoints, logs, and rendered plots so a first push stays focused on source code.

If you have already created a GitLab project, add the remote and push:

```bash
git remote add origin git@gitlab.com:<namespace>/pokeenv_transformer.git
git push -u origin main
```

If you prefer HTTPS:

```bash
git remote add origin https://gitlab.com/<namespace>/pokeenv_transformer.git
git push -u origin main
```
