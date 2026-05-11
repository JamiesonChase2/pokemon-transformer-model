# pokeenv_transformer

Train, evaluate, and run a Transformer-powered Pokemon Showdown bot for Gen 9 Random Battles.

`pokeenv_transformer` turns `poke-env` battle state into structured tensors, feeds them through a policy/value Transformer, and closes the loop with behavior cloning, self-play, checkpointing, evaluation, and analysis tooling. It is built for local research on partially observed battle decision-making, not just one-off inference.

## Features

- Transformer policy/value model with a fixed 14-action policy space.
- `poke-env` player integration for ladder, challenge, self-play, and baseline matches.
- Deterministic Gen 9 Random Battles tokenizer with hidden-information masking.
- Disk-backed replay shards for long-running self-play and imitation experiments.
- Behavior cloning from `SimpleHeuristicsPlayer`.
- PPO-style league self-play with current-vs-best promotion.
- Evaluation against Random, MaxDamage, and SimpleHeuristics baselines.
- Utility scripts for plotting logs, dashboards, rollout benchmarking, and token debugging.
- Unit tests for tokenizer shapes, model forward passes, action masks, reward shaping, and training helpers.

## Project Structure

```text
src/
  bot.py          # poke-env player adapters and trajectory collection
  checkpoint.py   # checkpoint save/load and model reconstruction
  model.py        # Transformer encoder and policy/value heads
  policy.py       # action masking and action-index conversion
  replay.py       # disk-backed replay buffer
  tokenizer.py    # battle-state observation extraction
  train.py        # PPO and imitation training utilities
  vocab.py        # observation vocabulary construction/loading

scripts/
  train_selfplay.py          # league-style self-play training
  train_imitation.py         # behavior-cloning trainer
  eval_baseline_suite.py     # baseline evaluation suite
  play.py                    # play ladder/challenge games with a checkpoint
  plot_training_log.py       # self-play log plotting
  plot_imitation_log.py      # imitation log plotting

tests/
  test_*.py                  # focused unit coverage
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd pokeenv_transformer
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run a local Pokemon Showdown server

Most training and evaluation workflows expect a local Showdown server. From a `pokemon-showdown` checkout:

```bash
node pokemon-showdown start --no-security --port 8000
```

Optional environment variables:

```bash
export SHOWDOWN_WS_URL=ws://localhost:8000/showdown/websocket
export SHOWDOWN_AUTH_URL=http://localhost:8000/action.php?
export SHOWDOWN_ROOT=/path/to/pokemon-showdown
```

`SHOWDOWN_ROOT` is optional, but useful when the tokenizer/vocabulary should read local Showdown metadata.

## Usage

### Run the test suite

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix avoids unrelated global pytest plugin issues.

### Smoke-test self-play training

```bash
python -m scripts.train_selfplay \
  --iterations 1 \
  --selfplay-battles 2 \
  --selfplay-steps 0 \
  --train-steps 2 \
  --eval-battles 2 \
  --batch-size 8 \
  --checkpoint-dir checkpoints_smoke
```

### Train with behavior cloning

```bash
python -m scripts.train_imitation \
  --iterations 1 \
  --demo-battles 4 \
  --demo-steps 0 \
  --train-steps 2 \
  --eval-battles 4 \
  --batch-size 8 \
  --checkpoint-dir checkpoints_imitation_smoke
```

### Evaluate a checkpoint

```bash
python -m scripts.eval_baseline_suite \
  --checkpoint checkpoints_smoke/current.pt \
  --n-battles 20
```

### Play with a trained checkpoint

```bash
python -m scripts.play \
  --checkpoint checkpoints_smoke/current.pt \
  --n-battles 1 \
  --battle-format gen9randombattle
```

### Plot training logs

```bash
python -m scripts.plot_training_log path/to/selfplay_output.txt
python -m scripts.plot_imitation_log path/to/imitation_output.txt
```

## 🔍 Model and Observation Notes

The default Gen 9 Random Battles observation is a structured flat vector that includes both teams, battle globals, transition history, and a legal-action mask. Internally, the model repacks the observation into a compact token sequence:

- 1 actor token
- 1 critic token
- 1 field/global token
- 12 Pokemon tokens

The policy head predicts over 14 actions:

- 4 move actions
- 4 Terastallized move actions
- 6 switch actions

The value head supports scalar or two-hot value targets, with the default two-hot configuration using 51 bins across `[-1.6, 1.6]`.

## 📄 License

This project is licensed under the MIT License. Add or update the repository `LICENSE` file with the full license text before publishing.
