#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_ppo_mps}"
SERVER_WS="${SERVER_WS:-ws://127.0.0.1:8000/showdown/websocket}"
SERVER_AUTH="${SERVER_AUTH:-http://127.0.0.1:8000/action.php?}"

python -m scripts.train_selfplay \
  --device auto \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --iterations "${ITERATIONS:-20}" \
  --selfplay-battles "${SELFPLAY_BATTLES:-40}" \
  --train-steps "${TRAIN_STEPS:-200}" \
  --eval-battles "${EVAL_BATTLES:-20}" \
  --promote-threshold "${PROMOTE_THRESHOLD:-0.55}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-1e-4}" \
  --ppo-clip-epsilon "${PPO_CLIP_EPS:-0.2}" \
  --ppo-value-clip "${PPO_VALUE_CLIP:-0.2}" \
  --selfplay-temperature "${SELFPLAY_TEMP:-1.0}" \
  --eval-temperature "${EVAL_TEMP:-0.0}" \
  --reward-damage-coef "${REWARD_DAMAGE_COEF:-0.10}" \
  --reward-turn-penalty "${REWARD_TURN_PENALTY:-0.002}" \
  --reward-discount "${REWARD_DISCOUNT:-0.99}" \
  --reward-target-clip "${REWARD_TARGET_CLIP:-1.0}" \
  --max-turns-before-forfeit "${MAX_TURNS_BEFORE_FORFEIT:-500}" \
  --server-ws "$SERVER_WS" \
  --server-auth "$SERVER_AUTH" \
  "$@"
