#!/bin/bash

# Training script for user-conditioned GRPO

echo "=================================="
echo "User-Conditioned GRPO Training"
echo "=================================="

# Set paths (modify these to match your setup)
BASE_MODEL_PATH="progen2hf/progen2-small"
TOKENIZER_PATH="progen2hf/"
ACTIVITY_CHECKPOINT="amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="personalization/checkpoints/stability_head.pth"

# Train with single persona
echo ""
echo "Training with single persona (BalancedDesigner)..."
python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --use-personalization \
  --persona-name BalancedDesigner \
  --persona-cycle-mode single \
  --output-dir grpo_runs/user_conditioned_single \
  --epochs 10 \
  --batch-size 32 \
  --steps 100 \
  --lr 2e-5 \
  --save-every 25 \
  --reward-penalty -10.0 \
  --min-charge 0.0

# Train with multi-persona cycling (random)
echo ""
echo "Training with multi-persona cycling (random mode)..."
python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --use-personalization \
  --persona-cycle-mode random \
  --output-dir grpo_runs/user_conditioned_multi \
  --epochs 20 \
  --batch-size 32 \
  --steps 100 \
  --lr 2e-5 \
  --save-every 25 \
  --reward-penalty -10.0 \
  --min-charge 0.0

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="

