#!/bin/bash

# Training script for user-conditioned GRPO

echo "=================================="
echo "User-Conditioned GRPO Training"
echo "=================================="

# Set paths (modify these to match your setup)
BASE_MODEL_PATH="hugohrban/progen2-large"  # Using large model (4096 dim) to match checkpoint
TOKENIZER_PATH="hugohrban/progen2-large"
LORA_CHECKPOINT="amp_design/grpo_ckpt"  # Pre-trained GRPO checkpoint (compatible with large model)
ACTIVITY_CHECKPOINT="amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="personalization/checkpoints/stability_head.pth"
NORMALIZATION_STATS="personalization/checkpoints/property_normalization.json"

# Train with single persona
echo ""
echo "Training with single persona (BalancedDesigner) - Starting from pre-trained GRPO checkpoint..."
python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --lora-checkpoint "$LORA_CHECKPOINT" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --normalization-stats-path "$NORMALIZATION_STATS" \
  --use-personalization \
  --persona-name BalancedDesigner \
  --persona-cycle-mode single \
  --output-dir grpo_runs/user_conditioned_large_single \
  --epochs 5 \
  --batch-size 16 \
  --steps 100 \
  --lr 1e-5 \
  --save-every 25 \
  --reward-penalty -10.0 \
  --min-charge 0.0

# Train with multi-persona cycling (random)
echo ""
echo "Training with multi-persona cycling (random mode) - Starting from pre-trained GRPO checkpoint..."
python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --lora-checkpoint "$LORA_CHECKPOINT" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --normalization-stats-path "$NORMALIZATION_STATS" \
  --use-personalization \
  --persona-cycle-mode random \
  --output-dir grpo_runs/user_conditioned_large_multi \
  --epochs 10 \
  --batch-size 16 \
  --steps 100 \
  --lr 1e-5 \
  --save-every 25 \
  --reward-penalty -10.0 \
  --min-charge 0.0

echo ""
echo "=================================="
echo "Training complete!"
echo "=================================="

