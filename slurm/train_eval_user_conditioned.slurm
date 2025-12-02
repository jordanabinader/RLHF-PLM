#!/bin/bash
#SBATCH --job-name=user_conditioned_grpo
#SBATCH --output=logs/user_conditioned_%j.out
#SBATCH --error=logs/user_conditioned_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=PARTITION_NAME_HERE

# User-Conditioned GRPO: Full Training & Evaluation Pipeline
# This script:
#   1. Trains a user-conditioned policy with multi-persona cycling
#   2. Evaluates the trained policy across all personas
#   3. Generates analysis and visualization

echo "========================================"
echo "User-Conditioned GRPO Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# Change to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source venv/bin/activate
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create output directories
mkdir -p logs
mkdir -p grpo_runs/user_conditioned_multi
mkdir -p evaluation_results/user_conditioned_multi

# Configuration
BASE_MODEL_PATH="progen2hf/progen2-small"
TOKENIZER_PATH="progen2hf/"
ACTIVITY_CHECKPOINT="amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="personalization/checkpoints/stability_head.pth"
OUTPUT_DIR="grpo_runs/user_conditioned_multi"
EVAL_DIR="evaluation_results/user_conditioned_multi"

# Training parameters
EPOCHS=20
BATCH_SIZE=32
STEPS=100
LR=2e-5
SAVE_EVERY=25
REWARD_PENALTY=-10.0
MIN_CHARGE=0.0

echo ""
echo "========================================"
echo "STAGE 1: Training User-Conditioned Policy"
echo "========================================"
echo "Configuration:"
echo "  - Mode: Multi-persona (random cycling)"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LR"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Run training
python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --use-personalization \
  --persona-cycle-mode random \
  --output-dir "$OUTPUT_DIR" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --steps $STEPS \
  --lr $LR \
  --save-every $SAVE_EVERY \
  --reward-penalty $REWARD_PENALTY \
  --min-charge $MIN_CHARGE \
  --tracker-project-name "user_conditioned_grpo" \
  --exp-name "multi_persona_${SLURM_JOB_ID}"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"

echo ""
echo "========================================"
echo "STAGE 2: Evaluating Trained Policy"
echo "========================================"
echo "Configuration:"
echo "  - Checkpoint: $OUTPUT_DIR/final_model"
echo "  - Sequences per persona: 200"
echo "  - Output: $EVAL_DIR"
echo ""

# Run evaluation
python personalization/evaluate_user_conditioned_policy.py \
  --checkpoint "$OUTPUT_DIR/final_model" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --activity-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --num-sequences 200 \
  --output-dir "$EVAL_DIR" \
  --device cuda

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "Training completed, but evaluation needs manual review."
fi

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $EVAL_DIR"
echo "========================================"

# Generate summary report
echo ""
echo "========================================"
echo "STAGE 3: Generating Summary Report"
echo "========================================"

REPORT_FILE="$EVAL_DIR/summary_report.txt"

cat > "$REPORT_FILE" <<EOF
User-Conditioned GRPO Training & Evaluation Report
==================================================
Job ID: $SLURM_JOB_ID
Date: $(date)
Node: $SLURM_NODELIST

Training Configuration
----------------------
Model: $BASE_MODEL_PATH
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LR
Persona Cycle Mode: random
Output Directory: $OUTPUT_DIR

Evaluation Configuration
------------------------
Sequences per Persona: 200
Checkpoint: $OUTPUT_DIR/final_model
Output Directory: $EVAL_DIR

Files Generated
---------------
- Training checkpoint: $OUTPUT_DIR/final_model/
- Evaluation CSV: $EVAL_DIR/persona_evaluation.csv
- Per-persona sequences: $EVAL_DIR/*_sequences.txt
- This report: $REPORT_FILE

EOF

# Add evaluation summary if available
if [ -f "$EVAL_DIR/persona_evaluation.csv" ]; then
    echo "Evaluation Summary (from CSV)" >> "$REPORT_FILE"
    echo "------------------------------" >> "$REPORT_FILE"
    head -20 "$EVAL_DIR/persona_evaluation.csv" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

echo "Summary report saved to: $REPORT_FILE"

# Display key metrics
echo ""
echo "========================================"
echo "Key Metrics"
echo "========================================"
if [ -f "$EVAL_DIR/persona_evaluation.csv" ]; then
    echo ""
    echo "Top 10 lines from evaluation results:"
    head -10 "$EVAL_DIR/persona_evaluation.csv"
    echo ""
fi

# Cleanup
echo ""
echo "========================================"
echo "Cleanup"
echo "========================================"
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "No torch available for cleanup"

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "End Time: $(date)"
echo "Duration: $SECONDS seconds"
echo ""
echo "Next steps:"
echo "  1. Review results in: $EVAL_DIR"
echo "  2. Check training logs in: logs/user_conditioned_${SLURM_JOB_ID}.out"
echo "  3. Compare sequences across personas"
echo "  4. Optionally run ablation studies"
echo "========================================"

# Exit with appropriate code
if [ $TRAIN_EXIT_CODE -eq 0 ] && [ $EVAL_EXIT_CODE -eq 0 ]; then
    exit 0
else
    exit 1
fi

