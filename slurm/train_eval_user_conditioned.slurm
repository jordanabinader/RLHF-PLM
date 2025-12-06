#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J user_conditioned_grpo             # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 6:00:00                         # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/user_conditioned_grpo_%j.out # STDOUT log
#SBATCH -e logs/user_conditioned_grpo_%j.err # STDERR log
#########################
# Environment setup     #
#########################

# Create logs directory
mkdir -p logs
# Enable wandb (comment this line to enable wandb logging)
# export WANDB_MODE=disabled
export WANDB_MODE=online
export WANDB_PROJECT="RLHF-PLM"
export WANDB_LOG_MODEL="false"  # Don't upload model checkpoints to wandb
export WANDB_WATCH="false"   
# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
# We use 'venv' (standard spelling), not 'vnev'
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 3. Activate the environment (Fixing the typo here: vnev -> venv)
source venv/bin/activate

# 4. CRITICAL: Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# 5. Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# 6. Install packages
# NOTE: We REMOVED 'pip uninstall'. You cannot uninstall system packages.
# Instead, we use --ignore-installed to force installing a fresh copy 
# into your local venv, shadowing the system version.

# echo "Installing dependencies..."
# python -m pip install --upgrade --ignore-installed --no-cache-dir \
#     "torch>=2.0" \
#     "transformers==4.45.2" \
#     "tokenizers==0.20.0" \
#     "peft==0.10.0" \
#     "numpy" \
#     "pandas" \
#     "scipy" \
#     "scikit-learn" \
#     "fair-esm" \
#     "biopython" \

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions in venv:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
python -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')"
python -c "import accelerate; print(f'accelerate: {accelerate.__version__}')"

# Create output directories
mkdir -p logs
mkdir -p grpo_runs/user_conditioned_multi
mkdir -p evaluation_results/user_conditioned_multi

# Configuration
# Configuration - Use absolute paths
REPO_ROOT="/orcd/home/002/jordancn/RLHF-PLM"  # Adjust to your actual path
BASE_MODEL_PATH="${REPO_ROOT}/amp_design/progen2hf/progen2-small"
TOKENIZER_PATH="${REPO_ROOT}/amp_design/progen2hf/progen2-small"
ACTIVITY_CHECKPOINT="${REPO_ROOT}/amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/stability_head.pth"
OUTPUT_DIR="grpo_runs/user_conditioned_multi"
EVAL_DIR="evaluation_results/user_conditioned_multi"

# Training parameters
EPOCHS=1
BATCH_SIZE=32
STEPS=150  # Increased from 100 to allow more stable convergence
LR=2e-5
SAVE_EVERY=25
REWARD_PENALTY=-0.5
MIN_CHARGE=0.0
BETA=0.15  # Increased from 0.1 to prevent KL divergence explosion
KL_CLIP_VALUE=0.1  # Increased from 0.0 to clip extreme divergence

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
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/amp_design:${PYTHONPATH}"

# Run training
python  amp_design/grpo.py \
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
  --beta $BETA \
  --kl-clip-value $KL_CLIP_VALUE \
  --save-every $SAVE_EVERY \
  --reward-penalty $REWARD_PENALTY \
  --min-charge $MIN_CHARGE \
  --esm-mode "650M" \
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
python  personalization/evaluate_user_conditioned_policy.py \
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

