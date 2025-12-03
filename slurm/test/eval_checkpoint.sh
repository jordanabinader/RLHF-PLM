#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J eval_checkpoint_50             # Job name
#SBATCH -p mit_normal_gpu                 # GPU partition
#SBATCH -c 4                              # CPU cores
#SBATCH --mem=32G                         # Memory
#SBATCH -t 0:30:00                        # Time limit (30 min)
#SBATCH -G 1                              # 1 GPU
#SBATCH -o logs/eval_checkpoint_50_%j.out # STDOUT log
#SBATCH -e logs/eval_checkpoint_50_%j.err # STDERR log

#########################
# Environment setup     #
#########################

# Create logs directory
mkdir -p logs

# Load the base system python
module load miniforge

# Activate the virtual environment
source venv/bin/activate

# Ensure we rely ONLY on this venv
export PYTHONNOUSERSITE=1

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"

#########################
# Configuration         #
#########################

# Use absolute paths
REPO_ROOT="/orcd/home/002/jordancn/RLHF-PLM"  # Adjust to your actual path
CHECKPOINT="${REPO_ROOT}/grpo_runs/user_conditioned_multi/checkpoint_step_50"
TOKENIZER_PATH="${REPO_ROOT}/amp_design/progen2hf/progen2-small"
ACTIVITY_CHECKPOINT="${REPO_ROOT}/amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/stability_head.pth"
OUTPUT_DIR="${REPO_ROOT}/evaluation_results/checkpoint_50_eval"
NUM_SEQUENCES=200  # Sequences per persona
ESM_MODE="650M"

# Create output directory
mkdir -p "$OUTPUT_DIR"

#########################
# Run Evaluation        #
#########################

echo ""
echo "========================================"
echo "Evaluating Checkpoint: checkpoint_step_50"
echo "========================================"
echo "Configuration:"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Sequences/persona: $NUM_SEQUENCES"
echo "  - ESM model: $ESM_MODE"
echo "  - Output: $OUTPUT_DIR"
echo ""

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/amp_design:${PYTHONPATH}"

# Add missing --esm-mode argument (temporary fix)
# Note: You may need to modify evaluate_user_conditioned_policy.py first

python personalization/evaluate_user_conditioned_policy.py \
  --checkpoint "$CHECKPOINT" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --activity-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --num-sequences $NUM_SEQUENCES \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

#########################
# Results Summary       #
#########################

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"

# Display summary if CSV exists
if [ -f "$OUTPUT_DIR/persona_evaluation.csv" ]; then
    echo ""
    echo "Summary Statistics:"
    echo "-------------------"
    head -20 "$OUTPUT_DIR/persona_evaluation.csv"
    echo ""
    echo "Full results: $OUTPUT_DIR/persona_evaluation.csv"
else
    echo "Warning: No CSV file generated"
fi

# Count sequences per persona
echo ""
echo "Generated Sequences:"
echo "-------------------"
for file in "$OUTPUT_DIR"/*_sequences.txt; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        basename=$(basename "$file")
        echo "  $basename: $count sequences"
    fi
done

# Generate quick report
REPORT_FILE="$OUTPUT_DIR/evaluation_summary.txt"
cat > "$REPORT_FILE" <<EOF
Checkpoint Evaluation Report
============================
Job ID: $SLURM_JOB_ID
Date: $(date)
Node: $SLURM_NODELIST

Configuration
-------------
Checkpoint: $CHECKPOINT
Sequences per Persona: $NUM_SEQUENCES
ESM Model: $ESM_MODE
Output Directory: $OUTPUT_DIR

Files Generated
---------------
- Summary CSV: $OUTPUT_DIR/persona_evaluation.csv
- Sequence files: $OUTPUT_DIR/*_sequences.txt

Status
------
EOF

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation: SUCCESS" >> "$REPORT_FILE"
else
    echo "Evaluation: FAILED (exit code: $EVAL_EXIT_CODE)" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "End Time: $(date)" >> "$REPORT_FILE"

echo ""
echo "Summary report saved to: $REPORT_FILE"

# Cleanup
echo ""
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "No torch available"

echo ""
echo "========================================"
echo "Evaluation job complete!"
echo "========================================"
echo "Duration: $SECONDS seconds"

exit $EVAL_EXIT_CODE