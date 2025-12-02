#!/bin/bash
#SBATCH -J single_persona_grpo
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -G 1
#SBATCH -o logs/single_persona_%j.out
#SBATCH -e logs/single_persona_%j.err

# Single Persona GRPO: Full Training & Evaluation Pipeline
# This script trains with one specific persona (faster, good for testing)

echo "========================================"
echo "Single Persona GRPO Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# Change to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create logs directory
mkdir -p logs

# Load miniforge module
module load miniforge

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# Upgrade pip
python -m pip install --upgrade pip

# Install/verify dependencies
echo "Verifying dependencies..."
python -m pip install --quiet --upgrade \
    "torch>=2.0" \
    "transformers>=4.40.0" \
    "tokenizers>=0.19.0" \
    "numpy" \
    "pandas" \
    "scipy" \
    "scikit-learn" \
    "fair-esm" \
    "biopython"

echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi || echo "nvidia-smi not available"

# Create output directories
mkdir -p logs
mkdir -p grpo_runs/single_persona
mkdir -p evaluation_results/single_persona

# Configuration
PERSONA="BalancedDesigner"  # Change this to test different personas
BASE_MODEL_PATH="progen2hf/progen2-small"
TOKENIZER_PATH="progen2hf/"
ACTIVITY_CHECKPOINT="amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="personalization/checkpoints/stability_head.pth"
OUTPUT_DIR="grpo_runs/single_persona_${PERSONA}"
EVAL_DIR="evaluation_results/single_persona_${PERSONA}"

# Training parameters
EPOCHS=10
BATCH_SIZE=32
STEPS=100
LR=2e-5

echo ""
echo "========================================"
echo "STAGE 1: Training with $PERSONA"
echo "========================================"

python amp_design/grpo.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --classifier-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --use-personalization \
  --persona-name "$PERSONA" \
  --persona-cycle-mode single \
  --output-dir "$OUTPUT_DIR" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --steps $STEPS \
  --lr $LR \
  --save-every 25 \
  --tracker-project-name "single_persona_grpo" \
  --exp-name "${PERSONA}_${SLURM_JOB_ID}"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "STAGE 2: Evaluating Trained Policy"
echo "========================================"

python personalization/evaluate_user_conditioned_policy.py \
  --checkpoint "$OUTPUT_DIR/final_model" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --activity-checkpoint "$ACTIVITY_CHECKPOINT" \
  --toxicity-checkpoint "$TOXICITY_CHECKPOINT" \
  --stability-checkpoint "$STABILITY_CHECKPOINT" \
  --num-sequences 200 \
  --output-dir "$EVAL_DIR" \
  --device cuda

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Results in: $EVAL_DIR"
echo "Duration: $SECONDS seconds"

