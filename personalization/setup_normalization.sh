#!/bin/bash

# Setup Normalization Statistics
# This script generates the property normalization statistics required to fix objective inversion.
# Run this ONCE before starting GRPO training.


#SBATCH -J setup_normalization             # Job name
#SBATCH -p mit_normal_gpu                 # GPU partition
#SBATCH -c 4                              # CPU cores
#SBATCH --mem=32G                         # Memory
#SBATCH -t 0:30:00                        # Time limit (30 min)
#SBATCH -G 1                              # 1 GPU
#SBATCH -o logs/normalization/setup_normalization_%j.out # STDOUT log
#SBATCH -e logs/normalization/setup_normalization_%j.err # STDERR log


echo "============================================================"
echo "Setting up Property Normalization Statistics"
echo "============================================================"
echo ""
echo "This will generate 5000 sequences from the base model and"
echo "calculate mean/std for all properties (activity, toxicity,"
echo "stability, length)."
echo ""
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
# Set paths

REPO_ROOT="/orcd/home/002/jordancn/RLHF-PLM" 
ACTIVITY_CHECKPOINT="${REPO_ROOT}/amp_design/best_new_4.pth"
TOXICITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/toxicity_head.pth"
STABILITY_CHECKPOINT="${REPO_ROOT}/personalization/checkpoints/stability_head.pth"
OUTPUT_PATH="${REPO_ROOT}/personalization/checkpoints/property_normalization.json"

# Check if normalization stats already exist
if [ -f "$OUTPUT_PATH" ]; then
    echo "⚠️  WARNING: Normalization statistics already exist at:"
    echo "   $OUTPUT_PATH"
    echo ""
    read -p "Do you want to regenerate them? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing statistics. Exiting."
        exit 0
    fi
    echo "Regenerating statistics..."
fi

# Check if checkpoints exist
missing_files=()
if [ ! -f "$ACTIVITY_CHECKPOINT" ]; then
    missing_files+=("$ACTIVITY_CHECKPOINT")
fi
if [ ! -f "$TOXICITY_CHECKPOINT" ]; then
    missing_files+=("$TOXICITY_CHECKPOINT")
fi
if [ ! -f "$STABILITY_CHECKPOINT" ]; then
    missing_files+=("$STABILITY_CHECKPOINT")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ ERROR: Missing required checkpoint files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please ensure all property head checkpoints are trained and available."
    exit 1
fi

echo "✓ All checkpoints found"
echo ""
echo "Starting statistics calculation..."
echo "This may take 10-20 minutes depending on your GPU."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to repo root for correct imports
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"
echo ""

# Set PYTHONPATH to ensure imports work (include amp_design for progen2hf)
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/amp_design:$PYTHONPATH"

# Run the calculation script
python personalization/calculate_normalization_stats.py \
    --activity_checkpoint "$ACTIVITY_CHECKPOINT" \
    --toxicity_checkpoint "$TOXICITY_CHECKPOINT" \
    --stability_checkpoint "$STABILITY_CHECKPOINT" \
    --output_path "$OUTPUT_PATH" \
    --num_sequences 5000 \
    --batch_size 8 \
    --model_name "hugohrban/progen2-xlarge" \
    --esm_model_size 650M \
    --device cuda

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ SUCCESS! Normalization statistics have been generated."
    echo "============================================================"
    echo ""
    echo "Statistics saved to: $OUTPUT_PATH"
    echo ""
    echo "Next steps:"
    echo "1. Review the statistics in the JSON file"
    echo "2. Run GRPO training with the normalization stats"
    echo "3. The objective inversion should now be fixed!"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ ERROR: Failed to generate normalization statistics"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and try again."
    exit 1
fi

