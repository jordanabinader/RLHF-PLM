#!/bin/bash
#SBATCH --job-name=train_property_heads
#SBATCH --output=logs/property_heads_%j.out
#SBATCH --error=logs/property_heads_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Print job information
echo "========================================================================"
echo "Training Both Property Heads (Toxicity + Stability)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Load modules (adjust based on your HPC environment)
# Uncomment and modify these based on your cluster's module system
# module load cuda/11.8
# module load python/3.9
# module load anaconda/2023

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
module load miniforge
# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify GPU availability
echo ""
echo "========================================================================"
echo "GPU Check"
echo "========================================================================"
echo ""

# Create checkpoint directory
mkdir -p personalization/checkpoints
mkdir -p logs

# ============================================================================
# Virtual Environment Setup
# ============================================================================
echo "========================================================================"
echo "Checking virtual environment"
echo "========================================================================"

# If venv does not exist → create it and install dependencies
if [ ! -d "venv" ]; then
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv

    echo "Activating virtual environment..."
    source venv/bin/activate

    echo "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "WARNING: requirements.txt not found. Skipping dependency installation."
    fi

else
    echo "Virtual environment already exists. Activating..."
    source venv/bin/activate
    pip install -r requirements.txt
fi

echo "Virtual environment ready."
echo "========================================================================"
echo ""

# ============================================================================
# Step 1: Train Toxicity Head
# ============================================================================
echo "========================================================================"
echo "Step 1/2: Training Toxicity Head"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

python personalization/train_toxicity.py \
    --data-dir ToxDL2-main/data \
    --output-dir personalization/checkpoints \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda \
    --patience 10

TOXICITY_EXIT_CODE=$?

if [ $TOXICITY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Toxicity head training completed successfully!"
    echo "  Checkpoint: personalization/checkpoints/toxicity_head.pth"
else
    echo ""
    echo "✗ Toxicity head training failed with exit code $TOXICITY_EXIT_CODE"
    echo "  Continuing to stability head anyway..."
fi

echo ""
echo "Finished at: $(date)"
echo ""

# ============================================================================
# Step 2: Create Stability Head Placeholder
# ============================================================================
echo "========================================================================"
echo "Step 2/2: Creating Stability Head Placeholder"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

python personalization/train_stability.py \
    --mode placeholder \
    --output-dir personalization/checkpoints \
    --device cuda

STABILITY_EXIT_CODE=$?

if [ $STABILITY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Stability head placeholder created successfully!"
    echo "  Checkpoint: personalization/checkpoints/stability_head.pth"
else
    echo ""
    echo "✗ Stability head creation failed with exit code $STABILITY_EXIT_CODE"
fi

echo ""
echo "Finished at: $(date)"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "Training Summary"
echo "========================================================================"
echo "Toxicity head:  $([ $TOXICITY_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Stability head: $([ $STABILITY_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""

# List created checkpoints
if [ -f "personalization/checkpoints/toxicity_head.pth" ]; then
    TOX_SIZE=$(du -h personalization/checkpoints/toxicity_head.pth | cut -f1)
    echo "Created: personalization/checkpoints/toxicity_head.pth ($TOX_SIZE)"
fi

if [ -f "personalization/checkpoints/stability_head.pth" ]; then
    STAB_SIZE=$(du -h personalization/checkpoints/stability_head.pth | cut -f1)
    echo "Created: personalization/checkpoints/stability_head.pth ($STAB_SIZE)"
fi

echo ""
echo "Job completed at: $(date)"
echo "========================================================================"

# Exit with failure if either training failed
if [ $TOXICITY_EXIT_CODE -ne 0 ] || [ $STABILITY_EXIT_CODE -ne 0 ]; then
    exit 1
fi

