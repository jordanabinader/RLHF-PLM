#!/bin/bash
#SBATCH --job-name=train_stability
#SBATCH --output=logs/stability_%j.out
#SBATCH --error=logs/stability_%j.err
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Print job information
echo "Starting stability placeholder creation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Load modules (adjust based on your HPC environment)
# module load cuda/11.8
# module load python/3.9
# module load anaconda/2023

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create checkpoint directory if it doesn't exist
mkdir -p personalization/checkpoints

# Run training script in placeholder mode
echo "Creating stability head placeholder..."
python personalization/train_stability.py \
    --mode placeholder \
    --output-dir personalization/checkpoints \
    --device cuda

# Check exit status
if [ $? -eq 0 ]; then
    echo "Placeholder created successfully!"
    echo "Checkpoint saved to: personalization/checkpoints/stability_head.pth"
else
    echo "Failed with exit code $?"
    exit 1
fi

echo "Job finished at: $(date)"

