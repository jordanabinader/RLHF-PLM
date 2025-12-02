#!/bin/bash
#SBATCH --job-name=train_toxicity
#SBATCH --output=logs/toxicity_%j.out
#SBATCH --error=logs/toxicity_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=PARTITION_NAME_HERE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
# TODO: Replace PARTITION_NAME_HERE with your cluster's GPU partition name

# Print job information
echo "Starting toxicity training job"
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
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Create checkpoint directory if it doesn't exist
mkdir -p personalization/checkpoints

# Run training script
echo "Starting toxicity head training..."
python personalization/train_toxicity.py \
    --data-dir ToxDL2-main/data \
    --output-dir personalization/checkpoints \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda \
    --patience 10

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Checkpoint saved to: personalization/checkpoints/toxicity_head.pth"
else
    echo "Training failed with exit code $?"
    exit 1
fi

echo "Job finished at: $(date)"

