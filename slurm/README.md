# SLURM Job Scripts for Property Head Training

## Quick Start

### 1. Setup Environment on HPC

```bash
# SSH to your HPC cluster
ssh your_username@your_cluster.edu

# Navigate to project directory
cd /path/to/RLHF-PLM

# Create logs directory
mkdir -p logs

# Setup virtual environment (if not already done)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Submit Jobs

**Option A: Train both property heads sequentially (recommended)**
```bash
sbatch slurm/train_both.slurm
```

**Option B: Train individually**
```bash
# Train toxicity head (takes ~2-3 hours)
sbatch slurm/train_toxicity.slurm

# Create stability placeholder (takes ~1 minute)
sbatch slurm/train_stability.slurm
```

### 3. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/property_heads_*.out

# Check for errors
tail -f logs/property_heads_*.err
```

### 4. After Completion

```bash
# Verify checkpoints were created
ls -lh personalization/checkpoints/

# Should see:
# - toxicity_head.pth (~2-5 MB)
# - stability_head.pth (~1-2 MB)
```

## Job Descriptions

### `train_both.slurm` (Recommended)
Trains both property heads in one job:
1. Toxicity head (~2-3 hours)
2. Stability placeholder (~1 minute)

**Resources:**
- Time: 6 hours
- GPU: 1 GPU
- Memory: 32 GB
- CPUs: 4

**Outputs:**
- `logs/property_heads_<JOBID>.out` - Training progress
- `logs/property_heads_<JOBID>.err` - Error messages
- `personalization/checkpoints/toxicity_head.pth`
- `personalization/checkpoints/stability_head.pth`

### `train_toxicity.slurm`
Trains toxicity head on ToxDL2 dataset.

**Resources:**
- Time: 6 hours
- GPU: 1 GPU
- Memory: 32 GB
- CPUs: 4

**Data required:** `ToxDL2-main/data/` directory must exist

### `train_stability.slurm`
Creates stability head placeholder (fast).

**Resources:**
- Time: 10 minutes
- GPU: 1 GPU (minimal usage)
- Memory: 8 GB
- CPUs: 2

## Customization

### Adjust Resource Requests

Edit the `#SBATCH` directives in the scripts:

```bash
#SBATCH --time=6:00:00        # Wall time (HH:MM:SS)
#SBATCH --partition=gpu        # Partition name (adjust for your cluster)
#SBATCH --gres=gpu:1           # Number of GPUs (can specify type: gpu:a100:1)
#SBATCH --cpus-per-task=4      # CPU cores
#SBATCH --mem=32G              # Memory
```

### Specify GPU Type

Some clusters allow specifying GPU type:

```bash
#SBATCH --gres=gpu:a100:1      # Request A100 GPU
#SBATCH --gres=gpu:v100:1      # Request V100 GPU
#SBATCH --gres=gpu:rtx8000:1   # Request RTX 8000 GPU
```

### Load Cluster-Specific Modules

Uncomment and adjust module loads in the scripts:

```bash
module load cuda/11.8
module load python/3.9
module load anaconda/2023
```

Check your cluster's available modules:
```bash
module avail
```

### Adjust Training Parameters

Modify the Python command in the scripts:

```bash
python personalization/train_toxicity.py \
    --data-dir ToxDL2-main/data \
    --output-dir personalization/checkpoints \
    --epochs 100 \              # Increase epochs
    --batch-size 64 \           # Larger batch size
    --lr 0.0005 \               # Lower learning rate
    --device cuda \
    --patience 20               # More patience for early stopping
```

## Common Issues

### Issue: "Permission denied"

**Solution:** Make scripts executable
```bash
chmod +x slurm/*.slurm
```

### Issue: "Module not found"

**Solution:** Check which modules are available
```bash
module avail python
module avail cuda
```

Then update the `module load` commands in the scripts.

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
--batch-size 16  # Instead of 32
```

Or request more GPU memory:
```bash
#SBATCH --mem=64G  # Instead of 32G
```

### Issue: "Job killed due to time limit"

**Solution:** Request more time
```bash
#SBATCH --time=12:00:00  # 12 hours instead of 6
```

### Issue: "Data directory not found"

**Solution:** Ensure ToxDL2-main is in the correct location
```bash
ls ToxDL2-main/data/
# Should show: domain_data/, esm_data/, pdb_data/, protein_sequences/
```

## Email Notifications

Add these lines to get email notifications:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu
```

## Job Arrays

To train multiple configurations in parallel:

```bash
#SBATCH --array=1-5

# Then in the script:
case $SLURM_ARRAY_TASK_ID in
    1) LR=0.001 ;;
    2) LR=0.0005 ;;
    3) LR=0.0001 ;;
    4) LR=0.00005 ;;
    5) LR=0.00001 ;;
esac

python personalization/train_toxicity.py --lr $LR ...
```

## Monitoring Progress

### Real-time monitoring
```bash
watch -n 10 'squeue -u $USER'
```

### Check GPU usage
```bash
ssh to_compute_node
nvidia-smi
```

### View training progress
```bash
tail -f logs/property_heads_*.out | grep "Epoch"
```

## After Training

### Verify Checkpoints

```bash
python -c "
import torch
checkpoint = torch.load('personalization/checkpoints/toxicity_head.pth')
print('Keys:', checkpoint.keys())
print('Model state dict size:', len(checkpoint['model_state_dict']))
"
```

### Test Property Function

```bash
python examples/property_model_demo.py
```

## Cluster-Specific Examples

### MIT Supercloud
```bash
#SBATCH --partition=xeon-p8
#SBATCH --gres=gpu:volta:1
```

### SLAC
```bash
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
```

### NERSC
```bash
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=your_account
```

Adjust the partition and GPU specifications based on your specific HPC cluster.

