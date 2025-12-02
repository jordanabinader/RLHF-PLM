# SLURM Scripts for User-Conditioned GRPO

This directory contains SLURM scripts for training and evaluating user-conditioned GRPO policies on HPC clusters.

## Available Scripts

### 1. `train_eval_user_conditioned.slurm` (Recommended)
**Full multi-persona training and evaluation pipeline**

- Trains with random persona cycling (learns to adapt to all personas)
- Evaluates across all personas
- Generates comprehensive reports
- Runtime: ~24 hours
- Memory: 64GB
- GPU: 1x

**Usage:**
```bash
# First, update the partition name (see setup below)
sbatch slurm/train_eval_user_conditioned.slurm
```

**What it does:**
1. Trains user-conditioned policy for 20 epochs
2. Saves checkpoints every 25 steps
3. Evaluates on all personas (200 sequences each)
4. Generates summary report with metrics
5. Saves per-persona sequences for analysis

### 2. `train_eval_single_persona.slurm`
**Single persona training (faster, for testing)**

- Trains with one specific persona
- Good for initial testing or persona-specific models
- Runtime: ~12 hours
- Memory: 48GB
- GPU: 1x

**Usage:**
```bash
# Edit the PERSONA variable in the script first
# Then submit:
sbatch slurm/train_eval_single_persona.slurm
```

## Setup Instructions

### Step 1: Find Your Cluster's GPU Partition

Run the discovery script:
```bash
bash slurm/check_cluster.sh
```

This will show available partitions and GPUs on your cluster.

### Step 2: Verify Cluster Configuration

The scripts are pre-configured for **MIT Supercloud** with:
- Partition: `mit_normal_gpu`
- Module: `miniforge`
- Virtual environment auto-creation

If you're on a different cluster, edit these lines in the SLURM script:

```bash
# Open the script
nano slurm/train_eval_user_conditioned.slurm

# Update partition (line ~3):
#SBATCH -p mit_normal_gpu  # Change to your GPU partition

# Update module loading (line ~35):
module load miniforge      # Change to your Python module

# Examples for different clusters:
# Generic cluster with conda:
module load anaconda3

# Cluster with system Python:
# (Remove module load line, ensure python3 is in PATH)

# Cluster with specific CUDA/Python:
module load cuda/11.8
module load python/3.10
```

### Step 3: Verify Virtual Environment

The script automatically:
1. Creates `venv/` if it doesn't exist
2. Installs all required packages
3. Uses `PYTHONNOUSERSITE=1` to avoid conflicts

No manual setup needed!

### Step 4: Prepare Checkpoints

Ensure these checkpoints exist before training:

**Required:**
- ✓ `amp_design/best_new_4.pth` - Activity classifier (should already exist)
- ✓ `personalization/checkpoints/toxicity_head.pth` - Run `sbatch slurm/train_toxicity.slurm` first
- ✓ `personalization/checkpoints/stability_head.pth` - Run `sbatch slurm/train_stability.slurm` first

**Optional (if using ProGen):**
- `progen2hf/progen2-small` - Base model
- `progen2hf/` - Tokenizer

**To train property heads first:**
```bash
# 1. Train toxicity head
sbatch slurm/train_toxicity.slurm

# 2. Train/load stability head
sbatch slurm/train_stability.slurm

# 3. Wait for both to complete, then train GRPO
sbatch slurm/train_eval_user_conditioned.slurm
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f logs/user_conditioned_JOBID.out
```

### View errors
```bash
tail -f logs/user_conditioned_JOBID.err
```

### Cancel a job
```bash
scancel JOBID
```

## Output Structure

After completion, you'll find:

```
grpo_runs/user_conditioned_multi/
├── checkpoint_step_25/         # Intermediate checkpoints
├── checkpoint_step_50/
├── ...
└── final_model/                # Final trained model

evaluation_results/user_conditioned_multi/
├── persona_evaluation.csv      # Summary statistics
├── summary_report.txt          # Full report
├── PotencyMaximizer_sequences.txt
├── SafetyFirst_sequences.txt
├── BalancedDesigner_sequences.txt
└── ...                         # One file per persona

logs/
└── user_conditioned_JOBID.out  # Training and evaluation logs
```

## Understanding Results

### Evaluation CSV Columns

- `persona`: Persona name
- `num_sequences`: Total sequences generated
- `validity_rate`: Fraction passing validity constraints
- `mean_activity`: Average activity score
- `mean_toxicity`: Average toxicity score
- `mean_stability`: Average stability score
- `mean_length`: Average sequence length
- `mean_reward`: Average personalized reward
- `uniqueness`: Fraction of unique sequences
- `aa_entropy`: Amino acid composition diversity

### What to Look For

**Good Results:**
- Validity rate > 70%
- Different personas show different property distributions
- High uniqueness (>80%)
- Persona-specific patterns in generated sequences

**Issues:**
- Low validity rate (<50%): Adjust `--reward-penalty` or `--min-charge`
- All personas generate similar sequences: Increase training time or adjust persona weights
- Low uniqueness (<50%): Adjust temperature or top-p sampling

## Customization

### Change Training Duration

Edit these lines in the script:
```bash
EPOCHS=20      # Number of training epochs
BATCH_SIZE=32  # Sequences per batch
STEPS=100      # Steps per epoch
```

### Adjust Validity Constraints

```bash
REWARD_PENALTY=-10.0  # Penalty for invalid sequences (more negative = stricter)
MIN_CHARGE=0.0        # Minimum net charge (higher = requires more positive charge)
```

### Change Persona Cycling

Edit `--persona-cycle-mode`:
- `random`: Random persona each batch (default)
- `round_robin`: Cycle through personas in order
- `single`: Use one persona only (with `--persona-name`)

## Troubleshooting

### Out of Memory (OOM)

Reduce resource usage:
```bash
BATCH_SIZE=16         # Smaller batches
#SBATCH --mem=48G     # Less memory
```

Or request more:
```bash
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1  # Request more powerful GPU
```

### Job Pending Forever

Check your partition name:
```bash
sinfo  # Show available partitions
squeue -u $USER  # Check job status
```

### CUDA Out of Memory

Add to the script before training:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Training Too Slow

Use multiple GPUs (if available):
```bash
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
```

And add to training command:
```bash
--world-size 2
```

## Advanced Usage

### Array Jobs for Multiple Personas

Create an array job to train different personas in parallel:

```bash
#!/bin/bash
#SBATCH --array=0-4  # 5 personas

PERSONAS=("PotencyMaximizer" "SafetyFirst" "BalancedDesigner" "StabilityFocused" "ShortPeptideFan")
PERSONA=${PERSONAS[$SLURM_ARRAY_TASK_ID]}

python amp_design/grpo.py \
  --persona-name "$PERSONA" \
  --persona-cycle-mode single \
  --output-dir "grpo_runs/${PERSONA}" \
  ...
```

### Hyperparameter Sweep

Test different learning rates:

```bash
#!/bin/bash
#SBATCH --array=0-4

LRS=(1e-5 2e-5 5e-5 1e-4 2e-4)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

python amp_design/grpo.py \
  --lr $LR \
  --exp-name "lr_sweep_${LR}" \
  ...
```

## Resource Recommendations

### For Testing
- Partition: Any GPU partition
- GPU: 1x (any type)
- Memory: 32GB
- Time: 4 hours
- Config: `--epochs 2 --batch-size 16`

### For Full Training
- Partition: GPU partition
- GPU: 1x V100 or A100
- Memory: 64GB
- Time: 24 hours
- Config: Default (20 epochs, batch 32)

### For Production
- Partition: High-priority GPU
- GPU: 1x A100
- Memory: 96GB
- Time: 48 hours
- Config: `--epochs 30 --batch-size 64`

## Getting Help

1. Check logs for error messages
2. Review `personalization/USER_CONDITIONED_GRPO.md` for detailed documentation
3. Run integration test: `python tests/test_integration_user_conditioned.py`
4. Verify checkpoints exist and are accessible
5. Check cluster-specific documentation for SLURM configuration

## Quick Start Checklist

- [ ] Run `slurm/check_cluster.sh` to find GPU partition
- [ ] Update partition name in SLURM script
- [ ] Verify all checkpoint paths exist
- [ ] Test with single persona script first
- [ ] Submit full multi-persona job
- [ ] Monitor logs for issues
- [ ] Review evaluation results
- [ ] Compare sequences across personas

---

**Last Updated**: December 2024  
**Compatible With**: User-Conditioned GRPO v1.0

