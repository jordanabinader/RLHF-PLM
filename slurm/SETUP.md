# HPC Setup Instructions

## Step 1: Find Your Cluster's Partition Name

Your cluster doesn't use the generic "gpu" partition name. Run this to find available partitions:

```bash
bash slurm/check_cluster.sh
```

This will show you:
- Available partitions with GPU resources
- GPU types available
- Your account permissions

## Step 2: Update SLURM Scripts

Once you know your partition name, update the scripts:

```bash
# Replace PARTITION_NAME_HERE in all scripts
sed -i 's/PARTITION_NAME_HERE/your_actual_partition/g' slurm/train_*.slurm
```

**Common partition names by institution:**
- **MIT Supercloud**: `sched_mit_hill`, `xeon-p8`, `xeon-g6-volta`
- **Harvard Cannon**: `gpu`, `gpu_requeue`, `serial_requeue`
- **TACC Frontera**: `rtx`, `a100`
- **NERSC Perlmutter**: `gpu` (but uses `--constraint=gpu`)
- **Princeton**: `gpu`, `datasci`
- **Generic SLURM**: Check with `sinfo -o "%P %G"`

## Step 3: Check GPU Type (Optional)

If your cluster requires specifying GPU type:

```bash
# See available GPU types
sinfo -o "%P %G" | grep gpu
```

Then update `--gres` line:
```bash
#SBATCH --gres=gpu:v100:1      # For V100
#SBATCH --gres=gpu:a100:1      # For A100
#SBATCH --gres=gpu:rtx8000:1   # For RTX 8000
```

## Step 4: Create Logs Directory

```bash
mkdir -p logs
```

## Step 5: Submit Job

```bash
sbatch slurm/train_both.slurm
```

## Quick Fix for Your Cluster

Since you're getting "invalid partition specified: gpu", try these common alternatives:

### Option 1: Find and replace automatically
```bash
# Check what partitions exist
sinfo -s

# Common alternatives to try (pick one that exists):
# For GPU partition:
PARTITION=$(sinfo -o "%P" | grep -E "gpu|cuda|ml|ai" | head -1 | tr -d '*')

# Replace in all scripts
if [ ! -z "$PARTITION" ]; then
    echo "Found GPU partition: $PARTITION"
    sed -i "s/PARTITION_NAME_HERE/$PARTITION/g" slurm/*.slurm
else
    echo "No GPU partition found automatically. Check manually with: sinfo"
fi
```

### Option 2: Manual replacement examples

**If your cluster uses standard partitions:**
```bash
sed -i 's/PARTITION_NAME_HERE/gpu_requeue/g' slurm/*.slurm
```

**If your cluster uses named queues:**
```bash
sed -i 's/PARTITION_NAME_HERE/normal/g' slurm/*.slurm
```

### Option 3: Use default partition
If your cluster has a default GPU partition, you can remove the partition line entirely:

```bash
# Comment out the partition line
sed -i 's/#SBATCH --partition=.*/#SBATCH --partition=DEFAULT/' slurm/*.slurm
```

Then edit to:
```bash
# Just comment it out
##SBATCH --partition=PARTITION_NAME_HERE
```

## Troubleshooting

### "Invalid partition name specified"
→ Run `sinfo -s` or `sinfo -o "%P"` to see available partitions

### "QOSMaxGRESPerUser" or similar
→ Your cluster limits GPU usage. Add:
```bash
#SBATCH --qos=normal
```

### "Unable to allocate resources"
→ Try:
```bash
#SBATCH --constraint=gpu  # Instead of --partition
```

### "Invalid account" 
→ Add your account:
```bash
#SBATCH --account=your_account_name
```

Find it with: `sacctmgr show user $USER`

## Example: MIT Supercloud

```bash
#SBATCH --partition=xeon-p8
#SBATCH --gres=gpu:volta:1
```

## Example: Harvard Cannon

```bash
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
```

## Example: Generic High-Memory GPU

```bash
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
```

## Still Having Issues?

Contact your cluster's support or check documentation:
```bash
# Most clusters have this
man sbatch

# Or check cluster-specific docs
ls /etc/slurm/

# Or ask support what GPU partitions are available
```

