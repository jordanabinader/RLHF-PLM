#!/bin/bash
# Script to check available partitions and GPU resources on your HPC cluster

echo "========================================================================"
echo "Checking Available Partitions and GPU Resources"
echo "========================================================================"
echo ""

echo "Available Partitions:"
echo "--------------------"
sinfo -o "%P %G %l %D %N" | head -20
echo ""

echo "GPU-enabled Partitions:"
echo "----------------------"
sinfo -o "%P %G" | grep -i gpu
echo ""

echo "Your Account Info:"
echo "-----------------"
sacctmgr show user $USER withassoc format=user,account%20,partition%20,qos%30 2>/dev/null || echo "sacctmgr not available"
echo ""

echo "Detailed Node Info (first 10 GPU nodes):"
echo "----------------------------------------"
sinfo -N -o "%N %P %G %C %m %f" | grep -i gpu | head -10
echo ""

echo "To submit your job, update the SLURM scripts with:"
echo "  #SBATCH --partition=<partition_name_from_above>"
echo "  #SBATCH --gres=gpu:<gpu_type>:1"
echo ""
echo "========================================================================"

