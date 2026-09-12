#!/usr/bin/env bash
# Example: mkdir -p logs; sbatch --chdir="$PWD" configs/slurm/gpu.sbatch python scripts/your_script.py ...
# Adjust partition and wall time for your own scheduler. Runs exactly the supplied command.
#SBATCH --job-name=blood-vlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "Pass a command and its arguments to this job." >&2
    exit 2
fi
"$@"
