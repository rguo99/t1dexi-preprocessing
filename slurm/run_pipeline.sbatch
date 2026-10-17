#!/bin/bash
# Example Slurm job for the full pipeline. Edit the partition and the
# environment activation for your cluster, then submit from the repo root:
#   sbatch slurm/run_pipeline.sbatch
#
# Typical resources (all 502 T1DEXI participants): 16 CPUs, 64 GB memory,
# about 20 minutes.
#SBATCH --job-name=t1dexi-preprocessing
#SBATCH --partition=<your_partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=t1dexi-preprocessing_%j.log

set -euo pipefail

# Activate the environment that has requirements.txt installed, e.g.
#   source /path/to/venv/bin/activate     or     conda activate t1dexi-preprocessing

export T1DEXI_RAW_DIR=/path/to/T1DEXI
export T1DEXI_OUTPUT_DIR=/path/to/t1dexi_output
export T1DEXI_NUM_CPUS="${SLURM_CPUS_PER_TASK}"

bash run_pipeline.sh
