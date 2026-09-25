#!/bin/bash
#SBATCH --job-name=Sionna985-4090
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --array=0-3%4
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail

SCRIPT_DIR="${RID_CLUSTER_GENERATOR_DIR:-${SLURM_SUBMIT_DIR:?}}"
cd "$SCRIPT_DIR"

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load miniforge3/26.3.2-3
module load cuda/12.8

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BASE="$HOME/miniconda3"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "Conda was not found. Load Miniforge and create the UAV_RM environment first." >&2
    exit 2
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${RID_CONDA_ENV:-UAV_RM}"

export PYTHONUNBUFFERED=1
export RID_TX_BATCH_SIZE="${RID_TX_BATCH_SIZE:-60}"
python -u run_cluster_5090.py
