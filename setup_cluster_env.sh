#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${RID_CONDA_ENV:-UAV_RM}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module load miniforge3/26.3.2-3
module load cuda/12.8

if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
else
    echo "Conda was not found after loading the Miniforge module." >&2
    exit 2
fi

CONDA_BASE="$($CONDA_EXE info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements-ubuntu.txt"
python -c "import sionna.rt, mitsuba, drjit; print('Sionna RT imports OK')"

echo "Environment ready. Submit with: sbatch $SCRIPT_DIR/submit_5090.slurm"
