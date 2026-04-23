#!/bin/bash
#SBATCH --account=pXXXXX
#SBATCH --partition=gengpu
#SBATCH --job-name=vae-train
#SBATCH --output=outputs/ml/logs/%x-%j.out
#SBATCH --error=outputs/ml/logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Quest GPU options:
# - Replace pXXXXX above with your Quest Slurm account from `groups`.
# - For Priority Access allocations, use your account's partition or `buyin`
#   instead of `gengpu`.
# - If you need a specific GPU, change `--gres=gpu:1` to `--gres=gpu:a100:1`
#   or `--gres=gpu:h100:1`.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

mkdir -p outputs/ml/logs outputs/ml/results outputs/ml/plots outputs/ml/models/Models1

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"

# Keep matplotlib headless-safe on compute nodes.
export MPLBACKEND=Agg

# Avoid TensorFlow over-subscribing CPU threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

# Quest uses LMOD modules. Mamba is the recommended Python environment manager.
module purge
module load mamba/24.3.0

PLOT_FILENAME="train_${SLURM_JOB_ID:-local}.pdf"

if [[ -n "${QUEST_CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${QUEST_CONDA_ENV}"
    python src/ML/Train.py --plot-filename "${PLOT_FILENAME}"
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
    python src/ML/Train.py --plot-filename "${PLOT_FILENAME}"
elif command -v uv >/dev/null 2>&1; then
    uv run python src/ML/Train.py --plot-filename "${PLOT_FILENAME}"
else
    echo "No Python environment found." >&2
    echo "Create/activate one on Quest, or submit with QUEST_CONDA_ENV=/path/to/env." >&2
    exit 1
fi

echo "Finished: $(date)"
