#!/bin/bash
#SBATCH --account=p52924
#SBATCH --partition=gengpu
#SBATCH --job-name=vae-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Quest notes:
# - Replace pXXXXX with your Quest allocation from `groups`.
# - General Access GPU jobs use the `gengpu` partition.
# - To pin a GPU type, use `#SBATCH --gres=gpu:a100:1` or `#SBATCH --gres=gpu:h100:1`.
# - To shard training across multiple GPUs on one node, submit with more GPUs, e.g.
#   `sbatch --gres=gpu:4 --export=ALL,DISTRIBUTION=mirrored jobs/train.sbatch`
# - Slurm writes top-level stdout/stderr in the submit directory, and this script
#   also mirrors them into data/outputs/<data-dir>/<run-id>/logs/.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

# Runtime knobs. Override with:
# sbatch --export=ALL,DATA_DIR=syndata,EPOCHS=200,SEED=42,BATCH_SIZE=12000,DISTRIBUTION=auto,PLOT_FILENAME=run.pdf,RUN_ID=my_run jobs/train.sbatch
DATA_DIR="${DATA_DIR:-syndata}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-12000}"
DISTRIBUTION="${DISTRIBUTION:-auto}"
PLOT_FILENAME="${PLOT_FILENAME:-train_${SLURM_JOB_ID:-local}.pdf}"
FORMAT_DATASET_FIRST="${FORMAT_DATASET_FIRST:-0}"
DEFAULT_RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}_job${SLURM_JOB_ID:-local}}"
RUN_OUTPUT_DIR="${PROJECT_DIR}/data/outputs/${DATA_DIR}/${RUN_ID}"
RUN_LOG_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p "${RUN_LOG_DIR}"

STDOUT_LOG="${RUN_LOG_DIR}/slurm-${SLURM_JOB_NAME:-vae-train}-${SLURM_JOB_ID:-local}.out"
STDERR_LOG="${RUN_LOG_DIR}/slurm-${SLURM_JOB_NAME:-vae-train}-${SLURM_JOB_ID:-local}.err"

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Requested GPUs: ${SLURM_GPUS_ON_NODE:-1}"
echo "Run ID: ${RUN_ID}"
echo "Run output directory: ${RUN_OUTPUT_DIR}"
echo "Run log directory: ${RUN_LOG_DIR}"
echo "Mirrored stdout log: ${STDOUT_LOG}"
echo "Mirrored stderr log: ${STDERR_LOG}"

# Keep matplotlib headless-safe and avoid CPU oversubscription.
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

module purge
module load mamba/24.3.0

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available in PATH." >&2
    echo "On the Quest login node, install it once with: python -m pip install uv" >&2
    exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
    echo "Missing repo-local .venv." >&2
    echo "On the Quest login node, run: uv sync" >&2
    exit 1
fi

echo "uv: $(command -v uv)"
echo "Project Python: ${PROJECT_DIR}/.venv/bin/python"
uv run python --version

if [[ "${FORMAT_DATASET_FIRST}" == "1" ]]; then
    echo "Formatting dataset: ${DATA_DIR}"
    uv run python src/ML/FormatDataset.py --data-dir "${DATA_DIR}"
fi

echo "Training dataset: ${DATA_DIR}"
uv run python src/ML/Train.py \
    --data-dir "${DATA_DIR}" \
    --run-id "${RUN_ID}" \
    --plot-filename "${PLOT_FILENAME}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --distribution "${DISTRIBUTION}"

echo "Finished: $(date)"
