#!/bin/bash
#SBATCH --account=p52924
#SBATCH --partition=gengpu
#SBATCH --job-name=vae-vimu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

# Override with, for example:
# sbatch --export=ALL,EPOCHS=8000,RUN_ID=vimu_full jobs/train_virtual_imus.sbatch
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-syndata}"
DATA_DIR="${DATA_DIR:-syndata_virtual_imus}"
FORMATTED_DATA_DIR="${FORMATTED_DATA_DIR:-${PROJECT_DIR}/data/ml/${DATA_DIR}}"
EPOCHS="${EPOCHS:-8000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-12000}"
DISTRIBUTION="${DISTRIBUTION:-auto}"
PLOT_FILENAME="${PLOT_FILENAME:-virtual_imu_${SLURM_JOB_ID:-local}.pdf}"
FORMAT_DATASET_FIRST="${FORMAT_DATASET_FIRST:-0}"
DEFAULT_RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_ID}_job${SLURM_JOB_ID:-local}}"
RUN_OUTPUT_DIR="${PROJECT_DIR}/data/outputs/${DATA_DIR}/${RUN_ID}"
RUN_LOG_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p "${RUN_LOG_DIR}"

STDOUT_LOG="${RUN_LOG_DIR}/slurm-${SLURM_JOB_NAME:-vae-vimu}-${SLURM_JOB_ID:-local}.out"
STDERR_LOG="${RUN_LOG_DIR}/slurm-${SLURM_JOB_NAME:-vae-vimu}-${SLURM_JOB_ID:-local}.err"

exec > >(tee -a "${STDOUT_LOG}")
exec 2> >(tee -a "${STDERR_LOG}" >&2)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "Source motion dataset: ${SOURCE_DATA_DIR}"
echo "Formatted ML dataset name: ${DATA_DIR}"
echo "Formatted ML dataset path: ${FORMATTED_DATA_DIR}"
echo "Run ID: ${RUN_ID}"
echo "Run output directory: ${RUN_OUTPUT_DIR}"
echo "Run log directory: ${RUN_LOG_DIR}"

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
    echo "Formatting virtual-IMU dataset from source: ${SOURCE_DATA_DIR}"
    uv run python src/ML/FormatDataset.py \
        --data-dir "${SOURCE_DATA_DIR}" \
        --output-dir "${FORMATTED_DATA_DIR}"
fi

if [[ ! -f "${FORMATTED_DATA_DIR}/train_data.npy" || ! -f "${FORMATTED_DATA_DIR}/test_data.npy" ]]; then
    echo "Missing formatted dataset under: ${FORMATTED_DATA_DIR}" >&2
    echo "Either scp it there or submit with FORMAT_DATASET_FIRST=1." >&2
    exit 1
fi

echo "Training virtual-IMU dataset: ${DATA_DIR}"
uv run python src/ML/Train.py \
    --data-dir "${DATA_DIR}" \
    --formatted-data-dir "${FORMATTED_DATA_DIR}" \
    --run-id "${RUN_ID}" \
    --plot-filename "${PLOT_FILENAME}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --distribution "${DISTRIBUTION}" \
    --eval-mode masked-kinematics-and-imus-by-location

echo "Finished: $(date)"
