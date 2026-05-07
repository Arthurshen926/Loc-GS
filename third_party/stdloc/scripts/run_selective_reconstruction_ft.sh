#!/usr/bin/env bash
set -euo pipefail

SCENE="${SCENE:-StMarysChurch}"
DATA_ROOT="${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
WORK_DATASET="${WORK_DATASET:-datasets/cambridge}"
BASE_MAP="${BASE_MAP:-map_cambridge_spgs/StMarysChurch_stream_fastsave}"
OUT_MAP="${OUT_MAP:-map_cambridge_spgs/StMarysChurch_selective_ft200}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU="${GPU:-0}"
FT_ITERS="${FT_ITERS:-200}"
LOAD_ITERATION="${LOAD_ITERATION:-30000}"
SELECTIVE_RECON_WEIGHT="${SELECTIVE_RECON_WEIGHT:-0.5}"
SELECTIVE_RECON_MIN_WEIGHT="${SELECTIVE_RECON_MIN_WEIGHT:-0.05}"
SELECTIVE_RECON_GAMMA="${SELECTIVE_RECON_GAMMA:-2.0}"
SELECTIVE_RECON_TOP_RATIO="${SELECTIVE_RECON_TOP_RATIO:-0.2}"
CFG="${CFG:-configs/stdloc_spgs_cambridge_detector10000.yaml}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$(dirname "${WORK_DATASET}")" "$(dirname "${OUT_MAP}")" "${LOG_DIR}"

if [ ! -e "${WORK_DATASET}" ] && [ -d "${DATA_ROOT}" ]; then
    ln -s "${DATA_ROOT}" "${WORK_DATASET}"
fi

if [ ! -d "${OUT_MAP}" ]; then
    cp -a "${BASE_MAP}" "${OUT_MAP}"
fi

images="processed"
if [ ! -d "${WORK_DATASET}/${SCENE}/${images}" ]; then
    images="."
fi

train_log="${LOG_DIR}/selective_ft_${SCENE}_$(date +%Y%m%d_%H%M%S).log"
eval_prefix="spgs_selective_${SCENE}_i${FT_ITERS}"

CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 \
"${PYTHON_BIN}" train.py \
    -s "${WORK_DATASET}/${SCENE}" \
    -m "${OUT_MAP}" \
    -r 1 -f sp -g 3dgs \
    --images "${images}" \
    --data_device cpu \
    --load_iteration "${LOAD_ITERATION}" \
    --iterations "${FT_ITERS}" \
    --selective_recon_weight "${SELECTIVE_RECON_WEIGHT}" \
    --selective_recon_min_weight "${SELECTIVE_RECON_MIN_WEIGHT}" \
    --selective_recon_gamma "${SELECTIVE_RECON_GAMMA}" \
    --selective_recon_top_ratio "${SELECTIVE_RECON_TOP_RATIO}" \
    --position_lr_init 0 \
    --position_lr_final 0 \
    --feature_lr 0 \
    --opacity_lr 0 \
    --scaling_lr 0 \
    --rotation_lr 0 \
    --densify_until_iter 0 \
    --test_iterations "${FT_ITERS}" \
    --save_iterations "${FT_ITERS}" \
    --stream_cameras \
    --train_only_cameras \
    2>&1 | tee "${train_log}"

CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 \
"${PYTHON_BIN}" stdloc.py \
    -s "${WORK_DATASET}/${SCENE}" \
    -m "${OUT_MAP}" \
    -r 1 -f sp -g 3dgs \
    --images "${images}" \
    --data_device cpu \
    --iteration "${FT_ITERS}" \
    --cfg "${CFG}" \
    --prefix "${eval_prefix}"
