#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
    SCENES=("$@")
else
    SCENES=("ShopFacade" "GreatCourt")
fi
DATA_ROOT="${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
WORK_DATASET="${WORK_DATASET:-datasets/cambridge}"
OUT_ROOT="${OUT_ROOT:-map_cambridge_spgs}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU_LIST="${GPU_LIST:-4 5}"
TRAIN_ITERS="${TRAIN_ITERS:-30000}"
DETECTOR_ITERS="${DETECTOR_ITERS:-30000}"
LOG_DIR="${LOG_DIR:-logs}"
LAMBDA_KEYPOINT_LOC="${LAMBDA_KEYPOINT_LOC:-0.0}"
LAMBDA_POSE_LOC="${LAMBDA_POSE_LOC:-0.001}"
POSE_LOC_MAX="${POSE_LOC_MAX:-32}"
POSE_LOC_LOCABILITY_WEIGHT="${POSE_LOC_LOCABILITY_WEIGHT:-0.2}"
STREAM_CAMERAS="${STREAM_CAMERAS:-1}"
TRAIN_ONLY_CAMERAS="${TRAIN_ONLY_CAMERAS:-1}"

mkdir -p "$(dirname "${WORK_DATASET}")" "${OUT_ROOT}" "${LOG_DIR}"

if [ ! -e "${WORK_DATASET}" ] && [ -d "${DATA_ROOT}" ]; then
    ln -s "${DATA_ROOT}" "${WORK_DATASET}"
fi

read -r -a GPUS <<< "${GPU_LIST}"

run_scene() {
    local scene="$1"
    local gpu="$2"
    local out="${OUT_ROOT}/${scene}"
    local log="${LOG_DIR}/train_spgs_${scene}_$(date +%Y%m%d_%H%M%S).log"
    local images="processed"
    if [ ! -d "${WORK_DATASET}/${scene}/${images}" ]; then
        images="."
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" train.py \
        -s "${WORK_DATASET}/${scene}" \
        -m "${out}" \
        -r 1 -f sp -g 3dgs \
        --iterations "${TRAIN_ITERS}" \
        --data_device cpu \
        --train_detector \
        --train_detector_iterations "${DETECTOR_ITERS}" \
        --lambda_keypoint_loc "${LAMBDA_KEYPOINT_LOC}" \
        --lambda_pose_loc "${LAMBDA_POSE_LOC}" \
        --pose_loc_max "${POSE_LOC_MAX}" \
        --pose_loc_locability_weight "${POSE_LOC_LOCABILITY_WEIGHT}" \
        --densify_grad_threshold 0.0004 \
        --images "${images}" \
        --position_lr_init 0.000016 \
        --scaling_lr 0.001 \
        --test_iterations 7000 "${TRAIN_ITERS}" \
        --save_iterations 7000 "${TRAIN_ITERS}" \
        --test_detector_iterations "${DETECTOR_ITERS}" \
        --save_detector_iterations "${DETECTOR_ITERS}" \
        $( [ "${STREAM_CAMERAS}" = "1" ] && printf '%s' '--stream_cameras' ) \
        $( [ "${TRAIN_ONLY_CAMERAS}" = "1" ] && printf '%s' '--train_only_cameras' ) \
        2>&1 | tee "${log}"
}

idx=0
for scene in "${SCENES[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    run_scene "${scene}" "${gpu}" &
    idx=$((idx + 1))
done
wait

for scene in "${SCENES[@]}"; do
    images="processed"
    if [ ! -d "${WORK_DATASET}/${scene}/${images}" ]; then
        images="."
    fi

    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" stdloc.py \
        -s "${WORK_DATASET}/${scene}" \
        -m "${OUT_ROOT}/${scene}" \
        -r 1 -f sp -g 3dgs \
        --images "${images}" \
        --data_device cpu \
        --cfg configs/stdloc_cambridge.yaml \
        --prefix "baseline_${scene}"

    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" stdloc.py \
        -s "${WORK_DATASET}/${scene}" \
        -m "${OUT_ROOT}/${scene}" \
        -r 1 -f sp -g 3dgs \
        --images "${images}" \
        --data_device cpu \
        --cfg configs/stdloc_spgs_cambridge.yaml \
        --prefix "spgs_geom_${scene}"
done
