#!/usr/bin/env bash
# Train correlation-specific ResNet-18s over several seeds, with at most one
# job per GPU, then aggregate their train-probe accuracies with error bars.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/outputs/spurious_correlations_cifar10_train_runs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/spurious_correlations_cifar10}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-$ROOT_DIR/outputs/spurious_correlations_cifar10_train.png}"
NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

correlations=(0.0 0.2 0.4 0.6 0.8 1.0)
seeds=(0 1 2 3 4)
pids=()
gpu=0

for seed in "${seeds[@]}"; do
    for correlation in "${correlations[@]}"; do
        while (( ${#pids[@]} >= NUM_GPUS )); do
            wait -n
            active_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    active_pids+=("$pid")
                fi
            done
            pids=("${active_pids[@]}")
        done

        CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT_DIR/exp-spurious_correlations_cifar10.py" \
            --correlations "$correlation" \
            --seed "$seed" \
            --probe-eval-split train \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --metrics-path "$RUN_DIR/corr_${correlation}_seed_${seed}.json" \
            --output "$RUN_DIR/corr_${correlation}_seed_${seed}.png" \
            --quiet \
            "$@" >"$RUN_DIR/corr_${correlation}_seed_${seed}.log" 2>&1 &
        pids+=("$!")
        gpu=$(( (gpu + 1) % NUM_GPUS ))
    done
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

python "$ROOT_DIR/exp-spurious_correlations_cifar10.py" \
    --probe-eval-split train \
    --aggregate-metrics "$RUN_DIR"/corr_*_seed_*.json \
    --output "$COMBINED_OUTPUT"
