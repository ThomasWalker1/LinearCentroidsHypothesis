#!/usr/bin/env bash

set -e

# Point this at a torchvision-compatible ImageNet root (train/, val/, meta.bin).
export IMAGENET_ROOT="${IMAGENET_ROOT:-./data}"

SEEDS="0 1 2"

# 1. Extract DINOv2 centroids and latents for the ten-dog-breed subset.
python dino-dogs/extraction.py

# 2. Train a TopK SAE on each, for several seeds so the comparison carries error bars.
for seed in $SEEDS; do
    for activation in centroids latents; do
        python dino-dogs/training.py \
            --base_dir outputs/dino-dogs \
            --activation_type "$activation" \
            --seed "$seed"
    done
done

# 3. Both arms converge by the same factor, so neither is advantaged by reconstruction fit.
python dino-dogs/plot_training_losses.py --sae_seeds $SEEDS

# 4. Quantify the Jaccard-neighbour claim: LCH retrieves neighbours in a more consistent
#    spatial configuration, scored by DINOv3 patch tokens compared position-wise.
python dino-dogs/evaluate_neighbor_coherence.py \
    --data_root "$IMAGENET_ROOT" \
    --split train \
    --n_queries 1000 \
    --sae_seeds $SEEDS

python dino/evaluate_with_probes.py
