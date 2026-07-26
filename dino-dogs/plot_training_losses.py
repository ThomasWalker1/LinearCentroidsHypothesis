# ======================================================================================
# Plots the SAE reconstruction-loss traces written by dino-dogs/training.py.
#
# The two dictionaries decompose different objects (centroids vs. latents) whose entries
# differ in scale, so absolute MSE is not comparable between the arms. Each curve is
# therefore normalised by its own first-epoch loss, which shows that both SAEs converge
# by the same amount over training -- i.e. neither arm is advantaged by being better fit.
# ======================================================================================
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "centroids": dict(label="LCH", color=plt.cm.winter(0.25), linestyle="-"),
    "latents": dict(label="LRH", color=plt.cm.autumn(0.25), linestyle="--"),
}


def main():
    parser = argparse.ArgumentParser(description="Plot normalised SAE training loss traces.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dino-dogs"))
    parser.add_argument("--model_type", type=str, default="v2")
    parser.add_argument("--tokens", type=str, default="cls")
    parser.add_argument("--expansion", type=int, default=10)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--sae_seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    run_dir = args.output_dir / args.model_type
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

    for activation_type, style in STYLES.items():
        curves = []
        for seed in args.sae_seeds:
            path = run_dir / (
                f"{args.model_type}-{args.tokens}-{activation_type}-"
                f"{args.expansion}exp-{args.topk}K-{seed}-losses.json"
            )
            if not path.exists():
                print(f"Missing {path} -- rerun dino-dogs/training.py for '{activation_type}'.")
                continue
            epoch_loss = np.array(json.loads(path.read_text())["epoch_loss"])
            curves.append(epoch_loss / epoch_loss[0])

        if not curves:
            continue
        curves = np.stack(curves)
        epochs = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        ax.plot(epochs, mean, **style)
        ax.fill_between(epochs, curves.min(axis=0), curves.max(axis=0), color=style["color"], alpha=0.2)
        print(f"{activation_type:>10}: final loss {mean[-1]:.3f} of initial (mean over {len(curves)} seeds)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss / first-epoch loss")
    ax.grid(True, linestyle="dashed", color="gray", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = args.output_dir / f"sae_training_loss-{args.topk}K.png"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
