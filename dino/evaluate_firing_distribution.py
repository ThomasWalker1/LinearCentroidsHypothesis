# ======================================================================================
# 1. Imports
# ======================================================================================
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ======================================================================================
# 2. Model Definitions
# ======================================================================================

def topk_sparsify(x: torch.Tensor, k: int) -> torch.Tensor:
    """Applies Top-K sparsification to the input tensor along the feature dimension."""
    if k <= 0:
        return torch.zeros_like(x)
    with torch.no_grad():
        _, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
    return x * mask


class TopKAutoencoder(nn.Module):
    """A sparse autoencoder that uses Top-K sparsification in its latent space."""
    def __init__(self, input_dim: int, hidden_dim: int, topk: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = topk
        self.hidden_dim = hidden_dim # Store hidden_dim for easy access

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        z = topk_sparsify(activated, self.topk)
        reconstructed = self.decoder(z)
        return reconstructed, z

# ======================================================================================
# 3. Core Logic Functions
# ======================================================================================

@torch.no_grad()
def get_active_feature_indices(
    sae_model: TopKAutoencoder, inputs: torch.Tensor
) -> List[Set[int]]:
    """
    Gets the indices of active (non-zero) features for each item in a batch.

    Args:
        sae_model (TopKAutoencoder): The sparse autoencoder model.
        inputs (torch.Tensor): The input batch tensor.

    Returns:
        List[Set[int]]: A list where each element is a set of active feature indices
                        for the corresponding item in the batch.
    """
    sae_model.eval()
    device = next(sae_model.parameters()).device
    inputs = inputs.to(device)
    
    _, z = sae_model(inputs)
    
    # Find indices where activation is greater than a small epsilon
    return [set(torch.where(act > 1e-8)[0].cpu().numpy()) for act in z]


def calculate_firing_counts(
    config: argparse.Namespace,
    activation_type: str,
    device: torch.device,
) -> np.ndarray:
    """
    Calculates the firing frequency for each neuron in a specified autoencoder.

    This function loads the appropriate dataset and pre-trained autoencoder,
    runs the data through the model to find active neurons, and counts their activations.
    It caches the results to a .npy file to speed up subsequent runs.
    """
    run_name = (
        f"{config.model_type}-{config.tokens}-{activation_type}-"
        f"{config.expansion}exp-{config.topk}K-{config.seed}"
    )
    
    # Define paths
    feature_path = config.output_dir / config.model_type / f"test-{config.tokens}"
    model_path = config.output_dir / config.model_type / f"{run_name}.pt"
    counts_cache_path = config.output_dir / f"{run_name}_feature_counts.npy"

    if counts_cache_path.exists():
        print(f"Loading cached firing counts for {activation_type} from {counts_cache_path}")
        return np.load(counts_cache_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    # Load data
    inputs = torch.load(feature_path / f"{activation_type}.pt", weights_only=True)
    labels = torch.load(feature_path / "labels.pt", weights_only=True)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Load model
    input_dim = inputs.size(1)
    hidden_dim = input_dim * config.expansion
    model = TopKAutoencoder(input_dim, hidden_dim=hidden_dim, topk=config.topk)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Calculate firing counts
    print(f"Calculating firing counts for {run_name}...")
    feature_counts = torch.zeros(model.hidden_dim, dtype=torch.long)
    for batch, _ in tqdm(loader, desc=f"Analyzing {activation_type}"):
        list_of_active_sets = get_active_feature_indices(model, batch)
        for active_set in list_of_active_sets:
            for feature_index in active_set:
                feature_counts[feature_index] += 1
    
    feature_counts_np = feature_counts.cpu().numpy()
    np.save(counts_cache_path, feature_counts_np)
    print(f"Saved firing counts to {counts_cache_path}")
    
    return feature_counts_np


def plot_distributions(
    latent_counts: np.ndarray,
    centroid_counts: np.ndarray,
    config: argparse.Namespace,
    output_path: Path,
):
    """Generates and saves a plot of the feature firing distributions."""
    l_active = latent_counts[latent_counts > 0]
    c_active = centroid_counts[centroid_counts > 0]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(
        range(len(l_active)), np.flip(np.sort(l_active)), 
        color=plt.cm.autumn(0.25), label=f'Latents ({len(l_active)} active)'
    )
    ax.plot(
        range(len(c_active)), np.flip(np.sort(c_active)), 
        color=plt.cm.winter(0.25), label=f'Centroids ({len(c_active)} active)'
    )
    
    ax.set_yscale("log")
    ax.set_xlabel("Feature Rank (Sorted by Frequency)", fontsize=12)
    ax.set_ylabel("Activation Frequency (Log Scale)", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to {output_path}")

# ======================================================================================
# 4. Main Execution
# ======================================================================================
def main():
    """Main function to run the firing distribution analysis."""
    parser = argparse.ArgumentParser(description="Analyze and plot feature firing distributions of sparse autoencoders.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dino"), help="Directory to save plots and results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    
    # Experiment-specific arguments
    parser.add_argument("--model_type", type=str, default="v2")
    parser.add_argument("--expansion", type=int, default=10)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--tokens", type=str, default="all", choices=["cls", "all"])
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Calculate counts for both latents and centroids ---
    latent_counts = calculate_firing_counts(args, "latents", args.device)
    centroid_counts = calculate_firing_counts(args, "centroids", args.device)
    
    # --- Plot the results ---
    plot_filename = (
        f"{args.tokens}-exp{args.expansion}-k{args.topk}-seed{args.seed}_firing_dist.png"
    )
    plot_distributions(latent_counts, centroid_counts, args, args.output_dir / plot_filename)


if __name__ == "__main__":
    main()
