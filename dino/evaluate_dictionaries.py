# ======================================================================================
# 1. Imports
# ======================================================================================
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 2. Model Definition
# ======================================================================================
def topk_sparsify(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Applies Top-K sparsification to the input tensor along the feature dimension.
    This function is part of the model architecture but not directly used in the
    comparison logic itself.
    """
    with torch.no_grad():
        _, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
    return x * mask


class TopKAutoencoder(nn.Module):
    """
    A sparse autoencoder that uses Top-K sparsification in its latent space.
    This class must match the one used for training to load the models correctly.
    """

    def __init__(self, input_dim: int, hidden_dim: int, topk: int):
        """
        Initializes the TopKAutoencoder.
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = topk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the autoencoder.
        """
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        z = topk_sparsify(activated, self.topk)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ======================================================================================
# 3. Helper Functions
# ======================================================================================
def load_model(model_path: Path, device: torch.device) -> TopKAutoencoder:
    """
    Loads a trained TopKAutoencoder model from a state dictionary.

    It infers the model's dimensions from the state dictionary keys.

    Args:
        model_path (Path): The path to the saved .pt model file.
        device (torch.device): The device to load the model onto.

    Returns:
        TopKAutoencoder: The loaded and initialized model.
    """
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Infer dimensions from the state dictionary
    input_dim = state_dict["decoder.weight"].shape[0]
    hidden_dim = state_dict["decoder.weight"].shape[1]
    
    model = TopKAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, topk=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded with Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")
    return model


def get_similarity_scores(model_v2: TopKAutoencoder, model_v3: TopKAutoencoder) -> np.ndarray:
    """
    Performs cosine similarity comparison and returns the best-match scores.

    Args:
        model_v2 (TopKAutoencoder): The model trained on DINOv2 features.
        model_v3 (TopKAutoencoder): The model trained on DINOv3 features.

    Returns:
        np.ndarray: An array of the highest cosine similarity score for each v2 feature.
    """
    print("Calculating similarity scores...")
    # --- Step 1: Extract and Normalize Dictionaries ---
    # The decoder weights are our feature dictionaries. Transpose to get (hidden_dim, input_dim).
    dict_v2 = model_v2.decoder.weight.T.detach()
    dict_v3 = model_v3.decoder.weight.T.detach()

    # Normalize each feature vector to have a unit norm (L2 norm = 1)
    dict_v2_norm = F.normalize(dict_v2, p=2, dim=1)
    dict_v3_norm = F.normalize(dict_v3, p=2, dim=1)

    # --- Step 2: Find Best Matches for each v2 feature in v3 ---
    # Cosine similarity between normalized vectors is their dot product.
    similarity_matrix = torch.matmul(dict_v2_norm, dict_v3_norm.T)

    # For each v2 feature (row), find the max similarity value
    best_matches_sim, _ = torch.max(similarity_matrix, dim=1)

    return best_matches_sim.cpu().numpy()


def plot_comparison_histogram(
    similarity_scores: Dict[str, np.ndarray],
    tokens: str,
    expansion: int,
    topk: int,
    output_dir: Path
):
    """
    Plots the distributions of similarity scores for different activation types
    on the same axes and prints summary statistics.

    Args:
        similarity_scores (Dict[str, np.ndarray]): A dictionary mapping activation
                                                   type to its similarity scores.
        tokens (str): The token type used (e.g., 'cls').
        expansion (int): The expansion factor of the autoencoder.
        topk (int): The top-k value for sparsity.
        output_dir (Path): Directory to save the plot.
    """
    print("\n--- Generating Combined Similarity Distribution Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(5, 4))
    
    colors = {'centroids': plt.cm.winter(0.25), 'latents': plt.cm.autumn(0.25)}

    for activation_type, scores in similarity_scores.items():
        if scores is None:
            continue

        # --- Analyze and Report Results ---
        mean_sim = scores.mean()
        median_sim = np.median(scores)
        std_sim = scores.std()

        print(f"\n--- Statistics for '{activation_type}' ---")
        print(f"Mean best-match similarity:   {mean_sim:.4f}")
        print(f"Median best-match similarity: {median_sim:.4f}")
        print(f"Std Dev of similarity:        {std_sim:.4f}")
        
        # --- Plotting ---
        sns.histplot(
            scores,
            kde=True,
            ax=ax,
            bins=50,
            stat="density",
            label=f'{activation_type.capitalize()}',
            color=colors.get(activation_type, 'gray'),
            alpha=0.6
        )

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{tokens}_comparison_{expansion}exp_{topk}K.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nCombined plot saved to {plot_path}")
    plt.show()


# ======================================================================================
# 4. Main Execution
# ======================================================================================
if __name__ == "__main__":
    # --- Configuration ---
    TOKENS = 'cls'
    EXPANSION = 10
    TOPK = 32
    ACTIVATION_TYPES = ['centroids', 'latents']
    BASE_OUTPUT_DIR = Path("outputs/dino")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Data Collection Loop ---
    all_similarity_scores = {}

    for activation in ACTIVATION_TYPES:
        print(f"--- Processing activation type: {activation.upper()} ---")
        
        # Construct model paths dynamically
        model_v2_path = BASE_OUTPUT_DIR / f'v2/v2-{TOKENS}-{activation}-{EXPANSION}exp-{TOPK}K-0.pt'
        model_v3_path = BASE_OUTPUT_DIR / f'v3/v3-{TOKENS}-{activation}-{EXPANSION}exp-{TOPK}K-0.pt'

        try:
            # Load the two models
            model_v2 = load_model(model_v2_path, device)
            model_v3 = load_model(model_v3_path, device)

            # Get similarity scores and store them
            with torch.no_grad():
                scores = get_similarity_scores(model_v2, model_v3)
                all_similarity_scores[activation] = scores
                print(f"Successfully processed and stored scores for '{activation}'.\n")

        except FileNotFoundError as e:
            print(f"Warning: Could not find a model file. Skipping '{activation}'.")
            print(f"Error details: {e}\n")
            all_similarity_scores[activation] = None

    # --- Plotting ---
    # Check if we have any data to plot before proceeding
    if any(scores is not None for scores in all_similarity_scores.values()):
        plot_comparison_histogram(
            similarity_scores=all_similarity_scores,
            tokens=TOKENS,
            expansion=EXPANSION,
            topk=TOPK,
            output_dir=BASE_OUTPUT_DIR
        )
    else:
        print("No data was successfully loaded. Exiting without plotting.")