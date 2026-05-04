# ======================================================================================
# 1. Imports
# ======================================================================================
import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
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
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        z = topk_sparsify(activated, self.topk)
        reconstructed = self.decoder(z)
        return reconstructed, z

# ======================================================================================
# 3. Core Logic Functions
# ======================================================================================

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculates the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 1.0


@torch.no_grad()
def get_active_sae_features(sae_model: TopKAutoencoder, inputs: torch.Tensor) -> List[Set[int]]:
    """Gets the indices of active (non-zero) features from the SAE for a batch."""
    sae_model.eval()
    device = next(sae_model.parameters()).device
    inputs = inputs.to(device)
    _, z = sae_model(inputs)
    return [set(torch.where(act > 1e-8)[0].cpu().numpy()) for act in z]


@torch.no_grad()
def find_similar_decompositions(
    sae_model: TopKAutoencoder, target_input: torch.Tensor, data_loader: DataLoader
) -> List[Tuple[float, int]]:
    """
    Finds items in a dataset with the most similar SAE feature decompositions
    to a target input, based on Jaccard similarity.
    """
    if target_input.dim() == 1:
        target_input = target_input.unsqueeze(0)
    
    target_active_set = get_active_sae_features(sae_model, target_input)[0]

    all_results = []
    absolute_idx = 0
    for batch_inputs, _ in tqdm(data_loader, desc="Finding similar SAE decompositions", leave=False):
        batch_active_sets = get_active_sae_features(sae_model, batch_inputs)
        for current_active_set in batch_active_sets:
            similarity = jaccard_similarity(target_active_set, current_active_set)
            all_results.append((similarity, absolute_idx))
            absolute_idx += 1
            
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results


def analyze_similarity_ratio(
    config: argparse.Namespace, activation_type: str, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Performs the core analysis for a given activation type (latents or centroids).
    
    Returns:
        A tuple of (SAE similarity scores, Original Model similarity scores, similar_ids).
    """
    run_name = (
        f"{config.model_type}-{config.tokens}-{activation_type}-"
        f"{config.expansion}exp-{config.topk}K-{config.seed}"
    )
    
    # --- Load Data and Models ---
    feature_path = config.output_dir / config.model_type / f"train-{config.tokens}"
    model_path = config.output_dir / config.model_type / f"{run_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"SAE model not found at {model_path}. Please train it first.")

    features = torch.load(feature_path / f"{activation_type}.pt", weights_only=True)
    labels = torch.load(feature_path / "labels.pt", weights_only=True)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    input_dim = features.size(1)
    hidden_dim = input_dim * config.expansion
    sae_model = TopKAutoencoder(input_dim, hidden_dim, config.topk)
    sae_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    sae_model.to(device)

    # --- Find Similar Decompositions based on SAE ---
    target_feature = features[config.target_id]
    similar_decompositions = find_similar_decompositions(sae_model, target_feature, loader)
    sae_jdists = np.array([v[0] for v in similar_decompositions])
    similar_ids = [v[1] for v in similar_decompositions]

    # --- Calculate Similarity in Original Model's Activation Space ---
    block = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768))
    block.load_state_dict(torch.load(config.output_dir / config.model_type / "block.pth", weights_only=True))
    block.to(device)
    block.eval()

    block_inputs = torch.load(config.output_dir / config.model_type / f"train-{config.tokens}/inputs.pt", weights_only=True)
    
    @torch.no_grad()
    def get_original_model_active_neurons(inputs_tensor: torch.Tensor) -> Set[int]:
        """Helper to get active neurons in the DINOv2 MLP block's hidden layer."""
        activations = block[:2](inputs_tensor.to(device))
        return set(torch.where(activations > 0)[0].cpu().numpy())

    target_input_for_block = block_inputs[config.target_id]
    target_active_neurons = get_original_model_active_neurons(target_input_for_block)
    
    similar_inputs_for_block = block_inputs[similar_ids]
    
    model_jdists = []
    for p in tqdm(similar_inputs_for_block, desc="Calculating original model similarity", leave=False):
        similar_active_neurons = get_original_model_active_neurons(p)
        model_jdists.append(jaccard_similarity(target_active_neurons, similar_active_neurons))
        
    return sae_jdists, np.array(model_jdists), similar_ids


def get_image_subset(data_root: str) -> Subset:
    """
    Recreates the exact dataset subset used during feature generation to map indices
    back to the original PIL images for visualization.
    """
    # We apply visual transforms only (no ToTensor/Normalize) to keep them as PIL images
    visual_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    dataset = torchvision.datasets.ImageNet(
        data_root, split="train", transform=visual_transform
    )
    
    # Set seed for reproducibility before extracting classes
    random.seed(0)
    
    # ImageNet classes 151 through 268 (inclusive) are domestic dogs. 
    # Randomly sample 10 from this specific range.
    dog_classes = list(range(151, 269))
    selected_classes = set(random.sample(dog_classes, 10))
    
    # Optional but helpful: Get the human-readable names of the breeds
    selected_class_names = [dataset.classes[i] for i in selected_classes]
    
    train_indices = [
        i for i, target in enumerate(dataset.targets) if target in selected_classes
    ]

    return Subset(dataset, train_indices)


def plot_image_grid(target_idx: int, similar_ids: List[int], sae_jdists: np.ndarray, dataset: Subset, output_path: Path):
    """
    Plots a 3x3 grid with the target image in the center and the top 8 most similar 
    images surrounding it.
    """
    # Filter out the target ID just in case it appears in the top results, 
    # then take the top 8
    top_8_indices = [i for i in range(len(similar_ids)) if similar_ids[i] != target_idx][:8]
    top_8_ids = [similar_ids[i] for i in top_8_indices]
    top_8_scores = [sae_jdists[i] for i in top_8_indices]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=200)
    
    # Grid coordinates surrounding the center (1,1)
    surrounding_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    
    # Plot Center (Target)
    target_img, target_label = dataset[target_idx]
    axes[1, 1].imshow(target_img)
    axes[1, 1].axis("off")
    
    # Plot Surrounding (Top 8 Similar)
    for pos, idx, score in zip(surrounding_positions, top_8_ids, top_8_scores):
        img, label = dataset[idx]
        axes[pos[0], pos[1]].imshow(img)
        axes[pos[0], pos[1]].axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"3x3 image grid saved to {output_path}")

# ======================================================================================
# 4. Main Execution
# ======================================================================================
def main():
    """Main function to run the similarity analysis."""
    parser = argparse.ArgumentParser(description="Analyze similarity ratios between SAE and original model spaces.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dino-dogs"), help="Directory to save plots.")
    parser.add_argument("--data_root", type=str, default="./data", help="Path to ImageNet dataset.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    
    # Experiment-specific arguments
    parser.add_argument("--model_type", type=str, default="v2", help="Model type/expansion factor.")
    parser.add_argument("--expansion", type=int, default=10, help="Expansion factor of the hidden layer.")
    parser.add_argument("--topk", type=int, default=32, help="Number of active neurons (K).")
    parser.add_argument("--tokens", type=str, default="cls", help="Token type used.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the model run.")
    parser.add_argument("--target_id", type=int, default=4, help="Index of the target sample in the dataset.")
    
    args = parser.parse_args()
    #args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'grid_latents').mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'grid_centroids').mkdir(parents=True, exist_ok=True)
    # --- Load image dataset for visualization ---
    print("Loading image dataset for grid plotting...")
    img_dataset = get_image_subset(args.data_root)
    for target_id in np.linspace(0,len(img_dataset)-1,16,dtype=int):
        args.target_id=target_id
        print("\n--- Analyzing Latents ---")
        l_sae_sim, l_model_sim, l_similar_ids = analyze_similarity_ratio(args, "latents", args.device)
        
        # --- Analyze Centroids ---
        print("\n--- Analyzing Centroids ---")
        c_sae_sim, c_model_sim, c_similar_ids = analyze_similarity_ratio(args, "centroids", args.device)

        # --- Plot 3x3 Image Grids ---
        print("\n--- Generating 3x3 Image Grids ---")
        latent_grid_filename = f"grid_latents/target{args.target_id}_{args.expansion}exp_{args.topk}K.png"
        plot_image_grid(args.target_id, l_similar_ids, l_sae_sim, img_dataset, args.output_dir / latent_grid_filename)
        
        centroid_grid_filename = f"grid_centroids/target{args.target_id}_{args.expansion}exp_{args.topk}K.png"
        plot_image_grid(args.target_id, c_similar_ids, c_sae_sim, img_dataset, args.output_dir / centroid_grid_filename)


if __name__ == "__main__":
    main()