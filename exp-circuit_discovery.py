# -*- coding: utf-8 -*-
"""
This script performs circuit discovery on a single MLP layer of a GPT-2 model.
It identifies important neurons by measuring the change in the feature centroid's
geometry when each neuron is individually ablated (removed). The script then
calculates an "attribution value" for each neuron and visualizes the results.
"""

# ======================================================================================
# 1. Imports
# ======================================================================================
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from centroids import Centroids


# ======================================================================================
# 1. Utility Functions
# ======================================================================================
def convert_mlp_to_sequential(mlp_module: nn.Module) -> nn.Sequential:
    """
    Converts a custom MLP module (like in nanoGPT) to a standard nn.Sequential.
    This makes it easier to manipulate and analyze individual layers.

    Args:
        mlp_module (nn.Module): The custom MLP module with `c_fc`, `gelu`, and `c_proj`.

    Returns:
        nn.Sequential: A standard PyTorch sequential model with the same weights.
    """
    return nn.Sequential(
        copy.deepcopy(mlp_module.c_fc),
        copy.deepcopy(mlp_module.gelu),
        copy.deepcopy(mlp_module.c_proj),
    )


def sample_neighborhood(x: torch.Tensor, radius: float, n_samples: int) -> torch.Tensor:
    """
    Samples points uniformly from a hypersphere of a given radius around a center point x.

    Args:
        x (torch.Tensor): The center point of the hypersphere (1D tensor).
        radius (float): The radius of the hypersphere.
        n_samples (int): The total number of samples to generate (including the center).

    Returns:
        torch.Tensor: A tensor of shape (n_samples, dim) containing the sampled points.
    """
    if x.dim() != 1:
        raise ValueError("Input tensor x must be a 1D vector.")
    if n_samples < 1:
        raise ValueError("Number of samples must be at least 1.")

    dim = x.shape[0]
    # Sample N-1 random directions on a unit hypersphere
    directions = torch.randn(n_samples - 1, dim, device=x.device)
    directions /= directions.norm(dim=1, keepdim=True)

    # Sample radii uniformly within the hypersphere volume
    u = torch.rand(n_samples - 1, 1, device=x.device)
    scales = u.pow(1.0 / dim) * radius
    samples = x + scales * directions

    # Add the original point and shuffle
    samples = torch.cat([samples, x.unsqueeze(0)], dim=0)
    return samples[torch.randperm(n_samples, device=x.device)]


def remove_neuron(
    mlp_sequential: nn.Sequential, layer_idx: int, neuron_idx: int
) -> nn.Sequential:
    """
    Creates a new nn.Sequential model with a specified neuron removed from an MLP.
    It correctly adjusts the weights of the affected layer and the subsequent layer.

    Args:
        mlp_sequential (nn.Sequential): The MLP model (must have Linear -> Act -> Linear structure).
        layer_idx (int): The index of the Linear layer from which to remove the neuron (e.g., 0).
        neuron_idx (int): The index of the neuron to remove.

    Returns:
        nn.Sequential: A new model with the neuron ablated.
    """
    if not (
        isinstance(mlp_sequential[0], nn.Linear)
        and isinstance(mlp_sequential[2], nn.Linear)
    ):
        raise TypeError("Expected MLP structure: Linear -> Activation -> Linear")

    # Deepcopy to avoid modifying the original model
    new_model = copy.deepcopy(mlp_sequential)
    
    # --- Step 1: Modify the first linear layer (remove output neuron) ---
    old_fc_layer = new_model[layer_idx]
    new_out_features = old_fc_layer.out_features - 1
    
    new_fc_layer = nn.Linear(
        old_fc_layer.in_features, new_out_features, bias=old_fc_layer.bias is not None
    )
    
    # Copy weights, excluding the row for the removed neuron
    new_fc_layer.weight.data = torch.cat([
        old_fc_layer.weight.data[:neuron_idx, :],
        old_fc_layer.weight.data[neuron_idx + 1:, :]
    ], dim=0)
    if old_fc_layer.bias is not None:
        new_fc_layer.bias.data = torch.cat([
            old_fc_layer.bias.data[:neuron_idx],
            old_fc_layer.bias.data[neuron_idx + 1:]
        ], dim=0)

    # --- Step 2: Modify the second linear layer (remove input connection) ---
    old_proj_layer = new_model[layer_idx + 2]
    new_in_features = old_proj_layer.in_features - 1

    new_proj_layer = nn.Linear(
        new_in_features, old_proj_layer.out_features, bias=old_proj_layer.bias is not None
    )

    # Copy weights, excluding the column corresponding to the removed neuron's input
    new_proj_layer.weight.data = torch.cat([
        old_proj_layer.weight.data[:, :neuron_idx],
        old_proj_layer.weight.data[:, neuron_idx + 1:]
    ], dim=1)
    if old_proj_layer.bias is not None:
        new_proj_layer.bias.data.copy_(old_proj_layer.bias.data)

    # Replace the layers in the new model
    new_model[layer_idx] = new_fc_layer
    new_model[layer_idx + 2] = new_proj_layer
    
    return new_model


def calculate_centroid_delta(
    original_func: nn.Module,
    pruned_func: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Calculates the normalized mean squared error between the centroids of an
    original function and a pruned version of it.

    Args:
        original_func (nn.Module): The original model.
        pruned_func (nn.Module): The model with a neuron removed.
        sample (torch.Tensor): The input sample to compute centroids for.
        device (torch.device): The computation device.

    Returns:
        float: The normalized attribution value.
    """
    original_func.to(device)
    pruned_func.to(device)
    
    centroids_orig = Centroids(original_func)(sample).get_centroids()
    centroids_pruned = Centroids(pruned_func)(sample).get_centroids()

    # Normalize by the norm of the original centroids to make the delta comparable
    delta = (centroids_orig - centroids_pruned).square().mean()
    norm = centroids_orig.norm()
    
    return (delta / norm).item() if norm > 1e-8 else 0.0


def plot_attributions(
    attribution_values: np.ndarray, highlighted_neuron: int, output_path: Path
):
    """
    Generates and saves a scatter plot of neuron attribution values.

    Args:
        attribution_values (np.ndarray): The calculated attribution for each neuron.
        highlighted_neuron (int): The index of a specific neuron to highlight.
        output_path (Path): The path to save the plot image.
    """
    normalized_values = attribution_values / np.max(attribution_values)
    highlight_val = normalized_values[highlighted_neuron]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.scatter(
        range(len(normalized_values)),
        normalized_values,
        color=plt.cm.autumn(0.25),
        s=5,
        alpha=0.8,
    )
    ax.scatter(
        [highlighted_neuron],
        [highlight_val],
        color="black",
        s=20,
        zorder=5,
    )
    ax.annotate(
        f"Neuron {highlighted_neuron}",
        xy=(highlighted_neuron, highlight_val),
        xytext=(highlighted_neuron + 50, highlight_val + 0.2),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.0),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8),
    )
    ax.set_ylabel("Normalized Attribution Value", fontsize=14)
    ax.set_xlabel("Neuron Index", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(size=8,labelsize=12)
    
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Plot saved to {output_path}")


# ======================================================================================
# 2. Main Execution
# ======================================================================================
def main():
    """Main script execution."""
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "gpt2-large"
    TARGET_LAYER_IDX = 31
    PROMPT = "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked"
    NEIGHBORHOOD_RADIUS = 0.25
    N_SAMPLES = 256
    HIGHLIGHTED_NEURON = 892

    # --- Setup Output Directory and Paths ---
    output_dir = Path("outputs/circuits")
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution_path = output_dir / "attribution_values.npy"
    plot_path = output_dir / "attribution_values.png"

    # --- Load Model and Tokenizer ---
    try:
        from gpt2 import GPT
        model = GPT.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
    except ImportError:
        print("Warning: `model.py` not found. Using Hugging Face's GPT2LMHeadModel as a placeholder.")
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)

    # --- Attribution Calculation ---
    if attribution_path.exists():
        print(f"Loading pre-computed attribution values from {attribution_path}")
        attribution_values = np.load(attribution_path)
    else:
        print("Calculating neuron attribution values...")
        # Extract the target MLP layer and convert it to a sequential model
        target_mlp_module = model.transformer.h[TARGET_LAYER_IDX].mlp
        func = convert_mlp_to_sequential(target_mlp_module).to(DEVICE)

        # Get the input embedding for the target MLP layer
        if hasattr(model, 'forward_with_hook'):
             with torch.no_grad():
                _, hidden_states = model.forward_with_hook(tokens, TARGET_LAYER_IDX)
        else:
            print("Warning: `forward_with_hook` not found. Using a forward hook to get intermediate activations.")
            activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output[0].detach()
                return hook
            
            hook_handle = model.transformer.h[TARGET_LAYER_IDX].register_forward_hook(get_activation('mlp_input'))
            with torch.no_grad():
                model(tokens)
            hook_handle.remove()
            hidden_states = activations['mlp_input']

        # Use the embedding of the last token as the point of interest
        embedding = hidden_states[:, -1, :].squeeze(0)

        # Sample points in the neighborhood of this embedding
        neighborhood_sample = sample_neighborhood(
            embedding.cpu(), NEIGHBORHOOD_RADIUS, N_SAMPLES
        )

        attribution_values = []
        num_neurons = func[0].out_features
        for neuron_idx in tqdm(range(num_neurons), desc="Attributing Neurons"):
            func_pruned = remove_neuron(func, 0, neuron_idx)
            delta = calculate_centroid_delta(func, func_pruned, neighborhood_sample, DEVICE)
            attribution_values.append(delta)

        attribution_values = np.array(attribution_values)
        np.save(attribution_path, attribution_values)
        print(f"Attribution values saved to {attribution_path}")

    # --- Analysis and Visualization ---
    plot_attributions(attribution_values, HIGHLIGHTED_NEURON, plot_path)
    
    # Print percentile of the highlighted neuron
    percentile = np.mean(attribution_values <= attribution_values[HIGHLIGHTED_NEURON]) * 100
    print(f"Neuron {HIGHLIGHTED_NEURON}'s attribution value is at the {percentile:.2f}th percentile.")


if __name__ == "__main__":
    main()