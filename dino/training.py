# -*- coding: utf-8 -*-
"""
This script trains a sparse autoencoder on features extracted from a DINOv2 model.
The autoencoder learns a compressed, sparse representation of the input features
(either latents or centroids) using a Top-K activation mechanism.

The script is designed to be run from the command line, allowing for easy
experimentation with different hyperparameters.
"""

# ======================================================================================
# 1. Imports
# ======================================================================================
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb as wb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ======================================================================================
# 2. Model & Helper Functions
# ======================================================================================
def set_seed(seed: int):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def topk_sparsify(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Applies Top-K sparsification to the input tensor along the feature dimension.

    Only the top 'k' activations for each item in the batch are kept, while the
    rest are zeroed out.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_features).
        k (int): The number of top activations to keep.

    Returns:
        torch.Tensor: The sparsified tensor with the same shape as the input.
    """
    with torch.no_grad():
        # Find the values and indices of the top k elements in dim 1
        _, indices = torch.topk(x, k, dim=1)
        mask = torch.zeros_like(x)
        # Create a mask by scattering 1s at the locations of the top k indices
        mask.scatter_(1, indices, 1.0)
    return x * mask


class TopKAutoencoder(nn.Module):
    """
    A sparse autoencoder that uses Top-K sparsification in its latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, topk: int):
        """
        Initializes the TopKAutoencoder.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden (latent) space.
            topk (int): The number of neurons to keep active in the latent space.
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = topk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the autoencoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed output tensor.
                - The sparse latent representation (z).
        """
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        z = topk_sparsify(activated, self.topk)
        reconstructed = self.decoder(z)
        return reconstructed, z


# ======================================================================================
# 3. Data Loading & Training
# ======================================================================================
def get_dataloaders(
    base_data_path: Path,
    token_type: str,
    activation_type: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Loads the pre-extracted features and creates PyTorch DataLoaders.

    Args:
        base_data_path (Path): The path to the directory containing the extracted features.
        token_type (str): The type of tokens used ('cls' or 'all').
        activation_type (str): The type of features ('latents' or 'centroids').
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): The number of worker processes for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, int]: A tuple containing:
            - The training DataLoader.
            - The testing DataLoader.
            - The input dimension of the features.
    """
    print("Loading pre-computed features...")
    train_path = base_data_path / f"train-{token_type}"
    test_path = base_data_path / f"test-{token_type}"

    train_inputs = torch.load(train_path / f"{activation_type}.pt", weights_only=True)
    test_inputs = torch.load(test_path / f"{activation_type}.pt", weights_only=True)

    # Labels are not used for training but are loaded to maintain dataset structure
    train_labels = torch.load(train_path / "labels.pt", weights_only=True)
    test_labels = torch.load(test_path / "labels.pt", weights_only=True)

    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    input_dim = train_inputs.size(1)
    return train_loader, test_loader, input_dim


def train(config: argparse.Namespace, device: torch.device):
    """
    The main training function.

    Args:
        config (argparse.Namespace): An object containing all hyperparameters.
        device (torch.device): The device to train the model on.
    """
    # --- Setup ---
    set_seed(config.seed)
    run_name = (
        f"{config.model_name}-{config.tokens}-{config.activation_type}-"
        f"{config.expansion}exp-{config.topk}K-{config.seed}"
    )
    run = wb.init(project="centroid_affinity_dino", config=config, name=run_name)
    
    run_dir=Path(config.base_dir) / config.model_name
    model_save_path = run_dir / f"{run_name}.pt"

    # --- Data and Model ---
    train_loader, _, input_dim = get_dataloaders(
        base_data_path=run_dir,
        token_type=config.tokens,
        activation_type=config.activation_type,
        batch_size=config.batch_size,
    )
    
    hidden_dim = input_dim * config.expansion
    model = TopKAutoencoder(input_dim, hidden_dim, config.topk).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    print(f"Starting training for run: {run_name}")
    pbar = tqdm(range(config.epochs), desc="Training Epochs")
    total_steps = 0
    for _ in pbar:
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed_x, _ = model(x)
            loss = criterion(reconstructed_x, x)
            loss.backward()
            optimizer.step()

            if total_steps % 100 == 0:
                wb.log({"step": total_steps, "loss": loss.item()})
            
            total_steps += 1
        
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    # --- Save Model and Log to WandB ---
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    artifact = wb.Artifact(name=f"{run_name}_model", type="model")
    artifact.add_file(str(model_save_path))
    run.log_artifact(artifact)
    run.finish()


# ======================================================================================
# 4. Main Execution
# ======================================================================================
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on DINOv2 Features")
    
    parser.add_argument("--base_dir", type=str, default="outputs/dino")
    parser.add_argument("--model_name", type=str, default="v2", choices=["v2","v3"])
    parser.add_argument("--tokens", type=str, default="cls", choices=["cls", "all"])
    parser.add_argument("--activation_type", type=str, choices=["latents", "centroids"])
    parser.add_argument("--expansion", type=int, default=10)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in ['v2','v3']:
        args.model_name=model_name

        args.tokens='all'
        for activation_type in ['centroids','latents']:
            args.activation_type=activation_type
            for topk in [8,16,32,64]:
                args.topk=topk
                train(args,device)

        args.tokens='cls'
        args.topk=32
        for activation_type in ['centroids','latents']:
            args.activation_type=activation_type
            train(args,device)
