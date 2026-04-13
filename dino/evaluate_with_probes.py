# ======================================================================================
# 1. Imports
# ======================================================================================
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        activated = F.relu(encoded)
        z = topk_sparsify(activated, self.topk)
        reconstructed = self.decoder(z)
        return reconstructed, z


class LinearProbe(nn.Module):
    """A simple linear classifier for probing representations."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# ======================================================================================
# 3. Core Logic Functions
# ======================================================================================

def get_dataloaders(
    base_data_path: Path, token_type: str, activation_type: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Loads pre-computed features and creates DataLoaders."""
    train_path = base_data_path / f"train-{token_type}"
    test_path = base_data_path / f"test-{token_type}"

    train_inputs = torch.load(train_path / f"{activation_type}.pt", weights_only=True)
    train_labels = torch.load(train_path / "labels.pt", weights_only=True)
    test_inputs = torch.load(test_path / f"{activation_type}.pt", weights_only=True)
    test_labels = torch.load(test_path / "labels.pt", weights_only=True)

    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_inputs.size(1)
    num_classes = len(torch.unique(train_labels))
    
    return train_loader, test_loader, input_dim, num_classes


@torch.no_grad()
def extract_latents(
    autoencoder: TopKAutoencoder, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts sparse latent representations from the autoencoder for a given dataset."""
    autoencoder.eval()
    all_latents, all_labels = [], []
    for x, y in tqdm(loader, desc="Extracting Latents"):
        x = x.to(device)
        _, latents = autoencoder(x)
        all_latents.append(latents.cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_latents), torch.cat(all_labels)


def train_and_evaluate_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> float:
    """Trains a linear probe on the features and evaluates its accuracy."""
    input_dim = train_features.shape[1]
    probe = LinearProbe(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- Training ---
    probe.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = probe(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def plot_results(stats: Dict[str, Dict[int, float]], output_path: Path):
    """Generates and saves a plot of the probe accuracies."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    
    colors = {"Latents": plt.cm.autumn(0.25), "Centroids": plt.cm.winter(0.25)}
    
    for label, data in stats.items():
        if not data: continue
        keys = sorted(data.keys())
        values = [data[k] for k in keys]
        ax.plot(keys, values, marker='o', linestyle='-', color=colors.get(label, 'gray'), label=label)

    ax.set_xlabel("Sparsity (K)", fontsize=12)
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xticks(list(next(iter(stats.values())).keys()))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")

# ======================================================================================
# 4. Main Execution
# ======================================================================================
def main():
    """Main function to run the linear probing experiments."""
    parser = argparse.ArgumentParser(description="Evaluate sparse autoencoders with linear probes.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dino"), help="Directory to save plots and results.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "probe_accuracy_stats.pt"

    # Load existing stats if available, otherwise initialize
    if stats_path.exists():
        print(f"Loading cached stats from {stats_path}")
        stats = torch.load(stats_path)
    else:
        print("No cached stats found. Starting from scratch.")
        stats = {"Latents": {}, "Centroids": {}}

    # Define the experiment runs to evaluate
    run_configs = [
        {'model_type': 'v2', 'tokens': 'all', 'activation_type': 'latents', 'expansion': 10, 'topk': k, 'seed': 0}
        for k in [8, 16, 32, 64]
    ] + [
        {'model_type': 'v2', 'tokens': 'all', 'activation_type': 'centroids', 'expansion': 10, 'topk': k, 'seed': 0}
        for k in [8, 16, 32, 64]
    ]

    needs_saving = False
    for config in tqdm(run_configs, desc="Evaluating Runs"):
        run_name = (
            f"{config['model_type']}-{config['tokens']}-{config['activation_type']}-"
            f"{config['expansion']}exp-{config['topk']}K-{config['seed']}"
        )
        model_path = args.output_dir / config['model_type'] / f"{run_name}.pt"
        
        # Check cache before proceeding
        stats_key = config['activation_type'].capitalize()
        if config['topk'] in stats.get(stats_key, {}):
            acc = stats[stats_key][config['topk']]
            print(f"Result for {run_name} found in cache. Accuracy: {acc:.2f}%. Skipping.")
            continue

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
        
        needs_saving = True  # A new value will be computed, so we need to save.

        # 1. Load data and model
        train_loader, test_loader, input_dim, num_classes = get_dataloaders(
            args.output_dir / config['model_type'], config['tokens'], config['activation_type'], batch_size=512
        )
        
        hidden_dim = input_dim * config['expansion']
        autoencoder = TopKAutoencoder(input_dim, hidden_dim, config['topk']).to(args.device)
        autoencoder.load_state_dict(torch.load(model_path, map_location=args.device))

        # 2. Extract sparse latents
        train_latents, train_labels = extract_latents(autoencoder, train_loader, args.device)
        test_latents, test_labels = extract_latents(autoencoder, test_loader, args.device)

        # 3. Train and evaluate the probe
        accuracy = train_and_evaluate_probe(
            train_latents, train_labels, test_latents, test_labels, num_classes, args.device
        )
        
        print(f"Run: {run_name} | Probe Accuracy: {accuracy:.2f}%")
        
        # Store result for plotting, ensuring the nested dictionary exists
        stats.setdefault(stats_key, {})[config['topk']] = accuracy

    # 4. Save and plot the final results
    if needs_saving:
        print(f"\nSaving updated stats to {stats_path}...")
        torch.save(stats, stats_path)
    else:
        print("\nNo new results computed. Stats file is already up-to-date.")

    plot_results(stats, args.output_dir / "probe_accuracy_vs_sparsity.png")


if __name__ == "__main__":
    main()