"""Measure color information in CIFAR-10 ResNet-18 representations.

This is the CIFAR-10 counterpart to ``exp-spurious_correlations.py``.  A
synthetic tint is correlated with the CIFAR-10 class, then linear probes are
trained to predict that tint from either the penultimate activation or the
gradient of the predicted logit with respect to that activation.

Example:
    python exp-spurious_correlations_cifar10.py --epochs 5 --probe-epochs 10
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SpuriousColorCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 with a class-correlated multiplicative RGB tint."""

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=True,
        num_colors=10,
        correlation=0.0,
        seed=42,
    ):
        # Apply the transform after tinting, since CIFAR-10 transforms normally
        # expect a PIL image whereas tinting is most convenient on a tensor.
        super().__init__(root=root, train=train, transform=None, download=download)
        self.custom_transform = transform
        self.num_colors = num_colors
        self.correlation = correlation
        rng = np.random.RandomState(seed)
        self.colors = torch.from_numpy(
            rng.uniform(0.2, 1.0, size=(num_colors, 3)).astype(np.float32)
        )
        self.label_to_color = {label: label % num_colors for label in range(10)}
        # Sample color assignments once: a sample's color does not change every
        # time it is visited during training or evaluation.
        self.color_indices = self._make_color_indices(rng)

    def _make_color_indices(self, rng):
        assigned = np.empty(len(self.targets), dtype=np.int64)
        for index, target in enumerate(self.targets):
            if rng.rand() < self.correlation:
                assigned[index] = self.label_to_color[target]
            else:
                assigned[index] = rng.randint(self.num_colors)
        return torch.from_numpy(assigned)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image = torchvision.transforms.functional.to_tensor(image)
        color_idx = self.color_indices[index]
        image = image * self.colors[color_idx].view(3, 1, 1)
        if self.custom_transform is not None:
            image = self.custom_transform(image)
        return image, target, color_idx


def make_loader(
    correlation,
    *,
    root,
    train,
    batch_size,
    workers,
    max_samples=None,
    seed=42,
):
    transform = torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    dataset = SpuriousColorCIFAR10(
        root=root,
        train=train,
        transform=transform,
        correlation=correlation,
        # Use a different fixed assignment for the held-out split.
        seed=seed + (0 if train else 10_000),
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


class CIFARResNet18(nn.Module):
    """ResNet-18 adjusted for 32x32 CIFAR images."""

    feature_dim = 512

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def features(self, images):
        model = self.model
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return torch.flatten(model.avgpool(x), 1)

    def logits_from_features(self, features):
        return self.model.fc(features)

    def forward(self, images):
        features = self.features(images)
        return self.logits_from_features(features)


def train_main_model(model, loader, device, epochs, learning_rate, quiet=False):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in tqdm(range(epochs), desc="Training ResNet-18", leave=False, disable=quiet):
        for images, labels, _ in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    """Return CIFAR-10 object-classification accuracy."""
    model.eval()
    correct = total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def extract_probe_features(model, images, probe_type):
    """Return penultimate activations or d(max predicted logit)/d(activation)."""
    if probe_type == "latents":
        with torch.no_grad():
            return model.features(images)

    with torch.enable_grad():
        with torch.no_grad():
            features = model.features(images)
        features.requires_grad_(True)
        logits = model.logits_from_features(features)
        maximum_logits = logits.max(dim=1).values
        return torch.autograd.grad(maximum_logits.sum(), features)[0].detach()


def train_probe(model, train_loader, eval_loader, probe_type, device, epochs, learning_rate, quiet=False):
    """Fit a color probe and return its color prediction accuracy on ``eval_loader``."""
    model.eval()
    probe = nn.Linear(model.feature_dim, 10).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(epochs), desc=f"{probe_type} probe", leave=False, disable=quiet):
        probe.train()
        for images, _, color_labels in train_loader:
            images = images.to(device, non_blocking=True)
            color_labels = color_labels.to(device, non_blocking=True)
            features = extract_probe_features(model, images, probe_type)
            loss = criterion(probe(features), color_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = total = 0
    for images, _, color_labels in eval_loader:
        images = images.to(device, non_blocking=True)
        color_labels = color_labels.to(device, non_blocking=True)
        features = extract_probe_features(model, images, probe_type)
        predictions = probe(features).argmax(dim=1)
        correct += (predictions == color_labels).sum().item()
        total += color_labels.numel()
    return 100.0 * correct / total


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 download directory")
    parser.add_argument("--output", default="outputs/spurious_correlations_cifar10.png")
    parser.add_argument("--checkpoint-dir", default="checkpoints/spurious_correlations_cifar10")
    parser.add_argument("--metrics-path", help="Write this single-correlation run's metrics as JSON")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore an existing checkpoint")
    parser.add_argument("--device", help="Torch device, for example cuda:0")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars")
    parser.add_argument("--epochs", type=int, default=5, help="ResNet training epochs per correlation")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--probe-eval-split",
        choices=("train", "test"),
        default="train",
        help="Dataset split used to evaluate each fitted probe (default: train)",
    )
    parser.add_argument("--correlations", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aggregate-metrics",
        nargs="+",
        help="Create a combined plot from JSON metrics files without training",
    )
    return parser.parse_args()


def get_checkpoint_path(checkpoint_dir, correlation, seed, epochs):
    return Path(checkpoint_dir) / f"resnet18_corr_{correlation:.1f}_seed_{seed}_epochs_{epochs}.pt"


def plot_results(results, output_path, probe_eval_split):
    grouped = {}
    for result in results:
        grouped.setdefault(result["correlation"], []).append(result)
    correlations = sorted(grouped)
    latent_means = [np.mean([result["latent_probe_accuracy"] for result in grouped[correlation]]) for correlation in correlations]
    gradient_means = [np.mean([result["gradient_probe_accuracy"] for result in grouped[correlation]]) for correlation in correlations]
    # Standard error communicates uncertainty in the estimate of the seed mean.
    latent_errors = [np.std([result["latent_probe_accuracy"] for result in grouped[correlation]], ddof=1) / np.sqrt(len(grouped[correlation])) if len(grouped[correlation]) > 1 else 0.0 for correlation in correlations]
    gradient_errors = [np.std([result["gradient_probe_accuracy"] for result in grouped[correlation]], ddof=1) / np.sqrt(len(grouped[correlation])) if len(grouped[correlation]) > 1 else 0.0 for correlation in correlations]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4), dpi=200)
    axis.errorbar(correlations, latent_means, yerr=latent_errors, color=plt.cm.autumn(0.25), marker="o", capsize=3, label="Latent activations")
    axis.errorbar(correlations, gradient_means, yerr=gradient_errors, color=plt.cm.winter(0.25), marker="o", capsize=3, label="Gradients / centroids")
    axis.set_xlabel("Color-label correlation")
    axis.set_ylabel(f"{probe_eval_split.title()} linear-probe accuracy (%)")
    axis.legend()
    axis.grid(linestyle="--", color="grey", alpha=0.25)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {output_path}")


def run_experiment(args):
    if args.metrics_path and len(args.correlations) != 1:
        raise ValueError("--metrics-path requires exactly one correlation")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    print(f"Running CIFAR-10 ResNet-18 experiment on {device}.")

    for correlation in args.correlations:
        train_loader = make_loader(correlation, root=args.data_dir, train=True, batch_size=args.batch_size, workers=args.workers, max_samples=args.max_train_samples, seed=args.seed)
        model = CIFARResNet18().to(device)
        saved_checkpoint = get_checkpoint_path(args.checkpoint_dir, correlation, args.seed, args.epochs)
        if saved_checkpoint.exists() and not args.force_retrain:
            checkpoint = torch.load(saved_checkpoint, map_location=device, weights_only=True)
            if checkpoint["seed"] != args.seed or checkpoint["correlation"] != correlation or checkpoint["epochs"] != args.epochs:
                raise ValueError(f"Checkpoint metadata does not match this run: {saved_checkpoint}")
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint: {saved_checkpoint}")
        else:
            train_main_model(model, train_loader, device, args.epochs, args.learning_rate, quiet=args.quiet)
            saved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"correlation": correlation, "seed": args.seed, "epochs": args.epochs, "model_state_dict": model.state_dict()}, saved_checkpoint)
            print(f"Saved checkpoint: {saved_checkpoint}")

        eval_loader = train_loader
        if args.probe_eval_split == "test":
            eval_loader = make_loader(correlation, root=args.data_dir, train=False, batch_size=args.batch_size, workers=args.workers, max_samples=args.max_test_samples, seed=args.seed)
        latent_accuracy = train_probe(model, train_loader, eval_loader, "latents", device, args.probe_epochs, args.probe_learning_rate, quiet=args.quiet)
        gradient_accuracy = train_probe(model, train_loader, eval_loader, "gradients", device, args.probe_epochs, args.probe_learning_rate, quiet=args.quiet)
        result = {
            "correlation": correlation,
            "seed": args.seed,
            "latent_probe_accuracy": latent_accuracy,
            "gradient_probe_accuracy": gradient_accuracy,
            "probe_eval_split": args.probe_eval_split,
        }
        results.append(result)
        print(f"Seed: {args.seed} | correlation: {correlation:.1f} | latent probe: {latent_accuracy:.1f}% | gradient probe: {gradient_accuracy:.1f}%")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(results[0], indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {metrics_path}")
    plot_results(results, args.output, args.probe_eval_split)


def aggregate_metrics(args):
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.aggregate_metrics]
    plot_results(results, args.output, args.probe_eval_split)
    for correlation in sorted({result["correlation"] for result in results}):
        group = [result for result in results if result["correlation"] == correlation]
        latent = np.array([result["latent_probe_accuracy"] for result in group])
        gradient = np.array([result["gradient_probe_accuracy"] for result in group])
        latent_sem = latent.std(ddof=1) / np.sqrt(len(latent)) if len(latent) > 1 else 0.0
        gradient_sem = gradient.std(ddof=1) / np.sqrt(len(gradient)) if len(gradient) > 1 else 0.0
        print(f"Correlation: {correlation:.1f} | latent probe: {latent.mean():.1f} ± {latent_sem:.1f}% | gradient probe: {gradient.mean():.1f} ± {gradient_sem:.1f}% (n={len(group)})")


if __name__ == "__main__":
    arguments = parse_args()
    aggregate_metrics(arguments) if arguments.aggregate_metrics else run_experiment(arguments)
