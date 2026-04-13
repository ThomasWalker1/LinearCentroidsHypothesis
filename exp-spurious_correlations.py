import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import os
from tqdm import tqdm

# --- 1. The Spurious Dataset ---
class SpuriousColorFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, download=True, 
                 num_colors=10, correlation=0.0, seed=42):
        super().__init__(root, train=train, transform=None, download=download)
        self.custom_transform = transform
        self.num_colors = num_colors
        self.correlation = correlation
        
        rng = np.random.RandomState(seed)
        self.colors = torch.from_numpy(rng.uniform(0.2, 1.0, (num_colors, 3))).to(torch.float32)
        
        self.label_to_color = {i: i % num_colors for i in range(10)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        if np.random.rand() < self.correlation:
            color_idx = self.label_to_color[target]
        else:
            color_idx = np.random.randint(0, self.num_colors)
            
        color = self.colors[color_idx].view(3, 1, 1)
        colored_img = img_tensor.repeat(3, 1, 1) * color
        
        if self.custom_transform:
            colored_img = self.custom_transform(colored_img)
            
        return colored_img, target, color_idx

def get_loader(correlation, batch_size=256):
    transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    
    ds = SpuriousColorFashionMNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        correlation=correlation
    )

    subset_indices = range(0, 10000) 
    ds = torch.utils.data.Subset(ds, subset_indices)
    
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# --- 2. Model Definition ---
def get_model(width=128, device='cuda'):
    backbone = nn.Sequential(
        nn.Conv2d(3, width, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(width, width, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(width, width, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ).to(device)

    classifier = nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 10)
    ).to(device)
    
    return backbone, classifier

# --- 3. Training Routines ---
def train_main_model(backbone, classifier, loader, device, epochs=10):
    """Trains the model to classify FashionMNIST objects."""
    opt = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    backbone.train()
    classifier.train()
    
    for _ in tqdm(range(epochs)):
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            z = backbone(imgs)
            out = classifier(z)
            loss = crit(out, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

def train_probe(backbone, classifier, loader, probe_type, device, epochs=3):
    # Freeze main model
    backbone.eval()
    classifier.eval()
    
    # Init Probe
    probe = nn.Linear(128, 10).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    for _ in range(epochs):
        for imgs, _, color_labels in loader:
            imgs, color_labels = imgs.to(device), color_labels.to(device)
            
            # 1. Get Latents (No gradients needed for backbone weights)
            with torch.no_grad():
                z = backbone(imgs)

            # 2. Extract Features (Latents or Gradients)
            if probe_type == 'latents':
                features = z
            elif probe_type == 'gradients':
                # We need a graph from z -> output, even though weights are frozen
                z.requires_grad_(True)
                outs = classifier(z)
                max_scores, _ = outs.max(dim=1)
                
                # Calculate gradient w.r.t z
                grads = torch.autograd.grad(
                    outputs=max_scores.sum(), 
                    inputs=z, 
                    create_graph=False
                )[0]
                features = grads.detach() # Detach so probe training doesn't affect backbone

            # 3. Train Probe
            probe_out = probe(features)
            loss = crit(probe_out, color_labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    correct = 0
    total = 0
    
    for imgs, _, color_labels in loader:
        imgs, color_labels = imgs.to(device), color_labels.to(device)
        
        with torch.no_grad():
            z = backbone(imgs)

        if probe_type == 'latents':
            features = z
        else:
            with torch.enable_grad():
                z.requires_grad_(True)
                outs = classifier(z)
                max_scores, _ = outs.max(dim=1)
                features = torch.autograd.grad(max_scores.sum(), z)[0].detach()
            
        with torch.no_grad():
            preds = probe(features).argmax(dim=1)
            correct += (preds == color_labels).sum().item()
            total += color_labels.size(0)
            
    return 100.0 * correct / total

# --- 4. Main Experiment Loop ---
def run_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correlations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    latent_accuracies = []
    gradient_accuracies = []
    
    print(f"Running Experiment on {device}...")
    
    for corr in tqdm(correlations):
        # 1. Setup Data
        loader = get_loader(correlation=corr)
        
        # 2. Train Main Model (on Object Classification)
        backbone, classifier = get_model(device=device)
        train_main_model(backbone, classifier, loader, device)
        
        # 3. Probe Latents (Predict Color)
        acc_lat = train_probe(backbone, classifier, loader, 'latents', device)
        latent_accuracies.append(acc_lat)
        
        # 4. Probe Gradients/Centroids (Predict Color)
        acc_grad = train_probe(backbone, classifier, loader, 'gradients', device)
        gradient_accuracies.append(acc_grad)
        
        print(f"Corr: {corr} | Latent Acc: {acc_lat:.1f}% | Grad Acc: {acc_grad:.1f}%")

    # --- 5. Plotting ---
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(5,4),dpi=200)
    ax.plot(correlations,latent_accuracies,color=plt.cm.autumn(0.25),marker='o',label='Latent Activations')
    ax.plot(correlations,gradient_accuracies,color=plt.cm.winter(0.25),marker='o',label='Centroids')
    ax.set_xlabel('Color-Label Correlation')
    ax.set_ylabel('Linear Probe Train Accuracy')
    ax.legend()
    ax.grid(linestyle='--',color='grey',alpha=0.25)
    plt.savefig('outputs/spurious_correlations.png',bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run_experiment()