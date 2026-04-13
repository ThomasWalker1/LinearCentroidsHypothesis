# ======================================================================================
# 1. Imports
# ======================================================================================
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel
import pickle
import os

# ======================================================================================
# 2. Configuration & Constants
# ======================================================================================
# --- Script Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = Path("outputs/dino")
BATCH_SIZE = 256
DATA_ROOT = "./data"

# --- Image Transformation ---
DINO_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ======================================================================================
# 3. Helper Class for Hooks
# ======================================================================================
class LayerInputHook:
    """
    A simple hook class to capture the input to a specific nn.Module.
    This avoids the use of global variables.
    """

    def __init__(self):
        self.input: torch.Tensor | None = None
        self.handle = None

    def hook_fn(self, module, model_input, output):
        """The actual hook function that PyTorch will call."""
        # We clone the input tensor to avoid issues with in-place modifications
        self.input = model_input[0].detach().clone()

    def register(self, module: nn.Module):
        """Registers the hook to the target module."""
        self.handle = module.register_forward_hook(self.hook_fn)

    def unregister(self):
        """Removes the hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None

    def get_input(self) -> torch.Tensor:
        """Returns the captured input."""
        if self.input is None:
            raise ValueError("Hook did not capture any input. Was the forward pass run?")
        return self.input

# ======================================================================================
# 4. Core Logic Functions
# ======================================================================================
def get_dataloaders() -> Dict[str, DataLoader]:
    """
    Prepares and returns the Imagenette data loaders for training and validation sets.
    """
    print("Loading Imagenette dataset...")
    train_dataset = torchvision.datasets.Imagenette(
        DATA_ROOT, split="train", transform=DINO_TRANSFORM, download=True
    )
    test_dataset = torchvision.datasets.Imagenette(
        DATA_ROOT, split="val", transform=DINO_TRANSFORM, download=True
    )
    return {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
    }


def setup_model_and_block(model_name, model_path) -> Tuple[AutoModel, nn.Sequential]:
    print(f"Loading pre-trained model: {model_name}...")
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Create a standalone MLP block identical to the last one in the encoder
    block = nn.Sequential(
        nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768)
    )
    block_path = SAVE_DIR / model_name / "block.pth"

    if block_path.exists():
        print(f"Loading MLP block weights from {block_path}")
        block.load_state_dict(torch.load(block_path, weights_only=True))
    else:
        print("Initializing MLP block weights from the model's final layer...")
        if model_name == "v2":
            final_mlp = model.encoder.layer[-1].mlp
            block[0].weight.data.copy_(final_mlp.fc1.weight.data)
            block[0].bias.data.copy_(final_mlp.fc1.bias.data)
            block[-1].weight.data.copy_(final_mlp.fc2.weight.data)
            block[-1].bias.data.copy_(final_mlp.fc2.bias.data)
            torch.save(block.state_dict(), block_path)
        elif model_name == "v3":
            final_mlp = model.layer[-1].mlp
            block[0].weight.data.copy_(final_mlp.up_proj.weight.data)
            block[0].bias.data.copy_(final_mlp.up_proj.bias.data)
            block[-1].weight.data.copy_(final_mlp.down_proj.weight.data)
            block[-1].bias.data.copy_(final_mlp.down_proj.bias.data)
            torch.save(block.state_dict(), block_path)

    return model.to(DEVICE), block.to(DEVICE)


def extract_and_save_features(
    model: AutoModel,
    model_name: str,
    block: nn.Sequential,
    loader: DataLoader,
    set_type: str,
    token_type: Literal["cls", "all"],
):
    """
    Runs the feature extraction process for a given dataset split and token type.
    """
    output_dir = SAVE_DIR / model_name / f"{set_type}-{token_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_inputs, all_latents, all_centroids, all_labels = [], [], [], []
    total_centroid_time, total_latent_time = 0.0, 0.0

    if model_name == "v2":
        target_module = model.encoder.layer[-1].mlp
    elif model_name == "v3":
        target_module = model.layer[-1].mlp
    hook = LayerInputHook()
    hook.register(target_module)

    for images, labels in tqdm(loader, desc=f"Processing {set_type} - {token_type}"):
        start_time=time.time()
        images = images.to(DEVICE)
        
        # 1. Get the input to the MLP block using the hook
        with torch.no_grad():
            _ = model(images)
        
        block_input = hook.get_input()

        # 2. Process inputs based on whether we use CLS token or all tokens
        if token_type == "cls":
            # Select only the CLS token's features
            processed_input = block_input[:, 0, :].clone().requires_grad_(True)
            processed_labels = labels.cpu()
        else: # token_type == 'all'
            processed_input = block_input.clone().requires_grad_(True)
            # Duplicate labels for each token in the sequence
            num_tokens = processed_input.shape[1]
            processed_labels = labels.cpu().unsqueeze(1).expand(-1, num_tokens).flatten()

        # 3. Compute latents (forward pass through the standalone block)
        latents = block(processed_input)
        total_latent_time += time.time() - start_time

        # 4. Compute centroids (gradients)
        
        (centroids,) = torch.autograd.grad(
            outputs=latents,
            inputs=processed_input,
            grad_outputs=torch.ones_like(latents),
        )
        total_centroid_time += time.time() - start_time

        all_inputs.append(processed_input.detach().cpu())
        all_latents.append(latents.detach().cpu())
        all_centroids.append(centroids.detach().cpu())
        all_labels.append(processed_labels)

    hook.unregister()

    # --- Concatenate and Save Results ---
    print("Concatenating and saving results...")
    final_inputs = torch.cat(all_inputs).view(-1, all_inputs[0].shape[-1])
    final_latents = torch.cat(all_latents).view(-1, all_latents[0].shape[-1])
    final_centroids = torch.cat(all_centroids).view(-1, all_centroids[0].shape[-1])
    final_labels = torch.cat(all_labels)

    torch.save(final_inputs, output_dir / "inputs.pt")
    torch.save(final_latents, output_dir / "latents.pt")
    torch.save(final_centroids, output_dir / "centroids.pt")
    torch.save(final_labels, output_dir / "labels.pt")

    print(f"[{set_type}-{token_type}] Centroid computation time: {total_centroid_time:.2f}s")
    print(f"[{set_type}-{token_type}] Latent computation time: {total_latent_time:.2f}s")
    print("-" * 50)

    if os.path.exists(SAVE_DIR / 'times.pkl'):
        with open(SAVE_DIR / 'times.pkl','rb') as file:
            times=pickle.load(file)
    else:
        times={}
    times[f"{model_name}-{set_type}-{token_type}"]={'Centroids':total_centroid_time,'Latents':total_latent_time}
    with open(SAVE_DIR / 'times.pkl','wb') as file:
        pickle.dump(times,file)


# ======================================================================================
# 5. Main Execution
# ======================================================================================
def main():
    """Main script execution."""
    
    models={
        "v2":"facebook/dinov2-base",
        "v3":"facebook/dinov3-vitb16-pretrain-lvd1689m"
    }

    for model_name, model_path in models.items():

        (SAVE_DIR / model_name).mkdir(parents=True, exist_ok=True)
        print(f"Using device: {DEVICE}")

        loaders = get_dataloaders()
        model, block = setup_model_and_block(model_name,model_path)

        for set_type, loader in loaders.items():
            for token_type in ["cls", "all"]:
                extract_and_save_features(model, model_name, block, loader, set_type, token_type)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
