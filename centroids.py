import torch
import torch.nn as nn
import torch.nn.functional as F

class Centroids:
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        
        try:
            param_device = next(model.parameters()).device
        except StopIteration:
            param_device = 'cpu'
            
        self.device = device if device is not None else param_device

        self.centroids = None
        self.radii = None
        self.inner_products = None
        self.norms = None
        self.alignments = None

    def __call__(self, x):
        x_tensor = x if torch.is_tensor(x) else torch.from_numpy(x).float()
        
        if x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(0)

        x_in = x_tensor.to(self.device).requires_grad_(True)
        
        output = self.model(x_in)
        self.centroids = self._compute_centroids_grad(x_in, output)

        flat_centroids = self.centroids.reshape(self.centroids.size(0), -1)
        flat_x = x_in.reshape(x_in.size(0), -1)

        self.inner_products = (flat_centroids * flat_x).sum(dim=1)
        self.norms = flat_centroids.norm(p=2, dim=1)
        
        denom = torch.clamp(self.norms * flat_x.norm(p=2, dim=1), min=1e-8)
        self.alignments = self.inner_products / denom

        return self.centroids

    def _compute_centroids_grad(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        grad_output = torch.ones_like(output)
        
        centroids = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            create_graph=False 
        )[0]
        
        return centroids

class LocalCentroids:
    def __init__(self, model, device=None, stdev_spread=0.15, n_samples=10):
        self.model = model
        self.centroid_calc = Centroids(model, device=device)
        self.device = self.centroid_calc.device
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        
        self.local_centroids = None
        self.alignments = None

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            
        stdev = self.stdev_spread * (torch.max(x) - torch.min(x)).item()
        
        x = x.to(self.device)
        
        total_centroids = torch.zeros_like(x)

        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * stdev
            x_plus_noise = x + noise

            centroid = self.centroid_calc(x_plus_noise)
            total_centroids += centroid

        local_centroids = total_centroids / self.n_samples
        self.local_centroids = local_centroids

        local_centroids_flat = local_centroids.view(x.size(0), -1)
        x_flat = x.view(x.size(0), -1)
        
        alignments = F.cosine_similarity(local_centroids_flat, x_flat, dim=1)
        self.alignments = alignments
        
        return self.local_centroids

class InternalModuleWrapper(nn.Module):
    def __init__(self, model, target_module_name):
        super().__init__()
        self.model = model
        self.target_module_name = target_module_name
        self.captured_output = None
        
        target_module = None
        try:
            modules_dict = {name: mod for name, mod in self.model.named_modules()}
            target_module = modules_dict[self.target_module_name]
        except KeyError:
            raise ValueError(f"Module '{self.target_module_name}' not found.")
            
        self.hook_handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input_tuple, output_tensor):
        self.captured_output = output_tensor
        
    def forward(self, x):
        self.model.eval() 
        self.captured_output = None 
        _ = self.model(x) 
        if self.captured_output is None:
            raise RuntimeError(f"Hook for {self.target_module_name} did not capture output.")
        return self.captured_output
        
    def remove_hook(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()