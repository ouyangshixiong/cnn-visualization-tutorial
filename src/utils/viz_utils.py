"""Visualization utilities for CNN models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import torchvision.transforms as transforms


def visualize_filters(model: torch.nn.Module, layer_name: str, max_filters: int = 16) -> None:
    """Visualize convolutional filters from a specific layer."""
    layer = dict(model.named_modules())[layer_name]
    if not isinstance(layer, torch.nn.Conv2d):
        raise ValueError(f"Layer {layer_name} is not a Conv2d layer")
    
    filters = layer.weight.data.cpu()
    n_filters = min(filters.shape[0], max_filters)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_filters):
        if i < len(axes):
            filter_img = filters[i].permute(1, 2, 0).numpy()
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
            axes[i].imshow(filter_img)
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_activations(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    max_activations: int = 16
) -> List[np.ndarray]:
    """Visualize feature map activations from a specific layer."""
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(input_tensor.unsqueeze(0))
    
    handle.remove()
    
    if activations:
        activation = activations[0].cpu()
        n_activations = min(activation.shape[1], max_activations)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_activations):
            if i < len(axes):
                act_img = activation[0, i].numpy()
                axes[i].imshow(act_img, cmap='viridis')
                axes[i].set_title(f'Activation {i}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return [act.numpy() for act in activations]


def get_layer_names(model: torch.nn.Module, layer_type: type = torch.nn.Conv2d) -> List[str]:
    """Get names of all layers of a specific type."""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, layer_type)
    ]