import torch
import torch.nn as nn
import pandas as pd 

def print_model_summary(model: nn.Module, show_weights: bool = True, indent: int = 2):
    """
    Prints a clean hierarchical summary of the model architecture with parameter shapes.
    
    Args:
        model (nn.Module): The model (e.g. UNet instance)
        show_weights (bool): Whether to show the tensor shapes for each Conv/Linear layer
        indent (int): Indentation spaces per nested module
    """
    print("🧠 Model structure overview:")
    print("───────────────────────────────")

    def recurse(module, prefix=""):
        for name, layer in module.named_children():
            layer_name = f"{prefix}{name}"
            layer_type = layer.__class__.__name__
            if show_weights and hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
                shape = tuple(layer.weight.shape)
                print(f"{layer_name:<40} ({layer_type:<15}) → {shape}")
            else:
                print(f"{layer_name:<40} ({layer_type:<15})")
            recurse(layer, prefix + " " * indent)

    recurse(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("───────────────────────────────")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable: {trainable_params:,}")
    print("───────────────────────────────")


def model_to_dataframe(model: nn.Module):
    """
    Convert a PyTorch model into a pandas DataFrame with detailed layer info.
    Keeps only 'real' layers (Conv2d, BatchNorm2d, ReLU, Linear, ConvTranspose2d, etc.)
    
    Returns:
        pd.DataFrame with columns:
        ['Layer', 'Type', 'Shape', 'In Ch', 'Out Ch', 'Num Params', 'Num Params (k)']
    """
    layers = []

    def recurse(module, prefix=""):
        for name, layer in module.named_children():
            layer_name = f"{prefix}{name}"
            layer_type = layer.__class__.__name__
            
            # Skip container-only layers
            if isinstance(layer, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                recurse(layer, prefix + name + ".")
                continue

            # Extract shape info
            shape = getattr(layer, "weight", None)
            shape = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else None
            num_params = sum(p.numel() for p in layer.parameters())

            # Extract input/output channels if present
            in_ch = getattr(layer, "in_channels", None)
            out_ch = getattr(layer, "out_channels", None)

            layers.append({
                "Layer": layer_name,
                "Type": layer_type,
                "Shape": shape,
                "In Ch": in_ch,
                "Out Ch": out_ch,
                "Num Params": num_params,
                "Num Params (k)": num_params / 1e3
            })

            recurse(layer, prefix + name + ".")

    recurse(model)

    df = pd.DataFrame(layers)

    # Keep only "real" layers
    keep_types = ["Conv2d", "ConvTranspose2d", "BatchNorm2d", "ReLU", "Linear"]
    df = df[df["Type"].isin(keep_types)].reset_index(drop=True)

    return df


def model_to_dataframe_with_l1(model: nn.Module, remove_nan_layers: bool = True):
    """
    Build a DataFrame summarizing all convolutional layers in a UNet-like model,
    including L1 statistics and parameter counts.

    Args:
        model (nn.Module): Model to inspect.
        remove_nan_layers (bool): If True, drops layers without L1 stats (e.g. container modules).

    Returns:
        pd.DataFrame: Layer summary with columns:
        ['Layer', 'Type', 'Shape', 'In Ch', 'Out Ch',
         'Num Params', 'Mean L1', 'Min L1', 'Max L1', 'L1 Std']
    """
    layers = []

    def recurse(module, prefix=""):
        for name, layer in module.named_children():
            layer_name = f"{prefix}{name}"
            layer_type = layer.__class__.__name__

            # Recurse into container layers (Sequential, ModuleList, etc.)
            if isinstance(layer, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                recurse(layer, prefix + name + ".")
                continue

            # Extract general attributes
            shape = getattr(layer, "weight", None)
            shape = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else None
            num_params = sum(p.numel() for p in layer.parameters())
            in_ch = getattr(layer, "in_channels", None)
            out_ch = getattr(layer, "out_channels", None)

            # Compute L1 stats for Conv layers
            mean_l1 = min_l1 = max_l1 = std_l1 = None
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                w = layer.weight.data.abs().view(layer.weight.size(0), -1)
                l1_vals = w.sum(dim=1)
                mean_l1 = l1_vals.mean().item()
                min_l1 = l1_vals.min().item()
                max_l1 = l1_vals.max().item()
                std_l1 = l1_vals.std().item()

            layers.append({
                "Layer": layer_name,
                "Type": layer_type,
                "Shape": shape,
                "In Ch": in_ch,
                "Out Ch": out_ch,
                "Num Params": num_params,
                "Mean L1": mean_l1,
                "Min L1": min_l1,
                "Max L1": max_l1,
                "L1 Std": std_l1
            })

            recurse(layer, prefix + name + ".")

    recurse(model)
    df = pd.DataFrame(layers)

    enc = df[df["Layer"].str.startswith("encoders")].copy()
    bott = df[df["Layer"].str.startswith("bottleneck")].copy()
    dec = df[df["Layer"].str.startswith("decoders") | df["Layer"].str.startswith("final_conv")].copy()
    df_sorted = pd.concat([enc, bott, dec], ignore_index=True)

    if remove_nan_layers:
        df_sorted = df_sorted.dropna(subset=["Mean L1"]).reset_index(drop=True)

    return df_sorted



def compare_models(model_before, model_after):
    """Print difference in layer shapes or number of filters."""
    print("🔍 Comparing model structures before and after pruning:")
    print("───────────────────────────────")

    def get_layer_info(module):
        info = {}
        for name, layer in module.named_children():
            if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
                info[name] = tuple(layer.weight.shape)
            else:
                info[name] = None
            info.update({f"{name}.{k}": v for k, v in get_layer_info(layer).items()})
        return info

    before_info = get_layer_info(model_before)
    after_info = get_layer_info(model_after)

    for layer_name in before_info.keys():
        before_shape = before_info[layer_name]
        after_shape = after_info.get(layer_name, None)
        if before_shape != after_shape:
            print(f"Layer: {layer_name}")
            print(f"  Before: {before_shape}")
            print(f"  After:  {after_shape}")
            print("───────────────────────────────")

def layer_sparsity_report(model):
    """Compute and print per-layer sparsity percentage."""
    print("📊 Layer-wise sparsity report:")
    print("───────────────────────────────")
    for name, layer in model.named_modules():
        if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
            total_params = layer.weight.numel()
            zero_params = torch.sum(layer.weight == 0).item()
            sparsity = (zero_params / total_params) * 100
            print(f"{name:<40} Sparsity: {sparsity:.2f}% ({zero_params}/{total_params})")
    print("───────────────────────────────")
