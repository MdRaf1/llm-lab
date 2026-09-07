"""
Model Quantizer and Layer Replacement Engine.
Recursively converts all linear projection layers in a Hugging Face Transformer
into FusedInt4Linear modules, achieving 4x-8x weight memory reduction.
"""

import torch
import torch.nn as nn
from typing import Set, Optional
from rich.console import Console
from llm_lab.core.layers import FusedInt4Linear

console = Console()

def convert_model_to_fused_int4(
    model: nn.Module,
    exclude_modules: Optional[Set[str]] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in the model with FusedInt4Linear,
    except modules specified in exclude_modules (typically 'lm_head').
    """
    if exclude_modules is None:
        exclude_modules = {"lm_head"}
        
    converted_count = 0
    total_saved_bytes = 0
    
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear) and full_name not in exclude_modules and child_name not in exclude_modules:
                # Calculate saved bytes
                orig_bytes = child_module.weight.nelement() * child_module.weight.element_size()
                
                # Convert
                fused = FusedInt4Linear.from_linear(child_module)
                fused_bytes = fused.w_packed.nelement() * fused.w_packed.element_size() + fused.scales.nelement() * fused.scales.element_size()
                
                total_saved_bytes += (orig_bytes - fused_bytes)
                setattr(module, child_name, fused)
                converted_count += 1

    if verbose:
        console.print(f"[bold green]Converted {converted_count} linear layers to FusedInt4Linear.[/bold green]")
        console.print(f"[bold cyan]Total Weight Memory Saved: {total_saved_bytes / (1024**2):.2f} MB[/bold cyan]")
        
    return model
