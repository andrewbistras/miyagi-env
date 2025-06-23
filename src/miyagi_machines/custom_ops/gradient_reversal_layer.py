# /src/miyagi_machines/custom_ops/gradient_reversal_layer.py

"""
Apparrently the only graphable implementation of GRL in Torch.
Not only is it autograd-compliant, it also compiles on CUDA GPUs.
I am genuinely considering doing pull request so no one else will need
to experience the pain and suffering to get this right.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.library import Library, impl, register_autograd

_lib = Library("miyagi_machines", "DEF")
_lib.define("grl(Tensor x, float alpha) -> Tensor")

@impl(_lib, "grl", "CompositeExplicitAutograd")
def _grl_fwd(x: Tensor, alpha: float) -> Tensor:    
    # considered functional but avoids ever actually operating on its tensor.
    return torch.ops.aten.alias(x)

def _setup(ctx, inputs, output):
    _, alpha = inputs
    ctx.alpha = alpha

def _grl_bwd(ctx, g_out: Tensor):
    return -ctx.alpha * g_out, None

register_autograd("miyagi_machines::grl", _grl_bwd, setup_context=_setup)


# cheap fake kernel make this traceable & compilable on CUDA
@torch.library.register_fake("miyagi_machines::grl")
def _grl_fake(x: Tensor, alpha: float):
    return torch.ops.aten.alias(x)


class GradientReversalLayer(nn.Module):
    """nn.Module-friendly wrapper"""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        return torch.ops.miyagi_machines.grl(x, self.alpha)
