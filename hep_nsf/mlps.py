"""
mlps.py
=======
Conditioner MLP architectures used inside the coupling layers.

Each coupling layer needs a small neural network (the "conditioner") that
maps the *fixed* half of the input to the spline parameters (widths,
heights, derivatives) for the *transformed* half.

All conditioners produce a flat parameter vector of size
``out_dims * (3 * num_bins + 1)`` which the flow model reshapes into
``(w, h, d)`` tensors ready for the RQS.

Classes
-------
ResidualBlock
    Single pre-activation residual block used inside ``ResNet``.
MLP
    Plain feed-forward network: Linear → Activation → … → Linear.
ResNet
    Residual network with optional layer normalisation.
ConditionerFactory
    Convenience factory: instantiate any conditioner by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional dropout.

    Architecture: ``LN → Act → Linear → LN → Act → Dropout → Linear``
    followed by a skip connection (+ projection if dimensions differ).
    """

    def __init__(self,
                 hidden_dim: int,
                 activation: nn.Module,
                 dropout: float = 0.0,
                 use_layer_norm: bool = True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [activation, nn.Linear(hidden_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Conditioner networks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Plain feed-forward conditioner.

    Parameters
    ----------
    in_dim : int
        Input dimension (number of conditioning variables).
    out_dim : int
        Total output size, i.e. ``out_dims * (3 * num_bins + 1)``.
    hidden_dim : int
        Width of each hidden layer (default 64).
    num_layers : int
        Number of hidden layers, not counting input/output projections
        (default 2).
    activation : str
        One of ``'relu'``, ``'tanh'``, ``'elu'``, ``'leaky_relu'``,
        ``'silu'`` (default ``'relu'``).
    dropout : float
        Dropout probability applied after each hidden activation
        (default 0.0 = disabled).
    """

    _ACTIVATIONS: dict[str, type[nn.Module]] = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "elu":        nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu":       nn.SiLU,
        "gelu":       nn.GELU,
    }

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(self._ACTIVATIONS.keys())}"
            )

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet(nn.Module):
    """Residual conditioner.

    A linear projection to ``hidden_dim``, followed by *N* residual blocks,
    followed by a final linear head.

    Parameters
    ----------
    in_dim : int
    out_dim : int
    hidden_dim : int
        Width of all residual blocks (default 128).
    num_blocks : int
        Number of residual blocks (default 2).
    activation : str
        Activation used inside each block (default ``'relu'``).
    dropout : float
    use_layer_norm : bool
        Whether to apply ``LayerNorm`` inside residual blocks (default True).
    """

    _ACTIVATIONS = MLP._ACTIVATIONS

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 128,
                 num_blocks: int = 2,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 use_layer_norm: bool = True):
        super().__init__()
        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'.")

        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, act_cls(), dropout, use_layer_norm)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ArchitectureType = Literal["mlp", "resnet"]


def build_conditioner(arch: ArchitectureType,
                      in_dim: int,
                      out_dim: int,
                      hidden_dim: int = 64,
                      num_layers: int = 2,
                      activation: str = "relu",
                      dropout: float = 0.0,
                      use_layer_norm: bool = True) -> nn.Module:
    """Factory function that instantiates a conditioner network by name.

    Parameters
    ----------
    arch : str
        ``'mlp'`` or ``'resnet'``.
    in_dim, out_dim : int
        Input / output sizes passed directly to the network constructor.
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Depth parameter.  For ``'mlp'`` this is the number of hidden
        layers; for ``'resnet'`` this is the number of residual blocks.
    activation : str
        Activation function name.
    dropout : float
        Dropout probability.
    use_layer_norm : bool
        Only meaningful for ``'resnet'`` (ignored by ``'mlp'``).

    Returns
    -------
    nn.Module

    Examples
    --------
    >>> net = build_conditioner('mlp', in_dim=1, out_dim=3*32+1,
    ...                         hidden_dim=64, num_layers=2)
    >>> net = build_conditioner('resnet', in_dim=2, out_dim=3*32+1,
    ...                         hidden_dim=128, num_layers=3)
    """
    arch = arch.lower()
    if arch == "mlp":
        return MLP(in_dim, out_dim,
                   hidden_dim=hidden_dim,
                   num_layers=num_layers,
                   activation=activation,
                   dropout=dropout)
    elif arch == "resnet":
        return ResNet(in_dim, out_dim,
                      hidden_dim=hidden_dim,
                      num_blocks=num_layers,
                      activation=activation,
                      dropout=dropout,
                      use_layer_norm=use_layer_norm)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose 'mlp' or 'resnet'.")
