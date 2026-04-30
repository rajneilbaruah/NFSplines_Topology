"""
utils.py
========
Shared utilities: coordinate transforms, data loading, normalisation,
device helpers, and seed management.

Functions
---------
set_seed
    Fix all random seeds for reproducibility.
get_device
    Return the best available ``torch.device``.
cartesian_to_spherical
    ``(px, py, pz)`` → ``(cos θ, φ)`` on the unit sphere.
spherical_to_cartesian
    ``(cos θ, φ)`` → ``(px, py, pz)`` at radius ``r``.
load_json_data
    Load a JSON file of 3-momentum vectors.
normalise
    Compute mean/std and return standardised tensor.
denormalise
    Reverse standardisation using saved mean/std.
make_dataloaders
    Wrap a tensor dataset into train/val ``DataLoader`` objects with an
    optional stratified validation fraction.
save_checkpoint
    Save model + optimiser state and training metadata.
load_checkpoint
    Restore a checkpoint into an existing model (and optionally optimiser).
count_parameters
    Return the total number of trainable parameters in a model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full determinism on GPU (may slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available ``torch.device``.

    Parameters
    ----------
    prefer : str
        ``'auto'``  — use CUDA if available, else MPS, else CPU.
        ``'cpu'``   — force CPU.
        ``'cuda'``  — force CUDA (raises if unavailable).
        ``'mps'``   — force Apple MPS (raises if unavailable).

    Returns
    -------
    torch.device
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if prefer == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def cartesian_to_spherical(p: torch.Tensor,
                            phi_range: str = "0_2pi"
                            ) -> torch.Tensor:
    """Convert 3-momentum Cartesian to angular coordinates on S².

    Parameters
    ----------
    p : Tensor, shape ``(N, 3)`` — columns are ``(px, py, pz)``.
    phi_range : str
        ``'0_2pi'``   → φ ∈ [0, 2π]   (default, used in normalising flow).
        ``'neg_pi_pi'`` → φ ∈ [−π, π].

    Returns
    -------
    Tensor, shape ``(N, 2)`` — columns are ``(cos θ, φ)``.
    """
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    p_mag = torch.sqrt(px ** 2 + py ** 2 + pz ** 2).clamp(min=1e-9)
    cos_theta = (pz / p_mag).clamp(-1.0, 1.0)
    phi = torch.atan2(py, px)
    if phi_range == "0_2pi":
        phi = (phi + np.pi) % (2.0 * np.pi)
    return torch.stack([cos_theta, phi], dim=1)


def spherical_to_cartesian(sph: torch.Tensor,
                            r: float = 500.0) -> torch.Tensor:
    """Convert ``(cos θ, φ)`` to Cartesian ``(px, py, pz)``.

    Parameters
    ----------
    sph : Tensor, shape ``(N, 2)`` — columns ``(cos θ, φ)``,
          φ is expected in ``[0, 2π]``.
    r : float
        Momentum magnitude in GeV (default 500 GeV).

    Returns
    -------
    Tensor, shape ``(N, 3)``
    """
    cos_theta, phi = sph[:, 0], sph[:, 1]
    sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-9))
    px = r * sin_theta * torch.cos(phi)
    py = r * sin_theta * torch.sin(phi)
    pz = r * cos_theta
    return torch.stack([px, py, pz], dim=1)


def cartesian_to_physics(p: torch.Tensor
                          ) -> dict[str, torch.Tensor]:
    """Derive common HEP variables from Cartesian 3-momentum.

    Parameters
    ----------
    p : Tensor, shape ``(N, 3)``

    Returns
    -------
    dict with keys: ``px, py, pz, pT, eta, phi, |p|``
    """
    p = p.detach().cpu()
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    p_mag = torch.norm(p, dim=1).clamp(min=1e-9)
    pT = torch.sqrt(px ** 2 + py ** 2 + 1e-9)
    cos_arg = (pz / p_mag).clamp(-0.999, 0.999)
    eta = -torch.log(torch.tan(torch.acos(cos_arg) / 2.0) + 1e-9)
    phi = torch.atan2(py, px)
    return {"px": px, "py": py, "pz": pz,
            "pT": pT, "eta": eta, "phi": phi, "|p|": p_mag}


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_json_data(path: str | Path,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load a JSON file of 3-momentum vectors.

    The JSON is expected to be a list of lists, e.g.::

        [[px1, py1, pz1], [px2, py2, pz2], ...]

    Parameters
    ----------
    path : str or Path
    dtype : torch.dtype

    Returns
    -------
    Tensor, shape ``(N, 3)``
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return torch.tensor(data, dtype=dtype)


def save_losses(losses: dict[str, list[float]],
                path: str | Path) -> None:
    """Save training / validation loss curves to JSON."""
    with open(path, "w") as f:
        json.dump(losses, f, indent=2)


def load_losses(path: str | Path) -> dict[str, list[float]]:
    """Load loss curves from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(data: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardise ``data`` to zero mean and unit variance (per feature).

    Parameters
    ----------
    data : Tensor, shape ``(N, D)``

    Returns
    -------
    data_norm : Tensor, shape ``(N, D)``
    mean : Tensor, shape ``(1, D)``
    std  : Tensor, shape ``(1, D)``  — floored at 1e-8 to avoid division by zero
    """
    mean = data.mean(0, keepdim=True)
    std  = data.std(0,  keepdim=True).clamp(min=1e-8)
    return (data - mean) / std, mean, std


def denormalise(data_norm: torch.Tensor,
                mean: torch.Tensor,
                std: torch.Tensor) -> torch.Tensor:
    """Reverse standardisation.

    Parameters
    ----------
    data_norm : Tensor, shape ``(N, D)``
    mean, std : Tensors from a prior :func:`normalise` call.

    Returns
    -------
    Tensor, shape ``(N, D)``
    """
    return data_norm * std.to(data_norm.device) + mean.to(data_norm.device)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(data: torch.Tensor,
                     batch_size: int = 8192,
                     val_frac: float = 0.2,
                     seed: int = 42
                     ) -> tuple[DataLoader, DataLoader]:
    """Wrap a data tensor into shuffled train / validation ``DataLoader``s.

    Parameters
    ----------
    data : Tensor, shape ``(N, D)``  — already normalised.
    batch_size : int
    val_frac : float  — fraction held out for validation (default 0.2).
    seed : int        — used for the random split.

    Returns
    -------
    train_loader, val_loader : DataLoader objects
    """
    dataset = TensorDataset(data)
    n_total = len(dataset)
    n_val   = int(val_frac * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str | Path,
                    model: nn.Module,
                    optimiser: torch.optim.Optimizer,
                    epoch: int,
                    best_val_loss: float,
                    extra: dict[str, Any] | None = None) -> None:
    """Save a full training checkpoint.

    Parameters
    ----------
    path : str or Path  — destination file (``*.pt``).
    model : nn.Module
    optimiser : torch.optim.Optimizer
    epoch : int         — last completed epoch (0-indexed).
    best_val_loss : float
    extra : dict, optional — any additional metadata to store.
    """
    payload: dict[str, Any] = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optim_state":    optimiser.state_dict(),
        "best_val_loss":  best_val_loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path,
                    model: nn.Module,
                    optimiser: torch.optim.Optimizer | None = None,
                    device: torch.device | None = None
                    ) -> dict[str, Any]:
    """Restore a checkpoint.

    Parameters
    ----------
    path : str or Path
    model : nn.Module       — must already be initialised with the same
                              architecture.
    optimiser : optional    — if provided, its state is also restored.
    device : torch.device   — map location (defaults to CPU).

    Returns
    -------
    dict with keys ``epoch``, ``best_val_loss``, and any extras.
    """
    map_loc = device if device is not None else torch.device("cpu")
    payload = torch.load(path, map_location=map_loc)
    model.load_state_dict(payload["model_state"])
    if optimiser is not None and "optim_state" in payload:
        optimiser.load_state_dict(payload["optim_state"])
    return payload


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_std_correction(std: torch.Tensor) -> torch.Tensor:
    """Log-correction term for standardised NLL.

    When training on standardised data, the true NLL includes a constant
    correction of ``−∑ log σ_i``.  This function returns that correction
    as a scalar, ready to be *added* to the batch-mean NLL loss.

    Parameters
    ----------
    std : Tensor, shape ``(1, D)``

    Returns
    -------
    Tensor scalar
    """
    return -torch.log(std).sum()
