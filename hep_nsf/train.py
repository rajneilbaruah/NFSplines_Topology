"""
train.py
========
Self-contained training loop shared by all flow models.

The loop implements:
  - NLL loss with standardisation correction.
  - EMA-smoothed validation loss for robust early stopping.
  - Gradient clipping.
  - Periodic console logging.
  - Automatic checkpoint saving (best model only).
  - Optional dual learning-rate scheduler (ReduceLROnPlateau + Cosine).
  - NaN/Inf batch filtering.

Entry point
-----------
``train_model(model, train_loader, val_loader, ...)``
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .utils import save_checkpoint, save_losses, log_std_correction


# ---------------------------------------------------------------------------
# NLL loss helper
# ---------------------------------------------------------------------------

def nll_loss(z: torch.Tensor,
             ldj: torch.Tensor,
             std_correction: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood under a standard-normal base.

    Parameters
    ----------
    z : Tensor, shape ``(B, D)`` — latent codes.
    ldj : Tensor, shape ``(B,)`` — log |det J| per sample.
    std_correction : Tensor scalar — ``−∑ log σ_i`` from normalisation.

    Returns
    -------
    Tensor scalar — batch-mean NLL (lower is better).
    """
    log_pz = -0.5 * (z ** 2 + math.log(2.0 * math.pi)).sum(dim=1)
    return -(log_pz + ldj).mean() + std_correction


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader:   torch.utils.data.DataLoader,
                std_tensor:   torch.Tensor,
                device:       torch.device,
                # ── model metadata (saved in checkpoint) ──
                model_type:   str   = "",
                num_layers:   int   = 2,
                arch:         str   = "mlp",
                activation:   str   = "relu",
                # ── optimisation ──
                lr:           float = 1e-3,
                weight_decay: float = 0.0,
                max_epochs:   int   = 10_000,
                patience:     int   = 20,
                ema_alpha:    float = 0.3,
                clip_grad:    float = 5.0,
                # ── LR schedulers ──
                use_plateau:  bool  = True,
                plateau_factor: float = 0.5,
                plateau_patience: int = 10,
                use_cosine:   bool  = False,
                cosine_t_max: int   = 500,
                # ── I/O ──
                log_every:    int   = 10,
                save_dir:     str   = "checkpoints",
                run_name:     str   = "model",
                resume_from:  Optional[str] = None,
                ) -> tuple[nn.Module, dict[str, list]]:
    """Train a normalising-flow model.

    Parameters
    ----------
    model : nn.Module
        Must implement ``forward(x) → (z, ldj)``.
    train_loader, val_loader : DataLoader
        Yield batches of *normalised* data tensors.
    std_tensor : Tensor, shape ``(1, D)``
        Standard deviations from pre-processing (used in the NLL
        correction term).
    device : torch.device
    lr : float          Learning rate (default 1e-3).
    weight_decay : float
        L2 regularisation for Adam (default 0).
    max_epochs : int    Maximum training epochs (default 10 000).
    patience : int      Early-stopping patience (default 20).
    ema_alpha : float   EMA smoothing factor for val loss (default 0.3).
    clip_grad : float   Gradient-clipping max-norm (default 5.0).
    use_plateau : bool  Reduce LR on plateau (default True).
    plateau_factor : float  LR reduction factor (default 0.5).
    plateau_patience : int  Epochs without improvement before LR drop.
    use_cosine : bool   Add cosine-annealing LR scheduler (default False).
    cosine_t_max : int  Period for cosine annealing (default 500).
    log_every : int     Print frequency in epochs (default 10).
    save_dir : str      Directory for checkpoints (default ``'checkpoints'``).
    run_name : str      Prefix for checkpoint filenames.
    resume_from : str, optional
        Path to a checkpoint file to resume training from.

    Returns
    -------
    model : nn.Module — best model (state restored from best checkpoint).
    losses : dict
        ``{"train": [...], "val": [...]}``
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(save_dir) / f"{run_name}_best.pt"

    model = model.to(device)
    std_correction = log_std_correction(std_tensor).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)

    schedulers: list = []
    if use_plateau:
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=plateau_factor,
                patience=plateau_patience))
    if use_cosine:
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max))

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    smoothed_val: Optional[float] = None

    epoch_train: list[float] = []
    epoch_val:   list[float] = []

    # --- Resume from checkpoint ------------------------------------------ #
    if resume_from is not None:
        from .utils import load_checkpoint
        meta = load_checkpoint(resume_from, model, optimizer, device=device)
        start_epoch   = meta.get("epoch", 0) + 1
        best_val_loss = meta.get("best_val_loss", float("inf"))
        epoch_train   = meta.get("train_losses", [])
        epoch_val     = meta.get("val_losses",   [])
        print(f"Resuming from epoch {start_epoch}, best val = {best_val_loss:.6f}")

    t0 = time.time()

    for epoch in range(start_epoch, max_epochs):
        # ------------------------------------------------------------------ #
        # Training phase
        # ------------------------------------------------------------------ #
        model.train()
        total_loss, n_batches = 0.0, 0

        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()

            loss = -model.log_prob(xb).mean() + std_correction

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

        avg_train = total_loss / max(n_batches, 1)
        epoch_train.append(avg_train)

        # ------------------------------------------------------------------ #
        # Validation phase
        # ------------------------------------------------------------------ #
        model.eval()
        val_total, val_batches = 0.0, 0

        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                val_l = -model.log_prob(xb).mean() + std_correction
                if torch.isfinite(val_l):
                    val_total  += val_l.item()
                    val_batches += 1

        avg_val = val_total / max(val_batches, 1)
        epoch_val.append(avg_val)

        # EMA-smoothed validation loss
        smoothed_val = (avg_val if smoothed_val is None
                        else ema_alpha * avg_val + (1.0 - ema_alpha) * smoothed_val)

        # ------------------------------------------------------------------ #
        # Schedulers
        # ------------------------------------------------------------------ #
        for sched in schedulers:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(smoothed_val)
            else:
                sched.step()

        # ------------------------------------------------------------------ #
        # Early stopping & checkpoint
        # ------------------------------------------------------------------ #
        if smoothed_val < best_val_loss - 1e-6:
            best_val_loss   = smoothed_val
            patience_counter = 0
            # Derive hidden_dim from model internals
            try:
                if hasattr(model, "phi_nets") and model.phi_nets:
                    net = model.phi_nets[0]
                elif hasattr(model, "phi_nets_A") and model.phi_nets_A:
                    net = model.phi_nets_A[0]
                elif hasattr(model, "z_nets_B") and model.z_nets_B:
                    net = model.z_nets_B[0]
                elif hasattr(model, "nets_phi") and model.nets_phi:
                    net = model.nets_phi[0]
                elif hasattr(model, "nets1") and model.nets1:
                    net = model.nets1[0]
                else:
                    net = None
                if net is None:
                    hd = 64
                elif hasattr(net, "proj"):   # ResNet
                    hd = net.proj.out_features
                elif hasattr(net, "net"):    # MLP
                    hd = net.net[0].out_features
                else:
                    hd = 64
            except Exception:
                hd = 64
            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_loss,
                extra={
                    "train_losses": epoch_train,
                    "val_losses":   epoch_val,
                    "model_type":   model_type,
                    "num_bins":     model.num_bins,
                    "num_splines":  model.num_splines,
                    "bound":        model.bound if hasattr(model, "bound") else None,
                    "hidden_dim":   hd,
                    "num_layers":   num_layers,
                    "arch":         arch,
                    "activation":   activation,
                })
        else:
            patience_counter += 1

        # ------------------------------------------------------------------ #
        # Logging
        # ------------------------------------------------------------------ #
        if epoch % log_every == 0:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:5d}/{max_epochs} "
                  f"| Train: {avg_train:9.4f} "
                  f"| Val: {avg_val:9.4f} "
                  f"| Smooth: {smoothed_val:9.4f} "
                  f"| LR: {lr_now:.2e} "
                  f"| Patience: {patience_counter}/{patience} "
                  f"| Elapsed: {elapsed:.0f}s")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    print(f"\nBest val loss = {best_val_loss:.6f}")
    print(f"Checkpoint saved at: {best_path}")

    # Restore best weights
    from .utils import load_checkpoint
    load_checkpoint(best_path, model, device=device)

    losses = {"train": epoch_train, "val": epoch_val}
    save_losses(losses, Path(save_dir) / f"{run_name}_losses.json")

    return model, losses
