"""
plotting.py
===========
All visualisation routines for the HEP normalising-flow package.

Functions
---------
plot_loss_curves          — training / validation loss vs epoch.
plot_marginal_1d          — 1-D histograms comparing data and flow samples.
plot_marginal_2d          — 2-D scatter plots of marginal pairs.
plot_mollweide_kde        — Mollweide sky-map density (KDE) on S².
plot_physics_comparison   — 6-panel (px, py, pz, pT, η, φ) comparison.
plot_base_mapping         — Visualise how the base space maps to data space.
plot_jacobian_map         — Colour-coded log-det-Jacobian scatter in data space.
plot_radius_distribution  — Histogram of generated |p| in GeV.

All functions accept a ``save`` / ``save_path`` argument:
  - ``save=True``  → auto-generate filename from title.
  - string         → explicit filename.
  - ``False``      → do not save.

Colour conventions follow your notebook:
  Data = blue  (#00234e)
  Flow = red   (#A61200)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde

# Prefer a non-interactive backend when running headless on clusters
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

TRUE_COLOR  = "#00234e"
MODEL_COLOR = "#A61200"
DPI         = 400


def _setup_style() -> None:
    """Apply consistent matplotlib style."""
    rcParams["font.family"]      = "serif"
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams["axes.grid"]        = True
    rcParams["grid.linestyle"]   = "--"
    rcParams["grid.alpha"]       = 0.5


_setup_style()


def _savefig(fig: plt.Figure,
             save: bool | str,
             default_name: str,
             output_dir: str = "outputs") -> None:
    """Conditionally save a figure."""
    if save is False:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = save if isinstance(save, str) else default_name
    fig.savefig(Path(output_dir) / fname, dpi=DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(train_losses: list[float],
                     val_losses: list[float],
                     title: str = "Training Curves",
                     save: bool | str = True,
                     output_dir: str = "outputs") -> plt.Figure:
    """Plot training and validation NLL loss vs epoch.

    Parameters
    ----------
    train_losses, val_losses : list of float
    title : str
    save : bool or str
    output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=TRUE_COLOR,  lw=2, label="Train")
    ax.plot(epochs, val_losses,   color=MODEL_COLOR, lw=2, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save, "loss_curves.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# 1-D marginals
# ---------------------------------------------------------------------------

def plot_marginal_1d(data: np.ndarray,
                     samples: np.ndarray,
                     labels: Optional[list[str]] = None,
                     title: str = "1-D Marginals",
                     density: bool = True,
                     n_bins: int = 50,
                     save: bool | str = True,
                     output_dir: str = "outputs") -> plt.Figure:
    """Overlay histograms of data vs flow samples for each feature.

    Parameters
    ----------
    data : ndarray, shape ``(N, D)``
    samples : ndarray, shape ``(M, D)``
    labels : list of str, length ``D`` (optional)
    title : str
    density : bool  — normalise histograms to density (default True).
    n_bins : int
    save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    D = data.shape[1]
    if labels is None:
        labels = [f"$x_{{{i}}}$" for i in range(D)]

    ncols = min(D, 3)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i in range(D):
        ax = axes[i]
        d = data[:, i][np.isfinite(data[:, i])]
        s = samples[:, i][np.isfinite(samples[:, i])]

        combined = np.concatenate([d, s])
        lo, hi = np.percentile(combined, 0.1), np.percentile(combined, 99.9)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(d, bins=bins, density=density, histtype="step",
                lw=2.5, color=TRUE_COLOR,  label="Data")
        ax.hist(s, bins=bins, density=density, histtype="step",
                lw=2.5, color=MODEL_COLOR, label="Flow")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("Density" if density else "Counts")
        ax.legend(fontsize=9)

    # Hide unused panels
    for j in range(D, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, save, "marginals_1d.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# 2-D scatter pairs
# ---------------------------------------------------------------------------

def plot_marginal_2d(data: np.ndarray,
                     samples: np.ndarray,
                     labels: Optional[list[str]] = None,
                     title: str = "2-D Marginals",
                     max_points: int = 5000,
                     save: bool | str = True,
                     output_dir: str = "outputs") -> plt.Figure:
    """Scatter-plot all pairs of features, data vs flow.

    Parameters
    ----------
    data : ndarray, shape ``(N, D)``
    samples : ndarray, shape ``(M, D)``
    max_points : int — sub-sample for faster rendering.

    Returns
    -------
    matplotlib Figure
    """
    D = data.shape[1]
    if labels is None:
        labels = [f"$x_{{{i}}}$" for i in range(D)]

    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    ncols = min(len(pairs), 3)
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    idx_d = np.random.choice(len(data),    min(max_points, len(data)),    replace=False)
    idx_s = np.random.choice(len(samples), min(max_points, len(samples)), replace=False)

    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        ax.scatter(data[idx_d, i],    data[idx_d, j],
                   s=4, alpha=0.4, color=TRUE_COLOR,  label="Data", rasterized=True)
        ax.scatter(samples[idx_s, i], samples[idx_s, j],
                   s=4, alpha=0.4, color=MODEL_COLOR, label="Flow", rasterized=True)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.legend(markerscale=2, fontsize=9)

    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, save, "marginals_2d.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Mollweide KDE
# ---------------------------------------------------------------------------

def plot_mollweide_kde(cyl_data: torch.Tensor | np.ndarray,
                       title: str = "KDE Density",
                       bw: float = 0.2,
                       grid_lon: int = 150,
                       grid_lat: int = 75,
                       save: bool | str = True,
                       output_dir: str = "outputs") -> plt.Figure:
    """Mollweide sky-map showing KDE density of ``(cos θ, φ)`` samples.

    Parameters
    ----------
    cyl_data : Tensor or ndarray, shape ``(N, 2)`` — ``(cos θ, φ)``.
    title : str
    bw : float — KDE bandwidth.
    grid_lon, grid_lat : int — resolution of the evaluation grid.
    save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    if isinstance(cyl_data, tuple):
        cyl_data = cyl_data[0]
    if isinstance(cyl_data, torch.Tensor):
        cyl_data = cyl_data.detach().cpu()

    # Mollweide coordinates
    lat = np.arcsin(np.clip(np.array(cyl_data[:, 0]), -1.0, 1.0))      # latitude [-π/2, π/2]
    lon = (np.array(cyl_data[:, 1]) - np.pi)                            # longitude [-π, π]

    lon_g, lat_g = np.meshgrid(
        np.linspace(-np.pi,    np.pi,    grid_lon),
        np.linspace(-np.pi/2, np.pi/2, grid_lat)
    )
    kde = gaussian_kde(np.vstack([lon, lat]), bw_method=bw)
    zi  = kde(np.vstack([lon_g.ravel(), lat_g.ravel()])).reshape(lon_g.shape)

    fig = plt.figure(figsize=(10, 5))
    ax  = fig.add_subplot(111, projection="mollweide")
    im  = ax.pcolormesh(lon_g, lat_g, zi, cmap="viridis", shading="auto")
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, orientation="horizontal", pad=0.1, label="Density")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    _savefig(fig, save, f"{title.replace(' ', '_')}.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Physics comparison (px, py, pz, pT, η, φ)
# ---------------------------------------------------------------------------

def plot_physics_comparison(data_cart: torch.Tensor,
                             sample_cart: torch.Tensor,
                             title: str = "Physics Comparison",
                             n_bins: int = 50,
                             density: bool = True,
                             save: bool | str = True,
                             output_dir: str = "outputs") -> plt.Figure:
    """Six-panel comparison of derived physics variables.

    Panels: ``px, py, pz, pT, η, φ``.

    Parameters
    ----------
    data_cart : Tensor, shape ``(N, 3)`` — true ``(px, py, pz)`` in GeV.
    sample_cart : Tensor, shape ``(M, 3)`` — generated momenta.
    title : str
    n_bins : int
    density : bool
    save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    from .utils import cartesian_to_physics

    def _get(p: torch.Tensor) -> list[np.ndarray]:
        d = cartesian_to_physics(p)
        return [d["px"].numpy(), d["py"].numpy(), d["pz"].numpy(),
                d["pT"].numpy(), d["eta"].numpy(), d["phi"].numpy()]

    d_vals = _get(data_cart)
    s_vals = _get(sample_cart)
    var_labels = [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]",
                  r"$p_T$ [GeV]", r"$\eta$",       r"$\phi$ [rad]"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, ax in enumerate(axes.flatten()):
        d = d_vals[i][np.isfinite(d_vals[i])]
        s = s_vals[i][np.isfinite(s_vals[i])]

        lo = np.percentile(d, 0.1);  hi = np.percentile(d, 99.9)
        if lo == hi:
            lo -= 1; hi += 1
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(d, bins=bins, density=density, histtype="step",
                lw=2.5, color=TRUE_COLOR,  label="Data")
        ax.hist(s, bins=bins, density=density, histtype="step",
                lw=2.5, color=MODEL_COLOR, label="Flow")
        ax.set_xlabel(var_labels[i], fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, save, "physics_comparison.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Base-to-data mapping
# ---------------------------------------------------------------------------

def plot_base_mapping(model: torch.nn.Module,
                      device: torch.device,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      dim: int = 2,
                      n_r: int = 50,
                      n_phi: int = 100,
                      r_max: float = 4.0,
                      title: str = "Base Mapping",
                      save: bool | str = True,
                      output_dir: str = "outputs") -> plt.Figure:
    """Visualise how a polar grid in the base space maps to data space.

    Works for 2-D models only (``dim=2``).

    Parameters
    ----------
    model : nn.Module with ``forward(z, inverse=True)``.
    device : torch.device
    mean, std : Tensors from ``normalise()``.
    dim : int — must be 2.
    n_r, n_phi : int — radial / angular grid resolution.
    r_max : float — maximum radial distance in base space (Gaussian sigmas).
    title, save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    if dim != 2:
        raise ValueError("plot_base_mapping only supports 2-D models.")

    r_vals   = np.linspace(0.01, r_max, n_r)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    R, PHI   = np.meshgrid(r_vals, phi_vals)
    z1 = (R * np.cos(PHI)).ravel()
    z2 = (R * np.sin(PHI)).ravel()
    z_t = torch.tensor(np.stack([z1, z2], axis=1), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        x_norm = model.forward(z_t, inverse=True)
        x_phys = x_norm * std.to(device) + mean.to(device)
        x_np   = x_phys.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(x_np[:, 0], x_np[:, 1],
                          c=R.ravel(), cmap="viridis", s=4, rasterized=True)
    axes[0].set_title(r"Coloured by base radius $r$")
    axes[0].set_xlabel(r"$x_0$");  axes[0].set_ylabel(r"$x_1$")
    plt.colorbar(sc1, ax=axes[0], label="Gaussian sigmas")

    sc2 = axes[1].scatter(x_np[:, 0], x_np[:, 1],
                          c=PHI.ravel(), cmap="hsv", s=4, rasterized=True)
    axes[1].set_title(r"Coloured by base angle $\phi_{\rm base}$")
    axes[1].set_xlabel(r"$x_0$")
    plt.colorbar(sc2, ax=axes[1], label="Base angle [rad]")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _savefig(fig, save, "base_mapping.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Jacobian map
# ---------------------------------------------------------------------------

def plot_jacobian_map(model: torch.nn.Module,
                      device: torch.device,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      n_samples: int = 10_000,
                      title: str = "Log-Det-Jacobian Map",
                      save: bool | str = True,
                      output_dir: str = "outputs") -> plt.Figure:
    """Scatter plot colour-coded by log-det-Jacobian value.

    Parameters
    ----------
    model : nn.Module with ``forward(x, inverse=False)`` → ``(z, ldj)``.
    device : torch.device
    mean, std : Tensors from ``normalise()``.
    n_samples : int — number of samples drawn from the base.
    title, save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    dim = sum(1 for _ in model.parameters())  # rough proxy — just sample
    model.eval()
    with torch.no_grad():
        try:
            z = model.sample(n_samples, device=device)
        except AttributeError:
            raise AttributeError("Model must implement a .sample() method.")
        x_phys = z * std.to(device) + mean.to(device)
        z_norm = (x_phys - mean.to(device)) / std.to(device)
        _, ldj = model.forward(z_norm, inverse=False)

    x_np   = x_phys.cpu().numpy()
    ldj_np = ldj.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x_np[:, 0], x_np[:, 1],
                    c=ldj_np, cmap="magma", s=3, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label=r"$\ln|\det J|$")
    ax.set_xlabel(r"$x_0$");  ax.set_ylabel(r"$x_1$")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    _savefig(fig, save, "jacobian_map.png", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Radius distribution (R³ model)
# ---------------------------------------------------------------------------

def plot_radius_distribution(samples_cart: np.ndarray | torch.Tensor,
                              title: str = "Generated |p| Distribution",
                              n_bins: int = 50,
                              save: bool | str = True,
                              output_dir: str = "outputs") -> plt.Figure:
    """Histogram of the generated momentum magnitude ``|p|``.

    Parameters
    ----------
    samples_cart : ndarray or Tensor, shape ``(N, 3)`` — physical GeV.
    title, n_bins, save, output_dir : str

    Returns
    -------
    matplotlib Figure
    """
    if isinstance(samples_cart, torch.Tensor):
        samples_cart = samples_cart.detach().cpu().numpy()
    radii = np.linalg.norm(samples_cart, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    # Fall back to 'auto' binning if the data range is too narrow for n_bins
    r_min, r_max = radii.min(), radii.max()
    bins = n_bins if (r_max - r_min) > 1e-6 * n_bins else "auto"
    ax.hist(radii, bins=bins, color=MODEL_COLOR, alpha=0.75,
            histtype="stepfilled", edgecolor=MODEL_COLOR, lw=1.5)
    ax.set_xlabel(r"$|p|$ [GeV]", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(0.97, 0.95,
            f"mean = {np.mean(radii):.1f} GeV\nstd = {np.std(radii):.1f} GeV",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    fig.tight_layout()
    _savefig(fig, save, "radius_distribution.png", output_dir)
    return fig
