"""
analysis.py
===========
Statistical analysis and model-evaluation routines.

Functions
---------
kde_s2
    Kernel-density estimate on S² using 2-D Gaussian KDE.
kl_divergence_kde
    KL divergence D_KL(data ‖ flow) estimated via KDE.
effective_sample_size
    Effective sample size (ESS) as an importance-weighting measure.
wasserstein_1d
    Per-feature 1-D Wasserstein-1 (earth-mover) distance.
js_divergence_1d
    Per-feature 1-D Jensen–Shannon divergence (histogram-based).
consistency_report
    Full console report: unphysical fraction, radius stats.
model_summary
    Print architecture, parameter count, and spline configuration.
"""

from __future__ import annotations

import textwrap
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.special import rel_entr


# ---------------------------------------------------------------------------
# KDE on S²
# ---------------------------------------------------------------------------

def kde_s2(cos_theta: np.ndarray,
           phi: np.ndarray,
           bw: float = 0.2) -> gaussian_kde:
    """Construct a 2-D Gaussian KDE for ``(cos θ, φ)`` data.

    The KDE operates in ``(φ, π/2 − θ)`` space, which maps the sphere
    to a flat rectangle and allows a standard Gaussian kernel.

    Parameters
    ----------
    cos_theta : ndarray, shape ``(N,)``
    phi : ndarray, shape ``(N,)``
    bw : float — KDE bandwidth (default 0.2).

    Returns
    -------
    scipy.stats.gaussian_kde
    """
    pts = np.vstack([phi, np.pi / 2.0 - np.arccos(np.clip(cos_theta, -1, 1))])
    return gaussian_kde(pts, bw_method=bw)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_divergence_kde(cos_theta_data: np.ndarray,
                       phi_data: np.ndarray,
                       cos_theta_flow: np.ndarray,
                       phi_flow: np.ndarray,
                       bw: float = 0.2,
                       n_eval: Optional[int] = None) -> float:
    """Estimate D_KL(data ‖ flow) using Gaussian KDE on S².

    D_KL(P ‖ Q) = E_P [ log P(x) − log Q(x) ]

    Evaluated at the *data* points.

    Parameters
    ----------
    cos_theta_data, phi_data : ndarray — true samples.
    cos_theta_flow, phi_flow : ndarray — flow samples.
    bw : float — KDE bandwidth.
    n_eval : int, optional — sub-sample evaluation points for speed.

    Returns
    -------
    float — KL divergence (nats).  Positive means the flow is a worse
    model; near-zero means good agreement.
    """
    kde_p = kde_s2(cos_theta_data, phi_data,  bw=bw)
    kde_q = kde_s2(cos_theta_flow, phi_flow,  bw=bw)

    # Evaluate at data points
    if n_eval is not None and n_eval < len(cos_theta_data):
        idx = np.random.choice(len(cos_theta_data), n_eval, replace=False)
        pts = np.vstack([phi_data[idx],
                         np.pi / 2.0 - np.arccos(np.clip(cos_theta_data[idx], -1, 1))])
    else:
        pts = np.vstack([phi_data,
                         np.pi / 2.0 - np.arccos(np.clip(cos_theta_data, -1, 1))])

    log_p = np.log(kde_p(pts) + 1e-12)
    log_q = np.log(kde_q(pts) + 1e-12)
    return float(np.mean(log_p - log_q))


# ---------------------------------------------------------------------------
# ESS
# ---------------------------------------------------------------------------

def effective_sample_size(cos_theta_data: np.ndarray,
                            phi_data: np.ndarray,
                            cos_theta_flow: np.ndarray,
                            phi_flow: np.ndarray,
                            bw: float = 0.2) -> float:
    """Importance-weighted Effective Sample Size (ESS).

    ESS = (∑ w_i)² / ∑ w_i²,   w_i = P(x_i) / Q(x_i)

    A high ESS (close to N) indicates that the flow closely matches the
    true density.

    Parameters
    ----------
    cos_theta_data, phi_data : ndarray — true samples.
    cos_theta_flow, phi_flow : ndarray — flow samples.
    bw : float

    Returns
    -------
    float — ESS value in [1, N_data].
    """
    kde_p = kde_s2(cos_theta_data, phi_data,  bw=bw)
    kde_q = kde_s2(cos_theta_flow, phi_flow,  bw=bw)

    pts = np.vstack([phi_data,
                     np.pi / 2.0 - np.arccos(np.clip(cos_theta_data, -1, 1))])
    w  = kde_p(pts) / (kde_q(pts) + 1e-12)
    w /= np.mean(w)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


# ---------------------------------------------------------------------------
# 1-D Wasserstein & Jensen–Shannon
# ---------------------------------------------------------------------------

def wasserstein_1d(data: np.ndarray,
                   samples: np.ndarray) -> np.ndarray:
    """Per-feature 1-D Wasserstein-1 (earth-mover) distance.

    Parameters
    ----------
    data : ndarray, shape ``(N, D)``
    samples : ndarray, shape ``(M, D)``

    Returns
    -------
    ndarray, shape ``(D,)``
    """
    D = data.shape[1]
    return np.array([wasserstein_distance(data[:, i], samples[:, i])
                     for i in range(D)])


def js_divergence_1d(data: np.ndarray,
                     samples: np.ndarray,
                     n_bins: int = 100) -> np.ndarray:
    """Per-feature 1-D Jensen–Shannon divergence (histogram-based).

    Uses a shared bin grid built from the combined range so that both
    distributions are defined on the same support.

    Parameters
    ----------
    data : ndarray, shape ``(N, D)``
    samples : ndarray, shape ``(M, D)``
    n_bins : int

    Returns
    -------
    ndarray, shape ``(D,)``  — values in [0, ln 2] nats.
    """
    D = data.shape[1]
    jsd = np.zeros(D)
    for i in range(D):
        lo = min(data[:, i].min(), samples[:, i].min())
        hi = max(data[:, i].max(), samples[:, i].max())
        bins = np.linspace(lo, hi, n_bins + 1)

        p_hist, _ = np.histogram(data[:, i],    bins=bins, density=True)
        q_hist, _ = np.histogram(samples[:, i], bins=bins, density=True)

        # Normalise to probability masses (integrate → 1)
        bin_w = bins[1] - bins[0]
        p = p_hist * bin_w + 1e-12
        q = q_hist * bin_w + 1e-12
        m = 0.5 * (p + q)
        jsd[i] = 0.5 * (np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m)))
    return jsd


# ---------------------------------------------------------------------------
# Consistency report
# ---------------------------------------------------------------------------

def consistency_report(cyl_samples: torch.Tensor,
                        target_r: float = 500.0) -> dict:
    """Analyse ``(cos θ, φ)`` samples for physical validity.

    Checks whether generated angular coordinates fall within their physical
    bounds (cos θ ∈ [−1, 1], φ ∈ [0, 2π]) and reports momentum-radius
    statistics for valid points.

    Parameters
    ----------
    cyl_samples : Tensor, shape ``(N, 2)``
    target_r : float — assumed momentum magnitude used for radius calc.

    Returns
    -------
    dict with keys:
        total, n_unphysical, frac_unphysical,
        n_polar_fail, n_azimuth_fail,
        radius_mean, radius_std
    """
    cos_theta = cyl_samples[:, 0]
    phi       = cyl_samples[:, 1]

    out_polar   = (cos_theta < -1.0) | (cos_theta >  1.0)
    out_azimuth = (phi        <  0.0) | (phi        > 2 * np.pi)
    unphysical  = out_polar | out_azimuth

    n_total    = len(cyl_samples)
    n_unphys   = int(unphysical.sum().item())
    n_polar    = int(out_polar.sum().item())
    n_azimuth  = int(out_azimuth.sum().item())
    frac       = n_unphys / max(n_total, 1)

    # Radius for valid points
    valid = ~unphysical
    ct_v  = cos_theta[valid]
    phi_v = phi[valid]
    sin_t = torch.sqrt((1.0 - ct_v ** 2).clamp(min=0.0))
    px = target_r * sin_t * torch.cos(phi_v)
    py = target_r * sin_t * torch.sin(phi_v)
    pz = target_r * ct_v
    radii = torch.norm(torch.stack([px, py, pz], dim=1), dim=1).numpy()

    result = {
        "total":             n_total,
        "n_unphysical":      n_unphys,
        "frac_unphysical":   frac,
        "n_polar_fail":      n_polar,
        "n_azimuth_fail":    n_azimuth,
        "radius_mean":       float(np.mean(radii)),
        "radius_std":        float(np.std(radii)),
    }

    print("\n" + "=" * 50)
    print("  Consistency Report")
    print("=" * 50)
    print(f"  Total samples      : {n_total}")
    print(f"  Unphysical points  : {n_unphys}  ({frac*100:.2f}%)")
    print(f"    Polar failure    : {n_polar}")
    print(f"    Azimuth failure  : {n_azimuth}")
    print(f"  Radius (valid pts) : {result['radius_mean']:.2f} ± "
          f"{result['radius_std']:.2f} GeV")
    print("=" * 50 + "\n")
    return result


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def evaluate(cos_theta_data: np.ndarray,
             phi_data: np.ndarray,
             cos_theta_flow: np.ndarray,
             phi_flow: np.ndarray,
             bw: float = 0.2,
             n_eval: Optional[int] = 5000,
             verbose: bool = True) -> dict:
    """Run the full suite of evaluation metrics.

    Computes KL divergence, ESS, and 1-D Wasserstein distances for
    ``(cos θ, φ)``.

    Parameters
    ----------
    cos_theta_data, phi_data : ndarray — true samples.
    cos_theta_flow, phi_flow : ndarray — flow samples.
    bw : float
    n_eval : int — sub-sample size for KDE evaluation.
    verbose : bool — print a formatted summary.

    Returns
    -------
    dict with keys: ``kl``, ``ess``, ``ess_frac``,
                    ``W1_costheta``, ``W1_phi``
    """
    kl  = kl_divergence_kde(cos_theta_data, phi_data,
                             cos_theta_flow, phi_flow, bw=bw, n_eval=n_eval)
    ess = effective_sample_size(cos_theta_data, phi_data,
                                 cos_theta_flow, phi_flow, bw=bw)
    n = len(cos_theta_data)

    data_2d    = np.stack([cos_theta_data, phi_data],   axis=1)
    samples_2d = np.stack([cos_theta_flow, phi_flow],   axis=1)
    w1 = wasserstein_1d(data_2d, samples_2d)
    jsd = js_divergence_1d(data_2d, samples_2d)

    result = {
        "kl":           kl,
        "ess":          ess,
        "ess_frac":     ess / max(n, 1),
        "W1_costheta":  float(w1[0]),
        "W1_phi":       float(w1[1]),
        "JSD_costheta": float(jsd[0]),
        "JSD_phi":      float(jsd[1]),
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  Evaluation Metrics")
        print("=" * 50)
        print(f"  KL(data ‖ flow)  : {kl:.4f} nats")
        print(f"  ESS              : {ess:.1f} / {n}  ({ess/n*100:.1f}%)")
        print(f"  W1(cos θ)        : {w1[0]:.4e}")
        print(f"  W1(φ)            : {w1[1]:.4e}")
        print(f"  JSD(cos θ)       : {jsd[0]:.4e}")
        print(f"  JSD(φ)           : {jsd[1]:.4e}")
        print("=" * 50 + "\n")

    return result


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def model_summary(model: nn.Module,
                  model_type: str = "",
                  verbose: bool = True) -> dict:
    """Print a concise model summary.

    Parameters
    ----------
    model : nn.Module
    model_type : str — descriptive label.
    verbose : bool

    Returns
    -------
    dict with keys: ``total_params``, ``trainable_params``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {"total_params": total, "trainable_params": trainable}

    if verbose:
        print("\n" + "=" * 50)
        print(f"  Model Summary  {model_type}")
        print("=" * 50)
        print(f"  Total parameters     : {total:,}")
        print(f"  Trainable parameters : {trainable:,}")
        # Print num_splines if the attribute exists
        for attr in ("num_splines", "num_bins", "bound"):
            if hasattr(model, attr):
                print(f"  {attr:<22}: {getattr(model, attr)}")
        print("=" * 50 + "\n")

    return result
