"""
splines.py
==========
Rational-Quadratic Spline (RQS) bijections for normalising flows.

Reference: Durkan et al., "Neural Spline Flows" (NeurIPS 2019).

Functions
---------
rqs               -- Unified entry point for multi-dim standardised flows (r2, r3).
                     inputs: (B, D)  w/h: (B, D, K)  d: (B, D, K+1)
rqs_with_bounds   -- 1-D spline with explicit asymmetric domains (s2, s2rec).
                     inputs: (B,)    w/h: (B, K)     d: (B, K+1)
rqs_circular      -- 1-D circular spline: d has K elements, d[0]==d[K] enforced.
                     inputs: (B,)    w/h: (B, K)     d: (B, K)
rqs_forward       -- Forward-only multi-dim (kept for backward compatibility).
rqs_inverse       -- Inverse-only multi-dim (kept for backward compatibility).
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rqs_1d_forward(inputs, widths, heights, derivatives, cum_widths, cum_heights, K):
    """Forward pass of 1-D RQS. All tensors already processed (softmax applied)."""
    bin_idx = (
        torch.searchsorted(
            cum_widths.contiguous(),
            inputs.unsqueeze(-1).contiguous(),
            right=True) - 1
    ).clamp(0, K - 1)                       # (B, 1)

    w_b   = torch.gather(widths,      -1, bin_idx).squeeze(-1)
    h_b   = torch.gather(heights,     -1, bin_idx).squeeze(-1)
    d_k   = torch.gather(derivatives, -1, bin_idx).squeeze(-1)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1).squeeze(-1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx).squeeze(-1)
    y_k   = torch.gather(cum_heights, -1, bin_idx).squeeze(-1)
    s_b   = h_b / w_b

    xi   = (inputs - x_k) / w_b
    den  = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    num_ = h_b * (s_b * xi ** 2 + d_k * xi * (1.0 - xi))
    outputs = y_k + num_ / den

    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k  * (1.0 - xi) ** 2
    )
    logabsdet = torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)
    return outputs, logabsdet


def _rqs_1d_inverse(inputs, widths, heights, derivatives, cum_widths, cum_heights, K):
    """Inverse pass of 1-D RQS. All tensors already processed (softmax applied)."""
    bin_idx = (
        torch.searchsorted(
            cum_heights.contiguous(),
            inputs.unsqueeze(-1).contiguous(),
            right=True) - 1
    ).clamp(0, K - 1)                       # (B, 1)

    w_b   = torch.gather(widths,      -1, bin_idx).squeeze(-1)
    h_b   = torch.gather(heights,     -1, bin_idx).squeeze(-1)
    d_k   = torch.gather(derivatives, -1, bin_idx).squeeze(-1)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1).squeeze(-1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx).squeeze(-1)
    y_k   = torch.gather(cum_heights, -1, bin_idx).squeeze(-1)
    s_b   = h_b / w_b

    y_rel  = inputs - y_k
    a      = h_b * (s_b - d_k) + y_rel * (d_kp1 + d_k - 2.0 * s_b)
    b_coef = h_b * d_k          - y_rel * (d_kp1 + d_k - 2.0 * s_b)
    c      = -s_b * y_rel

    xi      = 2.0 * c / (-b_coef - torch.sqrt(b_coef ** 2 - 4.0 * a * c + 1e-9))
    outputs = xi * w_b + x_k

    den       = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k  * (1.0 - xi) ** 2
    )
    logabsdet = -(torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9))
    return outputs, logabsdet


# ---------------------------------------------------------------------------
# Public API — 1-D splines  (used by s2, s2norm, s2rec)
# ---------------------------------------------------------------------------

def rqs_with_bounds(
        inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
        b_x: tuple[float, float] = (-1.0, 1.0),
        b_y: tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D Rational-Quadratic Spline with explicit asymmetric domains.

    Used for cos_theta ∈ (-1, 1) in s2 and s2rec.

    Parameters
    ----------
    inputs : Tensor, shape ``(B,)``
    w, h   : Tensor, shape ``(B, K)`` — unnormalised width/height logits.
    d      : Tensor, shape ``(B, K+1)`` — unnormalised derivative logits.
    inverse : bool
        ``False`` = data→base (training).  ``True`` = base→data (sampling).
    b_x : (left, right) — input domain.
    b_y : (bottom, top) — output domain.
    eps : float — clamping margin.

    Returns
    -------
    outputs   : Tensor, shape ``(B,)``
    logabsdet : Tensor, shape ``(B,)``  — log |dy/dx| (forward) or log |dx/dy| (inverse).
    """
    left, right   = b_x
    bottom, top   = b_y
    K             = w.shape[-1]

    widths      = F.softmax(w, dim=-1) * (right - left)           # (B, K)
    heights     = F.softmax(h, dim=-1) * (top - bottom)           # (B, K)
    derivatives = F.softplus(d) + 1e-3                            # (B, K+1)

    cum_widths  = F.pad(torch.cumsum(widths,  dim=-1), (1, 0), value=0.0) + left
    cum_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    if not inverse:
        inputs = inputs.clamp(left + eps, right - eps)
        return _rqs_1d_forward(inputs, widths, heights, derivatives,
                                cum_widths, cum_heights, K)
    else:
        inputs = inputs.clamp(bottom + eps, top - eps)
        return _rqs_1d_inverse(inputs, widths, heights, derivatives,
                                cum_widths, cum_heights, K)


def rqs_circular(
        inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
        b_x: tuple[float, float] = (0.0, 2.0 * math.pi),
        b_y: tuple[float, float] = (0.0, 2.0 * math.pi),
        eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D CIRCULAR Rational-Quadratic Spline for periodic φ.

    Enforces d[0] == d[K] (equal derivatives at both boundaries), making
    the density smooth across the wrap-around point φ=0 / φ=2π.

    Used for phi ∈ (0, 2π) in s2, s2norm, s2rec.

    Parameters
    ----------
    inputs : Tensor, shape ``(B,)``
    w, h   : Tensor, shape ``(B, K)``
    d      : Tensor, shape ``(B, K)``  — K derivatives.
              ``d[:, K]`` is set equal to ``d[:, 0]`` internally.
    inverse : bool
    b_x, b_y : domains (both default to (0, 2π)).
    eps : float

    Returns
    -------
    outputs   : Tensor, shape ``(B,)``
    logabsdet : Tensor, shape ``(B,)``
    """
    # Enforce periodicity: derivative at right boundary = derivative at left boundary
    d_ext = torch.cat([d, d[:, 0:1]], dim=-1)   # (B, K) → (B, K+1)
    return rqs_with_bounds(inputs, w, h, d_ext,
                           inverse=inverse, b_x=b_x, b_y=b_y, eps=eps)


# ---------------------------------------------------------------------------
# Public API — multi-dim splines  (used by r2, r3, s2norm cos_theta dim)
# ---------------------------------------------------------------------------

def rqs_forward(
        inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        bound: float = 1.0,
        eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: data → base, domain ``[-bound, bound]`` (symmetric).

    Parameters
    ----------
    inputs : Tensor, shape ``(B,)`` or ``(B, D)``
    w, h   : Tensor, shape ``(B, K)`` or ``(B, D, K)``
    d      : Tensor, shape ``(B, K+1)`` or ``(B, D, K+1)``
    bound  : float — spline half-domain.

    Returns
    -------
    outputs   : same shape as ``inputs``
    logabsdet : same shape as ``inputs``
    """
    left = right = bottom = top = None   # suppress lint
    left, right  = -bound, bound
    bottom, top  = -bound, bound
    K = w.shape[-1]

    inputs = inputs.clamp(left + eps, right - eps)

    widths      = F.softmax(w, dim=-1) * (right - left)
    heights     = F.softmax(h, dim=-1) * (top - bottom)
    derivatives = F.softplus(d) + 1e-3

    cum_widths  = F.pad(torch.cumsum(widths,  dim=-1), (1, 0), value=0.0) + left
    cum_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    bin_idx = (
        torch.searchsorted(cum_widths, inputs.unsqueeze(-1).contiguous(), right=True) - 1
    ).clamp(0, K - 1)

    w_b   = torch.gather(widths,      -1, bin_idx)
    h_b   = torch.gather(heights,     -1, bin_idx)
    d_k   = torch.gather(derivatives, -1, bin_idx)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx)
    y_k   = torch.gather(cum_heights, -1, bin_idx)
    s_b   = h_b / w_b

    xi   = (inputs.unsqueeze(-1) - x_k) / w_b
    den  = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    num_ = h_b * (s_b * xi ** 2 + d_k * xi * (1.0 - xi))
    outputs = (y_k + num_ / den).squeeze(-1)

    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k  * (1.0 - xi) ** 2
    )
    logabsdet = (torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


def rqs_inverse(
        inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        bound: float = 1.0,
        eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse pass: base → data, domain ``[-bound, bound]`` (symmetric).

    Parameters
    ----------
    inputs : Tensor, shape ``(B,)`` or ``(B, D)``
    w, h   : Tensor, shape ``(B, K)`` or ``(B, D, K)``
    d      : Tensor, shape ``(B, K+1)`` or ``(B, D, K+1)``
    bound  : float

    Returns
    -------
    outputs   : same shape as ``inputs``
    logabsdet : same shape as ``inputs``  — log |dx/dy|.
    """
    left, right  = -bound, bound
    bottom, top  = -bound, bound
    K = w.shape[-1]

    inputs = inputs.clamp(bottom + eps, top - eps)

    widths      = F.softmax(w, dim=-1) * (right - left)
    heights     = F.softmax(h, dim=-1) * (top - bottom)
    derivatives = F.softplus(d) + 1e-3

    cum_widths  = F.pad(torch.cumsum(widths,  dim=-1), (1, 0), value=0.0) + left
    cum_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) + bottom

    bin_idx = (
        torch.searchsorted(cum_heights, inputs.unsqueeze(-1).contiguous(), right=True) - 1
    ).clamp(0, K - 1)

    w_b   = torch.gather(widths,      -1, bin_idx)
    h_b   = torch.gather(heights,     -1, bin_idx)
    d_k   = torch.gather(derivatives, -1, bin_idx)
    d_kp1 = torch.gather(derivatives, -1, bin_idx + 1)
    x_k   = torch.gather(cum_widths,  -1, bin_idx)
    y_k   = torch.gather(cum_heights, -1, bin_idx)
    s_b   = h_b / w_b

    y_rel  = inputs.unsqueeze(-1) - y_k
    a      = h_b * (s_b - d_k) + y_rel * (d_kp1 + d_k - 2.0 * s_b)
    b_coef = h_b * d_k          - y_rel * (d_kp1 + d_k - 2.0 * s_b)
    c      = -s_b * y_rel

    xi      = 2.0 * c / (-b_coef - torch.sqrt(b_coef ** 2 - 4.0 * a * c + 1e-9))
    outputs = (xi * w_b + x_k).squeeze(-1)

    den       = s_b + (d_kp1 + d_k - 2.0 * s_b) * xi * (1.0 - xi)
    deriv_num = s_b ** 2 * (
        d_kp1 * xi ** 2
        + 2.0 * s_b * xi * (1.0 - xi)
        + d_k  * (1.0 - xi) ** 2
    )
    logabsdet = -(torch.log(deriv_num + 1e-9) - 2.0 * torch.log(den + 1e-9)).squeeze(-1)
    return outputs, logabsdet


def rqs(
        inputs: torch.Tensor,
        w: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        inverse: bool = False,
        bound: float = 1.0,
        eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified RQS: calls forward or inverse based on ``inverse`` flag.

    Handles both 1-D ``(B,)`` and multi-dim ``(B, D)`` inputs automatically.
    Domain is symmetric ``[-bound, bound]``.

    Parameters
    ----------
    inputs : Tensor, shape ``(B,)`` or ``(B, D)``
    w, h   : Tensor, shape ``(B, K)`` or ``(B, D, K)``
    d      : Tensor, shape ``(B, K+1)`` or ``(B, D, K+1)``
    inverse : bool
    bound : float

    Returns
    -------
    outputs   : same shape as ``inputs``
    logabsdet : same shape as ``inputs``
    """
    if inverse:
        return rqs_inverse(inputs, w, h, d, bound=bound, eps=eps)
    return rqs_forward(inputs, w, h, d, bound=bound, eps=eps)
