"""
networks.py
===========
Flow model classes.

Naming convention
-----------------
s2  --  RecursiveSphereFlow
        Works in PHYSICAL (cos_theta, phi) coordinates.
        Base distribution: UNIFORM ON S2 (cos_theta ~ U[-1,1], phi ~ U[0,2pi]).
        cos_theta: free-parameter Cartesian RQS (no MLP).
        phi:       MLP-conditioned CIRCULAR RQS (d[0] == d[K]).
        No standardisation applied.

r2  --  AngularSphereFlow
        Works in STANDARDISED (cos_theta, phi) coordinates.
        Base distribution: STANDARD GAUSSIAN in R2.
        Both dimensions: MLP-conditioned standard RQS.

r3  --  CartesianNSF
        Works in STANDARDISED (px, py, pz) coordinates.
        Base distribution: STANDARD GAUSSIAN in R3.
        All dimensions: MLP-conditioned standard RQS.

All models support:
    forward(x, inverse=False)  ->  (z, log_det)  or  x_reconstructed
    log_prob(x)                ->  Tensor (B,)
    sample(n, device)          ->  Tensor (n, dim)

All models accept num_splines=N to stack N coupling blocks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .splines import rqs, rqs_with_bounds, rqs_circular
from .mlps import build_conditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack(net_out, out_dims, num_bins):
    """Reshape flat conditioner output into (w, h, d) for rqs()."""
    p = net_out.view(-1, out_dims, 3 * num_bins + 1)
    return p[..., :num_bins], p[..., num_bins:2*num_bins], p[..., 2*num_bins:]


# ---------------------------------------------------------------------------
# S2  --  RecursiveSphereFlow  (physical coords, uniform S2 base)
# ---------------------------------------------------------------------------

class RecursiveSphereFlow(nn.Module):
    """Normalising flow on S2 with UNIFORM base distribution.

    Works in PHYSICAL coordinates:
        cos_theta in (-1, 1)
        phi       in (0, 2*pi)

    Base distribution:
        cos_theta ~ Uniform[-1, 1]
        phi       ~ Uniform[0, 2*pi]
        => log p(z) = -log(4*pi)  (constant, flat over the sphere)

    Architecture per block
    ----------------------
    cos_theta : FREE-PARAMETER Cartesian RQS. No MLP.
    phi       : MLP-conditioned CIRCULAR RQS. d[0] == d[K].

    Parameters
    ----------
    num_bins : int
    num_splines : int
    hidden_dim : int
    num_layers : int
    arch : str
    activation : str
    dropout : float
    """

    TWO_PI = 2.0 * np.pi
    LOG_BASE = -np.log(4.0 * np.pi)   # log(1 / 4pi) -- uniform on S2

    def __init__(self,
                 num_bins: int = 32,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.num_splines = num_splines

        # Free parameters for cos_theta: (3K+1,) per block
        self.z_params = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins + 1))
            for _ in range(num_splines)
        ])

        # MLP conditioners for circular phi: output 3K (not 3K+1)
        self.phi_nets = nn.ModuleList([
            build_conditioner(arch, in_dim=1, out_dim=3 * num_bins,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(num_splines)
        ])

    def _z_whd(self, k, B):
        p = self.z_params[k]
        K = self.num_bins
        w = p[:K].unsqueeze(0).expand(B, -1)
        h = p[K:2*K].unsqueeze(0).expand(B, -1)
        d = p[2*K:].unsqueeze(0).expand(B, -1)
        return w, h, d

    def _phi_whd(self, k, z):
        out = self.phi_nets[k](z)
        K   = self.num_bins
        return out[:, :K], out[:, K:2*K], out[:, 2*K:]

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in range(self.num_splines):
                # Save DATA-SPACE cos_theta BEFORE transforming — phi_net
                # must always be conditioned on data-space cos_theta, matching
                # the original notebook's architecture.
                cos_theta_cond = cos_theta

                # Step A: transform cos_theta (data → base)
                w, h, d = self._z_whd(k, B)
                cos_theta, ldj = rqs_with_bounds(
                    cos_theta, w, h, d,
                    inverse=False, b_x=(-1, 1), b_y=(-1, 1))
                log_det += ldj

                # Step B: transform phi conditioned on PRE-TRANSFORM (data-space) cos_theta
                w, h, d = self._phi_whd(k, cos_theta_cond.unsqueeze(1))
                phi, ldj = rqs_circular(
                    phi, w, h, d,
                    inverse=False,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))
                log_det += ldj

            return torch.stack([cos_theta, phi], dim=1), log_det

        else:
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in reversed(range(self.num_splines)):
                # Inverse of forward block k:
                # Forward was: cos_theta_cond = cos_theta_k (data-space)
                #              cos_theta_{k+1} = rqs_fwd(cos_theta_k)
                #              phi_{k+1}       = rqs_circ_fwd(phi_k | phi_net(cos_theta_k))
                #
                # Inverse: first recover data-space cos_theta_k via rqs_inv,
                #          then invert phi conditioned on that same cos_theta_k.
                w, h, d = self._z_whd(k, B)
                cos_theta_data, _ = rqs_with_bounds(
                    cos_theta, w, h, d,
                    inverse=True, b_x=(-1, 1), b_y=(-1, 1))

                # Condition phi on data-space cos_theta (consistent with forward)
                w, h, d = self._phi_whd(k, cos_theta_data.unsqueeze(1))
                phi, _ = rqs_circular(
                    phi, w, h, d,
                    inverse=True,
                    b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

                cos_theta = cos_theta_data

            return torch.stack([cos_theta, phi], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        # Base is uniform on S2: log p(z) = -log(4*pi) for all z
        log_pz = torch.full((x.shape[0],), self.LOG_BASE,
                            device=x.device, dtype=x.dtype)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        # Sample from uniform S2 base
        cos_theta = torch.rand(n, device=device) * 2.0 - 1.0
        phi       = torch.rand(n, device=device) * self.TWO_PI
        z = torch.stack([cos_theta, phi], dim=1)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# R2  --  AngularSphereFlow  (standardised coords, Gaussian R2 base)
# ---------------------------------------------------------------------------

class AngularSphereFlow(nn.Module):
    """Normalising flow in R2 for standardised (cos_theta, phi).

    Works in STANDARDISED angular space.
    Base distribution: standard Gaussian in R2.
    Both dimensions use MLP-conditioned standard RQS.

    Parameters
    ----------
    num_bins : int
    bound : float -- spline half-domain in standardised space.
    num_splines : int
    hidden_dim, num_layers, arch, activation, dropout : conditioner settings.
    """

    def __init__(self,
                 num_bins: int = 32,
                 bound: float = 5.0,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.bound       = bound
        self.num_splines = num_splines
        out = 3 * num_bins + 1

        self.nets_phi = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])
        self.nets_costheta = nn.ModuleList([
            build_conditioner(arch, 1, out, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            ct, phi = x[:, 0:1], x[:, 1:2]
            for k in range(self.num_splines):
                w, h, d = _unpack(self.nets_phi[k](ct), 1, self.num_bins)
                phi, ldj = rqs(phi, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.squeeze(-1)
                w, h, d = _unpack(self.nets_costheta[k](phi), 1, self.num_bins)
                ct, ldj = rqs(ct, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.squeeze(-1)
            return torch.cat([ct, phi], dim=1), log_det
        else:
            ct, phi = x[:, 0:1], x[:, 1:2]
            for k in reversed(range(self.num_splines)):
                w, h, d = _unpack(self.nets_costheta[k](phi), 1, self.num_bins)
                ct, _ = rqs(ct, w, h, d, inverse=True, bound=self.bound)
                w, h, d = _unpack(self.nets_phi[k](ct), 1, self.num_bins)
                phi, _ = rqs(phi, w, h, d, inverse=True, bound=self.bound)
            return torch.cat([ct, phi], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, 2, device=device)
        return self.forward(z, inverse=True)


# ---------------------------------------------------------------------------
# R3  --  CartesianNSF  (standardised Cartesian, Gaussian R3 base)
# ---------------------------------------------------------------------------

class CartesianNSF(nn.Module):
    """Normalising flow in R3 for standardised (px, py, pz).

    Base distribution: standard Gaussian in R3.

    Parameters
    ----------
    num_bins : int
    bound : float
    num_splines : int
    hidden_dim, num_layers, arch, activation, dropout : conditioner settings.
    """

    def __init__(self,
                 num_bins: int = 32,
                 bound: float = 5.0,
                 num_splines: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 arch: str = "mlp",
                 activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.bound       = bound
        self.num_splines = num_splines

        out2 = 2 * (3 * num_bins + 1)
        out1 = 1 * (3 * num_bins + 1)

        self.nets1 = nn.ModuleList([
            build_conditioner(arch, 1, out2, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])
        self.nets2 = nn.ModuleList([
            build_conditioner(arch, 2, out1, hidden_dim, num_layers,
                              activation, dropout)
            for _ in range(num_splines)])

    def forward(self, x, inverse=False):
        B = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not inverse:
            px, pyz = x[:, 0:1], x[:, 1:3]
            for k in range(self.num_splines):
                w, h, d = _unpack(self.nets1[k](px), 2, self.num_bins)
                pyz, ldj = rqs(pyz, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.sum(dim=-1)
                w, h, d = _unpack(self.nets2[k](pyz), 1, self.num_bins)
                px, ldj = rqs(px, w, h, d, inverse=False, bound=self.bound)
                log_det += ldj.sum(dim=-1)
            return torch.cat([px, pyz], dim=1), log_det
        else:
            px, pyz = x[:, 0:1], x[:, 1:3]
            for k in reversed(range(self.num_splines)):
                w, h, d = _unpack(self.nets2[k](pyz), 1, self.num_bins)
                px, _ = rqs(px, w, h, d, inverse=True, bound=self.bound)
                w, h, d = _unpack(self.nets1[k](px), 2, self.num_bins)
                pyz, _ = rqs(pyz, w, h, d, inverse=True, bound=self.bound)
            return torch.cat([px, pyz], dim=1)

    def log_prob(self, x):
        z, log_det = self.forward(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, 3, device=device)
        return self.forward(z, inverse=True)


# ============================================================================
# S2NORM  --  NormSphereFlow
# ============================================================================
# Works on NORMALISED (cos_theta, phi) — same data pipeline as r2.
# Base: STANDARD GAUSSIAN (both dims).
# cos_theta: free-parameter interval RQS in normalised space.
# phi:       MLP-conditioned CIRCULAR RQS in normalised space (d[0]==d[K]).
#            Conditioned on PRE-TRANSFORM (data-space) cos_theta — original
#            notebook convention.
# Key differences from r2:
#   - circular constraint preserved on phi (smoothness at spline boundaries)
#   - cos_theta uses free params (no MLP), more like original s2 structure
#   - phi_net conditioned on data-space cos_theta (not post-transform)
# ============================================================================

class NormSphereFlow(nn.Module):
    """
    s2norm: normalised angular flow with circular phi constraint.

    Input space   : NORMALISED (cos_theta, phi)  — same data prep as r2.
    Base          : Standard Gaussian in R2.
    cos_theta dim : free-parameter interval RQS on (-bound, bound).
    phi dim       : MLP-conditioned CIRCULAR RQS on (-bound, bound).
                    d[0] == d[K] enforces smooth wrap at spline boundaries.
    Conditioning  : phi_net receives PRE-TRANSFORM (data-space) cos_theta.

    Parameters
    ----------
    num_bins    : int   -- spline segments K (Ks in paper). Default 32.
    num_splines : int   -- number of stacked coupling blocks. Default 1.
    bound       : float -- spline half-domain in normalised space. Default 5.0.
    hidden_dim  : int   -- conditioner MLP hidden size. Default 64.
    num_layers  : int   -- conditioner MLP depth. Default 2.
    arch        : str   -- 'mlp' or 'resnet'. Default 'mlp'.
    activation  : str   -- activation function. Default 'relu'.
    dropout     : float -- dropout rate. Default 0.0.
    """

    def __init__(self,
                 num_bins:    int   = 32,
                 num_splines: int   = 1,
                 bound:       float = 5.0,
                 hidden_dim:  int   = 64,
                 num_layers:  int   = 2,
                 arch:        str   = "mlp",
                 activation:  str   = "relu",
                 dropout:     float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.num_splines = num_splines
        self.bound       = bound
        TWO_PI           = 2.0 * np.pi   # kept for clarity; not used geometrically
        _ = TWO_PI

        # Free parameters for cos_theta: one (3K+1,) vector per block
        self.z_params = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins + 1))
            for _ in range(num_splines)
        ])

        # MLP conditioners for phi: output size 3K (circular: no extra deriv)
        self.phi_nets = nn.ModuleList([
            build_conditioner(arch, 1, 3 * num_bins,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(num_splines)
        ])

    # ------------------------------------------------------------------ #
    def _z_whd(self, k, B):
        """Unpack free cos_theta params for block k."""
        p = self.z_params[k]
        w = p[:self.num_bins].unsqueeze(0).expand(B, -1)
        h = p[self.num_bins:2*self.num_bins].unsqueeze(0).expand(B, -1)
        d = p[2*self.num_bins:].unsqueeze(0).expand(B, -1)
        return w, h, d

    def _phi_whd(self, k, cos_theta_in):
        """Run phi_net[k] on cos_theta_in; return (w, h, d) for circular RQS."""
        out = self.phi_nets[k](cos_theta_in)              # (B, 3K)
        K   = self.num_bins
        return out[:, :K], out[:, K:2*K], out[:, 2*K:]   # w, h, d  (K each)

    # ------------------------------------------------------------------ #
    def forward(self, x, inverse=False):
        B       = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)
        b       = self.bound

        if not inverse:
            # data → base (training direction)
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in range(self.num_splines):
                # Save PRE-TRANSFORM cos_theta for phi conditioning
                cos_theta_cond = cos_theta

                # Transform cos_theta with free params
                w, h, d = self._z_whd(k, B)
                cos_theta, ldj = rqs(
                    cos_theta, w, h, d,
                    inverse=False, bound=b)
                log_det += ldj

                # Transform phi with circular RQS conditioned on
                # pre-transform (data-space) cos_theta
                w, h, d = self._phi_whd(k, cos_theta_cond.unsqueeze(1))
                phi, ldj = rqs_circular(
                    phi, w, h, d,
                    inverse=False,
                    b_x=(-b, b), b_y=(-b, b))
                log_det += ldj

            return torch.stack([cos_theta, phi], dim=1), log_det

        else:
            # base → data (sampling direction)
            cos_theta = x[:, 0]
            phi       = x[:, 1]

            for k in reversed(range(self.num_splines)):
                # Invert cos_theta first to recover data-space value
                w, h, d = self._z_whd(k, B)
                cos_theta_data, _ = rqs(
                    cos_theta, w, h, d,
                    inverse=True, bound=b)

                # Invert phi using data-space cos_theta for conditioning
                w, h, d = self._phi_whd(k, cos_theta_data.unsqueeze(1))
                phi, _ = rqs_circular(
                    phi, w, h, d,
                    inverse=True,
                    b_x=(-b, b), b_y=(-b, b))

                cos_theta = cos_theta_data

            return torch.stack([cos_theta, phi], dim=1)

    # ------------------------------------------------------------------ #
    def log_prob(self, x):
        """Log-probability under Gaussian base."""
        z, log_det = self(x, inverse=False)
        log_pz = -0.5 * (z ** 2 + np.log(2.0 * np.pi)).sum(dim=1)
        return log_pz + log_det

    def sample(self, n, device=None):
        """Sample n points from Gaussian base, map to data space."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, 2, device=device)
        with torch.no_grad():
            return self(z, inverse=True)


# ============================================================================
# S2REC  --  RecursiveS2Flow
# ============================================================================
# Implements the recursive S2 flow from Rezende et al. 2020, Appendix J.
# Physical (cos_theta, phi) coordinates. Uniform S2 base.
#
# Blocks alternate between two types:
#
#   Type A (even k):
#     cos_theta -> interval RQS  (free params, no MLP)
#     phi       -> circular RQS  (MLP conditioned on TRANSFORMED cos_theta)
#
#   Type B (odd k):
#     phi       -> circular RQS  (free params, no MLP)
#     cos_theta -> interval RQS  (MLP conditioned on TRANSFORMED phi)
#
# num_splines=2 replicates paper's Appendix J: 1 A-block + 1 B-block.
#
# Why alternating?
#   A-type captures phi | cos_theta structure.
#   B-type captures cos_theta | phi structure.
#   Together they model the full joint, same as the paper's stacked flows.
#
# Ks (segments) in the paper = num_bins here.
# Paper Appendix J used Ks=80 (num_bins=80) and two stacked flows (num_splines=2).
# ============================================================================

class RecursiveS2Flow(nn.Module):
    """
    s2rec: Paper's recursive S2 flow (Rezende et al. 2020, Appendix J).

    Input space   : PHYSICAL (cos_theta, phi). No normalisation.
    Base          : UNIFORM ON S2  (log p = -log(4*pi)).
    Alternating blocks:
        A-type (even k): cos_theta free-param, phi MLP on transformed cos_theta.
        B-type (odd k):  phi free-param, cos_theta MLP on transformed phi.

    To replicate paper Appendix J: num_bins=80, num_splines=2.

    Parameters
    ----------
    num_bins    : int   -- spline segments K (= Ks in paper). Default 32.
    num_splines : int   -- total alternating blocks. 2 = paper's setup. Default 2.
    hidden_dim  : int   -- MLP hidden size. Default 64.
    num_layers  : int   -- MLP depth. Default 2.
    arch        : str   -- 'mlp' or 'resnet'. Default 'mlp'.
    activation  : str   -- activation. Default 'relu'.
    dropout     : float -- dropout. Default 0.0.
    """

    TWO_PI   = 2.0 * np.pi
    LOG_BASE = -np.log(4.0 * np.pi)   # log(1/4pi) -- uniform on S2

    def __init__(self,
                 num_bins:    int   = 32,
                 num_splines: int   = 2,
                 hidden_dim:  int   = 64,
                 num_layers:  int   = 2,
                 arch:        str   = "mlp",
                 activation:  str   = "relu",
                 dropout:     float = 0.0):
        super().__init__()
        self.num_bins    = num_bins
        self.num_splines = num_splines
        self.bound       = None   # s2rec uses physical domains, no bound needed

        # ── A-type blocks (even k): free cos_theta, MLP phi ─────────── #
        # cos_theta params: (3K+1,) per A block
        # phi MLP output:   3K   (circular — K derivatives, not K+1)
        n_A = (num_splines + 1) // 2   # number of A-type blocks

        self.z_params_A = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins + 1))
            for _ in range(n_A)
        ])
        self.phi_nets_A = nn.ModuleList([
            build_conditioner(arch, 1, 3 * num_bins,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(n_A)
        ])

        # ── B-type blocks (odd k): free phi, MLP cos_theta ──────────── #
        # phi params:       (3K,)   (circular)
        # cos_theta MLP output: (3K+1,) (interval)
        n_B = num_splines // 2

        self.phi_params_B = nn.ParameterList([
            nn.Parameter(torch.randn(3 * num_bins))   # circular: K derivatives
            for _ in range(n_B)
        ])
        self.z_nets_B = nn.ModuleList([
            build_conditioner(arch, 1, 3 * num_bins + 1,
                              hidden_dim=hidden_dim, num_layers=num_layers,
                              activation=activation, dropout=dropout)
            for _ in range(n_B)
        ])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _A_z_whd(self, a_idx, B):
        """A-type block: free-param cos_theta."""
        p = self.z_params_A[a_idx]
        K = self.num_bins
        w = p[:K].unsqueeze(0).expand(B, -1)
        h = p[K:2*K].unsqueeze(0).expand(B, -1)
        d = p[2*K:].unsqueeze(0).expand(B, -1)
        return w, h, d

    def _A_phi_whd(self, a_idx, cos_theta_in):
        """A-type block: MLP phi conditioned on transformed cos_theta."""
        out = self.phi_nets_A[a_idx](cos_theta_in)   # (B, 3K)
        K   = self.num_bins
        return out[:, :K], out[:, K:2*K], out[:, 2*K:]

    def _B_phi_whd(self, b_idx, B):
        """B-type block: free-param phi (circular)."""
        p = self.phi_params_B[b_idx]
        K = self.num_bins
        w = p[:K].unsqueeze(0).expand(B, -1)
        h = p[K:2*K].unsqueeze(0).expand(B, -1)
        d = p[2*K:].unsqueeze(0).expand(B, -1)
        return w, h, d

    def _B_z_whd(self, b_idx, phi_in):
        """B-type block: MLP cos_theta conditioned on transformed phi."""
        out = self.z_nets_B[b_idx](phi_in)     # (B, 3K+1)
        K   = self.num_bins
        return out[:, :K], out[:, K:2*K], out[:, 2*K:]

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _apply_A_forward(self, cos_theta, phi, B, a_idx):
        """A-type block: data→base direction."""
        # 1. Transform cos_theta (free params)
        w, h, d = self._A_z_whd(a_idx, B)
        cos_theta, ldj_z = rqs_with_bounds(
            cos_theta, w, h, d, inverse=False, b_x=(-1, 1), b_y=(-1, 1))

        # 2. Transform phi conditioned on TRANSFORMED cos_theta (paper convention)
        w, h, d = self._A_phi_whd(a_idx, cos_theta.unsqueeze(1))
        phi, ldj_phi = rqs_circular(
            phi, w, h, d, inverse=False,
            b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

        return cos_theta, phi, ldj_z + ldj_phi

    def _apply_B_forward(self, cos_theta, phi, B, b_idx):
        """B-type block: data→base direction."""
        # 1. Transform phi (free params, circular)
        w, h, d = self._B_phi_whd(b_idx, B)
        phi, ldj_phi = rqs_circular(
            phi, w, h, d, inverse=False,
            b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

        # 2. Transform cos_theta conditioned on TRANSFORMED phi
        w, h, d = self._B_z_whd(b_idx, phi.unsqueeze(1))
        cos_theta, ldj_z = rqs_with_bounds(
            cos_theta, w, h, d, inverse=False, b_x=(-1, 1), b_y=(-1, 1))

        return cos_theta, phi, ldj_z + ldj_phi

    def _apply_A_inverse(self, cos_theta, phi, B, a_idx):
        """A-type block: base→data direction."""
        # Reverse of forward A:
        # Forward was: cos_theta' = rqs(cos_theta), phi' = circ(phi | cos_theta')
        # Inverse:
        #   1. Invert phi using CURRENT (post-transform) cos_theta for conditioning
        #   2. Invert cos_theta
        w, h, d = self._A_phi_whd(a_idx, cos_theta.unsqueeze(1))
        phi, _ = rqs_circular(
            phi, w, h, d, inverse=True,
            b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

        w, h, d = self._A_z_whd(a_idx, B)
        cos_theta, _ = rqs_with_bounds(
            cos_theta, w, h, d, inverse=True, b_x=(-1, 1), b_y=(-1, 1))

        return cos_theta, phi

    def _apply_B_inverse(self, cos_theta, phi, B, b_idx):
        """B-type block: base→data direction."""
        # Forward was: phi' = circ(phi), cos_theta' = rqs(cos_theta | phi')
        # Inverse:
        #   1. Invert cos_theta using CURRENT (post-transform) phi for conditioning
        #   2. Invert phi
        w, h, d = self._B_z_whd(b_idx, phi.unsqueeze(1))
        cos_theta, _ = rqs_with_bounds(
            cos_theta, w, h, d, inverse=True, b_x=(-1, 1), b_y=(-1, 1))

        w, h, d = self._B_phi_whd(b_idx, B)
        phi, _ = rqs_circular(
            phi, w, h, d, inverse=True,
            b_x=(0, self.TWO_PI), b_y=(0, self.TWO_PI))

        return cos_theta, phi

    # ------------------------------------------------------------------ #

    def forward(self, x, inverse=False):
        B       = x.shape[0]
        log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

        cos_theta = x[:, 0]
        phi       = x[:, 1]

        # Track sub-indices for A and B parameter lists
        a_idx = 0
        b_idx = 0

        if not inverse:
            # data → base: apply blocks 0, 1, 2, ... in order
            for k in range(self.num_splines):
                if k % 2 == 0:   # A-type
                    cos_theta, phi, ldj = self._apply_A_forward(
                        cos_theta, phi, B, a_idx)
                    a_idx += 1
                else:             # B-type
                    cos_theta, phi, ldj = self._apply_B_forward(
                        cos_theta, phi, B, b_idx)
                    b_idx += 1
                log_det += ldj

            return torch.stack([cos_theta, phi], dim=1), log_det

        else:
            # base → data: apply blocks in REVERSE order
            # Pre-compute which type each block is
            block_types = ['A' if k % 2 == 0 else 'B'
                           for k in range(self.num_splines)]
            # Count total A/B used so far
            total_A = sum(1 for t in block_types if t == 'A')
            total_B = sum(1 for t in block_types if t == 'B')
            a_idx = total_A - 1
            b_idx = total_B - 1

            for k in reversed(range(self.num_splines)):
                if block_types[k] == 'A':
                    cos_theta, phi = self._apply_A_inverse(
                        cos_theta, phi, B, a_idx)
                    a_idx -= 1
                else:
                    cos_theta, phi = self._apply_B_inverse(
                        cos_theta, phi, B, b_idx)
                    b_idx -= 1

            return torch.stack([cos_theta, phi], dim=1)

    # ------------------------------------------------------------------ #

    def log_prob(self, x):
        """Log-probability under uniform S2 base."""
        z, log_det = self(x, inverse=False)
        log_pz = torch.full((x.shape[0],), self.LOG_BASE,
                            device=x.device, dtype=x.dtype)
        return log_pz + log_det

    def sample(self, n, device=None):
        """Sample n points from uniform S2 base, map to data space."""
        device = device or next(self.parameters()).device
        cos_theta = torch.rand(n, device=device) * 2 - 1
        phi       = torch.rand(n, device=device) * self.TWO_PI
        z = torch.stack([cos_theta, phi], dim=1)
        with torch.no_grad():
            return self(z, inverse=True)



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "s2":     RecursiveSphereFlow,   # physical (cos_theta, phi), uniform S2 base
    "s2norm": NormSphereFlow,        # normalised (cos_theta, phi), Gaussian base,
                                     #   circular phi spline
    "s2rec":  RecursiveS2Flow,       # paper's recursive S2 (Rezende et al. 2020),
                                     #   physical coords, alternating A/B blocks,
                                     #   num_splines=2 replicates Appendix J
    "r2":     AngularSphereFlow,     # standardised (cos_theta, phi), Gaussian R2 base
    "r3":     CartesianNSF,          # standardised (px, py, pz), Gaussian R3 base
}


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a flow model by registry key.

    Parameters
    ----------
    model_type : str
        One of 's2', 's2norm', 's2rec', 'r2', 'r3'.
    **kwargs
        Passed to the model constructor.

    Notes
    -----
    s2    : physical coords, no normalisation, uniform S2 base.
    s2norm: normalised coords (same data prep as r2), Gaussian base,
            circular phi constraint.
    s2rec : physical coords, alternating A/B blocks (Rezende 2020).
            num_splines=2, num_bins=80 replicates paper Appendix J.
            NO bound argument (uses physical domains directly).
    r2    : normalised angular coords, Gaussian base.
    r3    : normalised Cartesian 3-momentum, Gaussian base.

    Examples
    --------
    >>> build_model('s2',     num_bins=32, num_splines=1)
    >>> build_model('s2norm', num_bins=32, num_splines=2, bound=5.0)
    >>> build_model('s2rec',  num_bins=80, num_splines=2)  # paper exact
    >>> build_model('r2',     num_bins=32, num_splines=2, bound=5.0)
    >>> build_model('r3',     num_bins=64, num_splines=3, hidden_dim=128)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)
