#!/usr/bin/env python
"""
plot_all.py
===========
Generate every diagnostic plot for all three trained models.

Every plot is generated for every model, including cross-space plots
(e.g. Cartesian pair plots for angular models via back-conversion,
Mollweide for r3 via forward-conversion).

Output structure
----------------
outputs/s2/   outputs/r2/   outputs/r3/
  <model>_loss_curves.png
  <model>_marginals_angular_1d.png
  <model>_marginals_angular_2d.png
  <model>_marginals_cartesian_1d.png
  <model>_marginals_cartesian_2d.png
  <model>_mollweide_data.png
  <model>_mollweide_flow.png
  <model>_physics_comparison.png
  <model>_radius_dist.png          (r3 only — angular has fixed r)
  <model>_base_mapping.png
  <model>_jacobian_map*.png

outputs/
  combined_physics_comparison.png

Usage
-----
    python plot_all.py --data ../datasets/NFSpheres/eemumu_mup.json
"""

import argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from hep_nsf.utils    import (load_json_data, cartesian_to_spherical,
                               spherical_to_cartesian, normalise, denormalise,
                               get_device, set_seed, load_losses,
                               cartesian_to_physics)
from hep_nsf.networks import build_model
import hep_nsf.plotting as P


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",            required=True)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--output_dir",      default="outputs")
    p.add_argument("--num_samples",     type=int,   default=10000)
    p.add_argument("--target_r",        type=float, default=500.0)
    p.add_argument("--device",          default="auto",
                   choices=["auto","cpu","cuda","mps"])
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--skip",            nargs="*",  default=[])
    return p.parse_args()


def load_model(ckpt_path, device):
    meta        = torch.load(ckpt_path, map_location=device)
    mtype       = meta["model_type"]
    num_bins    = meta["num_bins"]
    num_splines = meta["num_splines"]
    bound       = meta.get("bound", 5.0)
    hidden_dim  = meta.get("hidden_dim", 64)

    num_layers  = meta.get("num_layers", 2)
    arch        = meta.get("arch", "mlp")
    activation  = meta.get("activation", "relu")
    kwargs = dict(num_bins=num_bins, num_splines=num_splines,
                  hidden_dim=hidden_dim, num_layers=num_layers,
                  arch=arch, activation=activation)
    if mtype in ("s2norm", "r2", "r3"):
        kwargs["bound"] = bound

    model = build_model(mtype, **kwargs).to(device)
    model.load_state_dict(meta["model_state"])
    model.eval()
    return model, mtype


def prepare(data_path, mtype):
    import torch
    raw = load_json_data(data_path)
    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        data = cartesian_to_spherical(raw, phi_range="0_2pi")
    else:
        data = raw

    if mtype in ("s2", "s2rec"):
        # physical coords — no normalisation
        mean = torch.zeros(1, 2)
        std  = torch.ones(1, 2)
        return raw, data, data, mean, std
    data_norm, mean, std = normalise(data)
    return raw, data, data_norm, mean, std


def plot_model(mtype, model, raw, data_phys, mean, std,
               device, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    s = str(out_dir)

    # ── Generate samples ─────────────────────────────────────────────────── #
    with torch.no_grad():
        z_norm = model.sample(args.num_samples, device=device)
        if mtype in ("s2", "s2rec"):
            # physical models already output (cos_theta, phi)
            x_phys = z_norm
        else:
            x_phys = denormalise(z_norm, mean, std)

    # Derive angular and Cartesian arrays for all models
    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        ang_data  = data_phys.numpy()
        ang_gen   = x_phys.cpu().numpy()
        cart_data = raw.numpy()
        cart_gen  = spherical_to_cartesian(
            torch.tensor(ang_gen), r=args.target_r).numpy()
    else:  # r3
        cart_data = data_phys.numpy()
        cart_gen  = x_phys.cpu().numpy()
        ang_data  = cartesian_to_spherical(raw).numpy()
        ang_gen   = cartesian_to_spherical(
            torch.tensor(cart_gen)).numpy()

    print(f"  [{mtype}] samples generated.")

    # ── Loss curves ──────────────────────────────────────────────────────── #
    loss_path = Path(args.checkpoints_dir) / f"{mtype}_losses.json"
    if loss_path.exists():
        losses = load_losses(loss_path)
        P.plot_loss_curves(losses["train"], losses["val"],
                           title=f"Loss Curves — {mtype.upper()}",
                           save=f"{mtype}_loss_curves.png", output_dir=s)
        print(f"  [{mtype}] loss curves.")

    # ── Angular 1D marginals ─────────────────────────────────────────────── #
    P.plot_marginal_1d(ang_data, ang_gen,
                       labels=[r"$\cos\theta$", r"$\phi$"],
                       title=f"{mtype.upper()} — Angular Marginals",
                       save=f"{mtype}_marginals_angular_1d.png", output_dir=s)
    print(f"  [{mtype}] angular 1D marginals.")

    # ── Angular 2D scatter ───────────────────────────────────────────────── #
    P.plot_marginal_2d(ang_data, ang_gen,
                       labels=[r"$\cos\theta$", r"$\phi$"],
                       title=f"{mtype.upper()} — Angular 2D",
                       save=f"{mtype}_marginals_angular_2d.png", output_dir=s)
    print(f"  [{mtype}] angular 2D scatter.")

    # ── Cartesian 1D marginals ───────────────────────────────────────────── #
    P.plot_marginal_1d(cart_data, cart_gen,
                       labels=[r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"],
                       title=f"{mtype.upper()} — Cartesian Marginals",
                       save=f"{mtype}_marginals_cartesian_1d.png", output_dir=s)
    print(f"  [{mtype}] Cartesian 1D marginals.")

    # ── Cartesian 2D pair plots ──────────────────────────────────────────── #
    P.plot_marginal_2d(cart_data, cart_gen,
                       labels=[r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"],
                       title=f"{mtype.upper()} — Cartesian 2D",
                       save=f"{mtype}_marginals_cartesian_2d.png", output_dir=s)
    print(f"  [{mtype}] Cartesian 2D pair plots.")

    # ── Mollweide KDE ────────────────────────────────────────────────────── #
    P.plot_mollweide_kde(torch.tensor(ang_data),
                         title=f"{mtype.upper()} Data KDE",
                         save=f"{mtype}_mollweide_data.png", output_dir=s)
    P.plot_mollweide_kde(torch.tensor(ang_gen),
                         title=f"{mtype.upper()} Flow KDE",
                         save=f"{mtype}_mollweide_flow.png", output_dir=s)
    print(f"  [{mtype}] Mollweide KDE.")

    # ── Physics comparison ───────────────────────────────────────────────── #
    P.plot_physics_comparison(
        torch.tensor(cart_data), torch.tensor(cart_gen),
        title=f"{mtype.upper()} Physics Comparison",
        save=f"{mtype}_physics_comparison.png", output_dir=s)
    print(f"  [{mtype}] physics comparison.")

    # ── Radius distribution (r3 only — angular models have fixed r) ──────── #
    if mtype == "r3":
        P.plot_radius_distribution(
            cart_gen,
            title=f"{mtype.upper()} |p| Distribution",
            save=f"{mtype}_radius_dist.png", output_dir=s)
        print(f"  [{mtype}] radius distribution.")
    else:
        print(f"  [{mtype}] radius distribution skipped "
              f"(angular model, r is fixed at {args.target_r} GeV).")

    # ── Base mapping ─────────────────────────────────────────────────────── #
    try:
        if mtype == "s2":
            P.plot_base_mapping_s2(
                model, device, mean, std,
                title="S2 Base Mapping (Uniform Lat-Lon Grid)",
                save=f"{mtype}_base_mapping.png", output_dir=s)
        elif mtype == "r2":
            P.plot_base_mapping(
                model, device, mean, std, dim=2,
                title="R2 Base Mapping (Gaussian Polar Grid)",
                save=f"{mtype}_base_mapping.png", output_dir=s)
        else:
            # r3: show base mapping as 2D projections of Gaussian grid
            # just use the Jacobian map projections as a substitute
            pass
        print(f"  [{mtype}] base mapping.")
    except Exception as e:
        print(f"  [{mtype}] base mapping skipped: {e}")

    # ── Jacobian map ─────────────────────────────────────────────────────── #
    try:
        if mtype in ("s2", "s2norm", "s2rec", "r2"):
            P.plot_jacobian_map(
                model, device, mean, std,
                n_samples=3000,
                title=f"{mtype.upper()} Jacobian Map",
                save=f"{mtype}_jacobian_map.png", output_dir=s)
            print(f"  [{mtype}] Jacobian map.")
        else:
            P.plot_jacobian_map_r3(
                model, device, mean, std,
                n_samples=3000,
                title="R3 Jacobian Map",
                save=f"{mtype}_jacobian_map.png", output_dir=s)
            print(f"  [{mtype}] Jacobian map (3 panels).")
    except Exception as e:
        print(f"  [{mtype}] Jacobian map skipped: {e}")

    return cart_gen


def plot_combined(results, raw, out_dir):
    """All models vs data on the same physics variable panels."""
    print("\nPlotting combined comparison...")
    raw_physics = cartesian_to_physics(raw)
    var_keys  = ["px","py","pz","pT","eta","phi"]
    var_labels= [r"$p_x$", r"$p_y$", r"$p_z$",
                 r"$p_T$", r"$\eta$", r"$\phi$"]
    colors    = {"s2":"#2ecc71", "s2norm":"#f39c12", "s2rec":"#9b59b6", "r2":"#00234e", "r3":"#A61200"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (vk, vl) in enumerate(zip(var_keys, var_labels)):
        ax = axes.flatten()[i]
        d  = raw_physics[vk].numpy()
        d  = d[np.isfinite(d)]
        lo = np.percentile(d, 0.5); hi = np.percentile(d, 99.5)
        bins = np.linspace(lo, hi, 51)

        ax.hist(d, bins=bins, density=True, histtype="step",
                lw=2.5, color="black", label="Data", linestyle="--")

        for mtype, cart_gen in results.items():
            g = cartesian_to_physics(torch.tensor(cart_gen))[vk].numpy()
            g = g[np.isfinite(g)]
            ax.hist(g, bins=bins, density=True, histtype="step",
                    lw=2, color=colors[mtype], label=mtype.upper())

        ax.set_xlabel(vl, fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("All Models vs Data", fontsize=14)
    fig.tight_layout()
    p = out_dir / "combined_physics_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


def main():
    args   = get_args()
    device = get_device(args.device)
    set_seed(args.seed)

    ckpt_dir = Path(args.checkpoints_dir)
    out_root = Path(args.output_dir)
    raw      = load_json_data(args.data)
    to_plot  = [m for m in ["s2","s2norm","s2rec","r2","r3"] if m not in args.skip]

    print("=" * 60)
    print(f"  Plotting: {to_plot}   Samples: {args.num_samples}")
    print("=" * 60)

    results = {}

    for mtype in to_plot:
        ckpt = ckpt_dir / f"{mtype}_best.pt"
        if not ckpt.exists():
            print(f"\n  [{mtype}] checkpoint not found — skipping.")
            continue

        print(f"\n--- {mtype.upper()} ---")
        model, _ = load_model(ckpt, device)
        raw_data, data_phys, _, mean, std = prepare(args.data, mtype)

        cart_gen = plot_model(mtype, model, raw, data_phys,
                              mean, std, device, args,
                              out_root / mtype)
        results[mtype] = cart_gen

    if len(results) > 1:
        plot_combined(results, raw, out_root)

    print(f"\nAll plots saved to: {out_root}/")


if __name__ == "__main__":
    main()
