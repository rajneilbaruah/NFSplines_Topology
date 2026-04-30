#!/usr/bin/env python
"""
test_plots.py
=============
Test every plotting function in hep_nsf/plotting.py.

Run without checkpoint (tests 9 out of 11 functions using fake data):
    python test_plots.py --data ../datasets/NFSpheres/eemumu_mup.json

Run with checkpoints (tests all functions including model-dependent ones):
    python test_plots.py --data ../datasets/NFSpheres/eemumu_mup.json \\
        --ckpt_s2 checkpoints/s2_best.pt \\
        --ckpt_r2 checkpoints/r2_best.pt \\
        --ckpt_r3 checkpoints/r3_best.pt

Each plot is saved to test_plot_outputs/ and a PASS/FAIL summary is printed.

Notes
-----
s2 : physical (cos_theta, phi), uniform S2 base, no denormalisation.
r2 : standardised (cos_theta, phi), Gaussian R2 base, denormalise after sample.
r3 : standardised (px, py, pz), Gaussian R3 base, denormalise after sample.
"""

import argparse
import traceback
import numpy as np
import torch
from pathlib import Path

from hep_nsf.utils import (
    load_json_data, cartesian_to_spherical, spherical_to_cartesian,
    normalise, denormalise, get_device, load_checkpoint)
from hep_nsf.networks import build_model
import hep_nsf.plotting as P


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",        required=True)
    p.add_argument("--ckpt_s2",     default=None, help="s2 checkpoint path")
    p.add_argument("--ckpt_r2",     default=None, help="r2 checkpoint path")
    p.add_argument("--ckpt_r3",     default=None, help="r3 checkpoint path")
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--output_dir",  default="test_plot_outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"
results = {}


def run(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results[name] = "PASS"
    except Exception:
        print(f"{FAIL}  {name}")
        traceback.print_exc()
        results[name] = "FAIL"


def skip(name, reason):
    print(f"{SKIP}  {name}  ({reason})")
    results[name] = "SKIP"


# ---------------------------------------------------------------------------
# Load model from checkpoint (reads architecture from metadata)
# ---------------------------------------------------------------------------

def load_model_from_ckpt(ckpt_path, device):
    meta        = torch.load(ckpt_path, map_location=device)
    mtype       = meta["model_type"]
    num_bins    = meta["num_bins"]
    num_splines = meta["num_splines"]
    bound       = meta.get("bound",      5.0)
    hidden_dim  = meta.get("hidden_dim", 64)
    num_layers  = meta.get("num_layers", 2)
    arch        = meta.get("arch",       "mlp")
    activation  = meta.get("activation", "relu")

    kwargs = dict(num_bins=num_bins, num_splines=num_splines,
                  hidden_dim=hidden_dim, num_layers=num_layers,
                  arch=arch, activation=activation)
    if mtype in ("r2", "r3"):
        kwargs["bound"] = bound

    model = build_model(mtype, **kwargs).to(device)
    model.load_state_dict(meta["model_state"])
    model.eval()
    return model, mtype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = get_device("cpu")
    OUT    = args.output_dir
    Path(OUT).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────── #
    print("\nLoading data...")
    raw                  = load_json_data(args.data)     # (N, 3) Cartesian GeV
    cyl                  = cartesian_to_spherical(raw, phi_range="0_2pi")  # (N, 2)
    cyl_norm, mean, std  = normalise(cyl)

    rng      = np.random.default_rng(42)
    N        = args.num_samples

    # ── Fake samples for data-only tests ──────────────────────────────────── #
    # Angular fake samples: add small noise to real data
    noise_ang  = torch.tensor(rng.normal(0, 0.03, (N, 2)), dtype=torch.float32)
    cyl_gen    = (cyl[:N] + noise_ang).clamp(
                    torch.tensor([-1.0, 0.0]),
                    torch.tensor([ 1.0, 2*np.pi]))

    # Cartesian fake samples: add noise to real Cartesian data
    noise_cart = torch.tensor(rng.normal(0, 5.0, (N, 3)), dtype=torch.float32)
    cart_gen   = raw[:N] + noise_cart

    data_ang_np  = cyl.numpy()[:N]
    gen_ang_np   = cyl_gen.numpy()
    data_cart_np = raw.numpy()[:N]
    gen_cart_np  = cart_gen.numpy()

    # Fake loss curves
    t_losses = list(np.linspace(3.0, 1.5, 60) + rng.normal(0, 0.04, 60))
    v_losses = list(np.linspace(3.1, 1.6, 60) + rng.normal(0, 0.04, 60))

    print(f"Output dir: {OUT}/\n")
    print("=" * 55)
    print("  Testing all plotting functions")
    print("=" * 55)

    # ═══════════════════════════════════════════════════════════════════════ #
    # DATA-ONLY TESTS (no model needed)
    # ═══════════════════════════════════════════════════════════════════════ #

    run("plot_loss_curves",
        lambda: P.plot_loss_curves(
            t_losses, v_losses,
            title="Test Loss Curves",
            save="test_loss_curves.png", output_dir=OUT))

    run("plot_marginal_1d  [angular]",
        lambda: P.plot_marginal_1d(
            data_ang_np, gen_ang_np,
            labels=[r"$\cos\theta$", r"$\phi$"],
            title="Test 1D Angular",
            save="test_marginals_angular_1d.png", output_dir=OUT))

    run("plot_marginal_1d  [cartesian]",
        lambda: P.plot_marginal_1d(
            data_cart_np, gen_cart_np,
            labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
            title="Test 1D Cartesian",
            save="test_marginals_cartesian_1d.png", output_dir=OUT))

    run("plot_marginal_2d  [angular]",
        lambda: P.plot_marginal_2d(
            data_ang_np, gen_ang_np,
            labels=[r"$\cos\theta$", r"$\phi$"],
            title="Test 2D Angular",
            save="test_marginals_angular_2d.png", output_dir=OUT))

    run("plot_marginal_2d  [cartesian]",
        lambda: P.plot_marginal_2d(
            data_cart_np, gen_cart_np,
            labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
            title="Test 2D Cartesian",
            save="test_marginals_cartesian_2d.png", output_dir=OUT))

    run("plot_mollweide_kde  [data]",
        lambda: P.plot_mollweide_kde(
            cyl[:N],
            title="Test Data KDE",
            save="test_mollweide_data.png", output_dir=OUT))

    run("plot_mollweide_kde  [flow]",
        lambda: P.plot_mollweide_kde(
            cyl_gen,
            title="Test Flow KDE",
            save="test_mollweide_flow.png", output_dir=OUT))

    run("plot_physics_comparison",
        lambda: P.plot_physics_comparison(
            raw[:N], cart_gen,
            title="Test Physics Comparison",
            save="test_physics_comparison.png", output_dir=OUT))

    run("plot_radius_distribution  [r3 — varied radii]",
        lambda: P.plot_radius_distribution(
            gen_cart_np,
            title="Test |p| Distribution",
            save="test_radius_dist.png", output_dir=OUT))

    # ═══════════════════════════════════════════════════════════════════════ #
    # MODEL-DEPENDENT TESTS
    # ═══════════════════════════════════════════════════════════════════════ #

    ckpt_map = {"s2": args.ckpt_s2, "r2": args.ckpt_r2, "r3": args.ckpt_r3}

    for mtype, ckpt_path in ckpt_map.items():
        if ckpt_path is None:
            skip(f"plot_base_mapping     [{mtype}]", "no checkpoint")
            skip(f"plot_jacobian_map     [{mtype}]", "no checkpoint")
            continue

        try:
            model, loaded_type = load_model_from_ckpt(ckpt_path, device)
            if loaded_type != mtype:
                print(f"  Warning: checkpoint reports model_type='{loaded_type}' "
                      f"but --ckpt_{mtype} was given. Using '{loaded_type}'.")
                mtype = loaded_type
        except Exception as e:
            skip(f"plot_base_mapping     [{mtype}]", f"checkpoint load failed: {e}")
            skip(f"plot_jacobian_map     [{mtype}]", f"checkpoint load failed: {e}")
            continue

        # Prepare mean/std for this model type
        if mtype in ("s2", "r2"):
            _, m, s = normalise(cyl)
        else:
            _, m, s = normalise(raw)

        # ── Base mapping ────────────────────────────────────────────────── #
        if mtype == "s2":
            run(f"plot_base_mapping_s2  [{mtype}]",
                lambda mt=mtype, mo=model, me=m, st=s: P.plot_base_mapping_s2(
                    mo, device, me, st,
                    title=f"Test S2 Base Mapping",
                    save=f"test_{mt}_base_mapping.png", output_dir=OUT))

        elif mtype == "r2":
            run(f"plot_base_mapping     [{mtype}]",
                lambda mt=mtype, mo=model, me=m, st=s: P.plot_base_mapping(
                    mo, device, me, st, dim=2,
                    title=f"Test R2 Base Mapping",
                    save=f"test_{mt}_base_mapping.png", output_dir=OUT))

        else:  # r3 — no 2D base mapping
            skip(f"plot_base_mapping     [{mtype}]",
                 "r3 is 3D — base mapping not applicable")

        # ── Jacobian map ────────────────────────────────────────────────── #
        if mtype in ("s2", "r2"):
            run(f"plot_jacobian_map     [{mtype}]",
                lambda mt=mtype, mo=model, me=m, st=s: P.plot_jacobian_map(
                    mo, device, me, st, n_samples=500,
                    title=f"Test {mt.upper()} Jacobian Map",
                    save=f"test_{mt}_jacobian_map.png", output_dir=OUT))

        else:  # r3 — three-panel jacobian
            run(f"plot_jacobian_map_r3  [{mtype}]",
                lambda mt=mtype, mo=model, me=m, st=s: P.plot_jacobian_map_r3(
                    mo, device, me, st, n_samples=500,
                    title="Test R3 Jacobian Map",
                    save=f"test_{mt}_jacobian_map.png", output_dir=OUT))

    # ═══════════════════════════════════════════════════════════════════════ #
    # Summary
    # ═══════════════════════════════════════════════════════════════════════ #
    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v == "FAIL")
    skipped= sum(1 for v in results.values() if v == "SKIP")

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}  (no checkpoint provided)")

    if failed == 0:
        print("\n  \033[92mAll tested functions passed.\033[0m")
    else:
        print("\n  \033[91mSome tests failed — see tracebacks above.\033[0m")
    print("=" * 55)
    print(f"\nPlots saved to: {OUT}/\n")


if __name__ == "__main__":
    main()
