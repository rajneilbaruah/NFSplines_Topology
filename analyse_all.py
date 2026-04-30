#!/usr/bin/env python
"""
analyse_all.py
==============
Full statistical analysis for s2, r2, r3 models.

  s2 — RecursiveSphereFlow  physical (cos_theta, phi)  uniform S2 base
  r2 — AngularSphereFlow    standardised (cos_theta, phi)  Gaussian R2 base
  r3 — CartesianNSF         standardised (px, py, pz)  Gaussian R3 base

Metrics
-------
  KL divergence, ESS, W1 (angular), JSD (angular),
  W1 (Cartesian px/py/pz), unphysical fraction, radius stats.

Outputs
-------
  outputs/analysis_summary.txt
  outputs/analysis_metrics.json
  outputs/analysis_metric_bars.png

Usage
-----
    python analyse_all.py --data ../datasets/NFSpheres/eemumu_mup.json
"""

import argparse, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from hep_nsf.utils    import (load_json_data, cartesian_to_spherical,
                               spherical_to_cartesian, normalise, denormalise,
                               get_device, set_seed)
from hep_nsf.networks import build_model
from hep_nsf.analysis import (kl_divergence_kde, effective_sample_size,
                               wasserstein_1d, js_divergence_1d,
                               consistency_report, model_summary)


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",            required=True)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--output_dir",      default="outputs")
    p.add_argument("--num_samples",     type=int,   default=10000)
    p.add_argument("--target_r",        type=float, default=500.0)
    p.add_argument("--kde_bw",          type=float, default=0.2)
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
    if mtype in ("s2","s2norm","s2rec","r2"):
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


def analyse_model(mtype, model, raw, data_phys, mean, std, device, args):
    print(f"\n{'='*60}\n  Analysing: {mtype.upper()}\n{'='*60}")

    with torch.no_grad():
        z_norm = model.sample(args.num_samples, device=device)
        if mtype in ("s2", "s2rec"):
            x_phys = z_norm
        else:
            x_phys = denormalise(z_norm, mean, std)

    metrics = {"model": mtype}
    arch    = model_summary(model, model_type=mtype, verbose=True)
    metrics.update(arch)

    # Derive angular and Cartesian for all models
    if mtype in ("s2","s2norm","s2rec","r2"):
        ang_data  = data_phys.numpy()
        ang_gen   = x_phys.cpu().numpy()
        cart_data = raw.numpy()
        cart_gen  = spherical_to_cartesian(
            torch.tensor(ang_gen), r=args.target_r).numpy()
    else:
        cart_data = data_phys.numpy()
        cart_gen  = x_phys.cpu().numpy()
        ang_data  = cartesian_to_spherical(raw).numpy()
        ang_gen   = cartesian_to_spherical(
            torch.tensor(cart_gen)).numpy()

    cos_d, phi_d = ang_data[:,0], ang_data[:,1]
    cos_g, phi_g = ang_gen[:,0],  ang_gen[:,1]

    # KL
    print("  Computing KL divergence...")
    kl = kl_divergence_kde(cos_d, phi_d, cos_g, phi_g,
                            bw=args.kde_bw, n_eval=5000)
    metrics["kl"] = float(kl)

    # ESS
    print("  Computing ESS...")
    ess = effective_sample_size(cos_d, phi_d, cos_g, phi_g, bw=args.kde_bw)
    metrics["ess"]      = float(ess)
    metrics["ess_frac"] = float(ess / len(cos_d))

    # W1 angular
    ang_d2 = np.stack([cos_d, phi_d], axis=1)
    ang_g2 = np.stack([cos_g, phi_g], axis=1)
    w1_ang = wasserstein_1d(ang_d2, ang_g2)
    metrics["W1_costheta"] = float(w1_ang[0])
    metrics["W1_phi"]      = float(w1_ang[1])

    # JSD angular
    jsd_ang = js_divergence_1d(ang_d2, ang_g2)
    metrics["JSD_costheta"] = float(jsd_ang[0])
    metrics["JSD_phi"]      = float(jsd_ang[1])

    # W1 Cartesian
    w1_cart = wasserstein_1d(cart_data, cart_gen)
    metrics["W1_px"] = float(w1_cart[0])
    metrics["W1_py"] = float(w1_cart[1])
    metrics["W1_pz"] = float(w1_cart[2])

    # Consistency / radius
    if mtype in ("s2","s2norm","s2rec","r2"):
        cons = consistency_report(torch.tensor(ang_gen), target_r=args.target_r)
        metrics["unphysical_frac"] = cons["frac_unphysical"]
        metrics["radius_mean"]     = cons["radius_mean"]
        metrics["radius_std"]      = cons["radius_std"]
    else:
        radii = np.linalg.norm(cart_gen, axis=1)
        metrics["unphysical_frac"] = 0.0
        metrics["radius_mean"]     = float(np.mean(radii))
        metrics["radius_std"]      = float(np.std(radii))

    print(f"\n  {'─'*50}")
    print(f"  {mtype.upper()} Summary")
    print(f"  {'─'*50}")
    print(f"  KL(data || flow)    : {kl:.4f} nats")
    print(f"  ESS                 : {ess:.1f}/{len(cos_d)} ({ess/len(cos_d)*100:.1f}%)")
    print(f"  W1 cos θ            : {w1_ang[0]:.4e}")
    print(f"  W1 φ                : {w1_ang[1]:.4e}")
    print(f"  JSD cos θ           : {jsd_ang[0]:.4e}")
    print(f"  JSD φ               : {jsd_ang[1]:.4e}")
    print(f"  W1 px               : {w1_cart[0]:.4e}")
    print(f"  W1 py               : {w1_cart[1]:.4e}")
    print(f"  W1 pz               : {w1_cart[2]:.4e}")
    print(f"  Unphysical          : {metrics['unphysical_frac']*100:.2f}%")
    print(f"  Radius mean ± std   : {metrics['radius_mean']:.2f} ± "
          f"{metrics['radius_std']:.2f} GeV")
    print(f"  {'─'*50}")

    return metrics


def print_table(all_metrics, output_dir):
    rows = [
        ("KL (nats)",         "kl",              "{:.4f}"),
        ("ESS%",              "ess_frac",         "{:.1%}"),
        ("W1 cos θ",          "W1_costheta",      "{:.3e}"),
        ("W1 φ",              "W1_phi",           "{:.3e}"),
        ("JSD cos θ",         "JSD_costheta",     "{:.3e}"),
        ("JSD φ",             "JSD_phi",          "{:.3e}"),
        ("W1 px",             "W1_px",            "{:.3e}"),
        ("W1 py",             "W1_py",            "{:.3e}"),
        ("W1 pz",             "W1_pz",            "{:.3e}"),
        ("Unphysical %",      "unphysical_frac",  "{:.2%}"),
        ("Radius mean (GeV)", "radius_mean",      "{:.1f}"),
        ("Radius std  (GeV)", "radius_std",       "{:.1f}"),
        ("Params",            "trainable_params", "{:,}"),
    ]
    models = [m["model"] for m in all_metrics]
    W = 18
    sep = "+" + "-"*22 + "+" + ("-"*W + "+")*len(models)

    lines = [sep]
    hdr = f"| {'Metric':<20} |"
    for m in models: hdr += f" {m.upper():^{W-2}} |"
    lines += [hdr, sep]

    for label, key, fmt in rows:
        row = f"| {label:<20} |"
        for m in all_metrics:
            val = m.get(key, float("nan"))
            try:    row += f" {fmt.format(val):^{W-2}} |"
            except: row += f" {'N/A':^{W-2}} |"
        lines.append(row)
    lines.append(sep)

    table = "\n".join(lines)
    print("\n\n" + "="*60 + "\n  COMPARISON TABLE\n" + "="*60)
    print(table)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    txt = Path(output_dir) / "analysis_summary.txt"
    with open(txt, "w") as f:
        f.write("HEP Neural Spline Flows — Analysis Summary\n\n")
        f.write("Model naming:\n")
        f.write("  s2 = RecursiveSphereFlow  physical (cos_theta,phi)  uniform S2 base\n")
        f.write("  r2 = AngularSphereFlow    standardised (cos_theta,phi)  Gaussian R2\n")
        f.write("  r3 = CartesianNSF         standardised (px,py,pz)  Gaussian R3\n\n")
        f.write(table)
        f.write("\n\nGuide:\n")
        f.write("  KL   : lower is better. 0 = perfect.\n")
        f.write("  ESS% : higher is better. 100% = perfect.\n")
        f.write("  W1   : earth-mover distance. lower is better.\n")
        f.write("  JSD  : 0=identical, ln2=worst.\n")

    json_p = Path(output_dir) / "analysis_metrics.json"
    with open(json_p, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved → {txt}")
    print(f"Saved → {json_p}")


def plot_bars(all_metrics, output_dir):
    models    = [m["model"].upper() for m in all_metrics]
    colors    = {"S2":"#2ecc71","R2":"#00234e","R3":"#A61200"}
    bar_colors= [colors.get(m,"steelblue") for m in models]

    to_plot = [
        ("KL Divergence (nats)",  "kl"),
        ("ESS %",                 "ess_frac"),
        ("W1 — cos θ",            "W1_costheta"),
        ("W1 — φ",                "W1_phi"),
        ("W1 — px",               "W1_px"),
        ("Unphysical Fraction",   "unphysical_frac"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (label, key) in zip(axes.flatten(), to_plot):
        vals = [m.get(key, 0.0) for m in all_metrics]
        bars = ax.bar(models, vals, color=bar_colors, edgecolor="black",
                      linewidth=0.8, alpha=0.85)
        ax.bar_label(bars, fmt="%.3e", fontsize=8, padding=2)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Model Comparison — Key Metrics", fontsize=14)
    fig.tight_layout()
    p = Path(output_dir) / "analysis_metric_bars.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


def main():
    args   = get_args()
    device = get_device(args.device)
    set_seed(args.seed)

    raw     = load_json_data(args.data)
    to_run  = [m for m in ["s2","s2norm","s2rec","r2","r3"] if m not in args.skip]

    print("=" * 60)
    print(f"  Models   : {to_run}")
    print(f"  Samples  : {args.num_samples}")
    print(f"  KDE bw   : {args.kde_bw}")
    print("=" * 60)

    all_metrics = []

    for mtype in to_run:
        ckpt = Path(args.checkpoints_dir) / f"{mtype}_best.pt"
        if not ckpt.exists():
            print(f"\n  [{mtype}] checkpoint not found — skipping.")
            continue
        model, _ = load_model(ckpt, device)
        raw_d, data_phys, _, mean, std = prepare(args.data, mtype)
        m = analyse_model(mtype, model, raw, data_phys, mean, std, device, args)
        all_metrics.append(m)

    if not all_metrics:
        print("No models analysed.")
        return

    print_table(all_metrics, args.output_dir)
    if len(all_metrics) > 1:
        plot_bars(all_metrics, args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
