#!/usr/bin/env python
"""
train_all.py — Train s2, r2, r3 on the same dataset.

  s2  RecursiveSphereFlow  physical (cos_theta, phi)  uniform S2 base
  r2  AngularSphereFlow    standardised (cos_theta, phi)  Gaussian R2 base
  r3  CartesianNSF         standardised (px, py, pz)  Gaussian R3 base

Usage
-----
    python train_all.py --data ../datasets/NFSpheres/eemumu_mup.json
"""

import argparse, time
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from hep_nsf.utils    import (load_json_data, normalise, cartesian_to_spherical,
                               make_dataloaders, get_device, set_seed)
from hep_nsf.networks import build_model
from hep_nsf.analysis import model_summary
from hep_nsf.train    import train_model


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",         required=True)
    p.add_argument("--num_bins",     type=int,   default=32)
    p.add_argument("--num_splines",  type=int,   default=1)
    p.add_argument("--hidden_dim",   type=int,   default=64)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--arch",         default="mlp", choices=["mlp","resnet"])
    p.add_argument("--activation",   default="relu")
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--bound",        type=float, default=5.0)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--batch_size",   type=int,   default=8192)
    p.add_argument("--epochs",       type=int,   default=10000)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--ema_alpha",    type=float, default=0.3)
    p.add_argument("--clip_grad",    type=float, default=5.0)
    p.add_argument("--val_frac",     type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="auto",
                   choices=["auto","cpu","cuda","mps"])
    p.add_argument("--save_dir",     default="checkpoints")
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--skip",         nargs="*",  default=[],
                   help="Models to skip e.g. --skip s2 r3")
    return p.parse_args()


def prepare(args, mtype):
    import torch
    raw = load_json_data(args.data)
    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        data = cartesian_to_spherical(raw, phi_range="0_2pi")
    else:
        data = raw

    if mtype in ("s2", "s2rec"):
        # physical coords — no normalisation
        mean = torch.zeros(1, 2)
        std  = torch.ones(1, 2)
        return data, mean, std
    return normalise(data)   # data_norm, mean, std


def model_kwargs(args, mtype):
    base = dict(num_bins=args.num_bins, num_splines=args.num_splines,
                hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                arch=args.arch, activation=args.activation, dropout=args.dropout)
    if mtype in ("s2norm", "r2", "r3"):
        base["bound"] = args.bound
    return base


def main():
    args   = get_args()
    device = get_device(args.device)
    set_seed(args.seed)

    to_train = [m for m in ["s2","s2norm","s2rec","r2","r3"] if m not in args.skip]

    print("=" * 60)
    print(f"  Models  : {to_train}")
    print(f"  Device  : {device}")
    print(f"  Bins    : {args.num_bins}   Splines: {args.num_splines}")
    print(f"  Epochs  : {args.epochs}   Patience: {args.patience}")
    print("=" * 60)

    all_losses = {}
    t0 = time.time()

    for mtype in to_train:
        print(f"\n{'='*60}\n  Model: {mtype.upper()}\n{'='*60}\n")

        data_norm, mean, std = prepare(args, mtype)
        train_loader, val_loader = make_dataloaders(
            data_norm, batch_size=args.batch_size,
            val_frac=args.val_frac, seed=args.seed)

        model = build_model(mtype, **model_kwargs(args, mtype))
        model_summary(model, model_type=mtype)

        model, losses = train_model(
            model, train_loader, val_loader,
            std_tensor=std, device=device,
            model_type=mtype,
            num_layers=args.num_layers,
            arch=args.arch,
            activation=args.activation,
            lr=args.lr, max_epochs=args.epochs,
            patience=args.patience, ema_alpha=args.ema_alpha,
            clip_grad=args.clip_grad, log_every=args.log_every,
            save_dir=args.save_dir, run_name=mtype)

        all_losses[mtype] = losses

    # Combined loss plot
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    colors = {"s2":"#2ecc71","s2norm":"#f39c12","s2rec":"#9b59b6","r2":"#00234e","r3":"#A61200"}

    fig, axes = plt.subplots(1, len(all_losses),
                              figsize=(7*len(all_losses), 5))
    if len(all_losses) == 1:
        axes = [axes]

    for ax, (mtype, losses) in zip(axes, all_losses.items()):
        ep = range(1, len(losses["train"]) + 1)
        ax.plot(ep, losses["train"], color=colors[mtype], lw=2, label="Train")
        ax.plot(ep, losses["val"],   color=colors[mtype], lw=2,
                linestyle="--", label="Val")
        ax.set_title(f"Model {mtype.upper()}", fontsize=13)
        ax.set_xlabel("Epoch"); ax.set_ylabel("NLL Loss")
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Training Curves — All Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/all_loss_curves.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.output_dir}/all_loss_curves.png")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
