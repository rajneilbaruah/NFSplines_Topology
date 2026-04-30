#!/usr/bin/env python
"""
main.py — CLI entry point for HEP Neural Spline Flows.

Models
------
  s2  RecursiveSphereFlow — physical (cos_theta, phi), uniform S2 base
  r2  AngularSphereFlow   — standardised (cos_theta, phi), Gaussian R2 base
  r3  CartesianNSF        — standardised (px, py, pz), Gaussian R3 base

Modes
-----
  train    — train from scratch or resume
  sample   — draw samples from a checkpoint
  evaluate — compute KL, ESS, Wasserstein
  plot     — generate all diagnostic plots

Note on s2
----------
s2 works in physical (cos_theta, phi) coordinates with no standardisation.
model.sample() returns physical coordinates directly.
Normalisation stats (mean, std) are still computed for the NLL correction
term during training, but are NOT applied to samples at inference time.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import yaml


def build_parser():
    p = argparse.ArgumentParser(
        prog="hep_nsf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--config",   type=str, default=None)
    p.add_argument("--mode",     choices=["train","sample","evaluate","plot"],
                   default="train")

    # Data
    p.add_argument("--data",     required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed",     type=int,   default=42)

    # Model
    p.add_argument("--model",    choices=["s2","s2norm","s2rec","r2","r3"], default="s2",
                   help="s2=RecursiveSphereFlow (uniform S2 base), "
                        "s2norm=NormSphereFlow (normalised+circular phi), "
                        "s2rec=RecursiveS2Flow (Rezende 2020 paper), "
                        "r2=AngularSphereFlow (Gaussian R2 base), "
                        "r3=CartesianNSF (Gaussian R3 base)")
    p.add_argument("--num_bins",    type=int,   default=32)
    p.add_argument("--num_splines", type=int,   default=1)
    p.add_argument("--bound",       type=float, default=5.0)
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--arch",        choices=["mlp","resnet"], default="mlp")
    p.add_argument("--activation",
                   choices=["relu","tanh","elu","leaky_relu","silu","gelu"],
                   default="relu")
    p.add_argument("--dropout",     type=float, default=0.0)

    # Training
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=0.0)
    p.add_argument("--batch_size",       type=int,   default=8192)
    p.add_argument("--epochs",           type=int,   default=10_000)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--ema_alpha",        type=float, default=0.3)
    p.add_argument("--clip_grad",        type=float, default=5.0)
    p.add_argument("--use_plateau",      action="store_true", default=True)
    p.add_argument("--no_plateau",       dest="use_plateau", action="store_false")
    p.add_argument("--plateau_factor",   type=float, default=0.5)
    p.add_argument("--plateau_patience", type=int,   default=10)
    p.add_argument("--use_cosine",       action="store_true", default=False)
    p.add_argument("--cosine_t_max",     type=int,   default=500)
    p.add_argument("--log_every",        type=int,   default=10)
    p.add_argument("--resume_from",      type=str,   default=None)

    # I/O
    p.add_argument("--run_name",    type=str, default="model")
    p.add_argument("--save_dir",    type=str, default="checkpoints")
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--checkpoint",  type=str, default=None)

    # Sampling
    p.add_argument("--num_samples", type=int,   default=10_000)
    p.add_argument("--target_r",    type=float, default=500.0)

    # Device / plotting
    p.add_argument("--device",   choices=["auto","cpu","cuda","mps"], default="auto")
    p.add_argument("--no_plots", action="store_true", default=False)
    p.add_argument("--dpi",      type=int, default=400)

    return p


def load_yaml_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(args, yaml_cfg):
    parser   = build_parser()
    defaults = vars(parser.parse_args(["--data", args.data or "dummy"]))
    for k, v in yaml_cfg.items():
        if k in defaults and vars(args)[k] == defaults[k]:
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_data(args, model_type=None):
    from hep_nsf.utils import (load_json_data, normalise,
                                cartesian_to_spherical, set_seed)
    set_seed(args.seed)
    mtype = model_type or args.model
    raw   = load_json_data(args.data)

    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        data = cartesian_to_spherical(raw, phi_range="0_2pi")
    else:
        data = raw

    if mtype in ("s2", "s2rec"):
        # Physical coords — NO normalisation.
        # std=ones so std_correction=0, denormalise is identity.
        import torch
        mean = torch.zeros(1, 2)
        std  = torch.ones(1, 2)
        return data, mean, std, data, raw
    else:
        # s2norm, r2, r3: normalise
        data_norm, mean, std = normalise(data)
        return data_norm, mean, std, data, raw


# ---------------------------------------------------------------------------
# Checkpoint loader — self-describing, no architecture flags needed
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(checkpoint_path, args, device):
    """Build and load a model entirely from checkpoint metadata.

    The checkpoint stores model_type, num_bins, num_splines, bound,
    hidden_dim — so no --model / --num_bins / --num_splines flags are
    needed at sample/evaluate/plot time.
    Falls back to CLI args if metadata is missing (old checkpoints).
    """
    from hep_nsf.networks import build_model
    meta        = torch.load(checkpoint_path, map_location=device)
    mtype       = meta.get("model_type",  args.model)
    num_bins    = meta.get("num_bins",    args.num_bins)
    num_splines = meta.get("num_splines", args.num_splines)
    bound       = meta.get("bound",       args.bound)
    hidden_dim  = meta.get("hidden_dim",  args.hidden_dim)

    num_layers  = meta.get("num_layers", args.num_layers)
    arch        = meta.get("arch",       args.arch)
    activation  = meta.get("activation", args.activation)

    kwargs = dict(num_bins=num_bins, num_splines=num_splines,
                  hidden_dim=hidden_dim, num_layers=num_layers,
                  arch=arch, activation=activation,
                  dropout=args.dropout)
    if mtype in ("s2norm", "r2", "r3") and bound is not None:
        kwargs["bound"] = bound

    model = build_model(mtype, **kwargs).to(device)
    model.load_state_dict(meta["model_state"])
    model.eval()
    return model, mtype


# ---------------------------------------------------------------------------
# Shared sampling helper — handles s2 physical vs r2/r3 normalised
# ---------------------------------------------------------------------------

def _sample_physical(model, mtype, n, device, mean, std):
    """Draw n samples and return physical-space tensor.

    s2 : model.sample() already returns physical (cos_theta, phi).
         No denormalisation needed.
    r2 : model.sample() returns normalised coords. Denormalise.
    r3 : model.sample() returns normalised coords. Denormalise.
    """
    from hep_nsf.utils import denormalise
    with torch.no_grad():
        z = model.sample(n, device=device)
        if mtype in ("s2", "s2rec"):
            return z          # already physical (cos_theta, phi)
        else:
            return denormalise(z, mean, std)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_train(args):
    from hep_nsf.networks import build_model
    from hep_nsf.utils    import make_dataloaders, get_device
    from hep_nsf.train    import train_model
    from hep_nsf.analysis import model_summary
    import hep_nsf.plotting as P

    device = get_device(args.device)
    print(f"Device: {device}\n")

    data_norm, mean, std, data_phys, raw = _prepare_data(args)
    train_loader, val_loader = make_dataloaders(
        data_norm, batch_size=args.batch_size,
        val_frac=args.val_frac, seed=args.seed)

    kwargs = dict(num_bins=args.num_bins, num_splines=args.num_splines,
                  hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                  arch=args.arch, activation=args.activation,
                  dropout=args.dropout)
    if args.model in ("s2norm", "r2", "r3"):
        kwargs["bound"] = args.bound

    model = build_model(args.model, **kwargs)
    model_summary(model, model_type=args.model)

    model, losses = train_model(
        model, train_loader, val_loader,
        std_tensor=std, device=device,
        model_type=args.model,
        num_layers=args.num_layers,
        arch=args.arch,
        activation=args.activation,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.epochs, patience=args.patience,
        ema_alpha=args.ema_alpha, clip_grad=args.clip_grad,
        use_plateau=args.use_plateau, plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        use_cosine=args.use_cosine, cosine_t_max=args.cosine_t_max,
        log_every=args.log_every, save_dir=args.save_dir,
        run_name=args.run_name, resume_from=args.resume_from)

    if not args.no_plots:
        P.plot_loss_curves(losses["train"], losses["val"],
                           title=f"Training Curves — {args.run_name}",
                           output_dir=args.output_dir)
    print("Training complete.")


def run_sample(args):
    from hep_nsf.utils import get_device
    import hep_nsf.plotting as P

    if args.checkpoint is None:
        sys.exit("--checkpoint is required for mode=sample")

    device = get_device(args.device)
    model, mtype = _load_model_from_checkpoint(args.checkpoint, args, device)
    _, mean, std, data_phys, raw = _prepare_data(args, model_type=mtype)

    x_phys = _sample_physical(model, mtype, args.num_samples, device, mean, std)

    x_np = x_phys.cpu().numpy()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"{args.run_name}_samples.npy"
    np.save(out_path, x_np)
    print(f"Saved {args.num_samples} samples → {out_path}")

    if not args.no_plots:
        _generate_all_plots(mtype, model, x_np, data_phys, raw,
                            mean, std, device, args)


def run_evaluate(args):
    from hep_nsf.utils   import (get_device, cartesian_to_spherical)
    from hep_nsf.analysis import evaluate, consistency_report

    if args.checkpoint is None:
        sys.exit("--checkpoint is required for mode=evaluate")

    device = get_device(args.device)
    model, mtype = _load_model_from_checkpoint(args.checkpoint, args, device)
    _, mean, std, data_phys, raw = _prepare_data(args, model_type=mtype)

    x_phys = _sample_physical(model, mtype, args.num_samples, device, mean, std)

    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        cyl_d = data_phys.numpy()
        cyl_s = x_phys.cpu().numpy()
    else:
        cyl_d = cartesian_to_spherical(raw).numpy()
        cyl_s = cartesian_to_spherical(x_phys.cpu()).numpy()

    evaluate(cyl_d[:, 0], cyl_d[:, 1], cyl_s[:, 0], cyl_s[:, 1])
    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        consistency_report(torch.tensor(cyl_s), target_r=args.target_r)


def run_plot(args):
    from hep_nsf.utils import load_losses
    if args.checkpoint is None:
        sys.exit("--checkpoint is required for mode=plot")
    loss_path = (Path(args.checkpoint)
                 .with_name(Path(args.checkpoint).stem
                            .replace("_best","_losses") + ".json"))
    if loss_path.exists():
        import hep_nsf.plotting as P
        losses = load_losses(loss_path)
        P.plot_loss_curves(losses["train"], losses["val"],
                           title="Training Curves",
                           output_dir=args.output_dir)
    run_sample(args)


# ---------------------------------------------------------------------------
# Shared plot dispatcher — called from run_sample and run_plot
# ---------------------------------------------------------------------------

def _generate_all_plots(mtype, model, x_np, data_phys, raw,
                        mean, std, device, args):
    import hep_nsf.plotting as P
    from hep_nsf.utils import spherical_to_cartesian, cartesian_to_spherical

    out = args.output_dir

    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        ang_data  = data_phys.numpy()
        ang_gen   = x_np
        cart_data = raw.numpy()
        cart_gen  = spherical_to_cartesian(
            torch.tensor(x_np), r=args.target_r).numpy()
    else:
        cart_data = data_phys.numpy()
        cart_gen  = x_np
        ang_data  = cartesian_to_spherical(raw).numpy()
        ang_gen   = cartesian_to_spherical(torch.tensor(x_np)).numpy()

    P.plot_marginal_1d(ang_data, ang_gen,
                       labels=[r"$\cos\theta$", r"$\phi$"],
                       title=f"{mtype.upper()} Angular Marginals",
                       save=f"{mtype}_marginals_angular_1d.png", output_dir=out)

    P.plot_marginal_2d(ang_data, ang_gen,
                       labels=[r"$\cos\theta$", r"$\phi$"],
                       title=f"{mtype.upper()} Angular 2D",
                       save=f"{mtype}_marginals_angular_2d.png", output_dir=out)

    P.plot_marginal_1d(cart_data, cart_gen,
                       labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
                       title=f"{mtype.upper()} Cartesian Marginals",
                       save=f"{mtype}_marginals_cartesian_1d.png", output_dir=out)

    P.plot_marginal_2d(cart_data, cart_gen,
                       labels=[r"$p_x$", r"$p_y$", r"$p_z$"],
                       title=f"{mtype.upper()} Cartesian 2D",
                       save=f"{mtype}_marginals_cartesian_2d.png", output_dir=out)

    P.plot_mollweide_kde(torch.tensor(ang_data),
                         title=f"{mtype.upper()} Data KDE",
                         save=f"{mtype}_mollweide_data.png", output_dir=out)
    P.plot_mollweide_kde(torch.tensor(ang_gen),
                         title=f"{mtype.upper()} Flow KDE",
                         save=f"{mtype}_mollweide_flow.png", output_dir=out)

    P.plot_physics_comparison(
        torch.tensor(cart_data), torch.tensor(cart_gen),
        title=f"{mtype.upper()} Physics Comparison",
        save=f"{mtype}_physics_comparison.png", output_dir=out)

    # Radius distribution only for r3
    if mtype == "r3":
        P.plot_radius_distribution(
            cart_gen,
            title=f"{mtype.upper()} |p| Distribution",
            save=f"{mtype}_radius_dist.png", output_dir=out)

    # Base mapping
    if mtype == "s2":
        P.plot_base_mapping_s2(model, device, mean, std,
                                title="S2 Base Mapping",
                                save="s2_base_mapping.png", output_dir=out)
    elif mtype == "s2rec":
        P.plot_base_mapping_s2(model, device, mean, std,
                                title="S2REC Base Mapping",
                                save="s2rec_base_mapping.png", output_dir=out)
    elif mtype == "s2norm":
        P.plot_base_mapping(model, device, mean, std, dim=2,
                            title="S2NORM Base Mapping",
                            save="s2norm_base_mapping.png", output_dir=out)
    elif mtype == "r2":
        P.plot_base_mapping(model, device, mean, std, dim=2,
                            title="R2 Base Mapping",
                            save="r2_base_mapping.png", output_dir=out)

    # Jacobian map
    if mtype in ("s2", "s2norm", "s2rec", "r2"):
        P.plot_jacobian_map(model, device, mean, std,
                            title=f"{mtype.upper()} Jacobian Map",
                            save=f"{mtype}_jacobian_map.png", output_dir=out)
    else:
        P.plot_jacobian_map_r3(model, device, mean, std,
                                title="R3 Jacobian Map",
                                save="r3_jacobian_map.png", output_dir=out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.config:
        args = merge_config(args, load_yaml_config(args.config))

    print("=" * 60)
    print(" HEP Neural Spline Flow")
    print(f" Mode      : {args.mode}")
    print(f" Model     : {args.model}")
    print(f" Data      : {args.data}")
    print(f" Bins      : {args.num_bins}   Splines: {args.num_splines}")
    print("=" * 60 + "\n")

    dispatch = {"train":    run_train,
                "sample":   run_sample,
                "evaluate": run_evaluate,
                "plot":     run_plot}
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
