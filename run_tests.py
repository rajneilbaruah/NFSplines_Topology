#!/usr/bin/env python
"""
run_tests.py
============
Exhaustive test suite for hep_nsf.

Tests every function, every argument combination, every model × arch ×
activation × num_splines × scheduler combination. Designed to run fast
using tiny models and few epochs.

Usage
-----
    python run_tests.py --data ../datasets/NFSpheres/eemumu_mup.json

Outputs
-------
    test_run.log        -- full log of every test with pass/fail + errors
    test_summary.md     -- readable summary grouped by section

Speed settings used throughout
-------------------------------
    num_bins=8, hidden_dim=16, num_layers=1,
    epochs=3,   batch_size=512, patience=50,
    num_samples=100
"""

import argparse
import subprocess
import sys
import os
import time
import traceback
import json
import math
from pathlib import Path
from datetime import datetime
from io import StringIO

import numpy as np
import torch

# ── Speed constants ──────────────────────────────────────────────────────── #
BINS    = 8
HDIM    = 16
LAYERS  = 1
EPOCHS  = 3
BS      = 512
PAT     = 50
NSAMP   = 100
KDE_BW  = 0.5    # faster KDE

TMPDIR  = Path("_test_tmp")
CKDIR   = TMPDIR / "checkpoints"
OUTDIR  = TMPDIR / "outputs"

# ── Result tracking ──────────────────────────────────────────────────────── #
results = []          # list of dicts: {section, name, status, error, elapsed}

LOG_FILE     = Path("test_run.log")
SUMMARY_FILE = Path("test_summary.md")

log_fh = None   # opened in main()


def _log(msg):
    print(msg)
    if log_fh:
        log_fh.write(msg + "\n")
        log_fh.flush()


def _record(section, name, status, error="", elapsed=0.0):
    results.append({
        "section": section, "name": name,
        "status": status, "error": error,
        "elapsed": round(elapsed, 2)
    })
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}[status]
    _log(f"  {icon} [{status:4s}] {name}  ({elapsed:.2f}s)")
    if error:
        for line in error.strip().splitlines()[-6:]:
            _log(f"         | {line}")


def run_test(section, name, fn, *args, **kwargs):
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        _record(section, name, "PASS", elapsed=time.time()-t0)
    except Exception:
        _record(section, name, "FAIL",
                error=traceback.format_exc(),
                elapsed=time.time()-t0)


def skip_test(section, name, reason):
    _record(section, name, "SKIP", error=reason)


def section_header(title):
    bar = "=" * 60
    _log(f"\n{bar}\n  {title}\n{bar}")


# ── CLI helper ───────────────────────────────────────────────────────────── #
def cli(section, name, cmd_args, extra_env=None):
    """Run a CLI command via subprocess and record pass/fail."""
    t0  = time.time()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "hep_nsf.main"] + cmd_args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=120, env=env)
        elapsed = time.time() - t0
        if r.returncode != 0:
            _record(section, name, "FAIL",
                    error=r.stderr[-1000:], elapsed=elapsed)
        else:
            _record(section, name, "PASS", elapsed=elapsed)
    except subprocess.TimeoutExpired:
        _record(section, name, "FAIL",
                error="TIMEOUT (>120s)", elapsed=time.time()-t0)
    except Exception:
        _record(section, name, "FAIL",
                error=traceback.format_exc(), elapsed=time.time()-t0)


def script_cli(section, name, script, cmd_args):
    """Run one of the helper scripts (train_all, plot_all, analyse_all)."""
    t0  = time.time()
    cmd = [sys.executable, script] + cmd_args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - t0
        if r.returncode != 0:
            _record(section, name, "FAIL",
                    error=r.stderr[-1000:], elapsed=elapsed)
        else:
            _record(section, name, "PASS", elapsed=elapsed)
    except subprocess.TimeoutExpired:
        _record(section, name, "FAIL",
                error="TIMEOUT (>300s)", elapsed=time.time()-t0)
    except Exception:
        _record(section, name, "FAIL",
                error=traceback.format_exc(), elapsed=time.time()-t0)


# ===========================================================================
# SECTION 1: Utils
# ===========================================================================

def test_utils(data_path):
    section_header("SECTION 1: Utils")
    from hep_nsf.utils import (
        load_json_data, cartesian_to_spherical, spherical_to_cartesian,
        normalise, denormalise, make_dataloaders, save_checkpoint,
        load_checkpoint, set_seed, get_device, cartesian_to_physics,
        count_parameters, log_std_correction, save_losses, load_losses)
    from hep_nsf.networks import build_model

    S = "Utils"

    run_test(S, "set_seed",         lambda: set_seed(42))
    run_test(S, "get_device(auto)", lambda: get_device("auto"))
    run_test(S, "get_device(cpu)",  lambda: get_device("cpu"))

    run_test(S, "load_json_data",
             lambda: load_json_data(data_path))

    def _cart_to_sph():
        raw = load_json_data(data_path)
        c   = cartesian_to_spherical(raw, phi_range="0_2pi")
        assert c.shape[1] == 2
        assert c[:, 0].min() >= -1 and c[:, 0].max() <= 1
        assert c[:, 1].min() >= 0  and c[:, 1].max() <= 2*math.pi

    run_test(S, "cartesian_to_spherical", _cart_to_sph)

    def _sph_to_cart():
        raw = load_json_data(data_path)
        cyl = cartesian_to_spherical(raw)
        c   = spherical_to_cartesian(cyl, r=500.0)
        assert c.shape[1] == 3

    run_test(S, "spherical_to_cartesian", _sph_to_cart)

    def _normalise():
        raw = load_json_data(data_path)
        n, m, s = normalise(raw)
        assert n.shape == raw.shape
        assert abs(n.mean().item()) < 0.1

    run_test(S, "normalise", _normalise)

    def _denormalise():
        raw = load_json_data(data_path)
        n, m, s = normalise(raw)
        r2 = denormalise(n, m, s)
        assert (raw - r2).abs().max() < 1e-4

    run_test(S, "denormalise (round-trip)", _denormalise)

    def _dataloaders():
        raw = load_json_data(data_path)
        n, m, s = normalise(raw)
        tl, vl = make_dataloaders(n, batch_size=BS, val_frac=0.2)
        xb = next(iter(tl))[0]
        assert xb.shape[1] == 3

    run_test(S, "make_dataloaders", _dataloaders)

    def _cart_phys():
        raw = load_json_data(data_path)
        d   = cartesian_to_physics(raw[:100])
        for k in ["px","py","pz","pT","eta","phi","|p|"]:
            assert k in d

    run_test(S, "cartesian_to_physics", _cart_phys)

    def _checkpoint():
        model = build_model("r2", num_bins=BINS, hidden_dim=HDIM,
                            num_layers=LAYERS, bound=5.0)
        opt   = torch.optim.Adam(model.parameters())
        p     = CKDIR / "test_utils_ckpt.pt"
        save_checkpoint(p, model, opt, 0, 99.0,
                        extra={"model_type":"r2","num_bins":BINS,
                               "num_splines":1,"bound":5.0,"hidden_dim":HDIM})
        m2 = build_model("r2", num_bins=BINS, hidden_dim=HDIM,
                          num_layers=LAYERS, bound=5.0)
        meta = load_checkpoint(p, m2)
        assert meta["model_type"] == "r2"

    run_test(S, "save/load_checkpoint", _checkpoint)

    def _losses():
        p = TMPDIR / "test_losses.json"
        save_losses({"train":[1.0,2.0],"val":[1.1,2.1]}, p)
        d = load_losses(p)
        assert d["train"] == [1.0, 2.0]

    run_test(S, "save/load_losses", _losses)

    run_test(S, "count_parameters",
             lambda: count_parameters(
                 build_model("r2", num_bins=BINS, hidden_dim=HDIM,
                             num_layers=LAYERS, bound=5.0)) > 0)

    run_test(S, "log_std_correction",
             lambda: log_std_correction(torch.ones(1, 3)))


# ===========================================================================
# SECTION 2: Splines
# ===========================================================================

def test_splines():
    section_header("SECTION 2: Splines")
    from hep_nsf.splines import rqs, rqs_with_bounds, rqs_circular
    S = "Splines"
    B, K = 64, BINS

    def _rqs_fwd_inv():
        x   = torch.rand(B, 2) * 1.6 - 0.8
        w   = torch.randn(B, 2, K)
        h   = torch.randn(B, 2, K)
        d   = torch.randn(B, 2, K+1)
        z, ldj = rqs(x, w, h, d, bound=1.0)
        xr, _  = rqs(z, w, h, d, inverse=True, bound=1.0)
        assert (x - xr).abs().max() < 1e-3, f"max err={(x-xr).abs().max()}"

    run_test(S, "rqs forward+inverse round-trip", _rqs_fwd_inv)

    def _rqs_ldj():
        x   = torch.rand(B, 1) * 1.6 - 0.8
        w   = torch.randn(B, 1, K)
        h   = torch.randn(B, 1, K)
        d   = torch.randn(B, 1, K+1)
        _, ldj = rqs(x, w, h, d, bound=1.0)
        assert torch.isfinite(ldj).all()

    run_test(S, "rqs log-det finite", _rqs_ldj)

    def _rqs_bounds_round():
        x  = torch.rand(B) * 1.8 - 0.9
        w  = torch.randn(B, K); h = torch.randn(B, K); d = torch.randn(B, K+1)
        z, _  = rqs_with_bounds(x, w, h, d, b_x=(-1,1), b_y=(-1,1))
        xr, _ = rqs_with_bounds(z, w, h, d, inverse=True, b_x=(-1,1), b_y=(-1,1))
        assert (x - xr).abs().max() < 1e-4

    run_test(S, "rqs_with_bounds round-trip  (cos_theta domain)", _rqs_bounds_round)

    def _rqs_bounds_phi():
        TWO_PI = 2*math.pi
        x  = torch.rand(B) * (TWO_PI - 0.2) + 0.1
        w  = torch.randn(B, K); h = torch.randn(B, K); d = torch.randn(B, K+1)
        z, _  = rqs_with_bounds(x, w, h, d, b_x=(0,TWO_PI), b_y=(0,TWO_PI))
        xr, _ = rqs_with_bounds(z, w, h, d, inverse=True,
                                 b_x=(0,TWO_PI), b_y=(0,TWO_PI))
        assert (x - xr).abs().max() < 1e-4

    run_test(S, "rqs_with_bounds round-trip  (phi domain)", _rqs_bounds_phi)

    def _rqs_circular_round():
        TWO_PI = 2*math.pi
        x  = torch.rand(B) * (TWO_PI - 0.2) + 0.1
        w  = torch.randn(B, K); h = torch.randn(B, K)
        d  = torch.randn(B, K)   # K not K+1
        z, _  = rqs_circular(x, w, h, d, b_x=(0,TWO_PI), b_y=(0,TWO_PI))
        xr, _ = rqs_circular(z, w, h, d, inverse=True,
                              b_x=(0,TWO_PI), b_y=(0,TWO_PI))
        assert (x - xr).abs().max() < 1e-4

    run_test(S, "rqs_circular round-trip  (periodic phi)", _rqs_circular_round)

    def _rqs_circular_periodicity():
        """d[0] == d[K] is enforced internally — verify LDJ is finite."""
        TWO_PI = 2*math.pi
        x  = torch.rand(B) * (TWO_PI - 0.2) + 0.1
        w  = torch.randn(B, K); h = torch.randn(B, K); d = torch.randn(B, K)
        _, ldj = rqs_circular(x, w, h, d, b_x=(0,TWO_PI), b_y=(0,TWO_PI))
        assert torch.isfinite(ldj).all()

    run_test(S, "rqs_circular LDJ finite (periodicity check)", _rqs_circular_periodicity)

    # Vary num_bins
    for K2 in [4, 16, 32]:
        def _fn(k=K2):
            x  = torch.rand(B) * 1.6 - 0.8
            w  = torch.randn(B, k); h = torch.randn(B, k); d = torch.randn(B, k+1)
            z, _ = rqs_with_bounds(x, w, h, d, b_x=(-1,1), b_y=(-1,1))
            xr,_ = rqs_with_bounds(z, w, h, d, inverse=True,
                                    b_x=(-1,1), b_y=(-1,1))
            assert (x-xr).abs().max() < 1e-3
        run_test(S, f"rqs_with_bounds  num_bins={K2}", _fn)


# ===========================================================================
# SECTION 3: MLP / ResNet configurations
# ===========================================================================

def test_mlps():
    section_header("SECTION 3: MLP / ResNet configurations")
    from hep_nsf.mlps import build_conditioner
    S = "MLPs"

    B = 32; in_d = 1; out_d = 3*BINS+1

    for act in ["relu","tanh","elu","leaky_relu","silu","gelu"]:
        def _fn(a=act):
            net = build_conditioner("mlp", in_d, out_d,
                                    hidden_dim=HDIM, num_layers=2,
                                    activation=a)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"MLP  activation={act}", _fn)

    for nl in [1, 2, 3, 4]:
        def _fn(n=nl):
            net = build_conditioner("mlp", in_d, out_d,
                                    hidden_dim=HDIM, num_layers=n)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"MLP  num_layers={nl}", _fn)

    for hd in [16, 32, 64, 128]:
        def _fn(h=hd):
            net = build_conditioner("mlp", in_d, out_d,
                                    hidden_dim=h, num_layers=2)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"MLP  hidden_dim={hd}", _fn)

    for drop in [0.0, 0.1, 0.3]:
        def _fn(dr=drop):
            net = build_conditioner("mlp", in_d, out_d,
                                    hidden_dim=HDIM, dropout=dr)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"MLP  dropout={drop}", _fn)

    for act in ["relu","tanh","elu","leaky_relu","silu","gelu"]:
        def _fn(a=act):
            net = build_conditioner("resnet", in_d, out_d,
                                    hidden_dim=HDIM, num_layers=2,
                                    activation=a)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"ResNet  activation={act}", _fn)

    for nl in [1, 2, 3]:
        def _fn(n=nl):
            net = build_conditioner("resnet", in_d, out_d,
                                    hidden_dim=HDIM, num_layers=n)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"ResNet  num_layers(blocks)={nl}", _fn)

    for hd in [16, 32, 64]:
        def _fn(h=hd):
            net = build_conditioner("resnet", in_d, out_d,
                                    hidden_dim=h, num_layers=2)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"ResNet  hidden_dim={hd}", _fn)

    for drop in [0.0, 0.2]:
        def _fn(dr=drop):
            net = build_conditioner("resnet", in_d, out_d,
                                    hidden_dim=HDIM, dropout=dr,
                                    use_layer_norm=True)
            y = net(torch.randn(B, in_d))
            assert y.shape == (B, out_d)
        run_test(S, f"ResNet  dropout={drop}  layer_norm=True", _fn)


# ===========================================================================
# SECTION 4: Model API
# ===========================================================================

def test_model_api():
    section_header("SECTION 4: Model API")
    from hep_nsf.networks import build_model
    S = "ModelAPI"

    def _make_data(mtype):
        if mtype in ("s2", "s2rec"):
            # physical (cos_theta, phi)
            x = torch.stack([
                torch.rand(50) * 1.8 - 0.9,
                torch.rand(50) * (2*math.pi - 0.2) + 0.1], dim=1)
        elif mtype in ("s2norm", "r2"):
            x = torch.randn(50, 2)
        else:
            x = torch.randn(50, 3)
        return x

    configs = []
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        for arch in ["mlp","resnet"]:
            for ns in [1, 2, 3]:
                configs.append((mtype, arch, ns))

    for mtype, arch, ns in configs:
        kwargs = dict(num_bins=BINS, num_splines=ns,
                      hidden_dim=HDIM, num_layers=LAYERS,
                      arch=arch, activation="relu", dropout=0.0)
        if mtype in ("s2norm", "r2","r3"):
            kwargs["bound"] = 5.0

        # forward
        def _fwd(m=mtype, kw=kwargs):
            model = build_model(m, **kw)
            x = _make_data(m)
            z, ldj = model(x)
            assert torch.isfinite(ldj).all()
        run_test(S, f"forward  {mtype}  {arch}  ns={ns}", _fwd)

        # inverse round-trip
        def _rt(m=mtype, kw=kwargs):
            model = build_model(m, **kw)
            x  = _make_data(m)
            z, _  = model(x)
            xr = model(z, inverse=True)
            err = (x - xr).abs().max().item()
            assert err < 1e-2, f"round-trip err={err:.2e}"
        run_test(S, f"round-trip  {mtype}  {arch}  ns={ns}", _rt)

        # log_prob
        def _lp(m=mtype, kw=kwargs):
            model = build_model(m, **kw)
            x  = _make_data(m)
            lp = model.log_prob(x)
            assert lp.shape == (50,)
            assert torch.isfinite(lp).all()
        run_test(S, f"log_prob  {mtype}  {arch}  ns={ns}", _lp)

        # sample
        def _samp(m=mtype, kw=kwargs):
            model = build_model(m, **kw)
            s  = model.sample(30)
            dim = 2 if m in ("s2","r2") else 3
            assert s.shape == (30, dim)
            assert torch.isfinite(s).all()
        run_test(S, f"sample  {mtype}  {arch}  ns={ns}", _samp)

    # build_model error
    def _bad():
        try:
            build_model("xyz")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
    run_test(S, "build_model unknown key raises ValueError", _bad)


# ===========================================================================
# SECTION 5: Python API training
# ===========================================================================

def test_python_api_training(data_path):
    section_header("SECTION 5: Python API training")
    from hep_nsf.utils import (load_json_data, normalise,
                                cartesian_to_spherical, make_dataloaders)
    from hep_nsf.networks import build_model
    from hep_nsf.train    import train_model
    S = "PyAPITrain"

    raw = load_json_data(data_path)
    cyl = cartesian_to_spherical(raw, phi_range="0_2pi")

    def _train(mtype, arch, act, ns, use_plateau, use_cosine, dropout=0.0):
        if mtype in ("s2","s2norm","s2rec","r2"):
            data = cyl
        else:
            data = raw

        if mtype in ("s2", "s2rec"):
            dn   = data
            mean = torch.zeros(1, 2)
            std  = torch.ones(1, 2)
        else:
            dn, mean, std = normalise(data)
        tl, vl = make_dataloaders(dn, batch_size=BS, val_frac=0.2)

        kwargs = dict(num_bins=BINS, num_splines=ns,
                      hidden_dim=HDIM, num_layers=LAYERS,
                      arch=arch, activation=act, dropout=dropout)
        if mtype in ("s2norm", "r2","r3"):
            kwargs["bound"] = 5.0

        model = build_model(mtype, **kwargs)
        tag   = f"{mtype}_{arch}_{act}_ns{ns}"
        rname = f"pyapi_{tag}"

        model, losses = train_model(
            model, tl, vl, std_tensor=std,
            device=torch.device("cpu"),
            model_type=mtype,
            num_layers=LAYERS,
            arch=arch,
            activation=act,
            max_epochs=EPOCHS, patience=PAT,
            log_every=999,
            use_plateau=use_plateau, use_cosine=use_cosine,
            save_dir=str(CKDIR), run_name=rname)

        assert len(losses["train"]) > 0
        assert (CKDIR / f"{rname}_best.pt").exists()

    # All 5 models × both archs
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        for arch in ["mlp","resnet"]:
            run_test(S, f"train  {mtype}  {arch}",
                     _train, mtype, arch, "relu", 1, True, False)

    # All activations on r2/mlp
    for act in ["relu","tanh","elu","leaky_relu","silu","gelu"]:
        run_test(S, f"train  r2  mlp  act={act}",
                 _train, "r2", "mlp", act, 1, True, False)

    # num_splines 1,2,3 for each model
    for mtype in ["s2","r2","r3"]:
        for ns in [1,2,3]:
            run_test(S, f"train  {mtype}  ns={ns}",
                     _train, mtype, "mlp", "relu", ns, True, False)

    # dropout
    run_test(S, "train  r2  mlp  dropout=0.2",
             _train, "r2", "mlp", "relu", 1, True, False, 0.2)

    # scheduler combinations
    for up, uc in [(True,False),(False,True),(True,True),(False,False)]:
        run_test(S, f"train  r2  plateau={up}  cosine={uc}",
                 _train, "r2", "mlp", "relu", 1, up, uc)


# ===========================================================================
# SECTION 6: CLI training
# ===========================================================================

def test_cli_training(data_path):
    section_header("SECTION 6: CLI training")
    S = "CLITrain"

    base = ["--data", data_path, "--epochs", str(EPOCHS),
            "--patience", str(PAT), "--batch_size", str(BS),
            "--num_bins", str(BINS), "--hidden_dim", str(HDIM),
            "--num_layers", str(LAYERS), "--log_every", "999",
            "--save_dir", str(CKDIR), "--output_dir", str(OUTDIR),
            "--no_plots", "--device", "cpu"]

    # All 5 models
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        cli(S, f"train  {mtype}  defaults",
            ["--mode","train","--model",mtype,
             "--run_name",f"cli_{mtype}"] + base)

    # Archs
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        for arch in ["mlp","resnet"]:
            cli(S, f"train  {mtype}  {arch}",
                ["--mode","train","--model",mtype,"--arch",arch,
                 "--run_name",f"cli_{mtype}_{arch}"] + base)

    # Activations (on r2)
    for act in ["relu","tanh","elu","leaky_relu","silu","gelu"]:
        cli(S, f"train  r2  act={act}",
            ["--mode","train","--model","r2","--activation",act,
             "--run_name",f"cli_r2_{act}"] + base)

    # num_splines
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        for ns in [1,2,3]:
            cli(S, f"train  {mtype}  ns={ns}",
                ["--mode","train","--model",mtype,
                 "--num_splines",str(ns),
                 "--run_name",f"cli_{mtype}_ns{ns}"] + base)

    # dropout
    cli(S, "train  r2  dropout=0.2",
        ["--mode","train","--model","r2","--dropout","0.2",
         "--run_name","cli_r2_drop"] + base)

    # Schedulers
    cli(S, "train  r2  no_plateau",
        ["--mode","train","--model","r2","--no_plateau",
         "--run_name","cli_r2_noplateau"] + base)
    cli(S, "train  r2  use_cosine",
        ["--mode","train","--model","r2","--use_cosine",
         "--run_name","cli_r2_cosine"] + base)
    cli(S, "train  r2  plateau+cosine",
        ["--mode","train","--model","r2","--use_cosine",
         "--run_name","cli_r2_both"] + base)
    cli(S, "train  r2  no_plateau+cosine",
        ["--mode","train","--model","r2","--no_plateau","--use_cosine",
         "--run_name","cli_r2_cosineonly"] + base)

    # bound variations (r2, r3)
    for bnd in [3.0, 5.0, 8.0]:
        cli(S, f"train  r2  bound={bnd}",
            ["--mode","train","--model","r2",
             "--bound",str(bnd),
             "--run_name",f"cli_r2_bnd{bnd}"] + base)

    # hidden_dim and num_layers
    for hd in [16, 32, 64]:
        cli(S, f"train  r2  hidden_dim={hd}",
            ["--mode","train","--model","r2",
             "--hidden_dim",str(hd),
             "--run_name",f"cli_r2_hd{hd}"] + base)
    for nl in [1, 2, 3]:
        cli(S, f"train  r2  num_layers={nl}",
            ["--mode","train","--model","r2",
             "--num_layers",str(nl),
             "--run_name",f"cli_r2_nl{nl}"] + base)

    # ResNet with all activations
    for act in ["relu","tanh","elu"]:
        cli(S, f"train  r2  resnet  act={act}",
            ["--mode","train","--model","r2","--arch","resnet",
             "--activation",act,
             "--run_name",f"cli_r2_resnet_{act}"] + base)


# ===========================================================================
# SECTION 7: Resume from checkpoint
# ===========================================================================

def test_resume(data_path):
    section_header("SECTION 7: Resume from checkpoint")
    S = "Resume"

    base = ["--data", data_path, "--epochs", str(EPOCHS+2),
            "--patience", str(PAT), "--batch_size", str(BS),
            "--num_bins", str(BINS), "--hidden_dim", str(HDIM),
            "--num_layers", str(LAYERS), "--log_every", "999",
            "--save_dir", str(CKDIR), "--output_dir", str(OUTDIR),
            "--no_plots", "--device", "cpu"]

    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        ckpt = CKDIR / f"cli_{mtype}_best.pt"
        if not ckpt.exists():
            skip_test(S, f"resume  {mtype}", "checkpoint not found (train section failed)")
            continue
        cli(S, f"resume  {mtype}",
            ["--mode","train","--model",mtype,
             "--run_name",f"cli_{mtype}_resumed",
             "--resume_from", str(ckpt)] + base)

    # Resume with MLP checkpoint
    ckpt_mlp = CKDIR / "cli_r2_mlp_best.pt"
    if ckpt_mlp.exists():
        cli(S, "resume  r2  mlp  checkpoint",
            ["--mode","train","--model","r2","--arch","mlp",
             "--run_name","cli_r2_mlp_resumed",
             "--resume_from", str(ckpt_mlp)] + base)
    else:
        skip_test(S, "resume  r2  mlp  checkpoint", "checkpoint not found")

    # Resume with ResNet checkpoint
    ckpt_resnet = CKDIR / "cli_r2_resnet_best.pt"
    if ckpt_resnet.exists():
        cli(S, "resume  r2  resnet  checkpoint",
            ["--mode","train","--model","r2","--arch","resnet",
             "--run_name","cli_r2_resnet_resumed",
             "--resume_from", str(ckpt_resnet)] + base)
    else:
        skip_test(S, "resume  r2  resnet  checkpoint", "checkpoint not found")


# ===========================================================================
# SECTION 8: YAML config
# ===========================================================================

def test_yaml_config(data_path):
    section_header("SECTION 8: YAML config")
    S = "YAML"

    cfg = {
        "mode": "train", "model": "r2",
        "num_bins": BINS, "num_splines": 1,
        "hidden_dim": HDIM, "num_layers": LAYERS,
        "bound": 5.0, "arch": "mlp", "activation": "relu",
        "lr": 1e-3, "batch_size": BS, "epochs": EPOCHS,
        "patience": PAT, "log_every": 999,
        "save_dir": str(CKDIR), "output_dir": str(OUTDIR),
        "no_plots": True, "device": "cpu",
        "run_name": "yaml_r2"
    }

    import yaml
    cfg_path = TMPDIR / "test_cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    cli(S, "train via YAML config",
        ["--config", str(cfg_path), "--data", data_path])

    # YAML + CLI override (override model to s2)
    cli(S, "YAML + CLI override model",
        ["--config", str(cfg_path), "--data", data_path,
         "--model","s2","--run_name","yaml_s2_override"])

    # Test the actual shipped config files
    for cfg_name, model_key in [
        ("s2_default.yaml", "s2"),
        ("r2_default.yaml", "r2"),
        ("r3_default.yaml", "r3"),
    ]:
        cfg_file = Path("hep_nsf/configs") / cfg_name
        if cfg_file.exists():
            cli(S, f"train via shipped config {cfg_name}",
                ["--config", str(cfg_file), "--data", data_path,
                 "--epochs", str(EPOCHS), "--patience", str(PAT),
                 "--batch_size", str(BS), "--num_bins", str(BINS),
                 "--hidden_dim", str(HDIM), "--num_layers", str(LAYERS),
                 "--log_every", "999", "--no_plots", "--device", "cpu",
                 "--save_dir", str(CKDIR), "--output_dir", str(OUTDIR),
                 "--run_name", f"yaml_shipped_{model_key}"])
        else:
            skip_test(S, f"train via shipped config {cfg_name}",
                      f"{cfg_file} not found")


# ===========================================================================
# SECTION 9: CLI sample / evaluate / plot
# ===========================================================================

def test_cli_inference(data_path):
    section_header("SECTION 9: CLI sample / evaluate / plot")
    S = "CLIInference"

    base_inf = ["--data", data_path,
                "--num_samples", str(NSAMP),
                "--output_dir", str(OUTDIR),
                "--device", "cpu"]

    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        ckpt = CKDIR / f"cli_{mtype}_best.pt"
        if not ckpt.exists():
            skip_test(S, f"sample   {mtype}", "checkpoint missing")
            skip_test(S, f"sample   {mtype}  --no_plots", "checkpoint missing")
            skip_test(S, f"evaluate {mtype}", "checkpoint missing")
            skip_test(S, f"plot     {mtype}", "checkpoint missing")
            continue

        # sample with plots
        cli(S, f"sample  {mtype}  with plots",
            ["--mode","sample","--checkpoint",str(ckpt)] + base_inf)

        # sample without plots
        cli(S, f"sample  {mtype}  --no_plots",
            ["--mode","sample","--checkpoint",str(ckpt),
             "--no_plots"] + base_inf)

        # evaluate
        cli(S, f"evaluate  {mtype}",
            ["--mode","evaluate","--checkpoint",str(ckpt)] + base_inf)

        # plot
        cli(S, f"plot  {mtype}",
            ["--mode","plot","--checkpoint",str(ckpt),
             "--run_name",f"plotmode_{mtype}"] + base_inf)

    # Ensure self-describing checkpoint (no --model flag needed)
    for mtype in ["s2","s2norm","s2rec","r2","r3"]:
        ckpt = CKDIR / f"cli_{mtype}_best.pt"
        if not ckpt.exists():
            skip_test(S, f"sample  {mtype}  no arch flags", "checkpoint missing")
            continue
        # Deliberately omit --model --num_bins --hidden_dim etc
        cli(S, f"sample  {mtype}  self-describing (no arch flags)",
            ["--mode","sample","--checkpoint",str(ckpt),
             "--no_plots"] + base_inf)

    # ns=2 checkpoint — sample without mentioning ns=2
    ckpt_ns2 = CKDIR / "cli_r2_ns2_best.pt"
    if ckpt_ns2.exists():
        cli(S, "sample  r2  ns=2  self-describing",
            ["--mode","sample","--checkpoint",str(ckpt_ns2),
             "--no_plots"] + base_inf)
    else:
        skip_test(S, "sample  r2  ns=2  self-describing", "checkpoint missing")

    # ResNet checkpoint — sample without mentioning arch
    ckpt_rn = CKDIR / "cli_r2_resnet_best.pt"
    if ckpt_rn.exists():
        cli(S, "sample  r2  resnet  self-describing",
            ["--mode","sample","--checkpoint",str(ckpt_rn),
             "--no_plots"] + base_inf)
    else:
        skip_test(S, "sample  r2  resnet  self-describing", "checkpoint missing")

    # Error: missing checkpoint
    def _no_ckpt():
        r = subprocess.run(
            [sys.executable,"-m","hep_nsf.main","--mode","sample",
             "--data",data_path,"--checkpoint","nonexistent.pt",
             "--device","cpu"],
            capture_output=True, text=True, timeout=30)
        assert r.returncode != 0, "Should have failed with missing checkpoint"
    run_test(S, "sample  missing checkpoint raises error", _no_ckpt)


# ===========================================================================
# SECTION 10: Analysis functions
# ===========================================================================

def test_analysis(data_path):
    section_header("SECTION 10: Analysis functions")
    from hep_nsf.utils import load_json_data, cartesian_to_spherical
    from hep_nsf.analysis import (kl_divergence_kde, effective_sample_size,
                                   wasserstein_1d, js_divergence_1d,
                                   consistency_report, evaluate, model_summary)
    from hep_nsf.networks import build_model
    S = "Analysis"

    raw = load_json_data(data_path)
    cyl = cartesian_to_spherical(raw, phi_range="0_2pi").numpy()
    N   = 500
    rng = np.random.default_rng(42)

    cos_d = cyl[:N, 0]; phi_d = cyl[:N, 1]
    cos_g = np.clip(cos_d + rng.normal(0, 0.05, N), -0.99, 0.99)
    phi_g = (phi_d + rng.normal(0, 0.1, N)) % (2*math.pi)

    run_test(S, "kl_divergence_kde",
             lambda: kl_divergence_kde(cos_d, phi_d, cos_g, phi_g,
                                        bw=KDE_BW, n_eval=200))

    run_test(S, "effective_sample_size",
             lambda: effective_sample_size(cos_d, phi_d, cos_g, phi_g,
                                            bw=KDE_BW))

    run_test(S, "wasserstein_1d",
             lambda: wasserstein_1d(
                 np.stack([cos_d,phi_d],axis=1),
                 np.stack([cos_g,phi_g],axis=1)))

    run_test(S, "js_divergence_1d",
             lambda: js_divergence_1d(
                 np.stack([cos_d,phi_d],axis=1),
                 np.stack([cos_g,phi_g],axis=1)))

    run_test(S, "consistency_report",
             lambda: consistency_report(
                 torch.tensor(np.stack([cos_g,phi_g],axis=1).astype(np.float32)),
                 target_r=500.0))

    run_test(S, "evaluate  (full suite)",
             lambda: evaluate(cos_d, phi_d, cos_g, phi_g,
                               bw=KDE_BW, n_eval=200))

    for mtype in ["s2","r2","r3"]:
        kwargs = dict(num_bins=BINS, hidden_dim=HDIM, num_layers=LAYERS)
        if mtype in ("r2","r3"):
            kwargs["bound"] = 5.0
        run_test(S, f"model_summary  {mtype}",
                 lambda m=mtype, kw=kwargs: model_summary(
                     build_model(m, **kw), model_type=m))


# ===========================================================================
# SECTION 11: All plotting functions
# ===========================================================================

def test_plotting(data_path):
    section_header("SECTION 11: All plotting functions")
    from hep_nsf.utils import (load_json_data, cartesian_to_spherical,
                                spherical_to_cartesian, normalise, denormalise)
    import hep_nsf.plotting as P
    S = "Plotting"

    raw                  = load_json_data(data_path)
    cyl                  = cartesian_to_spherical(raw, phi_range="0_2pi")
    cyl_norm, mean2, std2 = normalise(cyl)
    _, mean3, std3        = normalise(raw)

    N   = 500
    rng = np.random.default_rng(42)

    ang_data  = cyl.numpy()[:N]
    ang_gen   = (cyl[:N] + torch.tensor(
                    rng.normal(0, 0.03, (N,2)), dtype=torch.float32)
                ).clamp(torch.tensor([-1.,0.]),
                        torch.tensor([1., 2*math.pi])).numpy()
    cart_data = raw.numpy()[:N]
    cart_gen  = (raw[:N] + torch.tensor(
                    rng.normal(0, 5., (N,3)), dtype=torch.float32)).numpy()
    t_losses  = list(np.linspace(3, 1.5, 40) + rng.normal(0,.04,40))
    v_losses  = list(np.linspace(3.1,1.6,40) + rng.normal(0,.04,40))

    OUT = str(OUTDIR / "plots")

    run_test(S, "plot_loss_curves",
             lambda: P.plot_loss_curves(t_losses, v_losses,
                 save="t_loss.png", output_dir=OUT))

    run_test(S, "plot_marginal_1d  [angular]",
             lambda: P.plot_marginal_1d(ang_data, ang_gen,
                 labels=[r"$\cos\theta$",r"$\phi$"],
                 save="t_ang1d.png", output_dir=OUT))

    run_test(S, "plot_marginal_1d  [cartesian]",
             lambda: P.plot_marginal_1d(cart_data, cart_gen,
                 labels=[r"$p_x$",r"$p_y$",r"$p_z$"],
                 save="t_cart1d.png", output_dir=OUT))

    run_test(S, "plot_marginal_2d  [angular]",
             lambda: P.plot_marginal_2d(ang_data, ang_gen,
                 save="t_ang2d.png", output_dir=OUT))

    run_test(S, "plot_marginal_2d  [cartesian]",
             lambda: P.plot_marginal_2d(cart_data, cart_gen,
                 save="t_cart2d.png", output_dir=OUT))

    run_test(S, "plot_mollweide_kde  [tensor input]",
             lambda: P.plot_mollweide_kde(cyl[:N],
                 title="Test KDE", save="t_moll.png", output_dir=OUT))

    run_test(S, "plot_mollweide_kde  [ndarray input]",
             lambda: P.plot_mollweide_kde(
                 torch.tensor(ang_gen),
                 save="t_moll2.png", output_dir=OUT))

    run_test(S, "plot_physics_comparison",
             lambda: P.plot_physics_comparison(
                 raw[:N], torch.tensor(cart_gen),
                 save="t_phys.png", output_dir=OUT))

    run_test(S, "plot_radius_distribution",
             lambda: P.plot_radius_distribution(
                 cart_gen, save="t_rad.png", output_dir=OUT))

    run_test(S, "plot_radius_distribution  [near-zero range  auto bins]",
             lambda: P.plot_radius_distribution(
                 spherical_to_cartesian(cyl[:N], r=500.0).numpy(),
                 save="t_rad_fixed.png", output_dir=OUT))

    # Model-dependent plots — use checkpoints if available
    from hep_nsf.networks import build_model

    for mtype, ckpt_name, m2, s2 in [
        ("s2", "cli_s2_best.pt", mean2, std2),
        ("r2", "cli_r2_best.pt", mean2, std2),
        ("r3", "cli_r3_best.pt", mean3, std3),
    ]:
        ckpt = CKDIR / ckpt_name
        if not ckpt.exists():
            skip_test(S, f"plot_base_mapping     [{mtype}]", "checkpoint missing")
            skip_test(S, f"plot_jacobian_map     [{mtype}]", "checkpoint missing")
            continue

        meta   = torch.load(ckpt, map_location="cpu")
        kwargs = dict(num_bins=meta["num_bins"],
                      num_splines=meta["num_splines"],
                      hidden_dim=meta.get("hidden_dim", HDIM),
                      num_layers=meta.get("num_layers", LAYERS),
                      arch=meta.get("arch", "mlp"),
                      activation=meta.get("activation", "relu"))
        if mtype in ("r2","r3"):
            kwargs["bound"] = meta.get("bound", 5.0)
        model = build_model(mtype, **kwargs)
        model.load_state_dict(meta["model_state"])
        model.eval()

        if mtype == "s2":
            run_test(S, f"plot_base_mapping_s2  [{mtype}]",
                     lambda mo=model, me=m2, st=s2:
                     P.plot_base_mapping_s2(mo, torch.device("cpu"), me, st,
                         save="t_s2_bmap.png", output_dir=OUT))

        elif mtype == "r2":
            run_test(S, f"plot_base_mapping     [{mtype}]",
                     lambda mo=model, me=m2, st=s2:
                     P.plot_base_mapping(mo, torch.device("cpu"), me, st, dim=2,
                         save="t_r2_bmap.png", output_dir=OUT))

        if mtype in ("s2","r2"):
            run_test(S, f"plot_jacobian_map     [{mtype}]",
                     lambda mo=model, me=m2, st=s2:
                     P.plot_jacobian_map(mo, torch.device("cpu"), me, st,
                         n_samples=200,
                         save=f"t_{mtype}_jmap.png", output_dir=OUT))
        else:
            run_test(S, f"plot_jacobian_map_r3  [{mtype}]",
                     lambda mo=model, me=m2, st=s2:
                     P.plot_jacobian_map_r3(mo, torch.device("cpu"), me, st,
                         n_samples=200,
                         save=f"t_r3_jmap.png", output_dir=OUT))


# ===========================================================================
# SECTION 12: Helper scripts
# ===========================================================================

def test_helper_scripts(data_path):
    section_header("SECTION 12: Helper scripts (train_all / plot_all / analyse_all)")
    S = "HelperScripts"

    base = ["--data", data_path,
            "--num_bins", str(BINS), "--hidden_dim", str(HDIM),
            "--num_layers", str(LAYERS), "--epochs", str(EPOCHS),
            "--patience", str(PAT), "--batch_size", str(BS),
            "--device", "cpu",
            "--save_dir", str(CKDIR/"scripts"),
            "--output_dir", str(OUTDIR/"scripts")]

    # train_all — all models
    script_cli(S, "train_all  s2+r2+r3",
               "train_all.py", base)

    # train_all — with --skip
    script_cli(S, "train_all  --skip r3",
               "train_all.py", base + ["--skip","r3"])

    script_cli(S, "train_all  --skip s2 r2",
               "train_all.py", base + ["--skip","s2","r2"])

    # plot_all
    plot_base = ["--data", data_path,
                 "--num_samples", str(NSAMP),
                 "--checkpoints_dir", str(CKDIR/"scripts"),
                 "--output_dir", str(OUTDIR/"scripts_plots"),
                 "--device", "cpu"]

    script_cli(S, "plot_all  s2+r2+r3",
               "plot_all.py", plot_base)

    script_cli(S, "plot_all  --skip r3",
               "plot_all.py", plot_base + ["--skip","r3"])

    # analyse_all
    ana_base = ["--data", data_path,
                "--num_samples", str(NSAMP),
                "--kde_bw", str(KDE_BW),
                "--checkpoints_dir", str(CKDIR/"scripts"),
                "--output_dir", str(OUTDIR/"scripts_ana"),
                "--device", "cpu"]

    script_cli(S, "analyse_all  s2+r2+r3",
               "analyse_all.py", ana_base)

    script_cli(S, "analyse_all  --skip s2",
               "analyse_all.py", ana_base + ["--skip","s2"])


# ===========================================================================
# SECTION 13: Devices
# ===========================================================================

def test_devices(data_path):
    section_header("SECTION 13: Device availability and model forward on device")
    from hep_nsf.networks import build_model
    from hep_nsf.utils    import get_device
    S = "Devices"

    for dname in ["cpu", "cuda", "mps"]:
        def _fn(d=dname):
            if d == "cpu":
                device = get_device("cpu")
            elif d == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available")
                device = torch.device("cuda")
            elif d == "mps":
                if not (hasattr(torch.backends,"mps") and
                        torch.backends.mps.is_available()):
                    raise RuntimeError("MPS not available")
                device = torch.device("mps")

            for mtype in ["s2","r2","r3"]:
                kwargs = dict(num_bins=BINS, hidden_dim=HDIM, num_layers=LAYERS)
                if mtype in ("r2","r3"):
                    kwargs["bound"] = 5.0
                model = build_model(mtype, **kwargs).to(device)
                x = model.sample(10, device=device)
                assert x.device.type == device.type

        if dname == "cpu":
            run_test(S, f"device={dname}  forward+sample  all models", _fn)
        else:
            # Record SKIP instead of FAIL when hardware not available
            try:
                if dname == "cuda" and not torch.cuda.is_available():
                    skip_test(S, f"device={dname}  forward+sample", "CUDA not available on this machine")
                elif dname == "mps" and not (hasattr(torch.backends,"mps") and
                                              torch.backends.mps.is_available()):
                    skip_test(S, f"device={dname}  forward+sample", "MPS not available on this machine")
                else:
                    run_test(S, f"device={dname}  forward+sample  all models", _fn)
            except Exception:
                skip_test(S, f"device={dname}  forward+sample", "device not available")


# ===========================================================================
# Summary writer
# ===========================================================================

def write_summary():
    sections   = sorted(set(r["section"] for r in results))
    total_pass = sum(1 for r in results if r["status"]=="PASS")
    total_fail = sum(1 for r in results if r["status"]=="FAIL")
    total_skip = sum(1 for r in results if r["status"]=="SKIP")
    total      = len(results)

    lines = []
    lines.append("# hep_nsf Test Suite — Summary")
    lines.append(f"\nRun at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"| Result | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✓ PASS | {total_pass} |")
    lines.append(f"| ✗ FAIL | {total_fail} |")
    lines.append(f"| ○ SKIP | {total_skip} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    for sec in sections:
        sec_results = [r for r in results if r["section"]==sec]
        sp = sum(1 for r in sec_results if r["status"]=="PASS")
        sf = sum(1 for r in sec_results if r["status"]=="FAIL")
        ss = sum(1 for r in sec_results if r["status"]=="SKIP")
        status_icon = "✓" if sf==0 else "✗"
        lines.append(f"\n## {status_icon} {sec}  ({sp} pass / {sf} fail / {ss} skip)\n")
        lines.append("| Status | Test | Time (s) |")
        lines.append("|--------|------|----------|")
        for r in sec_results:
            icon = {"PASS":"✓","FAIL":"✗","SKIP":"○"}[r["status"]]
            lines.append(f"| {icon} {r['status']} | `{r['name']}` | {r['elapsed']} |")

    if total_fail > 0:
        lines.append("\n---\n")
        lines.append("## ✗ Failures — Details\n")
        for r in results:
            if r["status"] == "FAIL":
                lines.append(f"### `{r['section']} / {r['name']}`\n")
                lines.append("```")
                lines.append(r["error"][-800:] if r["error"] else "(no traceback)")
                lines.append("```\n")

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines))

    _log(f"\n{'='*60}")
    _log(f"  TOTAL: {total_pass} pass  {total_fail} fail  {total_skip} skip  / {total}")
    _log(f"  Log     → {LOG_FILE}")
    _log(f"  Summary → {SUMMARY_FILE}")
    _log(f"{'='*60}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    global log_fh

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to JSON data file")
    p.add_argument("--skip_sections", nargs="*", default=[],
                   help="Section numbers to skip e.g. --skip_sections 5 6")
    args = p.parse_args()

    TMPDIR.mkdir(exist_ok=True)
    CKDIR.mkdir(exist_ok=True)
    OUTDIR.mkdir(exist_ok=True)

    log_fh = open(LOG_FILE, "w")
    log_fh.write(f"hep_nsf Test Suite\n")
    log_fh.write(f"Run: {datetime.now()}\n")
    log_fh.write(f"Data: {args.data}\n")
    log_fh.write("="*60 + "\n\n")

    skip_s = set(args.skip_sections)
    t_start = time.time()

    _log(f"\nhep_nsf EXHAUSTIVE TEST SUITE")
    _log(f"Data : {args.data}")
    _log(f"Speed settings: bins={BINS} hdim={HDIM} layers={LAYERS} "
         f"epochs={EPOCHS} bs={BS}")

    try:
        if "1" not in skip_s:  test_utils(args.data)
        if "2" not in skip_s:  test_splines()
        if "3" not in skip_s:  test_mlps()
        if "4" not in skip_s:  test_model_api()
        if "5" not in skip_s:  test_python_api_training(args.data)
        if "6" not in skip_s:  test_cli_training(args.data)
        if "7" not in skip_s:  test_resume(args.data)
        if "8" not in skip_s:  test_yaml_config(args.data)
        if "9" not in skip_s:  test_cli_inference(args.data)
        if "10" not in skip_s: test_analysis(args.data)
        if "11" not in skip_s: test_plotting(args.data)
        if "12" not in skip_s: test_helper_scripts(args.data)
        if "13" not in skip_s: test_devices(args.data)
    except KeyboardInterrupt:
        _log("\nInterrupted by user.")
    finally:
        write_summary()
        elapsed = time.time() - t_start
        _log(f"\nTotal elapsed: {elapsed/60:.1f} min")
        log_fh.close()


if __name__ == "__main__":
    main()
