# HEP Neural Spline Flows (`hep_nsf`)

A production-ready Python package for learning probability densities of
high-energy physics (HEP) 3-momentum data using **Rational-Quadratic Spline
(RQS) normalising flows**. Supports five flow architectures including the
recursive S² flow from Rezende et al. (2020).

---

## Model Naming Convention

| Key | Class | Input space | Base distribution | Notes |
|---|---|---|---|---|
| `s2` | `RecursiveSphereFlow` | Physical `(cos θ, φ)` | Uniform on S² | Free-param cos θ, MLP circular φ. No normalisation. Original notebook architecture. |
| `s2norm` | `NormSphereFlow` | Normalised `(cos θ, φ)` | Gaussian R² | Free-param cos θ, MLP circular φ in normalised space. Better gradient flow. |
| `s2rec` | `RecursiveS2Flow` | Physical `(cos θ, φ)` | Uniform on S² | Rezende et al. 2020 Appendix J. Alternating A/B coupling blocks. `num_splines=2, num_bins=80` = paper exact. |
| `r2` | `AngularSphereFlow` | Normalised `(cos θ, φ)` | Gaussian R² | Standard RQS both dims. MLP-conditioned. |
| `r3` | `CartesianNSF` | Normalised `(px, py, pz)` | Gaussian R³ | Full 3-momentum flow in Cartesian space. |

**Physical models** (`s2`, `s2rec`): no data normalisation, uniform S² base.
Sampling returns `(cos θ, φ)` directly.

**Normalised models** (`s2norm`, `r2`): data normalised to zero mean/unit
variance, Gaussian base. Sampling returns normalised coordinates — call
`denormalise()` before use.

**`s2rec` coupling blocks:**
- A-type (even k): cos θ → interval RQS (free params), φ → circular RQS (MLP on transformed cos θ)
- B-type (odd k): φ → circular RQS (free params), cos θ → interval RQS (MLP on transformed φ)
- `num_splines=2` replicates paper Appendix J: 1 A-block + 1 B-block

---

## Installation

### Option 1 — pip install (recommended)

```bash
git clone https://github.com/<your-username>/hep_nsf.git
cd hep_nsf
pip install -e .
```

After this, `import hep_nsf` works from any directory and the CLI
`python -m hep_nsf.main` is available everywhere.

For a clean environment first:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
pip install -e .
```

### Option 2 — without installation (run directly from the repo)

If you do not want to install, run scripts from the `NF_Sphere/` root
directory (the folder that contains `hep_nsf/` as a subdirectory):

```bash
cd ~/work/NF_Sphere

# All python commands run from here
python -m hep_nsf.main --mode train --model s2rec \
    --data ../datasets/NFSpheres/eemumu_mup.json \
    --num_bins 80 --num_splines 2

python train_all.py  --data ../datasets/NFSpheres/eemumu_mup.json
python plot_all.py   --data ../datasets/NFSpheres/eemumu_mup.json
python analyse_all.py --data ../datasets/NFSpheres/eemumu_mup.json
```

Python finds the package via the implicit namespace — as long as
`hep_nsf/` is a subdirectory of your working directory this works
without any `sys.path` manipulation.

### Dependencies

```
torch >= 2.0
numpy
scipy
matplotlib
pyyaml
```

Install manually if not using `pip install -e .`:

```bash
pip install torch numpy scipy matplotlib pyyaml
```

---

## Package Structure

```
hep_nsf/
├── __init__.py          Public API re-exports
├── splines.py           RQS bijections: rqs, rqs_with_bounds, rqs_circular
├── mlps.py              Conditioners: MLP, ResNet, build_conditioner
├── networks.py          Flow models: all five classes + build_model registry
├── utils.py             Data I/O, coordinate transforms, normalisation,
│                        DataLoader factory, checkpoint helpers
├── train.py             Training loop (model.log_prob, early stopping, schedulers)
├── analysis.py          KL, ESS, Wasserstein, JSD, consistency report
├── plotting.py          All matplotlib visualisation functions
├── main.py              CLI entry point (train / sample / evaluate / plot)
├── configs/
│   ├── s2_default.yaml
│   ├── s2norm_default.yaml
│   ├── s2rec_default.yaml
│   ├── r2_default.yaml
│   └── r3_default.yaml
├── requirements.txt
└── setup.py

# Scripts in NF_Sphere/ (parent directory)
train_all.py      Train all five models on the same dataset
plot_all.py       Generate all plots for all models
analyse_all.py    Full metric suite comparison across models
run_tests.py      Exhaustive test suite (256+ tests)
test_plots.py     Targeted plotting tests
run_all.sh        Full pipeline bash script
```

---

## Quick Start

### Train a single model

```bash
# s2rec — paper's recursive S2 flow (Rezende 2020 Appendix J)
python -m hep_nsf.main --mode train --model s2rec \
    --data eemumu_mup.json \
    --num_bins 80 --num_splines 2 \
    --run_name mup_s2rec

# s2norm — normalised angular with circular phi
python -m hep_nsf.main --mode train --model s2norm \
    --data eemumu_mup.json \
    --num_bins 32 --num_splines 2 --bound 5.0 \
    --run_name mup_s2norm

# s2 — original notebook architecture (physical coords)
python -m hep_nsf.main --mode train --model s2 \
    --data eemumu_mup.json \
    --num_bins 32 --num_splines 1 \
    --batch_size 128 --patience 10 \
    --run_name mup_s2

# r2 — normalised angular, Gaussian base
python -m hep_nsf.main --mode train --model r2 \
    --data eemumu_mup.json \
    --num_bins 32 --num_splines 2 --bound 5.0 \
    --run_name mup_r2

# r3 — Cartesian 3-momentum flow
python -m hep_nsf.main --mode train --model r3 \
    --data eemumu_mup.json \
    --num_bins 32 --num_splines 2 --bound 5.0 \
    --run_name mup_r3
```

### Sample, evaluate, plot (no architecture flags needed — self-describing checkpoint)

```bash
python -m hep_nsf.main --mode sample \
    --data eemumu_mup.json \
    --checkpoint checkpoints/mup_s2rec_best.pt \
    --num_samples 50000

python -m hep_nsf.main --mode evaluate \
    --data eemumu_mup.json \
    --checkpoint checkpoints/mup_s2rec_best.pt

python -m hep_nsf.main --mode plot \
    --data eemumu_mup.json \
    --checkpoint checkpoints/mup_s2rec_best.pt \
    --run_name mup_s2rec
```

### Train all five models at once

```bash
python train_all.py --data eemumu_mup.json --seed 42
```

Skip specific models:

```bash
python train_all.py --data eemumu_mup.json --skip s2 r3
```

### Full pipeline script

```bash
bash run_all.sh --data eemumu_mup.json --device cuda --seed 42
```

All outputs go into a timestamped run directory under `runs/`.

---

## Multi-Spline Stacking

All five models accept `--num_splines N`.

| Model | What one spline block does | Recommended range |
|---|---|---|
| `s2` | cos θ free spline + MLP circular φ conditioned on cos θ | 1–3 |
| `s2norm` | same but in normalised space | 1–4 |
| `s2rec` | one alternating A+B block pair (A: cos θ first; B: φ first) | 2–6 |
| `r2` | cos θ ↔ φ autoregressive pair | 1–4 |
| `r3` | px ↔ (py,pz) autoregressive pair | 1–4 |

For `s2rec`: `num_splines=2` = 1 A-block + 1 B-block = paper's Appendix J.
Odd `num_splines` end with an extra A-block.

---

## CLI Reference

```
python -m hep_nsf.main [--config CONFIG]
    --mode   {train, sample, evaluate, plot}
    --data   DATA_PATH

Model:
    --model         {s2, s2norm, s2rec, r2, r3}   default: s2
    --num_bins      int    default: 32   (Ks in paper)
    --num_splines   int    default: 1
    --bound         float  default: 5.0  (s2norm, r2, r3 only — ignored for s2/s2rec)
    --hidden_dim    int    default: 64
    --num_layers    int    default: 2
    --arch          {mlp, resnet}        default: mlp
    --activation    {relu,tanh,elu,leaky_relu,silu,gelu}
    --dropout       float  default: 0.0

Training:
    --lr            float  default: 1e-3
    --weight_decay  float  default: 0.0
    --batch_size    int    default: 8192
    --epochs        int    default: 10000
    --patience      int    default: 20
    --ema_alpha     float  default: 0.3
    --clip_grad     float  default: 5.0
    --use_plateau / --no_plateau
    --plateau_factor    float  default: 0.5
    --plateau_patience  int    default: 10
    --use_cosine
    --cosine_t_max  int    default: 500
    --log_every     int    default: 10
    --resume_from   PATH

I/O:
    --run_name      str    default: model
    --save_dir      str    default: checkpoints
    --output_dir    str    default: outputs
    --checkpoint    PATH   (required for sample/evaluate/plot)

Sampling:
    --num_samples   int    default: 10000
    --target_r      float  default: 500.0 (GeV)

Device:
    --device        {auto, cpu, cuda, mps}
    --no_plots
```

---

## Config Files (YAML)

Any CLI argument can be set in a YAML file and passed via `--config`.
CLI flags override YAML values.

```bash
# Use shipped config
python -m hep_nsf.main \
    --config configs/s2rec_default.yaml \
    --data eemumu_mup.json

# Override specific values on the command line
python -m hep_nsf.main \
    --config configs/s2rec_default.yaml \
    --data eemumu_mup.json \
    --num_bins 32 \
    --run_name s2rec_small
```

Shipped configs:

| Config | Model | Key settings |
|---|---|---|
| `s2_default.yaml` | `s2` | bins=32, ns=1, no bound |
| `s2norm_default.yaml` | `s2norm` | bins=32, ns=1, bound=5.0 |
| `s2rec_default.yaml` | `s2rec` | bins=80, ns=2, no bound — paper exact |
| `r2_default.yaml` | `r2` | bins=32, ns=1, bound=5.0 |
| `r3_default.yaml` | `r3` | bins=32, ns=1, bound=5.0 |

---

## Cluster Usage (SLURM) (EXPERIMENTAL)

```bash
#!/bin/bash
#SBATCH --job-name=hep_nsf
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

source /path/to/.venv/bin/activate
cd /path/to/NF_Sphere

python -m hep_nsf.main \
    --config configs/s2rec_default.yaml \
    --data /scratch/$USER/eemumu_mup.json \
    --num_splines $NUM_SPLINES \
    --run_name s2rec_ns${NUM_SPLINES}_${SLURM_JOB_ID} \
    --save_dir /scratch/$USER/checkpoints \
    --output_dir /scratch/$USER/outputs \
    --device cuda \
    --no_plots
```

Sweep:
```bash
for K in 2 4 6; do
    sbatch --export=ALL,NUM_SPLINES=$K submit.sh
done
```

---

## Python API

```python
import torch
from hep_nsf import (
    build_model, load_json_data, cartesian_to_spherical,
    normalise, denormalise, make_dataloaders,
    train_model, get_device, set_seed
)

set_seed(42)
device = get_device("auto")
raw = load_json_data("eemumu_mup.json")   # (N, 3) Cartesian GeV

# ── s2rec: paper's recursive flow, physical coords ──────────────────────── #
cyl = cartesian_to_spherical(raw, phi_range="0_2pi")  # (N, 2) physical

import torch as t
mean_phys = t.zeros(1, 2)
std_phys  = t.ones(1, 2)   # no normalisation — std_correction = 0
train_loader, val_loader = make_dataloaders(cyl, batch_size=8192)

model = build_model("s2rec", num_bins=80, num_splines=2,
                    hidden_dim=64, num_layers=2)

model, losses = train_model(
    model, train_loader, val_loader,
    std_tensor=std_phys, device=device,
    model_type="s2rec", num_layers=2, arch="mlp", activation="relu",
    run_name="mup_s2rec")

# Sampling: returns physical (cos_theta, phi) directly
samples = model.sample(50_000, device=device)  # (50000, 2)

# ── s2norm: normalised angular, circular phi ─────────────────────────────── #
cyl_norm, mean2, std2 = normalise(cyl)
train_loader2, val_loader2 = make_dataloaders(cyl_norm, batch_size=8192)

model2 = build_model("s2norm", num_bins=32, num_splines=2, bound=5.0,
                     hidden_dim=64, num_layers=2)

model2, _ = train_model(
    model2, train_loader2, val_loader2,
    std_tensor=std2, device=device,
    model_type="s2norm", num_layers=2, arch="mlp", activation="relu",
    run_name="mup_s2norm")

# Sampling: returns normalised coords — denormalise before use
z_norm = model2.sample(50_000, device=device)
samples2 = denormalise(z_norm, mean2, std2)  # physical (cos_theta, phi)
```

---

## Module Reference

### `networks.py`

| Class | Key | Base | Input |
|---|---|---|---|
| `RecursiveSphereFlow` | `s2` | Uniform S² | Physical `(cos θ, φ)` |
| `NormSphereFlow` | `s2norm` | Gaussian R² | Normalised `(cos θ, φ)` |
| `RecursiveS2Flow` | `s2rec` | Uniform S² | Physical `(cos θ, φ)` |
| `AngularSphereFlow` | `r2` | Gaussian R² | Normalised `(cos θ, φ)` |
| `CartesianNSF` | `r3` | Gaussian R³ | Normalised `(px, py, pz)` |

All models implement:
- `forward(x, inverse=False)` → `(z, ldj)` or `x_reconstructed`
- `log_prob(x)` → log-probability under the model's own base distribution
- `sample(n, device)` → n new samples

`build_model(key, **kwargs)` instantiates any model by key.

**Note on `bound`:**
`s2` and `s2rec` do not accept a `bound` argument — they use physical
coordinate ranges directly. `s2norm`, `r2`, `r3` all require `bound`
(half-domain of the spline in normalised space, default 5.0).

### `train.py`

`train_model(model, train_loader, val_loader, std_tensor, device, ...)`

Uses `model.log_prob(x)` directly so each model's own base distribution
is used automatically. For physical models (`s2`, `s2rec`), pass
`std_tensor = torch.ones(1, 2)` so the normalisation correction is zero.

Saves checkpoint with full architecture metadata:
`model_type`, `num_bins`, `num_splines`, `bound`, `hidden_dim`,
`num_layers`, `arch`, `activation`.

### `utils.py`

| Function | Description |
|---|---|
| `set_seed(seed)` | Fix all random seeds |
| `get_device(prefer)` | Best available device |
| `cartesian_to_spherical(p, phi_range)` | `(px,py,pz)` → `(cos θ, φ)` |
| `spherical_to_cartesian(sph, r)` | `(cos θ, φ)` → `(px,py,pz)` |
| `cartesian_to_physics(p)` | Dict: px, py, pz, pT, η, φ, \|p\| |
| `load_json_data(path)` | JSON → float32 Tensor |
| `normalise(data)` | Standardise; returns `(norm, mean, std)` |
| `denormalise(norm, mean, std)` | Reverse standardisation |
| `make_dataloaders(data, ...)` | Train/val split + DataLoader |
| `save_checkpoint / load_checkpoint` | Full checkpoint I/O |
| `save_losses / load_losses` | JSON loss curve I/O |

### `analysis.py`

| Function | Description |
|---|---|
| `kl_divergence_kde(...)` | KDE-based KL divergence on S² |
| `effective_sample_size(...)` | Importance-weighted ESS |
| `wasserstein_1d(data, samples)` | Per-feature W₁ |
| `js_divergence_1d(data, samples)` | Per-feature JSD |
| `consistency_report(cyl_samples)` | Unphysical fraction + radius stats |
| `evaluate(...)` | Full metric suite |
| `model_summary(model)` | Parameter count + config |

### `plotting.py`

| Function | Models |
|---|---|
| `plot_loss_curves` | all |
| `plot_marginal_1d/2d` | all |
| `plot_mollweide_kde` | angular models |
| `plot_physics_comparison` | all |
| `plot_base_mapping_s2` | `s2`, `s2rec` |
| `plot_base_mapping` | `s2norm`, `r2` |
| `plot_jacobian_map` | all angular |
| `plot_jacobian_map_r3` | `r3` |
| `plot_radius_distribution` | `r3` |

---

## Evaluation Metrics

| Metric | Interpretation |
|---|---|
| **KL(data ‖ flow)** | 0 = perfect. |
| **ESS%** | 100% = perfect density match. |
| **W₁** | Earth-mover distance per feature. Smaller is better. |
| **JSD** | 0 = identical. ln2 ≈ 0.693 = maximally different. |
| **Unphysical %** | Fraction of angular samples where cos θ ∉ [−1,1] or φ ∉ [0,2π]. |

**Note on training loss comparison across models:**
`s2`/`s2rec` use uniform S² base (floor = +log(4π) ≈ +2.531).
`s2norm`/`r2` use Gaussian base (floor ≈ +0.92 for D=2).
These are incomparable numbers. Use KL, ESS, W₁ from `analyse_all.py`
to compare models on equal footing.

A **negative** NLL (e.g. `s2rec` val loss = −7.5) is physically correct
— it means the flow has learned to concentrate probability mass well
beyond the flat base distribution.

---

## Architecture Details

### Spline bijection

The Rational-Quadratic Spline (Durkan et al. 2019) maps `x ∈ [left, right]`
to `y ∈ [bottom, top]` via K piecewise rational-quadratic segments.
Parameters: K widths + K heights + (K+1) derivatives = 3K+1 scalars.
Forward and inverse are both analytic.

### Circular spline for φ (`s2`, `s2norm`, `s2rec`)

φ is periodic — φ=0 and φ=2π are the same point. The circular constraint
enforces `d[0] = d[K]` (equal derivatives at both ends), guaranteeing
the density is smooth across the wrap-around point. Implemented as:
```python
d_circular = torch.cat([d, d[:, 0:1]], dim=-1)  # (B, K) → (B, K+1)
```

### NLL loss

```
L = −E_x[log p_θ(x)]
  = −E_x[log p_z(f(x)) + log|det J_f(x)|] + Σ log σ_i
```

`Σ log σ_i` is zero for physical models (σ=1). Each model's `log_prob()`
uses its own correct base distribution, so `train_model()` just calls
`model.log_prob(x)` directly.

---

## Test Suite

```bash
python run_tests.py --data eemumu_mup.json
```

256+ tests covering all five models × all architectures × all argument
combinations × checkpoint save/load/resume × CLI modes × YAML configs.

Skip sections for faster runs:
```bash
python run_tests.py --data eemumu_mup.json --skip_sections 6 12
```

---

## Citation

```bibtex
@article{durkan2019neural,
  title   = {Neural Spline Flows},
  author  = {Durkan, Conor and Bekasov, Artur and Murray, Iain and Papamakarios, George},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2019}
}

@inproceedings{rezende2020normalizing,
  title     = {Normalizing Flows on Tori and Spheres},
  author    = {Rezende, Danilo J. and Papamakarios, George and Raca{\~n}i{\`e}re, S{\'e}bastien
               and Albergo, Michael S. and Kanwar, Gurtej and Shanahan, Phiala E. and Cranmer, Kyle},
  booktitle = {International Conference on Machine Learning},
  year      = {2020}
}
```

---

## License

MIT License.
