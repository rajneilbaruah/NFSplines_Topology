#!/bin/bash
# =============================================================================
# run_all.sh  —  hep_nsf full pipeline
# HPC/SLURM ready version
#
# Usage (interactive or inside a SLURM script)
# ---------------------------------------------
#   bash run_all.sh                          # all defaults
#   bash run_all.sh --device cuda            # GPU
#   bash run_all.sh --num_splines 2          # deeper model
#   bash run_all.sh --skip_models r3         # skip one model
#   bash run_all.sh --run_name my_exp_01     # named run
#
# SLURM example
# -------------
#   #SBATCH --gres=gpu:1
#   bash run_all.sh --device cuda --run_name ${SLURM_JOB_ID}
# =============================================================================

# ---------------------------------------------------------------------------
# DEFAULTS — edit these for your cluster
# ---------------------------------------------------------------------------

DATA="/home/rajneil/work/datasets/NFSpheres/eemumu_mup.json"

# Use SLURM job ID if available, else timestamp
if [[ -n "${SLURM_JOB_ID}" ]]; then
    DEFAULT_RUN_NAME="job_${SLURM_JOB_ID}"
else
    DEFAULT_RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi
RUN_NAME="$DEFAULT_RUN_NAME"

# Architecture
NUM_BINS=32
NUM_SPLINES=1
HIDDEN_DIM=64
NUM_LAYERS=2
ARCH="mlp"
ACTIVATION="relu"
BOUND=5.0

# Training
LR=0.001
BATCH_SIZE=8192
EPOCHS=10000
PATIENCE=20
SEED=42

# Inference / analysis
NUM_SAMPLES=10000
TARGET_R=500.0
KDE_BW=0.2

# Device: auto / cpu / cuda / mps
DEVICE="auto"

# Models to skip (space-separated), e.g. "r3" or "s2 r3"
SKIP=""

# ---------------------------------------------------------------------------
# HPC ENVIRONMENT — edit for your cluster's module system
# ---------------------------------------------------------------------------

# Uncomment and edit the lines relevant to your cluster:

# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9

# If using conda:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate torch_env

# If using venv:
# source /path/to/.venv/bin/activate

# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)          DATA="$2";        shift 2 ;;
        --run_name)      RUN_NAME="$2";    shift 2 ;;
        --num_bins)      NUM_BINS="$2";    shift 2 ;;
        --num_splines)   NUM_SPLINES="$2"; shift 2 ;;
        --hidden_dim)    HIDDEN_DIM="$2";  shift 2 ;;
        --num_layers)    NUM_LAYERS="$2";  shift 2 ;;
        --arch)          ARCH="$2";        shift 2 ;;
        --activation)    ACTIVATION="$2";  shift 2 ;;
        --bound)         BOUND="$2";       shift 2 ;;
        --lr)            LR="$2";          shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";  shift 2 ;;
        --epochs)        EPOCHS="$2";      shift 2 ;;
        --patience)      PATIENCE="$2";    shift 2 ;;
        --seed)          SEED="$2";        shift 2 ;;
        --num_samples)   NUM_SAMPLES="$2"; shift 2 ;;
        --target_r)      TARGET_R="$2";    shift 2 ;;
        --kde_bw)        KDE_BW="$2";      shift 2 ;;
        --device)        DEVICE="$2";      shift 2 ;;
        --skip_models)   SKIP="$2";        shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Available: --data --run_name --num_bins --num_splines"
            echo "           --hidden_dim --num_layers --arch --activation"
            echo "           --bound --lr --batch_size --epochs --patience"
            echo "           --seed --num_samples --target_r --kde_bw"
            echo "           --device --skip_models"
            exit 1 ;;
    esac
done

# Derived paths — all outputs live under RUN_DIR
RUN_DIR="runs/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
PLOT_DIR="${RUN_DIR}/plots"
ANALYSIS_DIR="${RUN_DIR}/analysis"
LOG_FILE="${RUN_DIR}/run.log"

SKIP_FLAG=""
if [[ -n "$SKIP" ]]; then
    SKIP_FLAG="--skip $SKIP"
fi

# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# ---------------------------------------------------------------------------

echo "Running pre-flight checks..."

# Check data file exists
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: data file not found: $DATA"
    exit 1
fi

# Check python and hep_nsf are importable
python -c "import hep_nsf" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "ERROR: hep_nsf package not importable."
    echo "       Run: pip install -e /path/to/hep_nsf"
    exit 1
fi

# Check GPU if requested
if [[ "$DEVICE" == "cuda" ]]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: --device cuda requested but CUDA not available."
        echo "       Check module loads and GPU allocation."
        exit 1
    fi
    GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "GPU: $GPU_INFO"
fi

# Create output directories
mkdir -p "$CHECKPOINT_DIR" "$PLOT_DIR" "$ANALYSIS_DIR"

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

{
echo "============================================================"
echo "  hep_nsf full pipeline"
echo "  Run name   : $RUN_NAME"
echo "  Data       : $DATA"
echo "  Models     : s2 r2 r3  (skip: ${SKIP:-none})"
echo "  Bins       : $NUM_BINS   Splines: $NUM_SPLINES"
echo "  Hidden dim : $HIDDEN_DIM   Layers: $NUM_LAYERS"
echo "  Arch       : $ARCH   Act: $ACTIVATION"
echo "  LR         : $LR   Batch: $BATCH_SIZE"
echo "  Epochs     : $EPOCHS   Patience: $PATIENCE"
echo "  Device     : $DEVICE   Seed: $SEED"
echo "  Output dir : $RUN_DIR"
if [[ -n "${SLURM_JOB_ID}" ]]; then
echo "  SLURM job  : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
fi
echo "  Started    : $(date)"
echo "============================================================"
echo ""
} | tee "$LOG_FILE"

# ---------------------------------------------------------------------------
# STEP 1: TRAIN
# ---------------------------------------------------------------------------

echo "[$(date +%H:%M:%S)] STEP 1: Training..." | tee -a "$LOG_FILE"

python train_all.py \
    --data          "$DATA" \
    --num_bins      "$NUM_BINS" \
    --num_splines   "$NUM_SPLINES" \
    --hidden_dim    "$HIDDEN_DIM" \
    --num_layers    "$NUM_LAYERS" \
    --arch          "$ARCH" \
    --activation    "$ACTIVATION" \
    --bound         "$BOUND" \
    --lr            "$LR" \
    --batch_size    "$BATCH_SIZE" \
    --epochs        "$EPOCHS" \
    --patience      "$PATIENCE" \
    --seed          "$SEED" \
    --device        "$DEVICE" \
    --save_dir      "$CHECKPOINT_DIR" \
    --output_dir    "$PLOT_DIR" \
    $SKIP_FLAG \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_STATUS=${PIPESTATUS[0]}
if [[ $TRAIN_STATUS -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] ERROR: training failed (exit $TRAIN_STATUS)" | tee -a "$LOG_FILE"
    echo "Check $LOG_FILE for details."
    exit $TRAIN_STATUS
fi

echo "" | tee -a "$LOG_FILE"
echo "[$(date +%H:%M:%S)] Training complete." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# STEP 2: PLOT
# ---------------------------------------------------------------------------

echo "[$(date +%H:%M:%S)] STEP 2: Plotting..." | tee -a "$LOG_FILE"

python plot_all.py \
    --data             "$DATA" \
    --checkpoints_dir  "$CHECKPOINT_DIR" \
    --output_dir       "$PLOT_DIR" \
    --num_samples      "$NUM_SAMPLES" \
    --target_r         "$TARGET_R" \
    --seed             "$SEED" \
    --device           "$DEVICE" \
    $SKIP_FLAG \
    2>&1 | tee -a "$LOG_FILE"

PLOT_STATUS=${PIPESTATUS[0]}
if [[ $PLOT_STATUS -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] WARNING: plotting failed (exit $PLOT_STATUS)" | tee -a "$LOG_FILE"
    echo "Continuing to analysis anyway..."
fi

echo "" | tee -a "$LOG_FILE"
echo "[$(date +%H:%M:%S)] Plotting complete." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# STEP 3: ANALYSE
# ---------------------------------------------------------------------------

echo "[$(date +%H:%M:%S)] STEP 3: Analysis..." | tee -a "$LOG_FILE"

python analyse_all.py \
    --data             "$DATA" \
    --checkpoints_dir  "$CHECKPOINT_DIR" \
    --output_dir       "$ANALYSIS_DIR" \
    --num_samples      "$NUM_SAMPLES" \
    --target_r         "$TARGET_R" \
    --kde_bw           "$KDE_BW" \
    --seed             "$SEED" \
    --device           "$DEVICE" \
    $SKIP_FLAG \
    2>&1 | tee -a "$LOG_FILE"

ANALYSIS_STATUS=${PIPESTATUS[0]}
if [[ $ANALYSIS_STATUS -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] WARNING: analysis failed (exit $ANALYSIS_STATUS)" | tee -a "$LOG_FILE"
fi

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

{
echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Finished   : $(date)"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Plots      : $PLOT_DIR"
echo "  Analysis   : $ANALYSIS_DIR"
echo "  Log        : $LOG_FILE"
echo "============================================================"
} | tee -a "$LOG_FILE"
