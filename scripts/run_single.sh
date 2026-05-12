#!/bin/bash
# Launch a single isolated RISE-MAPPO training run.
#
# Usage:
#   ./scripts/run_single.sh <config> <seed> [--gpu 0|1|cpu] [--max-updates N]
#
# Examples:
#   ./scripts/run_single.sh configs/default.yaml 123 --gpu 0
#   ./scripts/run_single.sh configs/ablation_no_rise.yaml 42 --gpu 1 --max-updates 500
#   ./scripts/run_single.sh configs/default.yaml 999 --gpu cpu --max-updates 2

set -euo pipefail

# ---- Resolve project root (script lives in <root>/scripts/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Positional args ----
if [ $# -lt 2 ]; then
    echo "Usage: $0 <config> <seed> [--gpu 0|1|cpu] [--max-updates N]" >&2
    exit 2
fi

CONFIG="$1"
SEED="$2"
shift 2

GPU="0"
MAX_UPDATES=""

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --max-updates)
            MAX_UPDATES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

# ---- Validation ----
if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "Seed must be integer, got: ${SEED}" >&2
    exit 2
fi
case "${GPU}" in
    0|1|cpu) ;;
    *)
        echo "--gpu must be 0, 1, or cpu, got: ${GPU}" >&2
        exit 2
        ;;
esac
if [ -n "${MAX_UPDATES}" ] && ! [[ "${MAX_UPDATES}" =~ ^[0-9]+$ ]]; then
    echo "--max-updates must be integer, got: ${MAX_UPDATES}" >&2
    exit 2
fi

# ---- Derive run name from config filename ----
# configs/default.yaml              -> full_seed${SEED}
# configs/ablation_no_rise.yaml     -> no_rise_seed${SEED}
# configs/ablation_no_attention.yaml-> no_attention_seed${SEED}
# configs/ablation_no_cvar_head.yaml-> no_cvar_seed${SEED}
CONFIG_BASE="$(basename "${CONFIG}" .yaml)"
case "${CONFIG_BASE}" in
    default)               RUN_PREFIX="full" ;;
    ablation_no_rise)      RUN_PREFIX="no_rise" ;;
    ablation_no_attention) RUN_PREFIX="no_attention" ;;
    ablation_no_cvar_head) RUN_PREFIX="no_cvar" ;;
    *)
        # Fallback: strip ablation_ prefix, use as-is
        RUN_PREFIX="${CONFIG_BASE#ablation_}"
        ;;
esac

RUN_NAME="${RUN_PREFIX}_seed${SEED}"
OUTDIR="results/runs/${RUN_NAME}"

# ---- Refuse to clobber a completed run ----
if [ -f "${OUTDIR}/DONE" ]; then
    echo "✓ ${RUN_NAME} already DONE at ${OUTDIR}/DONE — skipping" >&2
    exit 0
fi

# ---- Create output dirs ----
mkdir -p "${OUTDIR}/checkpoints"

# ---- Snapshot the original config for provenance ----
cp "${CONFIG}" "${OUTDIR}/config_used.yaml"

# ---- Build run-specific config (override save_dir, n_training_updates) ----
RUN_CONFIG="${OUTDIR}/run_config.yaml"
OVERRIDE_UPDATES="${MAX_UPDATES}" \
OVERRIDE_SAVE_DIR="${OUTDIR}/checkpoints" \
SRC_CONFIG="${CONFIG}" \
DEST_CONFIG="${RUN_CONFIG}" \
python3 - <<'PY'
import os
import yaml

src = os.environ["SRC_CONFIG"]
dest = os.environ["DEST_CONFIG"]
save_dir = os.environ["OVERRIDE_SAVE_DIR"]
updates = os.environ.get("OVERRIDE_UPDATES", "")

with open(src) as f:
    cfg = yaml.safe_load(f) or {}

training = dict(cfg.get("training", {}) or {})
training["save_dir"] = save_dir
if updates:
    training["n_training_updates"] = int(updates)
cfg["training"] = training

with open(dest, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

# ---- GPU / device selection ----
if [ "${GPU}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    DEVICE_ARG="cpu"
else
    export CUDA_VISIBLE_DEVICES="${GPU}"
    DEVICE_ARG="cuda"
fi

# ---- Activate venv (if present) ----
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ---- Banner ----
echo "═══════════════════════════════════════════════════════"
echo " RUN:    ${RUN_NAME}"
echo " CONFIG: ${CONFIG}"
echo " SEED:   ${SEED}"
echo " GPU:    ${GPU}  (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}')"
echo " DEVICE: ${DEVICE_ARG}"
echo " OUTDIR: ${OUTDIR}"
[ -n "${MAX_UPDATES}" ] && echo " UPDATES: ${MAX_UPDATES}"
echo " STARTED: $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════"

# ---- Launch training ----
# Disable pipefail+errexit around the pipe so a nonzero training exit
# still lets us write the FAILED marker. PIPESTATUS[0] preserves the
# python exit code through `tee`.
set +e
set +o pipefail
python scripts/train.py \
    --config "${RUN_CONFIG}" \
    --seed "${SEED}" \
    --device "${DEVICE_ARG}" 2>&1 | tee "${OUTDIR}/train.log"
EXIT_CODE="${PIPESTATUS[0]}"
set -e
set -o pipefail

# ---- Completion marker ----
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "$(date -Iseconds)|${RUN_NAME}|${CONFIG}|${SEED}" > "${OUTDIR}/DONE"
    echo "✓ ${RUN_NAME} finished cleanly → ${OUTDIR}/DONE"
else
    echo "$(date -Iseconds)|${RUN_NAME}|exit_code=${EXIT_CODE}" > "${OUTDIR}/FAILED"
    echo "✗ ${RUN_NAME} failed (exit ${EXIT_CODE}) → ${OUTDIR}/FAILED" >&2
fi

exit "${EXIT_CODE}"
