#!/bin/bash
# Master batch launcher for Paper 3 multi-seed + ablation runs.
#
# Usage:
#   ./scripts/run_all_seeds.sh                       # Sequential on GPU 0
#   ./scripts/run_all_seeds.sh --parallel            # 2 jobs at a time, GPU 0+1
#   ./scripts/run_all_seeds.sh --phase a             # Only main seeds
#   ./scripts/run_all_seeds.sh --phase b             # Only ablation seeds
#   ./scripts/run_all_seeds.sh --max-updates 500     # Cap every run
#   ./scripts/run_all_seeds.sh --parallel --dry-run  # Show plan, launch nothing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Defaults ----
PARALLEL=0
DRY_RUN=0
PHASE="all"
MAX_UPDATES=""

# ---- Arg parsing ----
while [ $# -gt 0 ]; do
    case "$1" in
        --parallel)     PARALLEL=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --phase)        PHASE="$2"; shift 2 ;;
        --max-updates)  MAX_UPDATES="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

case "${PHASE}" in
    a|b|all) ;;
    *) echo "--phase must be a, b, or all" >&2; exit 2 ;;
esac

MAX_UPDATES_FLAG=()
if [ -n "${MAX_UPDATES}" ]; then
    MAX_UPDATES_FLAG=(--max-updates "${MAX_UPDATES}")
fi

# ---- Run definitions ----
RUNS_A=(
    "configs/default.yaml:42"
    "configs/default.yaml:123"
    "configs/default.yaml:456"
    "configs/default.yaml:789"
    "configs/default.yaml:1024"
)

RUNS_B=(
    "configs/ablation_no_rise.yaml:42"
    "configs/ablation_no_rise.yaml:123"
    "configs/ablation_no_rise.yaml:456"
    "configs/ablation_no_attention.yaml:42"
    "configs/ablation_no_attention.yaml:123"
    "configs/ablation_no_attention.yaml:456"
    "configs/ablation_no_cvar_head.yaml:42"
    "configs/ablation_no_cvar_head.yaml:123"
    "configs/ablation_no_cvar_head.yaml:456"
)

case "${PHASE}" in
    a)   ALL_RUNS=("${RUNS_A[@]}") ;;
    b)   ALL_RUNS=("${RUNS_B[@]}") ;;
    all) ALL_RUNS=("${RUNS_A[@]}" "${RUNS_B[@]}") ;;
esac

# ---- Helper: derive run name ----
derive_run_name() {
    local cfg="$1" seed="$2"
    local base
    base="$(basename "${cfg}" .yaml)"
    local prefix
    case "${base}" in
        default)               prefix="full" ;;
        ablation_no_rise)      prefix="no_rise" ;;
        ablation_no_attention) prefix="no_attention" ;;
        ablation_no_cvar_head) prefix="no_cvar" ;;
        *)                     prefix="${base#ablation_}" ;;
    esac
    echo "${prefix}_seed${seed}"
}

# ---- Helper: tmux session 'train' alive? ----
tmux_train_alive() {
    tmux has-session -t train 2>/dev/null
}

# ---- Migrate seed42 results from old location if training is finished ----
maybe_migrate_seed42() {
    local target="results/runs/full_seed42"
    if [ -f "${target}/DONE" ]; then
        return 0
    fi
    if tmux_train_alive; then
        return 0  # still running, do not migrate
    fi
    if [ ! -f "train_seed42.log" ]; then
        return 0  # nothing to migrate
    fi
    echo "↪ Migrating pre-existing seed42 results into ${target}/"
    mkdir -p "${target}/checkpoints"
    cp -n train_seed42.log "${target}/train.log" || true
    cp -n configs/default.yaml "${target}/config_used.yaml" || true
    # Original checkpoints lived in results/checkpoints/
    cp -n results/checkpoints/mappo_upd*.pt "${target}/checkpoints/" 2>/dev/null || true
    echo "$(date -Iseconds)|full_seed42|configs/default.yaml|42|migrated" > "${target}/DONE"
    echo "✓ Migrated full_seed42"
}

# ---- Build list of pending runs (after skip logic) ----
PENDING=()
SKIP_REASONS=()  # parallel-arrays-of-strings, one per skipped run

for run in "${ALL_RUNS[@]}"; do
    cfg="${run%%:*}"
    seed="${run##*:}"
    name="$(derive_run_name "${cfg}" "${seed}")"
    outdir="results/runs/${name}"

    # Special handling for the actively-running seed42 default
    if [ "${cfg}" = "configs/default.yaml" ] && [ "${seed}" = "42" ]; then
        if [ -f "${outdir}/DONE" ]; then
            SKIP_REASONS+=("SKIP ${name}: already DONE")
            continue
        fi
        if tmux_train_alive; then
            SKIP_REASONS+=("SKIP ${name}: tmux session 'train' active")
            continue
        fi
        # tmux gone + no DONE → migrate, then skip
        maybe_migrate_seed42
        SKIP_REASONS+=("SKIP ${name}: migrated from pre-existing run")
        continue
    fi

    if [ -f "${outdir}/DONE" ]; then
        SKIP_REASONS+=("SKIP ${name}: already DONE")
        continue
    fi
    if [ -f "${outdir}/FAILED" ]; then
        SKIP_REASONS+=("SKIP ${name}: FAILED marker present (delete it to retry)")
        continue
    fi
    PENDING+=("${run}")
done

# ---- Dry-run path ----
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY RUN — no jobs will be launched"
    for r in "${SKIP_REASONS[@]}"; do
        echo "  ${r}"
    done
    if [ "${#PENDING[@]}" -eq 0 ]; then
        echo "  (nothing pending)"
        exit 0
    fi
    if [ "${PARALLEL}" -eq 1 ]; then
        i=0
        batch=1
        while [ $i -lt ${#PENDING[@]} ]; do
            r0="${PENDING[$i]}"
            n0="$(derive_run_name "${r0%%:*}" "${r0##*:}")"
            if [ $((i+1)) -lt ${#PENDING[@]} ]; then
                r1="${PENDING[$((i+1))]}"
                n1="$(derive_run_name "${r1%%:*}" "${r1##*:}")"
                echo "  BATCH ${batch}: GPU0 → ${n0} | GPU1 → ${n1}"
                i=$((i+2))
            else
                echo "  BATCH ${batch}: GPU0 → ${n0} (odd, no pair)"
                i=$((i+1))
            fi
            batch=$((batch+1))
        done
        total_batches=$((batch-1))
        echo "  Total: ${#PENDING[@]} runs in ${total_batches} batches (~$((total_batches*80))h wall time)"
    else
        idx=1
        for r in "${PENDING[@]}"; do
            n="$(derive_run_name "${r%%:*}" "${r##*:}")"
            echo "  RUN ${idx}/${#PENDING[@]}: GPU0 → ${n}"
            idx=$((idx+1))
        done
        echo "  Total: ${#PENDING[@]} runs sequential (~$(( ${#PENDING[@]} * 80 ))h wall time)"
    fi
    exit 0
fi

# ---- Print skip reasons before launching ----
for r in "${SKIP_REASONS[@]}"; do
    echo "  ${r}"
done

if [ "${#PENDING[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

# ---- Counters ----
COMPLETED=0
FAILED=0
TOTAL=${#PENDING[@]}

launch_one() {
    # $1=run "config:seed", $2=gpu, $3=banner_prefix
    local run="$1" gpu="$2" prefix="$3"
    local cfg="${run%%:*}" seed="${run##*:}"
    local name
    name="$(derive_run_name "${cfg}" "${seed}")"
    echo "═══════════════════════════════════════════════════════"
    echo " ${prefix} | ${name} | GPU ${gpu}"
    echo " Started: $(date -Iseconds)"
    echo "═══════════════════════════════════════════════════════"
    ./scripts/run_single.sh "${cfg}" "${seed}" --gpu "${gpu}" "${MAX_UPDATES_FLAG[@]}"
}

if [ "${PARALLEL}" -eq 1 ]; then
    i=0
    batch=1
    while [ $i -lt ${TOTAL} ]; do
        r0="${PENDING[$i]}"
        if [ $((i+1)) -lt ${TOTAL} ]; then
            r1="${PENDING[$((i+1))]}"
            launch_one "${r0}" 0 "PARALLEL BATCH ${batch} — JOB A" &
            PID0=$!
            launch_one "${r1}" 1 "PARALLEL BATCH ${batch} — JOB B" &
            PID1=$!
            if wait "${PID0}"; then COMPLETED=$((COMPLETED+1)); else FAILED=$((FAILED+1)); echo "⚠ GPU0 job failed"; fi
            if wait "${PID1}"; then COMPLETED=$((COMPLETED+1)); else FAILED=$((FAILED+1)); echo "⚠ GPU1 job failed"; fi
            i=$((i+2))
        else
            if launch_one "${r0}" 0 "SINGLE BATCH ${batch}"; then
                COMPLETED=$((COMPLETED+1))
            else
                FAILED=$((FAILED+1))
            fi
            i=$((i+1))
        fi
        batch=$((batch+1))
        sleep 30  # GPU memory cleanup between batches
    done
else
    idx=1
    for run in "${PENDING[@]}"; do
        if launch_one "${run}" 0 "RUN ${idx}/${TOTAL}"; then
            COMPLETED=$((COMPLETED+1))
        else
            FAILED=$((FAILED+1))
        fi
        idx=$((idx+1))
        sleep 30
    done
fi

# ---- Final summary ----
echo
echo "All runs finished."
echo "Completed: ${COMPLETED} | Failed: ${FAILED} | Skipped: ${#SKIP_REASONS[@]}"
echo "Run: python scripts/collect_results.py"
