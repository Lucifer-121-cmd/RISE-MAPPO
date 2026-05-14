# Training Guide

## Single Run

```bash
python scripts/train.py --config configs/default.yaml --seed 42 --device cuda
```

**Flags:**
- `--config` — YAML config file path
- `--seed` — random seed (default: 42)
- `--device` — `cuda`, `cpu`, or `cuda:0` / `cuda:1` for specific GPU
- `--updates` — override number of training updates
- `--smoke` — quick test (2 updates, 1 env, minimal rollout)

**Output:** Training logs to stdout (redirect with `tee`). Checkpoints saved every 50 updates to `training.save_dir` (default: `results/checkpoints/`).

## Multi-Seed Batch Training

### Run plan
- 5 seeds (42, 123, 456, 789, 1024) for full RISE-MAPPO
- 3 seeds (42, 123, 456) × 3 ablation configs = 9 ablation runs
- Total: 14 runs

### Launch

```bash
# Preview what will run (no jobs launched)
./scripts/run_all_seeds.sh --parallel --dry-run

# Run all on 2 GPUs (inside tmux)
tmux new -s batch
./scripts/run_all_seeds.sh --parallel 2>&1 | tee batch_run.log

# Run only main seeds
./scripts/run_all_seeds.sh --parallel --phase a

# Run only ablation
./scripts/run_all_seeds.sh --parallel --phase b

# Limit to 500 updates (if convergence is clear)
./scripts/run_all_seeds.sh --parallel --max-updates 500
```

### Monitor

```bash
# One-shot status
python scripts/check_status.py

# Auto-refresh every 5 minutes
python scripts/check_status.py --watch 300
```

### Collect results

```bash
python scripts/collect_results.py
# Produces:
#   results/all_training_curves.csv
#   results/final_metrics_summary.csv
```

## Output Directory Structure

Each run is isolated under `results/runs/`:

```
results/runs/
├── full_seed42/
│   ├── run_config.yaml       # Config used (with overrides)
│   ├── config_used.yaml      # Original config copy
│   ├── train.log             # Full training log
│   ├── checkpoints/          # Isolated checkpoints
│   │   ├── mappo_upd50.pt
│   │   ├── mappo_upd100.pt
│   │   └── ...
│   └── DONE                  # Completion marker
├── no_rise_seed42/
└── ...
```

## Log Format

```
2026-05-14 11:18:09 paper3.runner INFO | update 823  R=459.562  cov=0.771  det=2.00  pl=-0.031  vl=0.092  H=1.662  kl=0.0282  264.9s
```

| Field | Meaning | Healthy Range |
|-------|---------|---------------|
| R | Episode reward (team total) | 300-500+ |
| cov | Coverage fraction | 0.6-0.85 |
| det | Targets detected per episode | 1.5-2.5 |
| pl | Policy loss | -0.05 to -0.02 |
| vl | Value loss | Decreasing over time |
| H | Policy entropy | 1.5-2.5 (not collapsing) |
| kl | KL divergence from old policy | 0.01-0.04 |

## Convergence

Training typically converges by update 300-500. Signs of convergence:
- Reward plateaus (variance remains due to stochastic rollouts)
- Value loss drops below 0.1
- Entropy stabilizes around 1.5-2.0

Evaluation with deterministic actions usually shows higher coverage than training logs (which use stochastic exploration).

## Known Issues and Fixes

1. **Risk cost scale mismatch**: Raw CVaR can be 75× reward scale → always normalize to [0,1]
2. **Gradient explosion with dual-head**: `max_grad_norm` must be 0.5 (not default 10)
3. **NaN in critic outputs**: `nan_to_num` guards + skip-on-nonfinite-grad
4. **Dead critic (vl=0.000)**: PopArt/nan guard interaction → separate PopArt per head
5. **GPU memory**: Each run needs ~4-5 GB. Two runs fit on 2× RTX 2080 Ti (11GB each)
