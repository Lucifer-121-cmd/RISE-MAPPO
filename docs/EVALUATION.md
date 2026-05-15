# Evaluation Guide

> **Important:** For production (paper-quality) evaluations, do **not** pass
> `--num-episodes` on the CLI.  The default of 50 episodes (from
> `configs/eval_default.yaml`) provides statistically meaningful results.
> The 3–5 episode runs in `results/eval/` were smoke tests only.


## Running Evaluations

### Trained policy
```bash
# Production evaluation (50 episodes, Phase-2 features enabled via eval config)
python scripts/evaluate.py \
    --policy trained \
    --checkpoint results/runs/full_seed42/checkpoints/mappo_upd1000.pt \
    --config configs/default.yaml \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml
# Do NOT pass --num-episodes for production runs; the default of 50 is correct.
```

### Baseline
```bash
python scripts/evaluate.py \
    --policy nearest_frontier \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml
```

### Ablation
```bash
python scripts/evaluate.py \
    --policy trained \
    --checkpoint results/runs/no_rise_seed42/checkpoints/mappo_upd1000.pt \
    --config configs/ablation_no_rise.yaml \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml \
    --name "Ours-no-CVaR"
```

## Scenarios

| Scenario | Config | Robots | Targets | Key Feature |
|----------|--------|--------|---------|-------------|
| Simple | `scenario_simple.yaml` | 3 | 5 | Open environment, minimal obstacles |
| Complex | `scenario_complex.yaml` | 5 | 10 | Corridors, hazard zones (main eval) |
| Scalability | `scenario_scalability.yaml` | 3-8 | 10 | Vary robot count |
| Energy | `scenario_energy.yaml` | 5 | 10 | Half energy budget |
| Comms Failure | `scenario_comms.yaml` | 5 | 10 | 50% GP fusion drop rate |

## Metrics

All metrics computed in `analysis/metrics.py`:

| Metric | Function | Description |
|--------|----------|-------------|
| Coverage Rate | `coverage_rate()` | Fraction of explorable area covered |
| Detection Success | `detection_success_rate()` | Fraction of targets found |
| Time to Detection | `time_to_full_detection()` | Steps until all targets found |
| Collision Rate | `collision_rate()` | Total collisions per episode |
| Energy Efficiency | `energy_efficiency()` | Coverage / total energy |
| Mean CVaR Risk | `mean_cvar_risk()` | Average tail risk encountered |
| Exploration Overlap | `exploration_overlap()` | Cells visited by >1 robot |
| Lyapunov Stability | `lyapunov_stability()` | Monotonic decrease fraction |

## Output Structure

```
results/eval/{scenario}/{policy_name}/
├── metrics_summary.json       # mean ± std for all metrics
├── metrics_per_episode.csv    # one row per episode
├── coverage_curves.npy        # (N_episodes, T) coverage over time
├── episode_data/              # full episode data for plotting
│   ├── ep_000.npz
│   └── ...
└── eval_config.yaml           # config used (reproducibility)
```

## Baselines

| Baseline | Class | Description |
|----------|-------|-------------|
| Random | `RandomPolicy` | Uniform random subgoal selection |
| Nearest Frontier | `NearestFrontierPolicy` | Greedy move to closest frontier cell |
| Voronoi Partition | `VoronoiPartitionPolicy` | Explore within Voronoi cell |

All baselines implement `BasePolicy` interface: `reset()`, `get_actions()`, `name`.

## Plotting

Publication-quality figures via `analysis/plotting.py`:

```python
from analysis.plotting import setup_ieee_style, plot_metric_comparison_bar

setup_ieee_style()  # call once

# Training curves
plot_training_curves("results/runs/full_seed42/train.log", "figures/")

# Metric comparison (all methods)
plot_metric_comparison_bar(results_dict, metrics, "figures/comparison.pdf")

# Coverage over time
plot_coverage_curves(coverage_dict, "figures/coverage.pdf")

# Ablation bar chart
plot_ablation_bar(ablation_dict, "figures/ablation.pdf")

# Scalability
plot_scalability(scalability_dict, "figures/scalability.pdf")
```

All plots follow IEEE two-column format: 3.5" (single) or 7.16" (double) width, 9pt serif fonts, 300 DPI, colorblind-safe palette.

## Statistical Testing

For paper results, use Wilcoxon signed-rank test (non-parametric, paired) with Cliff's delta effect size across 5 seeds. Significance threshold: p < 0.05.
