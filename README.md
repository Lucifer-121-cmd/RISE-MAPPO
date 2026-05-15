# RISE-MAPPO

**Risk-Sensitive Multi-Agent Policy Optimization for Multi-Robot Exploration**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 78/78](https://img.shields.io/badge/Tests-78%2F78-brightgreen.svg)](tests/)

A hierarchical framework coupling risk-sensitive multi-agent reinforcement learning (RISE-MAPPO) with Lyapunov-stable model predictive control and distributed Gaussian process perception for cooperative robotic search in unknown, hazardous environments.

<p align="center">
  <img src="paper/fig/architecture.jpg" alt="RISE-MAPPO Architecture" width="100%">
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage Guide](#usage-guide)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Running Baselines](#running-baselines)
   - [Parallel Execution](#parallel-execution)
7. [Configuration Guide](#configuration-guide)
   - [Environment (`env`)](#environment-env)
   - [Reward Weights (`reward`)](#reward-weights-reward)
   - [MAPPO / RISE (`mappo`)](#mappo--rise-mappo)
   - [Lyapunov-MPC (`mpc`)](#lyapunov-mpc-mpc)
   - [Gaussian Process (`gp`)](#gaussian-process-gp)
   - [Training Runner (`training`)](#training-runner-training)
   - [Evaluation (`eval`)](#evaluation-eval)
   - [Scenario Configs](#scenario-configs)
   - [Ablation Configs](#ablation-configs)
8. [Key Hyperparameters](#key-hyperparameters)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Baselines](#baselines)
11. [Robot Platform](#robot-platform)
12. [Recent Implementations & Architecture Changes](#recent-implementations--architecture-changes)
    - [Energy Model Change](#energy-model-change)
    - [CVaR Dual-Head Critic](#cvar-dual-head-critic)
    - [GP-Uncertainty Attention](#gp-uncertainty-attention)
    - [Phase-1 vs Phase-2 Training](#phase-1-vs-phase-2-training)
    - [MPC Config Propagation Fix](#mpc-config-propagation-fix)
    - [Exploration Overlap Fix](#exploration-overlap-fix)
    - [Lyapunov Reference Fix](#lyapunov-reference-fix)
    - [Coverage Curve NaN Padding](#coverage-curve-nan-padding)
13. [Logging and Results](#logging-and-results)
    - [Training Logs](#training-logs)
    - [Evaluation Logs](#evaluation-logs)
    - [Output Directory Structure](#output-directory-structure)
    - [Key Results from Recent Runs](#key-results-from-recent-runs)
14. [Directory Structure](#directory-structure)
15. [Limitations and Known Issues](#limitations-and-known-issues)
16. [Future Work](#future-work)
17. [Development Status](#development-status)
18. [Citation](#citation)
19. [License](#license)

---

## Overview

RISE-MAPPO addresses the problem of deploying teams of mobile robots to cooperatively search unknown or hazardous environments. The framework introduces three key innovations:

1. **Dual-Head CVaR Critic** — A centralized critic that simultaneously estimates expected returns (*V*_mean) and conditional value-at-risk (*V*_CVaR), producing a risk-adjusted advantage *A = A*_mean *− λ · A*_risk that makes the joint policy inherently risk-averse at the optimization level rather than through indirect reward shaping.

2. **GP-Uncertainty-Weighted Attention** — An attention mechanism in the centralized critic that modulates agent-level feature aggregation by the local Gaussian process posterior variance. Agents in high-uncertainty regions receive more attention weight, directing the team's representational focus toward poorly explored or hazardous areas.

3. **Hierarchical Control Architecture** — Three-layer design operating at different temporal scales:
   - **Planning layer** (RISE-MAPPO): selects subgoals every *Kₛ* = 25 steps
   - **Tracking layer** (Lyapunov-MPC): per-robot CasADi/IPOPT nonlinear MPC with Lyapunov contraction constraint *V(x_{k+1}) ≤ (1 − α_L) V(x_k)*
   - **Perception layer** (Distributed GP + BCM fusion): per-robot sparse GPs fused via Bayesian committee machine for communication-efficient shared environmental belief

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  RISE-MAPPO Policy (every Kₛ = 25 steps)             │
│                                                       │
│  Actor πθ  →  GP-Uncertainty Attention  →  Dual-Head  │
│               wᵢ ∝ exp(qᵀhᵢ·(1+ησ̃ᵢ))     Critic    │
│                                           ┌───┐┌───┐ │
│                                           │V_m││V_c│ │
│                                           └─┬─┘└─┬─┘ │
│                                         A = Am - λ·Ar │
└───────────────────────┬──────────────────────────────┘
                        │ Subgoals
┌───────────────────────▼──────────────────────────────┐
│  Per-Robot Lyapunov-MPC (every Δt = 0.1s)            │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ MPC₁    │  │ MPC₂    │  │ MPC₃    │  CasADi/IPOPT│
│  │V(k+1)≤  │  │V(k+1)≤  │  │V(k+1)≤  │              │
│  │(1-α)V(k)│  │(1-α)V(k)│  │(1-α)V(k)│              │
│  └────┬────┘  └────┬────┘  └────┬────┘              │
└───────┼─────────────┼───────────┼────────────────────┘
        │ Controls    │           │       ▲ Obstacles
┌───────▼─────────────▼───────────▼───────┼────────────┐
│  Distributed GP + BCM Fusion (continuous)            │
│                                                       │
│  GP₁  GP₂  GP₃  →  BCM Fusion → σ_BCM, CVaR cᵗ     │
│  σ⁻²_BCM = Σσᵢ⁻² − (N−1)σ⁻²_prior                  │
└──────────────────────────────────────────────────────┘
```

### Data Flow

**Training (CTDE — Centralized Training, Decentralized Execution):**

1. Environment provides local obs `o_i` per robot and global state `s`
2. Actor selects subgoal `a_i` from local obs (decentralized)
3. MPC tracks subgoal for `K_s` low-level steps
4. GP updates from sensor data, BCM fuses
5. Critic estimates `V_mean(s)` and `V_CVaR(s)` (centralized)
6. Risk cost: `c_t = mean(CVaR_α(μ_i, σ_i))` normalized to [0,1]
7. GAE computes `A_mean` and `A_risk` separately
8. Risk-adjusted advantage: `A = A_mean − λ_risk · A_risk`
9. PPO clipped surrogate update using `A`
10. Dual critic loss: `L = L_V_mean + β · L_V_CVaR`

**Evaluation (Decentralized):**

Same as training but: actor uses argmax (deterministic), no critic needed, metrics collected at every timestep.

### Reward Structure

```
reward = (w1 * delta_coverage      # exploration progress
        + w2 * new_targets         # target detection bonus
        - w3 * cvar_risk_cost      # GP-based tail risk penalty
        - w4 * energy_consumed     # energy efficiency
        + w5 * spread_bonus        # inter-robot spacing
        - w6 * collision_penalty)  # safety violation
```

---

## Project Structure

```
paper3/
├── configs/                          # YAML configuration files
│   ├── default.yaml                  # Full RISE-MAPPO training config
│   ├── eval_default.yaml             # Evaluation settings (50 eps, Phase-2 toggles)
│   ├── turtlebot3.yaml               # Robot physical parameters
│   ├── scenario_simple.yaml          # 5×5 m, 3 robots, 5 targets, 3 hazards
│   ├── scenario_complex.yaml         # 10×10 m, 5 robots, 10 targets, 3 hazards
│   ├── scenario_scalability.yaml     # Vary robots 3→8
│   ├── scenario_energy.yaml          # Half energy budget (50.0)
│   ├── scenario_comms.yaml           # 50% GP fusion drop rate
│   ├── ablation_no_rise.yaml         # MAPPO-only (no RISE modules)
│   ├── ablation_no_attention.yaml    # No GP-uncertainty attention (η=0)
│   └── ablation_no_cvar_head.yaml    # No CVaR dual-head (λ_risk=0, β=0)
├── envs/                             # Multi-robot search environment
│   ├── multi_robot_search_env.py     # Main env (PettingZoo-parallel API)
│   ├── grid_world.py                 # Continuous 2D world with obstacles, targets, hazards
│   ├── robot_dynamics.py             # TurtleBot3 Burger kinematics + dynamics
│   ├── observations.py               # Ego-centric observation builder
│   ├── integrations.py               # Controller/GP wiring helpers
│   └── wrappers.py                   # Environment wrappers
├── marl/mappo/                       # RISE-MAPPO algorithm
│   ├── actor.py                      # Shared decentralized actor (CNN + MLP)
│   ├── critic.py                     # Dual-head CVaR critic + GP-uncertainty attention
│   ├── algorithm.py                  # PPO update with risk-adjusted advantage
│   ├── buffer.py                     # Rollout storage (supports RISE dual-advantage)
│   └── runner.py                     # Training loop (collect → buffer → update)
├── marl/
│   └── utils.py                      # PopArt normalizer, seed utilities
├── mpc/                              # Low-level model predictive control
│   ├── lyapunov_mpc.py               # CasADi/IPOPT Lyapunov-stable MPC
│   ├── backstepping.py               # Backstepping controller + Lyapunov function
│   └── utils.py                      # ControllerFeedback dataclass, proportional fallback
├── gp/                               # Distributed Gaussian process perception
│   ├── distributed_gp.py             # BCM fusion, CVaR computation, info gain
│   └── local_gp.py                   # Per-robot sparse variational GP
├── baselines/                        # Evaluation baselines
│   ├── base_policy.py                # Abstract policy interface
│   ├── random_policy.py              # Uniform random subgoal selection
│   ├── nearest_frontier.py           # Yamauchi (1997) frontier-based exploration
│   ├── voronoi_partition.py          # Voronoi-based area partitioning
│   └── trained_policy.py             # Wrapper for loading trained checkpoints
├── analysis/                         # Evaluation metrics and visualization
│   ├── metrics.py                    # 10 evaluation metrics (coverage, CVaR, Lyapunov, etc.)
│   └── plotting.py                   # IEEE publication-quality figures (PDF + PNG)
├── scripts/                          # Entry points
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation pipeline
│   ├── ablation.py                   # Ablation study runner
│   ├── visualize.py                  # Trajectory visualization
│   ├── run_single.sh                 # Single training run launcher
│   ├── run_all_seeds.sh              # Multi-seed batch launcher
│   ├── check_status.py               # Training status dashboard
│   ├── collect_results.py            # Log parser → CSV
│   └── log_parser.py                 # Training log parser
├── tests/                            # Test suite (78 tests)
│   ├── test_env.py                   # Environment tests
│   ├── test_dynamics.py              # Robot dynamics tests
│   ├── test_mpc.py                   # Lyapunov-MPC tests
│   ├── test_gp.py                    # Distributed GP tests
│   ├── test_metrics.py               # Evaluation metrics tests
│   ├── test_baselines.py             # Baseline policy tests
│   ├── test_evaluate.py              # Evaluation pipeline tests
│   └── test_integration.py           # Integration tests
├── paper/                            # LaTeX manuscript
│   ├── main.tex                      # IEEEtran paper source
│   └── fig/                          # Figures (architecture.pdf, etc.)
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # Detailed architecture guide
│   ├── CONTRIBUTIONS.md              # Novelty claims and differentiation table
│   ├── TRAINING.md                   # Training guide and log format
│   └── EVALUATION.md                 # Evaluation guide and metrics reference
├── plans/                            # Implementation plans
│   ├── bug_fix_plan.md               # Original bug analysis and fix plan
│   ├── pre_production_fixes.md       # Pre-production fix execution plan
│   └── paper_update.md               # Paper update and production run plan
├── results/                          # Outputs (gitignored except tracked samples)
│   ├── phase1_seed42/                # Phase-1 trained checkpoints (1000 updates)
│   │   └── mappo_upd{50..1000}.pt    # 20 checkpoints at 50-update intervals
│   ├── checkpoints/                  # Phase-2 training checkpoints (created at runtime)
│   ├── eval/                         # Evaluation outputs (per scenario / per policy)
│   ├── runs/                         # Training run logs and configs
│   ├── gp_fusion_test.png            # GP fusion test output
│   ├── integration_test.png          # Integration test output
│   └── lyapunov_convergence.png      # Lyapunov convergence test output
├── setup.py                          # Package setup
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

## Installation

### Requirements

- **Python:** 3.10+
- **GPU:** CUDA-capable GPU recommended for training (tested on 2× RTX 2080 Ti, 11 GB each)
- **OS:** Linux (tested on Ubuntu 22.04 / WSL2)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Lucifer-121-cmd/RISE-MAPPO.git
cd RISE-MAPPO

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Neural networks, GPU training |
| CasADi | ≥ 3.6 | Symbolic NLP formulation for MPC |
| GPyTorch | ≥ 1.11 | Sparse variational Gaussian processes |
| Gymnasium | ≥ 0.29 | Environment interface (PettingZoo-compatible) |
| NumPy | ≥ 1.24 | Numerical computation |
| SciPy | ≥ 1.10 | Statistical functions (CVaR coefficient) |
| Matplotlib | ≥ 3.7 | Plotting and rendering |
| PyYAML | ≥ 6.0 | Configuration file parsing |
| WandB | ≥ 0.16 | Optional experiment tracking |
| pytest | ≥ 7.4 | Test framework |
| pytest-cov | ≥ 4.1 | Test coverage |

### Platform-Specific Notes

- **WSL2:** Ensure CUDA toolkit is installed on the Windows host and WSL2 CUDA support is enabled.
- **Headless servers:** Matplotlib uses the `Agg` backend automatically — no display required.
- **IPOPT:** CasADi bundles IPOPT; no separate IPOPT installation is needed.

---

## Quick Start

### Run the test suite

```bash
python -m pytest tests/ -v
# Expected: 78 passed in ~300s
```

### Smoke training (2 updates, CPU, minimal rollout)

```bash
python scripts/train.py --config configs/default.yaml --seed 42 --smoke
```

### Smoke evaluation (1 episode, Random policy)

```bash
python scripts/evaluate.py \
    --policy random \
    --scenario configs/scenario_simple.yaml \
    --eval-config configs/eval_default.yaml \
    --num-episodes 1
```

---

## Usage Guide

### Training

#### Full Phase-2 training (RISE-MAPPO with real GP + Lyapunov-MPC)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --seed 42 \
    --device cuda \
    --updates 500 2>&1 | tee train_phase2_seed42.log
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file path |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | `cuda` | `cuda`, `cpu`, or `cuda:0` / `cuda:1` for specific GPU |
| `--updates` | from config | Override `n_training_updates` |
| `--smoke` | `False` | Quick test: 2 updates, 1 env, minimal rollout |
| `--resume` | `None` | Path to checkpoint for resuming training |
| `--diag` | `False` | Enable critic diagnostic prints |
| `--log-level` | `INFO` | Logging level |

**GPU selection for multi-GPU systems:**

```bash
# Use GPU 1 specifically (GPU 0 may be busy)
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/default.yaml --seed 42 --device cuda
```

#### Resuming from a checkpoint

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --seed 42 \
    --device cuda \
    --resume results/phase1_seed42/mappo_upd500.pt
```

#### Using Phase-1 checkpoints

The `results/phase1_seed42/` directory contains checkpoints trained with Phase-1 features (synthetic GP decay grid, proportional controller). These serve as the MAPPO baseline for ablation studies. To evaluate them:

```bash
python scripts/evaluate.py \
    --policy trained \
    --checkpoint results/phase1_seed42/mappo_upd1000.pt \
    --config configs/default.yaml \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml
```

**Note:** Phase-1 checkpoints were trained with `use_real_gp=False` and `use_lyap_mpc=False`. Evaluating them with Phase-2 enabled is an out-of-distribution test. For the strongest results, use Phase-2 trained checkpoints.

#### Multi-seed batch training

```bash
# Preview what will run (no jobs launched)
./scripts/run_all_seeds.sh --parallel --dry-run

# Launch all on 2 GPUs (inside tmux)
tmux new -s batch
./scripts/run_all_seeds.sh --parallel 2>&1 | tee batch_run.log

# Run only main seeds (5 seeds for full RISE-MAPPO)
./scripts/run_all_seeds.sh --parallel --phase a

# Run only ablation (3 seeds × 3 configs = 9 runs)
./scripts/run_all_seeds.sh --parallel --phase b

# Limit to 500 updates
./scripts/run_all_seeds.sh --parallel --max-updates 500
```

#### Monitor training

```bash
# One-shot status
python scripts/check_status.py

# Auto-refresh every 5 minutes
python scripts/check_status.py --watch 300
```

### Evaluation

#### Trained policy (Phase-2 checkpoint)

```bash
# Production evaluation (50 episodes, Phase-2 features enabled via eval config)
python scripts/evaluate.py \
    --policy trained \
    --checkpoint results/checkpoints/mappo_upd500.pt \
    --config configs/default.yaml \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml
# Do NOT pass --num-episodes for production runs; the default of 50 is correct.
```

#### Baseline policy

```bash
python scripts/evaluate.py \
    --policy nearest_frontier \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml
```

Available `--policy` values: `trained`, `random`, `nearest_frontier`, `voronoi_partition`.

#### Ablation evaluation

```bash
python scripts/evaluate.py \
    --policy trained \
    --checkpoint results/runs/no_rise_seed42/checkpoints/mappo_upd1000.pt \
    --config configs/ablation_no_rise.yaml \
    --scenario configs/scenario_complex.yaml \
    --eval-config configs/eval_default.yaml \
    --name "Ours-no-CVaR"
```

#### Switching between Phase-1 and Phase-2 GP data

The evaluation config (`configs/eval_default.yaml`) controls whether real GP or synthetic decay grid data is used:

```yaml
# eval_default.yaml — Phase-2 features enabled
eval:
  num_episodes: 50
  deterministic: true
  device: "cpu"
env:
  use_real_gp: true      # Set to false for Phase-1 synthetic decay grid
  use_lyap_mpc: true     # Set to false for proportional controller
```

These `env` overrides are merged into the environment config during evaluation. The merge logic is in [`scripts/evaluate.py`](scripts/evaluate.py:319).

### Running Baselines

All baselines implement the `BasePolicy` interface:

```python
class BasePolicy(ABC):
    def reset(self, num_robots: int) -> None: ...
    def get_actions(self, observations, global_state) -> np.ndarray: ...
    @property
    def name(self) -> str: ...
```

| Baseline | Class | Description |
|----------|-------|-------------|
| Random | `RandomPolicy` | Uniform random subgoal selection (lower bound) |
| Nearest Frontier | `NearestFrontierPolicy` | Greedy move to closest frontier cell (Yamauchi 1997) |
| Voronoi Partition | `VoronoiPartitionPolicy` | Explore within Voronoi cell |

### Parallel Execution

Two processes can run simultaneously without resource conflicts:

```bash
# Terminal 1: Evaluation (CPU-only, no GPU needed)
export OMP_NUM_THREADS=8
python scripts/evaluate.py --policy random --scenario configs/scenario_simple.yaml \
    --eval-config configs/eval_default.yaml 2>&1 | tee eval_risk7.log

# Terminal 2: Training (GPU 1)
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
python scripts/train.py --config configs/default.yaml --seed 42 \
    --device cuda --updates 500 2>&1 | tee train_phase2_seed42.log
```

**Resource allocation for 2× RTX 2080 Ti + 40 CPU cores:**

| Resource | Eval | Training | Conflict? |
|----------|------|----------|-----------|
| GPU | None (CPU-only) | GPU 1 (11 GB) | No |
| CPU threads | 8 (OMP) | 8 (OMP) | No (16/40 used) |
| Disk I/O | `results/eval/` | `results/checkpoints/` | No (separate paths) |

---

## Configuration Guide

All hyperparameters live in YAML files under `configs/`. The training script merges the base config with CLI flags. The evaluation script merges base + scenario + eval config.

### Environment (`env`)

```yaml
env:
  num_robots: 3              # Number of robots in the team
  world_size: 10.0            # Square world side length (m)
  max_steps: 500              # MARL high-level steps per episode
  sensor_range: 1.5           # Robot sensor range (m)
  dt: 0.1                     # Low-level control timestep (s)
  difficulty: "medium"        # "easy", "medium", or "hard" (scales obstacles)
  num_targets: 5              # Number of hidden targets to find
  num_obstacles: 10           # Number of obstacles
  num_hazards: 3              # Number of hazard zones (must be >0 for CVaR)
  subgoal_steps: 25           # Low-level ticks per MARL action (K_s)
  detect_range: 0.4           # Target detection radius (m)
  robot_radius: 0.105         # TurtleBot3 Burger footprint radius (m)
  energy_budget: 100.0        # Per-robot energy budget
  add_noise: false            # Add process noise to dynamics
  use_dynamic_step: false     # Use first-order dynamic model (vs kinematic)

  # Phase-2 toggles — required for RISE-MAPPO CVaR head and GP-attention
  use_real_gp: true           # Use real GP posterior (vs synthetic decay grid)
  use_lyap_mpc: true          # Use Lyapunov-MPC (vs proportional controller)
  gp_update_interval: 10      # GP fusion every N low-level ticks
  gp_obs_noise_std: 0.05      # Observation noise std for hazard sensing
  cvar_alpha: 0.95            # CVaR confidence level (α)
  obstacle_margin_scale: 0.2  # Scale factor for GP-CVaR obstacle margins
```

### Reward Weights (`reward`)

```yaml
reward:
  w_coverage: 1.0             # Weight for coverage delta
  w_detection: 5.0            # Weight for new target detection
  w_cvar_risk: 0.5            # Weight for CVaR risk penalty
  w_energy: 0.3               # Weight for energy consumption
  w_coordination: 0.2         # Weight for inter-robot spacing bonus
  w_collision: 10.0           # Weight for collision penalty
```

### MAPPO / RISE (`mappo`)

```yaml
mappo:
  actor_lr: 5.0e-4            # Actor learning rate
  critic_lr: 5.0e-4           # Critic learning rate
  gamma: 0.99                 # Discount factor
  gae_lambda: 0.95            # GAE λ parameter
  ppo_epoch: 10               # PPO epochs per update
  clip_param: 0.2             # PPO clipping ε
  entropy_coef: 0.01          # Entropy bonus coefficient
  value_loss_coef: 0.5        # Value loss weight
  max_grad_norm: 0.5          # Gradient clipping norm
  num_mini_batch: 4           # Mini-batches per epoch
  use_popart: true            # PopArt value normalization

  # RISE-MAPPO settings
  use_rise: true              # Enable RISE modules (dual-head + GP attention)
  lambda_risk: 0.05           # Risk penalty weight (λ_risk)
  cvar_loss_coef: 0.25        # CVaR head loss weight (β)
  gp_attention_eta: 0.5       # GP attention temperature (η)
  gp_attention_heads: 1       # Number of attention heads
```

### Lyapunov-MPC (`mpc`)

All MPC parameters are YAML-configurable — no hardcoded values in the controller construction path. They flow from YAML → `EnvConfig.mpc` → `make_controller()` → `LyapunovMPCConfig`.

```yaml
mpc:
  horizon: 20                 # MPC prediction horizon (H)
  dt: 0.1                     # Control timestep (s)
  Q_diag: [10.0, 10.0, 1.0]  # State cost diagonal (x, y, θ)
  R_diag: [1.0, 0.5]         # Input cost diagonal (v, ω)
  S_du_diag: [1.0, 0.5]      # Input rate cost diagonal
  alpha_lyap: 0.1             # Lyapunov contraction rate (α_L)
  d_safe: 0.5                 # Minimum obstacle clearance (m)
  soft_lyap_penalty: 1000.0   # Slack penalty for soft Lyapunov constraint
  max_iter: 100               # IPOPT max iterations
  max_cpu_time: 1.0           # IPOPT max CPU time per solve (s)
  max_obstacles: 8            # Max obstacle slots in NLP
  goal_tolerance: 0.1         # Subgoal reached threshold (m)
  P_terminal_scale: 5.0       # Terminal cost scaling
  w_energy: 0.1               # Energy cost weight in MPC objective
```

### Gaussian Process (`gp`)

```yaml
gp:
  kernel: "rbf"               # GP kernel type
  lengthscale: 1.0            # RBF kernel lengthscale
  noise: 0.01                 # Observation noise
  n_inducing: 50              # Number of inducing points
  cvar_alpha: 0.05            # CVaR confidence level
```

### Training Runner (`training`)

```yaml
training:
  seed: 42                    # Random seed
  num_envs: 4                 # Number of parallel environments
  rollout_length: 256         # Steps per rollout
  n_training_updates: 1000    # Total training updates
  log_interval: 1             # Log every N updates
  save_interval: 50           # Save checkpoint every N updates
  eval_interval: 25           # Evaluation interval (not yet implemented)
  device: "cuda"              # Training device
  save_dir: "results/checkpoints"  # Checkpoint save directory
  use_wandb: false            # Enable Weights & Biases logging
  wandb_project: "paper3-marl-search"
```

### Evaluation (`eval`)

```yaml
eval:
  num_episodes: 50            # Episodes per (policy, scenario) cell
  deterministic: true         # Use argmax for trained policies
  device: "cpu"               # MUST stay CPU while training holds GPU
  save_trajectories: true     # Persist per-episode .npz blobs
  save_videos: false          # Video recording (off by default)
  results_dir: "results/eval" # Output directory
  seeds: [100, 200, 300, 400, 500]  # Evaluation seeds

env:                          # Phase-2 overrides merged into env config
  use_real_gp: true
  use_lyap_mpc: true
  gp_update_interval: 10
  gp_obs_noise_std: 0.05
```

### Scenario Configs

| Scenario | Config | Robots | Targets | Hazards | World Size | Key Feature |
|----------|--------|--------|---------|---------|------------|-------------|
| Simple | `scenario_simple.yaml` | 3 | 5 | 3 | 5×5 m | Open environment, fast baseline |
| Complex | `scenario_complex.yaml` | 5 | 10 | 3 | 10×10 m | Corridors, hazard zones (main eval) |
| Scalability | `scenario_scalability.yaml` | 3–8 | 10 | 2 | 10×10 m | Vary robot count |
| Energy | `scenario_energy.yaml` | 5 | 10 | 2 | 10×10 m | Half energy budget (50.0) |
| Comms Failure | `scenario_comms.yaml` | 5 | 10 | 2 | 10×10 m | 50% GP fusion drop rate |

### Ablation Configs

| Config | What Changes |
|--------|-------------|
| `ablation_no_rise.yaml` | `use_rise: false` — standard MAPPO |
| `ablation_no_attention.yaml` | `gp_attention_eta: 0.0` — no GP attention |
| `ablation_no_cvar_head.yaml` | `lambda_risk: 0.0, cvar_loss_coef: 0.0` — no CVaR head |

---

## Key Hyperparameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Risk penalty weight | λ_risk | 0.05 | Risk–return trade-off in advantage |
| CVaR head loss weight | β | 0.25 | Relative weight of CVaR critic head |
| GP attention temperature | η | 0.5 | Strength of uncertainty modulation in attention |
| CVaR confidence level | α | 0.05 | Tail probability for risk computation |
| Lyapunov contraction rate | α_L | 0.1 | MPC tracking convergence rate |
| MPC horizon | H | 20 | Prediction steps |
| Subgoal interval | K_s | 25 | Low-level steps per MARL decision |
| Max gradient norm | — | 0.5 | Gradient clipping (critical for dual-head stability) |
| Discount factor | γ | 0.99 | Return discounting |
| GAE parameter | λ | 0.95 | Advantage estimation |
| MPC obstacle clearance | d_safe | 0.5 | Minimum distance to obstacles (m) |
| Soft Lyapunov penalty | — | 1000.0 | Slack penalty weight |

---

## Evaluation Metrics

All metrics are computed in [`analysis/metrics.py`](analysis/metrics.py) from per-episode data collected during evaluation.

| Metric | Function | Description | Better |
|--------|----------|-------------|--------|
| Coverage Rate | `coverage_rate()` | Fraction of explorable area covered by episode end | Higher |
| Detection Success | `detection_success_rate()` | Fraction of targets found within episode budget | Higher |
| Time to Detection | `time_to_full_detection()` | Steps until all targets found (or max_steps) | Lower |
| Collision Rate | `collision_rate()` | Total collision events per episode | Lower |
| Energy Efficiency | `energy_efficiency()` | Coverage per unit total energy consumed | Higher |
| Mean CVaR Risk | `mean_cvar_risk()` | Average per-robot per-step CVaR of GP hazard posterior | Lower |
| Exploration Overlap | `exploration_overlap()` | Fraction of visited cells touched by >1 robot | Lower |
| Lyapunov Stability | `lyapunov_stability()` | Fraction of steps with monotonic decrease in V(t) | Higher |

**Important notes on metrics:**

- **Energy efficiency** is controller-dependent. The Lyapunov-MPC uses a quadratic power model (*P(v,ω) = c₁v² + c₂ω² + c₃|v||ω| + c₄|v| + c₅*) while baselines use a linear heuristic (*P ∝ |v| + 0.1|ω|*). Absolute values are not directly comparable across controller types. See [Energy Model Change](#energy-model-change) below.
- **Exploration overlap** uses fine-grained per-tick robot positions (recorded at every low-level controller tick) for accurate computation.
- **Lyapunov stability** for baselines uses a fixed spawn-position reference rather than the per-step subgoal, preventing the degenerate case where policies that change subgoals every step trivially report monotonic_fraction = 1.0.
- **Coverage curves** are NaN-padded for episodes that terminate early (crashes, energy depletion, all targets found). Downstream plotting uses nan-aware aggregation.

---

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| Random | Non-learning | Uniform random subgoal selection (lower bound) |
| Nearest Frontier | Classical | Yamauchi (1997), greedy nearest frontier |
| Voronoi Partition | Classical | Voronoi-based area partitioning |
| MAPPO | Ablation | Standard MAPPO without RISE modules (`use_rise: false`) |
| Ours w/o CVaR | Ablation | λ_risk = 0, β = 0 (removes dual-head risk critic) |
| Ours w/o GP-Attn | Ablation | η = 0 (removes uncertainty-weighted attention) |
| Ours w/o Lyap | Ablation | Proportional controller replaces MPC |

---

## Robot Platform

TurtleBot3 Burger with unicycle kinematics:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Max linear velocity | v_max | 0.22 m/s |
| Max angular velocity | ω_max | 2.84 rad/s |
| Wheel radius | r | 0.033 m |
| Wheel separation | L | 0.160 m |
| Mass | m | 1.0 kg |
| Moment of inertia | I | 8.7×10⁻³ kg·m² |
| Linear time constant | τ_v | 0.20 s |
| Angular time constant | τ_ω | 0.10 s |
| Sensor range | — | 1.5 m |
| Footprint radius | — | 0.105 m |

State: `[x, y, θ]` with θ ∈ (−π, π]. Control: `[v, ω]` saturated to actuator limits.

---

## Recent Implementations & Architecture Changes

### Energy Model Change

**Status:** Implemented (2026-05-14) | **Files:** [`mpc/lyapunov_mpc.py`](mpc/lyapunov_mpc.py:147), [`mpc/utils.py`](mpc/utils.py:104)

The energy consumption model was upgraded from a linear heuristic to a physically-motivated quadratic power model.

**Old model (proportional controller):**
```python
# mpc/utils.py line 104
energy_used = abs(v) + 0.1 * abs(omega)
```

**New model (Lyapunov-MPC):**
```python
# mpc/lyapunov_mpc.py lines 147-155
P_k = (pwr_c1 * v² + pwr_c2 * ω² + pwr_c3 * |v|·|ω| + pwr_c4 * |v| + pwr_c5)
```

Coefficients (in [`LyapunovMPCConfig`](mpc/lyapunov_mpc.py:65)): `pwr_c1=15.0`, `pwr_c2=0.5`, `pwr_c3=1.0`, `pwr_c4=3.0`, `pwr_c5=0.5`.

**Impact:** Energy efficiency values differ by ~10× between controller types. The paper (Sections IV-A, IV-B, V) documents this as a controller-dependent metric. See [`plans/paper_update.md`](plans/paper_update.md) for the full analysis.

### CVaR Dual-Head Critic

**Status:** Implemented | **Files:** [`marl/mappo/critic.py`](marl/mappo/critic.py:117), [`marl/mappo/algorithm.py`](marl/mappo/algorithm.py:182)

The critic outputs two value estimates from a shared CNN+MLP backbone:
- `V_mean(s)` — expected discounted return
- `V_CVaR(s)` — expected discounted risk cost

Each head has its own PopArt normalizer. The risk-adjusted advantage `A = A_mean − λ_risk · A_risk` drives the PPO policy update. The risk cost `c_t` is derived from the GP posterior CVaR at robot positions, normalized to [0,1].

### GP-Uncertainty Attention

**Status:** Implemented | **Files:** [`marl/mappo/critic.py`](marl/mappo/critic.py:58)

The `GPUncertaintyAttention` module modulates per-agent feature aggregation by GP posterior variance:

```
Q_i = W_Q(h_i + η · s_i)     # s_i = ReLU(W_σ · σ_i + b_σ)
K_j = W_K(h_j + η · s_j)
e_ij = Q_iᵀK_j / √d + η · σ_j   # direct column bias for uncertainty
w_ij = softmax(e_ij)
```

Agents in high-uncertainty regions both issue stronger queries and attract more attention. When `η = 0`, the mechanism reduces to standard scaled dot-product attention.

### Phase-1 vs Phase-2 Training

**Status:** Phase-1 complete, Phase-2 in progress | **Checkpoints:** `results/phase1_seed42/`

- **Phase-1** (`use_real_gp=False`, `use_lyap_mpc=False`): Trained with synthetic GP decay grid and proportional controller. 1000 updates completed in ~72 hours. Checkpoints saved in `results/phase1_seed42/` (renamed from `results/checkpoints/` to avoid overwriting).
- **Phase-2** (`use_real_gp=True`, `use_lyap_mpc=True`): Currently training with real GP posterior and Lyapunov-MPC. Estimated ~6 days for 500 updates. Checkpoints saved in `results/checkpoints/`.

Phase-1 checkpoints serve as the MAPPO baseline for ablation studies. Phase-2 checkpoints will be used for the main RISE-MAPPO results.

### MPC Config Propagation Fix

**Status:** Fixed (2026-05-14) | **Files:** [`scripts/train.py`](scripts/train.py:70), [`scripts/evaluate.py`](scripts/evaluate.py:69)

The `mpc` YAML section was previously ignored because it lives at the top level of the config (outside the `env` key). The controller silently used `LyapunovMPCConfig` dataclass defaults (`d_safe=0.3` instead of `0.5`, `soft_lyap_penalty=100.0` instead of `1000.0`).

**Fix:** `_make_env_cfg()` in `train.py` now passes `mpc=raw.get("mpc", {})`. `_build_env_cfg()` in `evaluate.py` now pulls `merged["mpc"]` into the env config. All 14 MPC parameters now flow from YAML → `EnvConfig.mpc` → `make_controller()` → `LyapunovMPCConfig`.

### Exploration Overlap Fix

**Status:** Fixed (2026-05-14) | **Files:** [`analysis/metrics.py`](analysis/metrics.py:102), [`envs/integrations.py`](envs/integrations.py:100)

The `exploration_overlap()` metric previously used robot positions sampled at MARL-step granularity (every 25 low-level ticks), producing overlap = 0.0 even when robots clearly crossed paths.

**Fix:** `run_low_level_loop()` now records per-tick positions in `positions_history`. The evaluation loop collects these into `EpisodeData.positions_per_tick`. `exploration_overlap()` uses the fine-grained positions when available, falling back to MARL-step positions for backward compatibility.

### Lyapunov Reference Fix

**Status:** Fixed (2026-05-14) | **Files:** [`scripts/evaluate.py`](scripts/evaluate.py:128), [`analysis/metrics.py`](analysis/metrics.py:127)

The Lyapunov metric previously computed V relative to the current subgoal. For baselines that change subgoals every MARL step (Nearest Frontier, Random), V ≈ 0 always, producing trivially monotonic_fraction = 1.0.

**Fix:** For baseline policies, `run_episode()` sets `EpisodeData.lyapunov_reference` to the spawn position. `lyapunov_stability()` recomputes V relative to this fixed reference, producing meaningful monotonicity values.

### Coverage Curve NaN Padding

**Status:** Fixed (2026-05-14) | **Files:** [`scripts/evaluate.py`](scripts/evaluate.py:262), [`analysis/plotting.py`](analysis/plotting.py:162)

Coverage curves were previously right-padded with the last coverage value for early-terminated episodes, artificially inflating aggregate statistics.

**Fix:** Padding uses `np.nan` instead of the last value. `plot_coverage_curves()` uses `np.nanmean` and `np.nanstd` for nan-safe aggregation.

---

## Logging and Results

### Training Logs

Training logs use the format:

```
2026-05-11 22:53:21,971 paper3.runner INFO | update 1  R=248.002  cov=0.376  det=0.82  pl=-0.018  vl=87.163  H=3.213  kl=0.0015  283.5s
```

| Field | Meaning | Healthy Range |
|-------|---------|---------------|
| R | Episode reward (team total) | 300–500+ |
| cov | Coverage fraction | 0.6–0.85 |
| det | Targets detected per episode | 1.5–2.5 |
| pl | Policy loss | −0.05 to −0.02 |
| vl | Value loss | Decreasing over time |
| H | Policy entropy | 1.5–2.5 (not collapsing) |
| kl | KL divergence from old policy | 0.01–0.04 |
| last number | Wall-clock seconds for this update | 150–350s |

**Convergence signs:** Reward plateaus, value loss drops below 0.1, entropy stabilizes around 1.5–2.0. Typically occurs by update 300–500.

### Evaluation Logs

Evaluation logs use the format:

```
2026-05-14 21:40:15 paper3.evaluate INFO | episode 1 / 50 (seed=100)
2026-05-14 21:45:30 paper3.evaluate INFO | episode 2 / 50 (seed=201)
...
2026-05-14 23:50:00 paper3.evaluate INFO | wrote results to results/eval/simple/Random
2026-05-14 23:50:00 paper3.evaluate INFO | summary: {'coverage_rate': 0.698, ...}
```

### Output Directory Structure

```
results/
├── phase1_seed42/                    # Phase-1 trained checkpoints
│   └── mappo_upd{50,100,...,1000}.pt
├── checkpoints/                      # Phase-2 training checkpoints
│   └── mappo_upd{50,100,...}.pt
├── eval/
│   └── {scenario}/
│       └── {policy_name}/
│           ├── metrics_summary.json   # mean ± std for all metrics
│           ├── metrics_per_episode.csv # one row per episode
│           ├── coverage_curves.npy    # (N_episodes, T) coverage over time
│           ├── scenario.yaml          # scenario config used
│           ├── eval.yaml              # eval config used
│           └── episode_data/          # full episode .npz blobs
│               ├── ep_000.npz
│               └── ...
└── runs/                             # Training run logs
    └── {run_name}/
        ├── train.log
        ├── run_config.yaml
        └── DONE                      # Completion marker
```

### Key Results from Recent Runs

**Phase-1 Training (seed 42, 1000 updates):**
- Completed in 71h 38m (~3 days)
- Final reward: ~460, coverage: ~0.67, value loss: ~0.13
- Converged by update ~300–500
- 20 checkpoints saved at 50-update intervals

**Smoke Evaluation (Random policy, Simple scenario, 2 episodes):**
- Collision rate: 1.5 (was 3.0 before fixes — Lyapunov-MPC avoids obstacles)
- Exploration overlap: 0.049 (was 0.0 — per-tick positions now recorded)
- Mean CVaR risk: 0.90 ± 0.82 (was ~0.05 constant — real GP active)
- Lyapunov monotonic: 0.972 (was ~0.51 — fixed spawn reference)

---

## Directory Structure

See [Project Structure](#project-structure) above for the complete annotated tree.

### Recently Added or Renamed

| Path | Status | Description |
|------|--------|-------------|
| `results/phase1_seed42/` | Renamed from `results/checkpoints/` | Phase-1 trained checkpoints preserved |
| `results/checkpoints/` | Recreated | Phase-2 training output directory |
| `plans/bug_fix_plan.md` | New | Original bug analysis and fix plan |
| `plans/pre_production_fixes.md` | New | Pre-production fix execution plan |
| `plans/paper_update.md` | New | Paper update and production run plan |
| `results/eval/simple/` | Deleted | Smoke test artifacts cleaned up |

---

## Limitations and Known Issues

1. **Checkpoint distribution mismatch:** Phase-1 checkpoints were trained with synthetic GP decay grid data. Evaluating them with Phase-2 enabled is an out-of-distribution test. Use Phase-2 trained checkpoints for the strongest results.

2. **Energy efficiency comparability:** The Lyapunov-MPC quadratic power model and the proportional controller linear heuristic produce energy values on different scales (~10×). Cross-controller energy efficiency comparisons are not meaningful.

3. **Scalability runtime:** The scalability scenario (8 robots) runs ~100,000 IPOPT solves per episode (~83 min/ep). Use `--num-episodes 10` for robot counts ≥ 6.

4. **IPOPT solver failures:** The Lyapunov contraction constraint can become infeasible near obstacle boundaries. The soft slack penalty ensures the NLP remains feasible, but excessive slack activation may degrade tracking performance.

5. **Fixed GP hyperparameters:** GP kernel hyperparameters (lengthscale, noise) are fixed rather than adapted online, which may limit performance in highly heterogeneous environments.

6. **2D simulation only:** The current evaluation uses a 2D simulated environment with idealized sensing. Real-world deployment requires ROS 2 integration and sensor noise modeling.

7. **Single-process training:** The runner uses a single-process loop (no multiprocessing). Multi-GPU training requires the `run_all_seeds.sh` script which launches independent processes.

---

## Future Work

1. **Phase-2 training completion:** The current Phase-2 training run (500 updates, ~6 days) will produce checkpoints with real GP posterior and Lyapunov-MPC for the main RISE-MAPPO results.

2. **Full production evaluation:** After the 14-hour verification test confirms the fixes, run the complete evaluation matrix (5 scenarios × 7 policies × 50 episodes × 5 seeds).

3. **Ablation studies:** Quantify the individual contribution of each RISE component (CVaR head, GP attention, Lyapunov-MPC) through controlled ablation experiments.

4. **ROS 2 deployment:** Port the trained policies to physical TurtleBot3 robots via ROS 2 for real-world validation.

5. **Adaptive GP hyperparameters:** Implement online kernel hyperparameter adaptation for heterogeneous environments.

6. **Risk-sensitive communication scheduling:** Extend the CVaR formulation to decide when agents should share GP data based on the marginal value of information.

---

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Environment + MAPPO + pipeline | ✅ Complete |
| Phase 2 | Lyapunov-MPC + GP + CVaR integration | ✅ Complete |
| Phase 2.5 | RISE-MAPPO novel modules (dual-head, GP-attention) | ✅ Complete |
| Phase 3 | Evaluation pipeline + baselines | ✅ Complete |
| Phase 4 | Bug fixes + config hardening | ✅ Complete |
| Phase 5 | Phase-2 training | 🔄 In progress |
| Phase 6 | Production evaluation | ⏳ Pending |
| Phase 7 | Paper finalization | ⏳ Pending |

---

## Citation

```bibtex
@article{dhakal2026rise,
  title={{RISE-MAPPO}: Risk-Sensitive Multi-Agent Policy Optimization
         for Multi-Robot Exploration},
  author={Dhakal, Nischal and {Shuaiyong Li}},
  journal={submitted to IEEE Robotics and Automation Letters},
  year={2026}
}
```

---

## License

This project is for academic research. Please cite our work if you use this codebase.
