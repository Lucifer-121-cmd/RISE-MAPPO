# Paper 3 — Risk-Aware Multi-Robot Cooperative Search

Hierarchical MARL (MAPPO, high level) + Lyapunov-MPC (low level) +
Distributed GP + CVaR risk on TurtleBot3 Burger robots.

## Status

Phase 1 (Foundation): environment, MAPPO, simple proportional low-level
controller. Phase 2 (Lyapunov-MPC, distributed GP, CVaR) wired as
placeholders behind clean interfaces; will be filled in subsequent phases
without changing call sites.

## Layout

```
paper3/
├── configs/      # YAML hyperparameters
├── envs/         # Gym multi-robot search environment + dynamics
├── marl/mappo/   # MAPPO actor / critic / buffer / algorithm / runner
├── mpc/          # Lyapunov-MPC (Phase 2 placeholder + simple controller)
├── gp/           # Distributed GP, BCM, CVaR (Phase 2 placeholders)
├── scripts/      # train / evaluate / visualize / ablation
├── tests/        # pytest suite
└── results/      # logs, figures, checkpoints
```

## Setup

```bash
source .venv/bin/activate          # symlinked to ../gp-cvar-mpc/.venv
pip install -r requirements.txt    # if anything is missing
pip install -e .                   # editable install of this package
```

## Quick start

```bash
# Run all tests
python -m pytest tests/ -v

# Smoke train (small)
python scripts/train.py --config configs/default.yaml --seed 42 --smoke
```

## Key design decisions

- **Hierarchy:** MAPPO selects discrete subgoal cells (5×5 grid, 1 m
  spacing) every `subgoal_steps=25` low-level steps. The low-level
  controller is a proportional pose regulator in Phase 1; it is swapped
  for Lyapunov-MPC in Phase 2 without changing the env interface.
- **Robot:** TurtleBot3 Burger (max v=0.22 m/s, max ω=2.84 rad/s).
- **CTDE:** Decentralised actors with local observations; centralised
  critic with global state (full occupancy + GP map + all poses).
- **Reward (team-shared):**
  `w1·Δcoverage + w2·targets - w3·CVaR - w4·energy + w5·spread - w6·collision`
