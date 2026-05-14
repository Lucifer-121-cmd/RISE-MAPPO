# RISE-MAPPO Architecture

## Hierarchical Overview

RISE-MAPPO operates at three temporal scales:

```
Decision Layer    │  RISE-MAPPO policy selects subgoals every Kₛ = 25 steps
                  │  Actor: shared MLP, decentralized execution
                  │  Critic: centralized, dual-head (V_mean + V_CVaR)
                  │  Attention: GP-uncertainty-weighted agent aggregation
                  ▼
Tracking Layer    │  Per-robot Lyapunov-MPC tracks subgoals every Δt = 0.1s
                  │  CasADi symbolic NLP, solved with IPOPT
                  │  Lyapunov contraction: V(x_{k+1}) ≤ (1 - α_L) V(x_k)
                  │  Warm-started from previous solution
                  ▼
Perception Layer  │  Local sparse GPs per robot, updated continuously
                  │  BCM fusion: σ⁻²_BCM = Σσᵢ⁻² − (N−1)σ⁻²_prior
                  │  Outputs: uncertainty grid, CVaR risk, info gain
```

## Module Interfaces

### Actor (`marl/mappo/actor.py`)

```python
actor.get_action(obs: Tensor) -> (action: Tensor, log_prob: Tensor)
# obs shape: (N, obs_dim) — local observation per robot
# action: (N,) — discrete subgoal index in [0, K-1]
# K = 25 (5×5 subgoal grid, 1m spacing)
```

### Dual-Head Critic (`marl/mappo/critic.py`)

```python
critic.forward(global_state, gp_sigmas) -> (v_mean, v_cvar)
# global_state: (batch, state_dim) — full map + all poses
# gp_sigmas: (batch, N) — GP posterior std per agent
# v_mean: (batch, 1) — expected return estimate
# v_cvar: (batch, 1) — CVaR risk estimate

# GP-Uncertainty Attention (internal):
# w_i ∝ exp(q^T h_i · (1 + η · σ̃_i))
# η = gp_attention_eta (default 0.5)
```

### Lyapunov-MPC (`mpc/lyapunov_mpc.py`)

```python
mpc.solve(current_state, subgoal, obstacles) -> ControlFeedback
# current_state: (3,) — [x, y, θ]
# subgoal: (2,) — [x_g, y_g]
# Returns: ControlFeedback(v, omega, solve_time, lyapunov_value)
```

### Distributed GP (`gp/distributed_gp.py`)

```python
gp.update(robot_id, position, observation)
gp.predict(query_points) -> (mu, sigma)
gp.fuse_bcm(all_local_predictions) -> (mu_bcm, sigma_bcm)
gp.compute_cvar(positions, alpha=0.05) -> cvar_values
gp.information_gain(positions) -> info_gain
```

## Data Flow

### Training (CTDE)

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

### Evaluation (Decentralized)

Same as training but: actor uses argmax (deterministic), no critic needed, metrics collected at every timestep.

## Reward Structure

```python
reward = (w1 * delta_coverage      # exploration progress
        + w2 * new_targets         # target detection bonus
        - w3 * cvar_risk_cost      # GP-based tail risk penalty
        - w4 * energy_consumed     # energy efficiency
        + w5 * spread_bonus        # inter-robot spacing
        - w6 * collision_penalty)  # safety violation
```

## Config System

All hyperparameters in `configs/default.yaml`. Ablation configs override specific fields:

| Config | What changes |
|--------|-------------|
| `ablation_no_rise.yaml` | `use_rise: false` — standard MAPPO |
| `ablation_no_attention.yaml` | `gp_attention_eta: 0.0` — no GP attention |
| `ablation_no_cvar_head.yaml` | `lambda_risk: 0.0, cvar_loss_coef: 0.0` — no CVaR head |

## Critical Implementation Notes

1. **Risk cost scale**: Raw CVaR can be ~75× reward scale. Always normalize to [0,1] before feeding to critic.
2. **Gradient norm**: Must be 0.5 (not default 10) for dual-head stability.
3. **PopArt**: Each critic head has its own PopArt normalizer to handle different output scales.
4. **NaN guards**: `nan_to_num` on critic outputs + skip-on-nonfinite-grad in optimizer.
5. **MPC warm-start**: Previous solution shifted by one timestep, critical for solve time.
