# RISE-MAPPO Contributions

## C1: RISE-MAPPO Core Algorithm (Primary Novelty)

**Dual-Head CVaR Critic + GP-Uncertainty Attention**

No prior work integrates CVaR directly into a policy-gradient (MAPPO-style) multi-agent critic. Existing risk-sensitive MARL methods (RMIX, RiskQ, DRIMA) operate within value-decomposition frameworks (QMIX). Our policy-gradient formulation avoids QMIX monotonicity constraints and naturally handles large discrete action spaces.

The GP-uncertainty attention is novel: instead of treating all agents equally in the centralized critic, agents in high-uncertainty regions receive more attention weight. This is a direct, differentiable connection between environmental perception (GP) and policy optimization (MARL critic) — not reward shaping.

**Key equations:**
- Risk-adjusted advantage: `A = A_mean − λ_risk · A_risk`
- Attention: `w_i ∝ exp(q^T h_i · (1 + η · σ̃_i))`

**Files:** `marl/mappo/critic.py`, `marl/mappo/algorithm.py`

## C2: Hierarchical RISE-MAPPO + Lyapunov-MPC

**What:** High-level MARL planner coupled with per-robot model-based controllers that provide kinodynamic feasibility and tracking stability guarantees.

**Why novel:** Studt & Schildbach (2025) combined MAPPO with MPC but without risk sensitivity or GP perception. Safe-RMM (Liu et al., 2024) used control barrier functions, not Lyapunov contraction. Our hierarchy uniquely combines (i) risk-sensitive high-level policy, (ii) Lyapunov-stable MPC, and (iii) GP-informed safety margins.

**Files:** `marl/mappo/runner.py` (hierarchy loop), `mpc/lyapunov_mpc.py`

## C3: Distributed GP with BCM Fusion

**What:** Per-robot sparse GPs fused via Bayesian committee machine for communication-efficient shared environmental belief.

**Why it matters:** The GP posterior feeds three signals into the system: (i) uncertainty grid → critic observation and attention, (ii) information gain → reward, (iii) CVaR risk → dual-head critic and reward. This creates a tight perception-planning loop absent from existing MARL exploration methods.

**Files:** `gp/distributed_gp.py`

## C4: Lyapunov-Stable Per-Robot Tracking

**What:** CasADi/IPOPT-based nonlinear MPC with Lyapunov contraction constraint `V(x_{k+1}) ≤ (1 − α_L) V(x_k)`.

**Why it matters:** Provides provable monotonic convergence to subgoals — a formal safety certificate that end-to-end MARL cannot provide. GP-CVaR-based safety margins tighten constraints in uncertain regions. The MPC also incorporates a quadratic power model `P(v,ω) = c₁v² + c₂ω² + c₃|v||ω| + c₄|v| + c₅` calibrated to the TurtleBot3 Burger platform, replacing the linear heuristic used in the proportional baseline controller.

**Files:** `mpc/lyapunov_mpc.py`

## Differentiation Table

| Method | Risk-Sensitive | GP Uncertainty | Hierarchical | Lyapunov MPC | Multi-Robot |
|--------|:-:|:-:|:-:|:-:|:-:|
| RMIX (NeurIPS 2021) | ✓ | | | | ✓ |
| RiskQ (NeurIPS 2023) | ✓ | | | | ✓ |
| ACE (AAMAS 2023) | | | | | ✓ |
| Safe-RMM (2024) | | | ✓ | CBF | ✓ |
| Studt (2025) | | | ✓ | ✓ | ✓ |
| **RISE-MAPPO (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |
