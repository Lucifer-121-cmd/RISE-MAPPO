# Phase-2 Training Progress Report

**Date**: 2026-05-20 (last log entry: 03:26 UTC, update 186 in-progress)  
**Log**: [`train_phase2_seed42.log`](../train_phase2_seed42.log) (17,397 lines)  
**Config**: [`configs/default.yaml`](../configs/default.yaml) — Phase-2 (real GP + Lyapunov-MPC, RISE-MAPPO enabled)  
**Status**: 🔄 **ACTIVE — running in tmux session `train`**  

---

## 1. Training Summary

| Metric | Value |
|--------|-------|
| Updates completed | 186 |
| Updates configured | 1,000 (at `--updates 500` flag → override, so goal is 500) |
| Wall time elapsed | 97.0 hours (~4.0 days) |
| Mean time per update | 1,878 seconds (~31.3 minutes) |
| Estimated time to 500 updates | ~164 more hours (~6.8 days from now) |
| Checkpoints saved | 3 (upd 50, 100, 150) |
| Next checkpoint | upd 200 |
| Errors/crashes | **0** — no ERROR, CRITICAL, or Traceback lines |
| NaN values | **0** |
| Infeasible MPC solves | 682 out of ~17,246 logged warnings (~4.0%) |

---

## 2. Metric Trends (25-update windows)

```
Upd    1-  25: R=   90.4±154  cov=0.196±.176  det=0.31  vl=6.638  H=3.089  kl=.0128  1461s
Upd   26-  50: R=  109.5±160  cov=0.186±.180  det=0.34  vl=0.476  H=2.981  kl=.0143  2571s
Upd   51-  75: R=  116.3±144  cov=0.286±.147  det=0.51  vl=0.326  H=2.966  kl=.0127  1973s
Upd   76- 100: R=  230.8±225  cov=0.328±.078  det=0.54  vl=0.338  H=3.081  kl=.0090  2357s  ← peak reward (R=798.8 at upd 80)
Upd  101- 125: R=  173.1±201  cov=0.344±.043  det=0.62  vl=0.276  H=3.092  kl=.0086  1710s
Upd  126- 150: R=  218.2±216  cov=0.355±.038  det=0.75  vl=0.298  H=3.131  kl=.0070  1604s  ← best detection
Upd  151- 175: R=  117.6±161  cov=0.351±.058  det=0.63  vl=0.252  H=3.112  kl=.0087  1440s
Upd  176- 186: R=  246.0±220  cov=0.350±.038  det=0.39  vl=0.311  H=3.109  kl=.0100  1949s
```

**Best single update**: R=798.8 at update 80.

---

## 3. Convergence Analysis

### Signs of Convergence ✅

- **Value loss stabilized**: From 89.2 at update 1 → 0.3 by update 50; currently oscillating between 0.1-0.5 with mean ~0.3. No further improvement since update 50.
- **KL divergence low**: <0.015 mean throughout, currently ~0.010. Policy changes are small — the actor is not making large distribution shifts.
- **Coverage plateaued**: Stabilized at ~0.35 by update 75, no improvement in 110+ updates.
- **Detection plateaued**: Peaked at 0.75 mean in window 126-150, then regressed to 0.39 in latest window.

### Signs of Non-Convergence ⚠️

- **Reward variance is high**: ±200 standard deviation across all windows. The policy alternates between high-reward episodes (R=400-800) and near-zero episodes (R=0-30).
- **Entropy has not collapsed**: Currently 3.11, close to initial 3.21. This is actually **healthy** — the policy is still exploring, not overfitting to a narrow strategy. Phase-1 convergence criterion is entropy at 1.5-2.0.
- **Binary episode outcome pattern persists**: Episodes with R=0, cov=0, det=0 appear regularly (~15% of updates). These are episodes where all 3 robots crashed or depleted energy before achieving any coverage. The policy has not learned to avoid this failure mode consistently.

### Verdict: NOT CONVERGED, but hitting a local plateau

The Phase-1 training (with synthetic GP, proportional controller) reached coverage ~0.67 and reward ~460 by update 300-500. Phase-2 is at coverage ~0.35 and reward ~200 at update 186. Three possible explanations:

1. **Phase-2 is genuinely harder**: Real GP posterior + Lyapunov-MPC create a much more complex state space. The policy needs more updates to learn effective behavior in this richer environment.

2. **Energy budget is the bottleneck**: The TurtleBot3 has 100.0 energy. At ~31 min per update (vs ~4 min for Phase-1), this means ~50 MPC solves per second. The quadratic power model may be consuming energy faster than the proportional controller's linear heuristic, leaving robots with less exploration time before energy depletion.

3. **The RISE-MAPPO CVaR head may be overly conservative**: With `lambda_risk=0.05`, the policy penalizes risky trajectories. Near hazard zones, the CVaR is high, so the advantage becomes negative — the policy avoids these regions entirely, limiting coverage. This is working as designed (risk-averse exploration) but may need tuning for the training phase.

---

## 4. Health Indicators

| Indicator | Status | Notes |
|-----------|--------|-------|
| NaN values | ✅ None | Training is numerically stable |
| Errors/crashes | ✅ None | No Python exceptions |
| Dead critic (vl<0.01) | ⚠️ 15.1% | Occurs in crashed episodes; not a bug |
| Entropy collapse | ✅ Not collapsing | 3.11, well above Phase-1 convergence target (1.5) |
| Gradient explosion | ✅ No signs | KL <0.02 throughout |
| Overfitting | N/A | No validation split; evaluation is post-hoc |
| MPC infeasibility | ⚠️ 4.0% | 682 infeasible solves — manageable but non-zero |
| Binary episode failures | ⚠️ Persisting | ~15% of updates: R=0, cov=0, det=0 |

---

## 5. Comparison: Phase-1 vs Phase-2 at Comparable Stages

| Metric | Phase-1 (Seed 42) at upd 200 | Phase-2 (Seed 42) at upd 186 |
|--------|------------------------------|------------------------------|
| Mean reward | ~350 | ~200-250 |
| Final coverage | ~0.67 (at upd 1000) | ~0.35 (plateaued since upd 75) |
| Value loss | ~0.13 | ~0.31 |
| Entropy | ~1.7 | ~3.11 |
| Time per update | ~4 min | ~31 min (7.8× slower) |
| Converged by | Update 300-500 | Not yet converged |

Phase-1 had **lower entropy and higher coverage** at the same point — suggesting Phase-2 needs significantly more updates due to the richer dynamics.

---

## 6. Recommendation

### DO NOT STOP TRAINING YET

The policy has not converged. Coverage plateaued at 0.35 but entropy remains high (3.11) — the policy is still exploring, and the RISE-MAPPO risk-averse behavior may simply require more samples to learn effective trade-offs in the real-GP environment.

**Continue training to at least update 500** as planned. The Phase-1 convergence occurred at update 300-500, and Phase-2 is 7.8× slower per update, so convergence may not occur until update 300-400 in Phase-2 time.

### Optional: Run parallel evaluation on current checkpoint

Without stopping training:

```bash
# Use CPU-only evaluation (doesn't touch GPU 0)
python scripts/evaluate.py --policy trained \
    --checkpoint results/checkpoints/mappo_upd150.pt \
    --config configs/default.yaml \
    --scenario configs/scenario_simple.yaml \
    --eval-config configs/eval_default.yaml \
    --num-episodes 10 2>&1 | tee eval_phase2_upd150.log
```

This gives a direct performance measurement of the current policy without affecting training. Run at update 200, 250, etc. to track convergence.

### If training stalls past update 400:

1. **Reduce `lambda_risk`** from 0.05 to 0.01 — less conservative CVaR head, more exploration
2. **Increase `entropy_coef`** from 0.01 to 0.02 — encourage broader exploration
3. **Switch to Phase-1 checkpoint as warm-start**: Resume from `results/phase1_seed42/mappo_upd1000.pt` which already has coverage 0.67 and let the GP+MPC fine-tune from a stronger initial policy

---

## 7. Key Metrics at a Glance

```
┌────────────┬──────────┬──────────┬──────────┐
│  Metric    │  Start   │ Current  │  Trend   │
├────────────┼──────────┼──────────┼──────────┤
│  Reward    │    0     │  ~200    │  ↗ slow  │
│  Coverage  │  0.00    │  0.35    │  → flat  │
│  Detection │  0.00    │  0.39    │  ↘ down  │
│  Val Loss  │  89.2    │  0.31    │  → stable│
│  Entropy   │  3.21    │  3.11    │  → stable│
│  KL div    │  0.007   │  0.010   │  → stable│
│  Sec/upd   │  1886    │  2150    │  ↗ slow  │
└────────────┴──────────┴──────────┴──────────┘
```

*Report generated from non-disruptive analysis of `train_phase2_seed42.log` (17,397 lines, 186 updates parsed). Training continues in tmux session `train` — no processes were modified.*
