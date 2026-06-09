# Trigger Design for the LP-Based Allocation Controller

Adopting the per-model replicas+weights LP from ENOVA, but replacing ENOVA's VAE/ELBO anomaly detector with HAS-GPU's lightweight trigger and extending it for heterogeneous LLM routing.

## HAS-GPU's Trigger (Reference)

**Kalman predictor on RPS.** Scalar Kalman filter on observed request rate:

```
R'  = R_{t-1}                  # predicted state (A = 1)
P'  = P_{t-1} + Q              # predicted uncertainty
K   = P' / (P' + D)            # Kalman gain
R   = (1 - K) · R' + K · R_t   # fused estimate
P   = (1 - K) · P'             # updated uncertainty
```

Reduces to a self-tuning EWMA. Q (process noise) and D (observation noise) are the only hyperparameters; K balances them automatically.

**Threshold rules with cooldown.**
- Scale-up:   `R > α · C_f`  (α ≈ 0.8–0.95, sensitivity knob)
- Scale-down: `R < β · C_f`  AND  `R > R_min`  AND  `t - t_last > T_cooldown`

Asymmetric on purpose: react fast to spikes, slowly to dips. Three knobs (α, β, T_cooldown) plus a keep-alive floor R_min.

## Why RPS-Only Doesn't Suffice Here

1. **Heterogeneous prompts** — a 50-token vs 4000-token prompt to the same model have wildly different cost.
2. **Heterogeneous models** — the bandit shifts load between small edge and large cloud models on the same RPS.
3. **Memory is a live constraint** — long-context requests can blow GPU memory at low RPS.

## Extended Trigger

Split signals by their nature:

| Signal                          | Nature                             | Treatment                            |
| ------------------------------- | ---------------------------------- | ------------------------------------ |
| Per-model RPS                   | Noisy time series, future matters  | Kalman + α/β threshold               |
| Per-(model, node) avg latency   | Noisy, deviation from prediction   | Kalman + deviation threshold (k·σ)   |
| Per-node free GPU memory        | Current state                      | Hard threshold                       |
| Per-node queue length           | Mostly redundant with RPS          | Optional hard threshold (safety net) |

Trigger LP re-run on any rule firing, all gated by one cooldown:

```
RERUN_LP  if  any of:
    predicted_RPS^(m)     > α · capacity^(m)                 # HAS-GPU rule
    all predicted_RPS^(m) < β · capacity^(m)  AND cooldown   # HAS-GPU rule
    observed_lat^(m,n)    > predicted_lat^(m,n) + k · σ      # model misbehaving
    free_mem^(n)          < min_required^(n)                 # memory pressure
```

`capacity^(m) = Σ_n (weight_{m,n} / avg_latency_{m,n})` over the model's currently-eligible nodes — all from existing telemetry.

> HAS-GPU's RPS-only trigger suffices for homogeneous serverless functions but is insufficient for heterogeneous LLM routing, where per-request work, model selection, and memory pressure decouple from arrival rate. We adopt HAS-GPU's Kalman + threshold + cooldown design and extend it with per-(model, node) latency-deviation and free-memory checks.

## Implementation

- One scalar Kalman per model
- Per-model capacity computation: reuses `_avg_latency_for_model`
- Per-model RPS: one counter on `_pick_queue_for_model`
- Trigger evaluator
- LP solver call on trigger: independent module

## Knobs

| Symbol      | Meaning                                   | Suggested start |
| ----------- | ----------------------------------------- | --------------- |
| α           | Scale-up threshold (fraction of capacity) | 0.85            |
| β           | Scale-down threshold                      | 0.5             |
| T_cooldown  | Min interval between scale-downs          | 30 s            |
| R_min       | Keep-alive floor on predicted RPS         | 0.1 req/s       |
| k           | Latency-deviation multiplier              | 2.0             |
| Q, D        | Kalman process/observation noise          | 0.01, 1.0       |

All tunable; defaults follow HAS-GPU's reported settings where applicable.

## What's Explicitly Out of Scope

- ENOVA's VAE/ELBO anomaly detector (mentioned in related work as the heavier alternative)
- HAS-GPU's vertical-scaling actuator (SM partitions, MPS) — replaced by the LP's eligibility map
- Learned latency prediction (RaPP) — replaced by the simple per-(model, node) rolling average already in `_avg_latency_for_model`
