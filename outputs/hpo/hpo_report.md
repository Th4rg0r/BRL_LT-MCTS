# Hyperparameter Optimization Report (`LA-MCTS + BRL`)

- **Total Optimization Wall Time**: `7h 19m`
- **Winning Candidate**: `Stage2-Best` (LCB = `+50.36` vs. Baseline `-175.13`)
- **Model Footprint**: `3,599,808` parameters, Peak VRAM `~0.53 GB`
- **Final 10-Seed Submission Run Estimate**: `~93.8 hours` (`3000` episodes × `1000` steps × `10` seeds, `3` parallel workers)

---

## Stage 4 Multi-Seed Finalist Leaderboard

| Rank | Candidate | Robust LCB ($\mu - 0.5\sigma$) | Raw Composite ($\mu \pm \sigma$) | Standard-Ant | OOD Suite Mean | Params | Step Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Stage2-Best` | **+50.36** | +68.89 ± 36.91 | +102.07 | +35.72 | 3.60M | 11.9 ms |
| 2 | `Stage2-RunnerUp` | **+41.20** | +50.68 ± 18.80 | +71.26 | +30.11 | 3.60M | 15.0 ms |
| 3 | `Stage1-Best` | **-2.78** | +17.77 ± 40.92 | +43.95 | -8.42 | 3.73M | 17.3 ms |
| 4 | `Stage3b-Best` | **-34.60** | -22.17 ± 24.70 | -6.06 | -38.28 | 3.60M | 12.3 ms |
| 5 | `Baseline-Default` | **-175.13** | -163.60 ± 22.73 | -143.65 | -183.55 | 4.81M | 29.2 ms |

---

## Blended Hyperparameter Importance Ranking (Prior Tier $\oplus$ PED-ANOVA $\oplus$ fANOVA)

| Hyperparameter | Prior Tier | Prior Weight | PED-ANOVA (Top Quantile) | fANOVA (Global) | Blended Importance | Active in Stage 2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `k_act` | Tier A | 1.00 | 5.4% | 5.4% | **5.2%** | Yes |
| `td_steps` | Tier A | 1.00 | 5.3% | 5.3% | **5.1%** | Yes |
| `batch_size` | Tier A | 1.00 | 2.9% | 2.9% | **3.9%** | Yes |
| `ema_beta_w` | Tier B | 0.55 | 5.1% | 5.1% | **3.9%** | Yes |
| `lambda_rep` | Tier A | 1.00 | 2.9% | 2.9% | **3.9%** | Yes |
| `lambda_r` | Tier B | 0.55 | 4.7% | 4.7% | **3.7%** | Yes |
| `unroll_steps` | Tier A | 1.00 | 2.3% | 2.3% | **3.6%** | Yes |
| `train_every_steps` | Tier A | 1.00 | 2.2% | 2.2% | **3.6%** | Yes |
| `lr` | Tier A | 1.00 | 2.2% | 2.2% | **3.5%** | Yes |
| `n_sim` | Tier A | 1.00 | 2.1% | 2.1% | **3.5%** | Yes |
| `c_1` | Tier B | 0.55 | 4.3% | 4.3% | **3.5%** | Yes |
| `root_tau_beta` | Tier B | 0.55 | 4.0% | 4.0% | **3.3%** | Yes |
| `tau_act_train` | Tier B | 0.55 | 3.8% | 3.8% | **3.3%** | Yes |
| `gamma` | Tier A | 1.00 | 1.5% | 1.5% | **3.2%** | Yes |
| `proj_dim` | Tier C | 0.20 | 5.4% | 5.4% | **3.2%** | Yes |
| `pred_dim` | Tier C | 0.20 | 5.3% | 5.3% | **3.1%** | Yes |
| `lambda_p` | Tier B | 0.55 | 2.9% | 2.9% | **2.8%** | Frozen after S1 |
| `balancer_noise_std` | Tier B | 0.55 | 2.6% | 2.6% | **2.6%** | Frozen after S1 |
| `c_reg` | Tier C | 0.20 | 3.8% | 3.8% | **2.4%** | Frozen after S1 |
| `beta_ent` | Tier C | 0.20 | 3.7% | 3.7% | **2.3%** | Frozen after S1 |
| `reanalyze_every` | Tier C | 0.20 | 3.6% | 3.6% | **2.3%** | Frozen after S1 |
| `model_dim` | Tier B | 0.55 | 1.8% | 1.8% | **2.2%** | Frozen after S1 |
| `inner_tau_beta` | Tier C | 0.20 | 3.4% | 3.4% | **2.2%** | Frozen after S1 |
| `lambda_entropy` | Tier C | 0.20 | 3.4% | 3.4% | **2.2%** | Frozen after S1 |
| `l_dyn` | Tier B | 0.55 | 1.6% | 1.6% | **2.1%** | Frozen after S1 |
| `num_peaks` | Tier B | 0.55 | 1.4% | 1.4% | **2.0%** | Frozen after S1 |
| `n_heads` | Tier B | 0.55 | 1.3% | 1.3% | **2.0%** | Frozen after S1 |
| `alpha_prog` | Tier C | 0.20 | 2.9% | 2.9% | **1.9%** | Frozen after S1 |
| `reanalyze_states` | Tier C | 0.20 | 2.8% | 2.8% | **1.9%** | Frozen after S1 |
| `k_tok` | Tier B | 0.55 | 0.8% | 0.8% | **1.8%** | Frozen after S1 |
| `l_enc` | Tier B | 0.55 | 0.0% | 0.0% | **1.4%** | Frozen after S1 |
| `lambda_v` | Tier B | 0.55 | 0.0% | 0.0% | **1.4%** | Frozen after S1 |
| `tau_s` | Tier B | 0.55 | 0.0% | 0.0% | **1.3%** | Frozen after S1 |
| `buffer_capacity` | Tier C | 0.20 | 1.1% | 1.1% | **1.1%** | Frozen after S1 |
| `alpha_eq` | Tier C | 0.20 | 1.1% | 1.1% | **1.0%** | Frozen after S1 |
| `balancer_lr` | Tier C | 0.20 | 0.8% | 0.8% | **0.9%** | Frozen after S1 |
| `eta_viol` | Tier C | 0.20 | 0.6% | 0.6% | **0.8%** | Frozen after S1 |
| `tweaker_lr` | Tier C | 0.20 | 0.6% | 0.6% | **0.8%** | Frozen after S1 |
| `sigma_zpd` | Tier C | 0.20 | 0.4% | 0.4% | **0.7%** | Frozen after S1 |
| `beta_mse` | Tier C | 0.20 | 0.1% | 0.1% | **0.5%** | Frozen after S1 |
