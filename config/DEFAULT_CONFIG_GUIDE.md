# Balanced Reinforcement Learning & LA-MCTS: Configuration Guide (`default_config.yaml`)

This document provides a comprehensive reference for every configuration parameter in [`config/default_config.yaml`](default_config.yaml). The parameters are grouped into four primary subsystems:
1. **`la_mcts`**: Generic Latent-Action Monte Carlo Tree Search, Transformer World Model, and Categorical Value Critic.
2. **`brl`**: Balanced Reinforcement Learning Tri-Agent Curriculum (Agent, Tweaker, Balancer).
3. **`ant_morph`**: Morphological MuJoCo Ant Environment and Dynamic XML Engine.
4. **`run`**: Experiment Execution, Parallelism, and Hardware Resource Allocation.

---

## 1. Latent-Action MCTS (`la_mcts`)

### 1.1 Architecture & Representation Dimensions

#### `obs_dim`
- **Type**: `int` | **Default**: `27`
- **Purpose**: Observation space dimensionality for the nominal 4-legged Ant ($z$-position, 4D orientation quaternion, 8 joint angles, 6D torso velocities, 8 joint velocities).
- **Training Impact**:
  - Must match the underlying environment's observation vector length.
  - Increasing or mismatching causes shape mismatch errors in the observation projection layer.

#### `max_action_dim`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Number of active continuous control channels for the nominal creature (2 actuators per leg $\times$ 4 legs = 8 torques).
- **Training Impact**:
  - Determines the action output dimensionality of the policy prior and the action input dimension for the dynamics model.
  - If morphology scaling is enabled (e.g. 6-legged Ant with 12 actuators), `max_action_capacity` in `ant_morph` determines total capacity, while `max_action_dim` defines the nominal baseline.

#### `k_tok`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Number of token slices into which the continuous input vector is divided for the Self-Attention Transformer representation network $h_\theta(o, w_{\text{ctx}})$.
- **Training Impact**:
  - **Higher**: Provides finer-grained attention between proprioceptive sub-groups (torso vs. individual limbs). Increases computational cost quadratically ($O(k_{\text{tok}}^2)$ attention matrix).
  - **Lower**: Blends features earlier, reducing expressiveness of spatial relationships across limbs.

#### `d_model`
- **Type**: `int` | **Default**: `256`
- **Purpose**: Internal embedding width of the Transformer representation backbone.
- **Training Impact**:
  - **Higher (e.g. 512)**: Increases capacity for complex multimodal dynamics and contact interactions, but slows down inference and MCTS expansion steps; risks overfitting if replay buffer is small.
  - **Lower (e.g. 128)**: Faster MCTS simulations, but may underfit complex terrain and contact dynamics.

#### `n_heads`
- **Type**: `int` | **Default**: `4`
- **Purpose**: Number of parallel attention heads in multi-head self-attention. Each head has dimension $d_k = d_{\text{model}} / n_{\text{heads}} = 64$.
- **Training Impact**:
  - Allows the agent to jointly attend to information from different representation subspaces (e.g. one head tracking velocity drift, another tracking leg contact phase).
  - Must evenly divide `d_model`.

#### `l_enc`
- **Type**: `int` | **Default**: `3`
- **Purpose**: Number of stacked Transformer encoder layers in the observation representation network $h_\theta$.
- **Training Impact**:
  - **Higher**: Deeper hierarchical reasoning over history and context, but increases latency during MCTS root node evaluation.
  - **Lower**: Minimal inference latency; 3 layers provides an optimal balance between representation depth and fast MCTS simulation throughput.

#### `latent_dim`
- **Type**: `int` | **Default**: `256`
- **Purpose**: Dimensionality of the abstract latent state vector $s \in \mathbb{R}^{\text{latent\_dim}}$ traversed by the MCTS tree search.
- **Training Impact**:
  - All recurrent tree unrolls happen in this latent space without simulating raw MuJoCo physics.
  - Should match `d_model` to avoid bottlenecking information transfer between the representation and dynamics networks.

#### `history_len`
- **Type**: `int` | **Default**: `8`
- **Purpose**: Temporal frame-stack horizon. The Transformer ingests a sliding window of past $(o_{t-H:t}, a_{t-H:t})$ transitions.
- **Training Impact**:
  - **Higher**: Enables robust system identification under noisy or delayed sensors (e.g., estimating leg mass and friction parameters from motion history). Increases memory footprint in replay buffer.
  - **Lower**: Struggles to disambiguate unobserved physical parameters (e.g. slippery ice vs. weak actuators) from instantaneous states.

#### `context_dim`
- **Type**: `int` | **Default**: `18`
- **Purpose**: Dimensionality of the perceived context/tweak vector $w_{\text{ctx}} \in \mathbb{R}^{18}$ appended to the state representation.
- **Training Impact**:
  - Must match `brl.num_params`. Informs the Transformer of current body scale distortions and environmental perturbations.

---

### 1.2 Policy Prior & Latent Action Space

#### `num_peaks`
- **Type**: `int` | **Default**: `6`
- **Purpose**: Number of Gaussian mixture modes in the Latent-Action policy prior network $p_\theta(a \mid s)$.
- **Training Impact**:
  - Locomotion often features multiple discrete gaits (trot, gallop, crawl, turn left, turn right). A multi-modal mixture of Gaussians prevents mode averaging (where the policy would output zero torque trying to average left and right turns).
  - **Higher**: Captures rich multi-gait behaviors; slightly increases policy sampling time.
  - **Lower (e.g. 1)**: Collapses to a single unimodal Gaussian, which can cause policy stagnation in complex obstacle or terrain traversal.

#### `log_std_bounds`
- **Type**: `list[float]` | **Default**: `[-3.0, 2.0]`
- **Purpose**: Clamping bounds $[\log \sigma_{\min}, \log \sigma_{\max}]$ applied to the predicted Gaussian standard deviations ($\sigma \in [e^{-3}, e^2] \approx [0.05, 7.39]$).
- **Training Impact**:
  - **Lower bound (-3.0)**: Prevents the policy covariance from collapsing to zero, which would cause infinite log-likelihood penalties and numerical instability.
  - **Upper bound (2.0)**: Prevents unbounded variance and pure noise action sampling.

---

### 1.3 Value Critic & Return Distribution

#### `use_categorical_critic`
- **Type**: `bool` | **Default**: `true`
- **Purpose**: When `true`, enables a two-hot categorical distributional value critic (similar to MuZero / DreamerV3) instead of scalar mean-squared error (MSE).
- **Training Impact**:
  - **`true` (Recommended)**: Dramatically stabilizes value estimation across varied reward scales and prevents outlier returns from destabilizing gradient updates.
  - **`false`**: Uses scalar MSE regression, which is prone to gradient explosions when the agent falls or transitions between high-reward locomotion and severe fall penalties.

#### `support_min` and `support_max`
- **Type**: `int` | **Default**: `support_min: -300`, `support_max: 300`
- **Purpose**: Discrete value boundaries for the return distribution $[V_{\min}, V_{\max}]$.
- **Training Impact**:
  - Returns falling outside $[support\_min, support\_max]$ are clipped to the boundary bins.
  - Must cover the range of cumulative discounted episode returns ($\gamma=0.99$). For Ant-v5 with step rewards $\approx [-1, +5]$ over an effective horizon of $1/(1-\gamma) = 100$ steps, $[-300, +300]$ provides ample dynamic range.

#### `num_bins`
- **Type**: `int` | **Default**: `601`
- **Purpose**: Number of discrete probability atoms spanning $[support\_min, support\_max]$.
- **Training Impact**:
  - Bin resolution is $\Delta = (300 - (-300)) / (601 - 1) = 1.0$ return units per bin.
  - **Higher**: Finer value granularity at the cost of larger softmax layers.
  - **Lower**: Coarser value predictions; 601 bins provides integer-level precision across the entire trajectory.

#### `gamma`
- **Type**: `float` | **Default**: `0.99`
- **Purpose**: Temporal discount factor $\gamma \in [0, 1)$ for future rewards: $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.
- **Training Impact**:
  - Effective planning horizon is $1 / (1 - \gamma) = 100$ simulation steps ($\approx 3.3$ seconds).
  - **Higher (e.g. 0.995)**: Encourages long-term forward velocity and energy preservation, but increases value variance.
  - **Lower (e.g. 0.95)**: Focuses myopically on immediate balance; may sacrifice forward progress to avoid falling in the next 10 steps.

#### `td_steps`
- **Type**: `int` | **Default**: `5`
- **Purpose**: Multi-step bootstrapping horizon for TD target targets: $z_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k})$.
- **Training Impact**:
  - **Higher (e.g. 10)**: Lower value bias, but higher variance due to environment stochasticity.
  - **Lower (e.g. 1)**: Classic TD(0); low variance but slow propagation of sparse distal rewards. 5 steps is the canonical sweet spot for continuous locomotion.

---

### 1.4 Dynamics & Self-Supervision Loss Weights

#### `l_dyn`
- **Type**: `int` | **Default**: `3`
- **Purpose**: Layer depth of the recurrent latent dynamics MLP $g_\theta(s_t, a_t) \to (\hat{s}_{t+1}, \hat{r}_t)$.
- **Training Impact**:
  - Must accurately model nonlinear contact mechanics, ground collisions, and friction. 3 layers with residual skip connections provides adequate expressive capacity without vanishing gradients across unroll steps.

#### `proj_dim` and `pred_dim`
- **Type**: `int` | **Default**: `proj_dim: 512`, `pred_dim: 256`
- **Purpose**: Dimensions of the non-linear projection and prediction heads used in the self-supervised representation consistency loss (SimSiam / BYOL architecture).
- **Training Impact**:
  - Prevents representation collapse (where all states map to a constant latent vector). The asymmetric predictor head ($pred\_dim < proj\_dim$) guarantees non-trivial latent representations.

#### `unroll_steps`
- **Type**: `int` | **Default**: `5`
- **Purpose**: Number of consecutive future latent steps unrolled during training from replay buffer trajectories.
- **Training Impact**:
  - Forces the dynamics model $g_\theta$ to remain accurate over multi-step trajectory rollouts, preventing compounding error during MCTS planning.
  - **Higher**: Improves long-horizon planning accuracy, but increases GPU memory and backpropagation time.

#### `lambda_rep`, `lambda_v`, `lambda_r`, `lambda_p`, `beta_mse`
- **Type**: `float`
- **Defaults**:
  - `lambda_rep: 2.0`: Weight of the self-supervised latent representation consistency loss.
  - `lambda_v: 0.5`: Weight of the value critic loss.
  - `lambda_r: 1.0`: Weight of the step reward prediction loss.
  - `lambda_p: 1.0`: Weight of the policy prior cross-entropy loss against MCTS search policy $\pi_{\text{mcts}}$.
  - `beta_mse: 1.0`: Value loss scale multiplier.
- **Training Impact**:
  - Balanced multi-task loss objective:
    $$\mathcal{L}_{\text{total}} = \lambda_p \mathcal{L}_{\text{policy}} + \lambda_v \mathcal{L}_{\text{value}} + \lambda_r \mathcal{L}_{\text{reward}} + \lambda_{\text{rep}} \mathcal{L}_{\text{rep}} + c_{\text{reg}} \|\theta\|_2^2$$
  - Increasing `lambda_rep` tightens latent dynamics alignment; increasing `lambda_p` speeds up distillation of MCTS search results into the fast policy prior.

#### `c_reg`
- **Type**: `float` | **Default**: `0.0001` ($10^{-4}$)
- **Purpose**: $L_2$ weight decay regularization penalty applied across network parameters.
- **Training Impact**:
  - Prevents weight explosion in high-dimensional Transformer and MLP layers over 3,000 training episodes.

---

### 1.5 MCTS Tree Search & Action Selection

#### `n_sim` and `n_sim_eval`
- **Type**: `int` | **Default**: `n_sim: 36`, `n_sim_eval: 64`
- **Purpose**: Number of MCTS simulation trajectories executed per environment step during training (`n_sim`) and evaluation (`n_sim_eval`).
- **Training Impact**:
  - **Higher**: Deeper and broader search tree; significantly improves action quality and stability over complex terrain. However, runtime scales linearly with simulation count.
  - **Training (36)**: Selected to maintain wall-clock throughput while giving substantial policy improvement over pure model-free policies.
  - **Eval (64)**: Increased simulation budget during evaluation to extract maximum performance from the learned world model.

#### `k_act`
- **Type**: `int` | **Default**: `24`
- **Purpose**: Number of candidate continuous actions sampled from the policy prior $p(a \mid s)$ when expanding a tree node.
- **Training Impact**:
  - Continuous action spaces cannot be exhaustively expanded. $k_{\text{act}} = 24$ candidates cover diverse gait variations.
  - **Higher**: Discovers better candidate actions at the cost of branching factor and memory.
  - **Lower (e.g. 8)**: Risks missing good recovery actions when the robot is perturbed.

#### `c_1` and `c_2`
- **Type**: `float` | **Default**: `c_1: 1.25`, `c_2: 19652.0`
- **Purpose**: Hyperparameters governing the polynomial upper confidence bound (pUCT) exploration bonus:
  $$U(s, a) = P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot \left(c_1 + \log\left(\frac{\sum_b N(s, b) + c_2 + 1}{c_2}\right)\right)$$
- **Training Impact**:
  - `c_1` sets base exploration weight.
  - `c_2` slowly increases exploration bonus as visit counts grow into the tens of thousands, preventing premature tree convergence.

#### `root_dirichlet_alpha` and `root_exploration_fraction`
- **Type**: `float` | **Default**: `root_dirichlet_alpha: 0.3`, `root_exploration_fraction: 0.25`
- **Purpose**: Injects Dirichlet noise $\text{Dir}(\alpha)$ into the root node policy prior:
  $$P(s_{\text{root}}, a) = (1 - \epsilon) p(a \mid s_{\text{root}}) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$
- **Training Impact**:
  - Ensures the tree search explores alternative branches from the current state even if the policy prior has high confidence in one mode.
  - Crucial during early training to discover locomotion gaits.

#### `root_tau_beta` and `inner_tau_beta`
- **Type**: `float` | **Default**: `root_tau_beta: 1.25`, `inner_tau_beta: 1.0`
- **Purpose**: Softmax temperature applied when sampling continuous actions from candidate proposal distributions at root vs. interior nodes.
- **Training Impact**:
  - Higher temperature at the root ($1.25$) diversifies sampled action candidates; interior nodes ($1.0$) follow tighter policy sampling.

#### `tau_act_train` and `tau_act_eval`
- **Type**: `float` | **Default**: `tau_act_train: 1.0`, `tau_act_eval: 0.05`
- **Purpose**: Temperature used when selecting the executed environment action from root visit counts:
  $$\pi(a \mid s_{\text{root}}) \propto N(s_{\text{root}}, a)^{1 / \tau_{\text{act}}}$$
- **Training Impact**:
  - **Training ($\tau = 1.0$)**: Samples actions stochastically in proportion to visit frequency $N$, providing exploration across similar-value actions.
  - **Evaluation ($\tau = 0.05$)**: Sharpens the distribution to near-deterministic greedy selection (essentially $\arg\max_a N(s, a)$) for benchmark evaluation.

---

### 1.6 Replay Buffer & MuZero Reanalyze

#### `lr`
- **Type**: `float` | **Default**: `0.0003` ($3 \times 10^{-4}$)
- **Purpose**: Adam optimizer learning rate for the Agent networks (Transformer representation, dynamics, value critic, and policy prior).

#### `batch_size`
- **Type**: `int` | **Default**: `128`
- **Purpose**: Minibatch size sampled from the replay buffer per gradient update step.

#### `buffer_capacity`
- **Type**: `int` | **Default**: `50000`
- **Purpose**: Maximum number of step transitions $(o_t, a_t, r_t, o_{t+1}, \pi_t)$ retained in the circular replay buffer.
- **Training Impact**:
  - At 1,000 steps per episode, retains transitions from the most recent ~50–100 episodes.
  - Keeps training focused on recent, high-performing policy data while discarding obsolete early random exploratory steps.

#### `train_every_steps`
- **Type**: `int` | **Default**: `80`
- **Purpose**: Frequency of agent gradient descent updates during episode execution. Every 80 environment steps, one training batch is sampled and backpropagated.
- **Training Impact**:
  - **Lower (e.g. 20)**: High sample efficiency (more updates per environment step), but slows down wall-clock simulation.
  - **Higher (e.g. 200)**: Faster wall-clock execution, but requires more environment interactions to learn.

#### `reanalyze_ratio`, `reanalyze_every`, `reanalyze_states`
- **Type**: `float` / `int` | **Default**: `reanalyze_ratio: 0.25`, `reanalyze_every: 2`, `reanalyze_states: 4`
- **Purpose**: Implements MuZero Reanalyze. Historical trajectories in the replay buffer have their search policies and value targets refreshed using the latest, more capable world model.
  - `reanalyze_every: 2`: Runs reanalyze pass every 2 episodes.
  - `reanalyze_states: 4`: Re-runs MCTS searches on 4 historical states per pass.
  - `reanalyze_ratio: 0.25`: 25% of training minibatches sample reanalyzed fresh targets.
- **Training Impact**:
  - Dramatically improves sample efficiency by reusing past physical interactions with newly improved search policies, preventing policy lag from stale buffer targets.

---

## 2. Balanced Reinforcement Learning (`brl`)

The BRL framework sets up an asymmetric three-agent game:
- **Agent**: Learns to locomote robustly under perturbations.
- **Tweaker**: Selects an adversarial perturbation vector $w \in [0, 1000]^{18}$ subject to knapsack cost constraints.
- **Balancer**: Regulates the environmental perturbation budget $B_e$ to maintain the agent's win rate in the **Zone of Proximal Development (ZPD)** ($\approx 50\%$).

---

### 2.1 Perturbation & Knapsack Parameter Space

#### `num_params`
- **Type**: `int` | **Default**: `18`
- **Purpose**: Dimensionality of the environmental and perceptual tweak vector $w \in \mathbb{R}^{18}$:
  - $w_1..w_3$: Leg geometry scaling (front, rear, lateral).
  - $w_4$: Leg 4 deactivation / amputation.
  - $w_5$: Torso payload / mass scaling.
  - $w_6$: Sliding surface friction (ice simulation).
  - $w_7$: Torsional and rolling friction.
  - $w_8$: Joint damping and stiffness.
  - $w_9$: Actuator motor gain reduction.
  - $w_{10}$: Proprioceptive sensor Gaussian noise.
  - $w_{11}$: Actuation latency / delay.
  - $w_{12}$: Terrain unevenness heightfield.
  - $w_{13}..w_{18}$: Perceptual distortion dimensions (observation shift, scale, and noise).

#### `param_max`
- **Type**: `float` | **Default**: `1000.0`
- **Purpose**: Maximum allowable intensity on each normalized perturbation channel ($w_i \in [0, 1000]$).

#### `budget_max`
- **Type**: `float` | **Default**: `18000.0`
- **Purpose**: Theoretical maximum environmental budget ($18 \times 1000$). The Balancer dynamically allocates a fraction $B_e \in [0, B_{\max}]$ to the Tweaker.

#### `cost_min` and `cost_max`
- **Type**: `float` | **Default**: `cost_min: 0.1`, `cost_max: 10.0`
- **Purpose**: Cost bounds for individual tweak dimensions in the continuous knapsack problem. Different perturbations consume different amounts of budget per unit intensity.

#### `tau_s`
- **Type**: `float` | **Default**: `500.0`
- **Purpose**: Softmax temperature applied to knapsack value-to-cost heuristic ratios during Tweaker perturbation selection.

#### `beta_score` and `eta_viol`
- **Type**: `float` | **Default**: `beta_score: 1.0`, `eta_viol: 2.0`
- **Purpose**: Exponent on the Tweaker's score-to-cost ratio (`beta_score`) and penalty factor applied if the Tweaker's selected tweaks exceed the Balancer's allocated budget (`eta_viol`).

#### `beta_ent`
- **Type**: `float` | **Default**: `0.01`
- **Purpose**: Entropy regularization weight for the Tweaker policy, ensuring diverse exploration across different failure dimensions rather than repeatedly exploiting only one weakness.

---

### 2.2 Equilibrium, Win Criteria & ZPD Tracking

#### `target_win_rate`
- **Type**: `float` | **Default**: `0.50`
- **Purpose**: The central equilibrium target $\mu^* = 0.50$ for the Zone of Proximal Development.
- **Training Impact**:
  - In educational psychology and optimal challenge curriculum learning, learning is fastest when tasks are neither trivially easy (win rate $\approx 1.0$) nor impossibly hard (win rate $\approx 0.0$). The Balancer dynamically adjusts task difficulty to keep the agent succeeding roughly 50% of the time.

#### `ema_beta_w`
- **Type**: `float` | **Default**: `0.05`
- **Purpose**: Exponential moving average (EMA) smoothing weight for the agent's tracked win rate:
  $$\mu_t = (1 - \beta_w) \mu_{t-1} + \beta_w \cdot \mathbf{1}_{\text{win}}$$
- **Training Impact**:
  - $\beta_w = 0.05$ creates an effective smoothing window of $1 / 0.05 = 20$ episodes. Prevents erratic budget fluctuations caused by single-episode luck.

#### `initial_r_ref` and `benchmark_r_ref`
- **Type**: `float` | **Default**: `initial_r_ref: 1000.0`, `benchmark_r_ref: 6169.54`
- **Purpose**: Dynamic reference return $R_{\text{ref}}$ used to define an episode "win":
  $$\text{Episode Win} \iff \text{Return } R \ge R_{\text{ref}}$$
- **Training Impact**:
  - `initial_r_ref = 1000.0`: Starting threshold for early training.
  - `benchmark_r_ref = 6169.54`: Nominal asymptotic performance benchmark for Ant-v5.
  - As the agent improves, $R_{\text{ref}}$ dynamically ratchets upward via quantile progression, preventing the agent from "winning" through mediocre performance.

#### `window_k_w`
- **Type**: `int` | **Default**: `20`
- **Purpose**: Sliding window history size (in episodes) used to compute recent failure correlations and agent win rate statistics.

#### `sigma_zpd`
- **Type**: `float` | **Default**: `0.15`
- **Purpose**: Standard deviation of the Zone of Proximal Development Gaussian bell curve centered at $\mu^* = 0.50$.
- **Training Impact**:
  - Defines the "sweet spot" bandwidth $[0.50 - \sigma, 0.50 + \sigma] = [0.35, 0.65]$. Win rates within this band receive high Balancer equilibrium rewards.

---

### 2.3 Balancer Meta-Controller (DDPG)

#### `balancer_lr`
- **Type**: `float` | **Default**: `0.0001` ($10^{-4}$)
- **Purpose**: Learning rate for the Balancer's DDPG actor and critic networks.

#### `balancer_tau`
- **Type**: `float` | **Default**: `0.005`
- **Purpose**: Polyak averaging soft update coefficient for target networks: $\theta_{\text{targ}} \leftarrow \tau \theta + (1-\tau)\theta_{\text{targ}}$.

#### `balancer_noise_std`
- **Type**: `float` | **Default**: `0.05`
- **Purpose**: Gaussian exploration noise $\mathcal{N}(0, 0.05)$ added to the Balancer's continuous budget action $B_e$.

#### `alpha_eq`, `alpha_inst`, `alpha_prog`, `lambda_entropy`
- **Type**: `float`
- **Defaults**:
  - `alpha_eq: 10.0`: Reward weight for keeping win rate near target: $-\alpha_{\text{eq}} (\mu - 0.5)^2$.
  - `alpha_inst: 2.0`: Penalty weight for abrupt win rate volatility: $-\alpha_{\text{inst}} |\mu_t - \mu_{t-1}|$.
  - `alpha_prog: 5.0`: Reward weight for raising the reference return $R_{\text{ref}}$ over time: $+\alpha_{\text{prog}} \Delta R_{\text{ref}}$.
  - `lambda_entropy: 0.5`: Entropy incentive for budget exploration.
- **Training Impact**:
  - Balancer reward function:
    $$r_{\text{bal}} = -\alpha_{\text{eq}} (\mu - 0.5)^2 - \alpha_{\text{inst}} |\Delta \mu| + \alpha_{\text{prog}} \Delta R_{\text{ref}} + \lambda_{\text{ent}} \mathcal{H}$$
  - Balances curriculum stability with continuous pressure to improve.

---

### 2.4 Curriculum Schedule & Warmup

#### `warmup_episodes`
- **Type**: `int` | **Default**: `50`
- **Purpose**: Number of initial episodes where environmental perturbations are strictly clamped.
- **Training Impact**:
  - Allows the agent to learn basic upright posture and stepping before facing severe ice, missing legs, or heavy payloads.
  - Setting this to 0 can cause premature policy collapse as untrained agents fail continuously under high perturbations.

#### `warmup_limit` and `warmup_budget`
- **Type**: `float` | **Default**: `warmup_limit: 50.0`, `warmup_budget: 100.0`
- **Purpose**: Caps on individual tweak intensity ($w_i \le 50.0$) and total budget ($B_e \le 100.0$) during the warmup phase.

#### `brl_episodes` and `total_episodes`
- **Type**: `int` | **Default**: `brl_episodes: 2500`, `total_episodes: 3000`
- **Purpose**:
  - `warmup_episodes`: 0 – 50
  - `brl_episodes`: 51 – 2550 (active tri-agent curriculum learning)
  - Remaining episodes: Consolidation / fine-tuning phase.

---

## 3. Ant Morphology Environment (`ant_morph`)

#### `env_id`
- **Type**: `str` | **Default**: `"Ant-v5"`
- **Purpose**: Base Gymnasium environment identifier.

#### `use_contact_forces`
- **Type**: `bool` | **Default**: `false`
- **Purpose**: Whether to include external contact force sensors (`cfrc_ext`, dimension $1 + 4 \times 3 \times 6$) in the observation vector.
- **Training Impact**:
  - When `true`, observation space expands significantly. Requires contact penalty cost adjustments. Keeping `false` aligns with standard benchmark benchmarks.

#### `max_episode_steps`
- **Type**: `int` | **Default**: `1000`
- **Purpose**: Truncation step horizon for each training episode ($\approx 33.3$ seconds at $\Delta t = 0.05$s).
- **Training Impact**:
  - Episodes reaching 1,000 steps without falling receive the `[TRUNCATED - MOVING]` designation and maximize return.

#### `xml_rebuild_threshold`
- **Type**: `float` | **Default**: `1.0`
- **Purpose**: Sensitivity threshold for triggering dynamic MJCF XML recompilation when morphological tweaks ($w_1..w_4$) or leg counts change.
- **Training Impact**:
  - Recompiling MuJoCo XML requires a few milliseconds. The threshold prevents tiny floating-point changes from triggering unnecessary model rebuilds while ensuring substantial morphology changes (e.g. tripod or hexapod) are compiled cleanly.

#### `healthy_z_range`
- **Type**: `list[float]` | **Default**: `[0.2, 1.0]`
- **Purpose**: Allowed vertical torso height $[z_{\min}, z_{\max}]$. If the torso drops below $0.2$m (falling over) or exceeds $1.0$m (flipping into the air), the episode terminates early as a fall.

#### `max_action_capacity`
- **Type**: `int` | **Default**: `12`
- **Purpose**: Maximum action buffer capacity allocated to support morphology scaling (e.g. up to 6 legs with 2 actuators each = 12 channels).

#### `default_num_legs`
- **Type**: `int` | **Default**: `4`
- **Purpose**: Default number of legs for the nominal creature.

---

## 4. Run Execution & Infrastructure (`run`)

#### `seeds` and `seed_offset`
- **Type**: `int` | **Default**: `seeds: 3`, `seed_offset: 42`
- **Purpose**: Number of independent training runs to execute and the starting seed index (runs seeds `42`, `43`, `44`).
- **Training Impact**:
  - Provides statistical rigor by verifying policy convergence across multiple stochastic seeds.

#### `episodes` and `max_steps`
- **Type**: `int` | **Default**: `0` (automatic)
- **Purpose**: Execution overrides:
  - `episodes: 0` $\implies$ uses `brl.total_episodes` (3000).
  - `max_steps: 0` $\implies$ uses `ant_morph.max_episode_steps` (1000).

#### `device`
- **Type**: `str` | **Default**: `"cuda"`
- **Purpose**: PyTorch device for neural network tensors (`"cuda"` or `"cpu"`).
- **Training Impact**:
  - CUDA dramatically accelerates Transformer representation and MCTS batched evaluations.

#### `eval_interval`
- **Type**: `int` | **Default**: `0` (automatic, every 10 episodes)
- **Purpose**: Frequency of benchmark evaluation runs across all 5 standard validation scenarios (Standard, Tripod, Spider, Heavy-Ice, Hexapod).

#### `checkpoint_interval`
- **Type**: `int` | **Default**: `500`
- **Purpose**: Saves permanent milestone checkpoints every 500 episodes (e.g. `checkpoint_ep_0500.pt`, `checkpoint_ep_1000.pt`), in addition to `latest_checkpoint.pt` and `best_model.pt`.

#### `parallel_seeds`
- **Type**: `int` | **Default**: `3`
- **Purpose**: Number of concurrent Python multiprocessing workers running independent seeds.
- **Training Impact**:
  - When set to 3, seeds 42, 43, and 44 train simultaneously across available CPU cores and GPU memory, reducing wall-clock experiment time by ~3x.

#### `gpu_mem_fraction`
- **Type**: `float` | **Default**: `0.0`
- **Purpose**: Per-process CUDA memory allocation limit fraction. `0.0` allows PyTorch to dynamically manage CUDA memory allocation.

---

## Quick Reference: Tuning Guide for Common Scenarios

| Objective | Key Parameters to Adjust | Recommended Change |
| :--- | :--- | :--- |
| **Speed up wall-clock training** | `n_sim`, `train_every_steps`, `history_len` | Decrease `n_sim` (e.g. 36 $\to$ 20), increase `train_every_steps` (80 $\to$ 120). |
| **Improve complex terrain locomotion** | `n_sim`, `k_act`, `unroll_steps` | Increase `n_sim` (36 $\to$ 50), increase `k_act` (24 $\to$ 32), increase `unroll_steps` (5 $\to$ 8). |
| **Prevent early falls / policy collapse** | `warmup_episodes`, `target_win_rate`, `alpha_eq` | Increase `warmup_episodes` (50 $\to$ 100), verify `target_win_rate = 0.50`. |
| **Encourage more aggressive gait discovery** | `num_peaks`, `root_exploration_fraction`, `tau_act_train` | Increase `num_peaks` (6 $\to$ 8), increase `root_exploration_fraction` (0.25 $\to$ 0.35). |
| **Fine-tune evaluation performance** | `n_sim_eval`, `tau_act_eval` | Increase `n_sim_eval` (64 $\to$ 128), lower `tau_act_eval` (0.05 $\to$ 0.01). |
