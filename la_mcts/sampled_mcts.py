"""Sampled Monte Carlo Tree Search (Sampled PUCT) over Continuous Action Samples."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from la_mcts.backbone import combine_cumulative_rep
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer


class MCTSNode:
    """Node in the Sampled Continuous MCTS search tree."""

    __slots__ = (
        "z_state",
        "z_cum_state",
        "value_prefix",
        "sampled_actions",
        "prior_weights",
        "visit_counts",
        "virtual_counts",
        "q_values",
        "rewards",
        "children",
        "total_visits",
        "v_value",
    )

    def __init__(
        self,
        z_state: torch.Tensor,
        value_prefix: float = 0.0,
        sampled_actions: Optional[torch.Tensor] = None,
        prior_weights: Optional[np.ndarray] = None,
        v_value: float = 0.0,
        z_cum_state: Optional[torch.Tensor] = None,
    ) -> None:
        self.z_state: torch.Tensor = z_state
        self.z_cum_state: torch.Tensor = z_cum_state if z_cum_state is not None else z_state
        self.value_prefix: float = float(value_prefix)
        self.v_value: float = float(v_value)

        if sampled_actions is not None and prior_weights is not None:
            k_act = sampled_actions.shape[0]
            self.sampled_actions: torch.Tensor = sampled_actions
            self.prior_weights: np.ndarray = prior_weights.astype(np.float64, copy=False)
        else:
            k_act = 0
            self.sampled_actions = torch.empty(0)
            self.prior_weights = np.empty(0, dtype=np.float64)

        self.visit_counts: np.ndarray = np.zeros(k_act, dtype=np.int32)
        self.virtual_counts: np.ndarray = np.zeros(k_act, dtype=np.int32)
        self.q_values: np.ndarray = np.zeros(k_act, dtype=np.float64)
        self.rewards: np.ndarray = np.zeros(k_act, dtype=np.float64)
        self.children: Dict[int, MCTSNode] = {}
        self.total_visits: int = 0


class MinMaxStats:
    """Tracks tree-wide minimum and maximum Q-values for [0, 1] PUCT normalization."""

    __slots__ = ("minimum", "maximum", "eps")

    def __init__(self, eps: float = 1e-6) -> None:
        self.minimum: float = float("inf")
        self.maximum: float = float("-inf")
        self.eps: float = eps

    def update(self, value: float) -> None:
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if self.maximum > self.minimum:
            return (values - self.minimum) / (self.maximum - self.minimum + self.eps)
        return values


class SampledContinuousMCTS:
    """Sampled MCTS Engine over K_act continuous actions per node via Sampled PUCT."""

    def __init__(
        self,
        actor_head: MultiPeakGMMActorHead,
        critic_head: SupportCriticHead,
        rep_trainer: ModelRepresentationTrainer,
        n_sim: int = 50,
        k_act: int = 16,
        gamma: float = 0.99,
        c_1: float = 1.25,
        c_2: float = 19652.0,
        root_tau_beta: float = 1.25,
        inner_tau_beta: float = 1.0,
        tau_act: float = 1.0,
        batch_leaf_eval: int = 16,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
        latent_ema_decay: float = 0.4,
    ) -> None:
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.rep_trainer = rep_trainer
        self.n_sim = n_sim
        self.k_act = k_act
        self.gamma = gamma
        self.c_1 = c_1
        self.c_2 = c_2
        self.root_tau_beta = root_tau_beta
        self.inner_tau_beta = inner_tau_beta
        self.tau_act = tau_act
        self.batch_leaf_eval = max(1, batch_leaf_eval)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_exploration_fraction = float(root_exploration_fraction)
        self.latent_ema_decay = float(latent_ema_decay)

    def _select_action_idx(self, node: MCTSNode, min_max_stats: MinMaxStats) -> int:
        """Selects action index j* in {0, ..., K_act-1} maximizing Sampled PUCT."""
        eff_counts = node.visit_counts + node.virtual_counts
        sum_n = float(node.total_visits)
        c_puct = self.c_1 + math.log((sum_n + self.c_2 + 1.0) / self.c_2)
        sqrt_sum_n = math.sqrt(max(sum_n, 1e-8))

        # For unvisited edges (N == 0), initialize Q to the node's value estimate
        q_raw = np.where(node.visit_counts > 0, node.q_values, node.v_value)
        q_norm = min_max_stats.normalize(q_raw)

        u_score = c_puct * node.prior_weights * (sqrt_sum_n / (1.0 + eff_counts))
        # Normalized virtual loss penalty ensures parallel batch trajectories explore distinct edges
        scores = q_norm + u_score - 1.5 * node.virtual_counts
        return int(np.argmax(scores))

    def _expand_node(
        self,
        z_state: torch.Tensor,
        temperature: float,
        value_prefix: float = 0.0,
        action_mask: Optional[torch.Tensor] = None,
        limb_kinematics: Optional[torch.Tensor] = None,
        z_cum_state: Optional[torch.Tensor] = None,
    ) -> MCTSNode:
        """Evaluates actor and critic heads on cumulative latent state to initialize an MCTSNode."""
        cum_state = z_cum_state if z_cum_state is not None else z_state
        sampled_actions, prior_weights_t = self.actor_head.sample_actions(
            cum_state,
            k_act=self.k_act,
            temperature=temperature,
            action_mask=action_mask,
            limb_kinematics=limb_kinematics,
        )
        v_val = float(self.critic_head(cum_state.unsqueeze(0)).squeeze().item())
        return MCTSNode(
            z_state=z_state,
            value_prefix=value_prefix,
            sampled_actions=sampled_actions,
            prior_weights=prior_weights_t.detach().cpu().numpy(),
            v_value=v_val,
            z_cum_state=cum_state,
        )

    def search(
        self,
        z_root: torch.Tensor,
        deterministic: bool = False,
        tau_act: Optional[float] = None,
        n_sim: Optional[int] = None,
        action_mask: Optional[torch.Tensor] = None,
        limb_kinematics: Optional[torch.Tensor] = None,
        root_tau_beta: Optional[float] = None,
        root_exploration_fraction: Optional[float] = None,
        prev_cum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, MCTSNode]:
        """Executes N_sim MCTS simulations from z_root using virtual-loss leaf batching.

        Dynamics transitions map cur_rep -> next_rep (z_state) without cumulative smearing;
        cumulative representations (z_cum_state) are computed on demand for actor/critic readout.

        Returns:
            chosen_action: Selected continuous action vector in [-1, 1]^N.
            policy_weights: Normalized root visit distribution pi^MCTS in Delta^{K_act}.
            root_sampled_actions: (K_act, N) candidate actions at the root.
            root_value: Root state value estimate V_MCTS(z_root).
            root_node: Expanded root MCTSNode.
        """
        if z_root.dim() == 2:
            z_root = z_root.squeeze(0)
        if prev_cum_state is not None and prev_cum_state.dim() == 2:
            prev_cum_state = prev_cum_state.squeeze(0)

        num_simulations = n_sim if n_sim is not None else self.n_sim
        temp_act = (0.0 if deterministic else self.tau_act) if tau_act is None else tau_act
        eff_root_tau_beta = self.root_tau_beta if root_tau_beta is None else float(root_tau_beta)
        eff_root_exp = (
            self.root_exploration_fraction
            if root_exploration_fraction is None
            else float(root_exploration_fraction)
        )

        with torch.inference_mode():
            min_max_stats = MinMaxStats()
            z_cum_root = combine_cumulative_rep(
                z_root, prev_cum_state, alpha=self.latent_ema_decay
            )
            root = self._expand_node(
                z_root,
                temperature=eff_root_tau_beta,
                value_prefix=0.0,
                action_mask=action_mask,
                limb_kinematics=limb_kinematics,
                z_cum_state=z_cum_root,
            )
            if not deterministic and eff_root_exp > 0.0:
                noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.k_act)
                root.prior_weights = (
                    (1.0 - eff_root_exp) * root.prior_weights
                    + eff_root_exp * noise
                ).astype(np.float64)
            min_max_stats.update(root.v_value)

            # Virtual-loss batched simulation loop for < 15 ms latency at N_sim=50
            sims_done = 0

            while sims_done < num_simulations:
                batch_limit = min(self.batch_leaf_eval, num_simulations - sims_done)
                pending_paths: List[List[Tuple[MCTSNode, int]]] = []
                parent_states: List[torch.Tensor] = []
                parent_cum_states: List[torch.Tensor] = []
                chosen_actions: List[torch.Tensor] = []
                seen_leaves: set[Tuple[int, int]] = set()

                for _ in range(batch_limit):
                    node = root
                    search_path: List[Tuple[MCTSNode, int]] = []

                    while True:
                        action_idx = self._select_action_idx(node, min_max_stats)
                        search_path.append((node, action_idx))

                        # Apply temporary virtual visit to diversify parallel traversals
                        node.virtual_counts[action_idx] += 1
                        node.total_visits += 1

                        if action_idx in node.children:
                            node = node.children[action_idx]
                        else:
                            leaf_key = (id(node), action_idx)
                            if leaf_key in seen_leaves:
                                # Revert virtual loss on duplicate leaf and break batch collection
                                for p_node, p_idx in reversed(search_path):
                                    p_node.virtual_counts[p_idx] -= 1
                                    p_node.total_visits -= 1
                                search_path = []
                            else:
                                seen_leaves.add(leaf_key)
                            break

                    if not search_path:
                        break

                    parent_node, edge_idx = search_path[-1]
                    pending_paths.append(search_path)
                    parent_states.append(parent_node.z_state)
                    parent_cum_states.append(parent_node.z_cum_state)
                    chosen_actions.append(parent_node.sampled_actions[edge_idx])

                if not pending_paths:
                    break

                # Vectorized transition cur_rep -> next_rep, plus on-demand next_cum_rep for Actor/Critic
                z_batch = torch.stack(parent_states, dim=0)
                z_cum_parent_batch = torch.stack(parent_cum_states, dim=0)
                a_batch = torch.stack(chosen_actions, dim=0)
                z_next_batch, r_hat_batch = self.rep_trainer(z_batch, a_batch)
                z_cum_next_batch = combine_cumulative_rep(
                    z_next_batch,
                    z_cum_parent_batch,
                    alpha=self.latent_ema_decay,
                    assume_nonzero=True,
                )
                v_next_batch = self.critic_head(z_cum_next_batch).squeeze(-1)
                sampled_acts_batch, prior_weights_batch = self.actor_head.sample_actions(
                    z_cum_next_batch,
                    k_act=self.k_act,
                    temperature=self.inner_tau_beta,
                    action_mask=action_mask,
                    limb_kinematics=limb_kinematics,
                )

                r_np = r_hat_batch.squeeze(-1).detach().cpu().numpy()
                v_np = v_next_batch.detach().cpu().numpy()
                priors_np = prior_weights_batch.detach().cpu().numpy()

                if r_np.ndim == 0:
                    r_np = np.array([float(r_np)])
                    v_np = np.array([float(v_np)])
                    priors_np = priors_np[np.newaxis, :]

                for b_idx, search_path in enumerate(pending_paths):
                    parent_node, edge_idx = search_path[-1]
                    r_val = float(r_np[b_idx])
                    v_val = float(v_np[b_idx])
                    parent_node.rewards[edge_idx] = r_val

                    child_node = MCTSNode(
                        z_state=z_next_batch[b_idx],
                        value_prefix=parent_node.value_prefix + r_val,
                        sampled_actions=sampled_acts_batch[b_idx],
                        prior_weights=priors_np[b_idx],
                        v_value=v_val,
                        z_cum_state=z_cum_next_batch[b_idx],
                    )
                    parent_node.children[edge_idx] = child_node

                    # Backpropagate bootstrapped L-k return and convert virtual visit to real visit
                    g_return = v_val
                    for node_k, idx_k in reversed(search_path):
                        g_return = float(node_k.rewards[idx_k]) + self.gamma * g_return
                        node_k.virtual_counts[idx_k] -= 1
                        n_prev = node_k.visit_counts[idx_k]
                        node_k.q_values[idx_k] = (
                            n_prev * node_k.q_values[idx_k] + g_return
                        ) / float(n_prev + 1)
                        node_k.visit_counts[idx_k] = n_prev + 1
                        min_max_stats.update(float(node_k.q_values[idx_k]))

                sims_done += len(pending_paths)

            # Compute normalized root visit distribution pi_t^MCTS
            counts_f = root.visit_counts.astype(np.float64)
            if deterministic or temp_act <= 1e-3:
                best_idx = int(np.argmax(counts_f))
                # Soft policy weights still reflect empirical visit fractions for training targets
                total_c = np.sum(counts_f)
                if total_c > 0:
                    policy_weights = counts_f / total_c
                else:
                    policy_weights = np.full(self.k_act, 1.0 / self.k_act, dtype=np.float64)
                chosen_idx = best_idx
            else:
                powered = np.power(counts_f, 1.0 / temp_act)
                sum_pow = float(np.sum(powered))
                if sum_pow > 0.0:
                    policy_weights = powered / sum_pow
                else:
                    policy_weights = np.full(self.k_act, 1.0 / self.k_act, dtype=np.float64)
                # Ensure exact normalization within floating-point precision
                policy_weights = policy_weights / np.sum(policy_weights)
                chosen_idx = int(np.random.choice(self.k_act, p=policy_weights))

            root_actions_np = root.sampled_actions.detach().cpu().numpy()
            chosen_action = root_actions_np[chosen_idx]
            visited_mask = root.visit_counts > 0
            if np.any(visited_mask):
                root_value = float(
                    np.sum(root.q_values[visited_mask] * root.visit_counts[visited_mask])
                    / np.sum(root.visit_counts[visited_mask])
                )
            else:
                root_value = root.v_value

            return (
                chosen_action.astype(np.float32),
                policy_weights.astype(np.float32),
                root_actions_np.astype(np.float32),
                root_value,
                root,
            )

    def search_batch(
        self,
        root_z_batch: torch.Tensor,
        deterministic: bool = False,
        tau_act: Optional[float] = None,
        n_sim: Optional[int] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Executes MCTS search across B_env parallel environments simultaneously."""
        bsz = root_z_batch.shape[0]
        num_simulations = n_sim if n_sim is not None else self.n_sim
        temp_act = (0.0 if deterministic else self.tau_act) if tau_act is None else tau_act

        with torch.inference_mode():
            sampled_acts_b, priors_b = self.actor_head.sample_actions(
                root_z_batch,
                k_act=self.k_act,
                temperature=self.root_tau_beta,
                action_mask=action_mask,
            )
            v_roots = self.critic_head(root_z_batch).squeeze(-1).detach().cpu().numpy()
            if v_roots.ndim == 0:
                v_roots = np.array([float(v_roots)])
            priors_np = priors_b.detach().cpu().numpy()

            roots: List[MCTSNode] = []
            stats_list: List[MinMaxStats] = []
            for b in range(bsz):
                st = MinMaxStats()
                v_b = float(v_roots[b])
                st.update(v_b)
                p_weights = priors_np[b].copy()
                if not deterministic and self.root_exploration_fraction > 0.0:
                    noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.k_act)
                    p_weights = (
                        (1.0 - self.root_exploration_fraction) * p_weights
                        + self.root_exploration_fraction * noise
                    ).astype(np.float64)
                node = MCTSNode(
                    z_state=root_z_batch[b],
                    value_prefix=0.0,
                    sampled_actions=sampled_acts_b[b],
                    prior_weights=p_weights,
                    v_value=v_b,
                )
                roots.append(node)
                stats_list.append(st)

            for _ in range(num_simulations):
                paths_per_env: List[List[Tuple[MCTSNode, int]]] = []
                parent_states: List[torch.Tensor] = []
                chosen_actions: List[torch.Tensor] = []

                for b in range(bsz):
                    node = roots[b]
                    path: List[Tuple[MCTSNode, int]] = []
                    while True:
                        a_idx = self._select_action_idx(node, stats_list[b])
                        path.append((node, a_idx))
                        if a_idx in node.children:
                            node = node.children[a_idx]
                        else:
                            break
                    parent_node, edge_idx = path[-1]
                    paths_per_env.append(path)
                    parent_states.append(parent_node.z_state)
                    chosen_actions.append(parent_node.sampled_actions[edge_idx])

                z_batch = torch.stack(parent_states, dim=0)
                a_batch = torch.stack(chosen_actions, dim=0)
                z_next_b, r_hat_b = self.rep_trainer(z_batch, a_batch)
                v_next_b = self.critic_head(z_next_b).squeeze(-1)
                c_acts_b, c_priors_b = self.actor_head.sample_actions(
                    z_next_b,
                    k_act=self.k_act,
                    temperature=self.inner_tau_beta,
                    action_mask=action_mask,
                )

                r_np = r_hat_b.squeeze(-1).detach().cpu().numpy()
                v_np = v_next_b.detach().cpu().numpy()
                cp_np = c_priors_b.detach().cpu().numpy()
                if r_np.ndim == 0:
                    r_np = np.array([float(r_np)])
                    v_np = np.array([float(v_np)])
                    cp_np = cp_np[np.newaxis, :]

                for b in range(bsz):
                    path = paths_per_env[b]
                    parent_node, edge_idx = path[-1]
                    r_val = float(r_np[b])
                    v_val = float(v_np[b])
                    parent_node.rewards[edge_idx] = r_val
                    child = MCTSNode(
                        z_state=z_next_b[b],
                        value_prefix=parent_node.value_prefix + r_val,
                        sampled_actions=c_acts_b[b],
                        prior_weights=cp_np[b],
                        v_value=v_val,
                    )
                    parent_node.children[edge_idx] = child

                    g_ret = v_val
                    for node_k, idx_k in reversed(path):
                        g_ret = float(node_k.rewards[idx_k]) + self.gamma * g_ret
                        n_prev = node_k.visit_counts[idx_k]
                        node_k.q_values[idx_k] = (
                            n_prev * node_k.q_values[idx_k] + g_ret
                        ) / float(n_prev + 1)
                        node_k.visit_counts[idx_k] = n_prev + 1
                        node_k.total_visits += 1
                        stats_list[b].update(float(node_k.q_values[idx_k]))

            chosen_actions_out = []
            policy_weights_out = []
            root_samples_out = []
            root_values_out = []

            for b in range(bsz):
                root = roots[b]
                counts_f = root.visit_counts.astype(np.float64)
                root_acts_np = root.sampled_actions.detach().cpu().numpy()
                if deterministic or temp_act <= 1e-3:
                    c_idx = int(np.argmax(counts_f))
                    pw = counts_f / max(np.sum(counts_f), 1.0)
                else:
                    powered = np.power(counts_f, 1.0 / temp_act)
                    pw = powered / np.sum(powered)
                    c_idx = int(np.random.choice(self.k_act, p=pw))

                chosen_actions_out.append(root_acts_np[c_idx])
                policy_weights_out.append(pw)
                root_samples_out.append(root_acts_np)
                vis = root.visit_counts > 0
                rv = (
                    float(
                        np.sum(root.q_values[vis] * root.visit_counts[vis])
                        / np.sum(root.visit_counts[vis])
                    )
                    if np.any(vis)
                    else root.v_value
                )
                root_values_out.append(rv)

            return (
                np.stack(chosen_actions_out, axis=0).astype(np.float32),
                np.stack(policy_weights_out, axis=0).astype(np.float32),
                np.stack(root_samples_out, axis=0).astype(np.float32),
                np.array(root_values_out, dtype=np.float32),
            )
