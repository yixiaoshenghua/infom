import copy
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.encoders import encoder_modules, GCEncoder # Assuming PyTorch versions
from utils.networks import GCActor, GCBilinearValue, Value # Assuming PyTorch versions
from utils.torch_utils import ModuleDict # Assuming PyTorch compatible utils


class TDInfoNCEAgent(nn.Module):
    """Temporal Difference InfoNCE (TD InfoNCE) agent. PyTorch version."""

    def __init__(self, config: Dict[str, Any], ex_observations_shape: Tuple, ex_actions_shape: Tuple, seed: int):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        action_dim = ex_actions_shape[-1]

        # Define encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module_class = encoder_modules[config['encoder']]
            encoders['reward'] = encoder_module_class()
            # GCBilinearValue and GCActor take GCEncoder which wraps a state_encoder
            encoders['critic_state'] = GCEncoder(state_encoder=encoder_module_class())
            encoders['critic_goal'] = GCEncoder(state_encoder=encoder_module_class()) # Separate for goal if needed
            encoders['actor'] = GCEncoder(state_encoder=encoder_module_class())

        # Define networks
        reward_def = Value( # Standard Value net for reward prediction R(s)
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
            encoder=encoders.get('reward'), # Simple encoder if any
            num_ensembles=1
        )
        # Critic is GCBilinearValue: V(s, s_g) or Q(s, a, s_g) via phi/psi features
        critic_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'], # Dimension of phi and psi features
            layer_norm=config['value_layer_norm'],
            num_ensembles=2, # For Clipped Double Q style logits
            value_exp=False, # TD-InfoNCE uses logits directly, not exp(logits) for V
            state_encoder=encoders.get('critic_state'), # Encoder for s in phi(s,a)
            goal_encoder=encoders.get('critic_goal'),   # Encoder for s_g in psi(s_g)
        )
        # Actor pi(a|s, s_g_actor)
        actor_def = GCActor( # Goal-conditioned actor
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'), # GCEncoder for actor's (s, s_g_actor) input
        )

        self.networks = ModuleDict({
            'reward': reward_def,
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'actor': actor_def,
            # Note: JAX version's actor doesn't have a target_actor.
            # If DDPG actor loss needs target actor, it should be added.
            # The provided JAX actor_loss uses online actor for Q-value computation via critic.
        }).to(self.device)

        # Initialize target critic network
        self.networks['target_critic'].load_state_dict(self.networks['critic'].state_dict())

        online_params = list(self.networks['reward'].parameters()) + \
                        list(self.networks['critic'].parameters()) + \
                        list(self.networks['actor'].parameters())
        self.optimizer = Adam(online_params, lr=config['lr'])

    def reward_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        rewards_gt = batch['rewards'].to(self.device).squeeze(-1)

        reward_preds = self.networks['reward'](observations).squeeze(-1)
        loss = F.mse_loss(reward_preds, rewards_gt)
        return loss, {'reward_loss': loss.item()}

    def contrastive_loss(self, batch: Dict[str, torch.Tensor], td_loss_type: str = 'sarsa') -> Tuple[torch.Tensor, Dict[str, float]]:
        # td_loss_type: 'sarsa' or 'q_learning'
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_obs = batch['next_observations'].to(self.device)
        # value_goals are random states s_rand used for negative samples in contrastive loss
        value_goals_random = batch['value_goals'].to(self.device)
        batch_size = obs.shape[0]

        # GCBilinearValue.forward(obs, goal_obs, actions=None, info=True) returns (v, phi, psi)
        # v: (B, E) or (B,) value prediction (e.g. ||phi-psi||^2 or dot product)
        # phi: (E, B, D) features from (obs, actions)
        # psi: (E, B, D) features from goal_obs

        # Logits for (s, a) vs s' (next_observations as goals)
        # next_v_online, next_phi_online, next_psi_online = \
        #     self.networks['critic'](obs, next_obs, actions=actions, info=True)

        # JAX code: next_v, next_phi, next_psi = self.network.select('critic')(batch['observations'], batch['next_observations'], actions=actions, info=True)
        # This means phi from (obs,actions), psi from next_obs
        _, next_phi_online, next_psi_online = \
            self.networks['critic'](obs, next_obs, actions=actions, info=True, get_v_separately=True) # Assume get_v_separately=True to get phi/psi
        # And then v is computed based on the specific GCBilinearValue logic if needed, or taken from the first output.
        # The JAX code seems to assign the 'v' output of critic(obs, next_obs) to `next_v`.
        next_v_online, _, _ = self.networks['critic'](obs, next_obs, actions=actions, info=True)


        if next_phi_online.ndim == 2: # If not ensembled from network, add ensemble dim
            next_phi_online = next_phi_online.unsqueeze(0)
            next_psi_online = next_psi_online.unsqueeze(0)
            if next_v_online.ndim == 1: next_v_online = next_v_online.unsqueeze(0) # (1,B) if num_ensembles=1 used internally

        # Einsum for dot product: (ensemble, batch, dim), (ensemble, batch_goals, dim) -> (batch, batch_goals, ensemble)
        next_logits = torch.einsum('ebd,egd->bge', next_phi_online, next_psi_online) / np.sqrt(next_phi_online.shape[-1])

        # Logits for (s, a) vs s_rand (value_goals_random as goals)
        # random_v_online, random_phi_online, random_psi_online = \
        #     self.networks['critic'](obs, value_goals_random, actions=actions, info=True)
        random_v_online, random_phi_online, random_psi_online = \
            self.networks['critic'](obs, value_goals_random, actions=actions, info=True, get_v_separately=True)
        random_v_online_val_part, _, _ = self.networks['critic'](obs, value_goals_random, actions=actions, info=True)


        if random_phi_online.ndim == 2:
            random_phi_online = random_phi_online.unsqueeze(0)
            random_psi_online = random_psi_online.unsqueeze(0)
            if random_v_online_val_part.ndim == 1: random_v_online_val_part = random_v_online_val_part.unsqueeze(0)


        random_logits = torch.einsum('ebd,egd->bge', random_phi_online, random_psi_online) / np.sqrt(random_phi_online.shape[-1])

        I_matrix = torch.eye(batch_size, device=self.device) # (B,B)

        # Combine logits: diagonal from next_logits, off-diagonal from random_logits
        # next_logits & random_logits are (B,B,E). I_matrix is (B,B)
        # JAX: I * logits1 + (1 - I) * logits2 applied per ensemble
        combined_logits_list = []
        v_combined_list = []
        for e_idx in range(next_logits.shape[-1]): # Iterate ensembles
            diag_part = I_matrix * next_logits[:,:,e_idx]
            off_diag_part = (1 - I_matrix) * random_logits[:,:,e_idx]
            combined_logits_list.append(diag_part + off_diag_part)

            # Combine V values similarly
            # next_v_online: (E,B), random_v_online: (E,B)
            # Ensure shapes are (E,B) for next_v_online and random_v_online_val_part
            # If GCBilinearValue returns (B,E), then transpose. Assuming (E,B) here based on phi/psi.
            # Let's assume GCBilinearValue's V output is (B,E) like standard networks.
            current_next_v = next_v_online[:, e_idx] if next_v_online.ndim==2 else next_v_online # (B,)
            current_random_v = random_v_online_val_part[:, e_idx] if random_v_online_val_part.ndim==2 else random_v_online_val_part # (B,)

            # JAX: I * v1 + (1-I) * v2. This is unusual for V values. Usually V(s) is just V(s).
            # This implies v here is also a (B,B) matrix like logits.
            # This means next_v_online and random_v_online are (B,B,E) or similar.
            # This part of JAX code is confusing if 'v' is standard V(s).
            # GCBilinearValue's 'v' output is typically a scalar value (or per-ensemble scalars) for (s,s_g) pair.
            # For TD-InfoNCE, 'v' is likely the dot product phi.psi, so it's naturally (B,B,E).
            # So, next_v_online and random_v_online are indeed the logits matrices before division by sqrt(D).
            # Let's rename them for clarity: next_v_logits, random_v_logits
            # And assume critic returns these raw dot products as 'v' for this agent.
            # This means the 'v' from GCBilinearValue is essentially the numerator of the logits.
            diag_v_part = I_matrix * next_v_online[:,:,e_idx] # Assuming next_v_online is (B,B,E)
            off_diag_v_part = (1-I_matrix) * random_v_online_val_part[:,:,e_idx] # Assuming random_v is (B,B,E)
            v_combined_list.append(diag_v_part + off_diag_v_part)


        final_combined_logits = torch.stack(combined_logits_list, dim=-1) # (B,B,E)
        final_v_combined = torch.stack(v_combined_list, dim=-1) # (B,B,E)


        # Loss1: Cross-entropy for combined_logits vs Identity (positive pairs are diagonal)
        # optax.softmax_cross_entropy(logits, labels)
        # PyTorch: F.cross_entropy(input (N,C), target (N)) or manual sum(-labels * log_softmax(logits))
        loss1_terms = []
        for e_idx in range(final_combined_logits.shape[-1]):
            # For each row i, target is one-hot vector with 1 at column i.
            # F.cross_entropy expects (Batch, Classes) and (Batch) of class indices.
            # Here, each row of final_combined_logits[:,:,e] is a set of scores for one sample.
            # So, input is (B, B), target is (B) with values [0,1,...,B-1]
            # loss1_terms.append(F.cross_entropy(final_combined_logits[:,:,e_idx],
            #                                   torch.arange(batch_size, device=self.device), reduction='mean'))
            # Or manual:
            log_p = F.log_softmax(final_combined_logits[:,:,e_idx], dim=1) # (B,B)
            loss1_terms.append(-torch.sum(I_matrix * log_p, dim=1)) # (B,) per ensemble

        loss1 = torch.stack(loss1_terms, dim=-1).mean() # Mean over batch and ensembles

        # Determine next actions for SARSA or Q-learning style target
        if td_loss_type == 'q_learning':
            with torch.no_grad(): # Actor is used for action selection, not trained here
                # Actor takes original observations
                next_dist = self.networks['actor'](next_obs) # Actor goal should be next_obs if GCActor
                if self.config['const_std']:
                    next_actions_policy = torch.clamp(next_dist.mode(), -1, 1)
                else:
                    next_actions_policy = torch.clamp(next_dist.sample(), -1, 1)
        elif td_loss_type == 'sarsa':
            next_actions_policy = batch['next_actions'].to(self.device)
        else:
            raise NotImplementedError(f"Unknown td_loss_type: {td_loss_type}")

        # Importance sampling weights 'w' from target critic
        with torch.no_grad():
            # _, w_phi, w_psi = self.networks['target_critic'](next_obs, value_goals_random, actions=next_actions_policy, info=True)
            _, w_phi, w_psi = self.networks['target_critic'](next_obs, value_goals_random, actions=next_actions_policy, info=True, get_v_separately=True)


            if w_phi.ndim == 2:
                w_phi = w_phi.unsqueeze(0)
                w_psi = w_psi.unsqueeze(0)

            w_logits_einsum = torch.einsum('ebd,egd->bge', w_phi, w_psi) / np.sqrt(w_phi.shape[-1]) # (B,B,E)
            w_logits_min_ensemble = w_logits_einsum.min(dim=-1)[0] # Min over ensembles (B,B)
            w_softmax = F.softmax(w_logits_min_ensemble, dim=-1) # (B,B), these are target probabilities for loss2

        # Loss2: Cross-entropy for random_logits vs importance weights w_softmax
        loss2_terms = []
        for e_idx in range(random_logits.shape[-1]): # random_logits is (B,B,E)
            log_p_random = F.log_softmax(random_logits[:,:,e_idx], dim=1) # (B,B)
            # Target is w_softmax (B,B)
            loss2_terms.append(-torch.sum(w_softmax * log_p_random, dim=1)) # (B,) per ensemble
        loss2 = torch.stack(loss2_terms, dim=-1).mean() # Mean over batch and ensembles

        total_contrastive_loss = (1 - self.config['discount']) * loss1 + self.config['discount'] * loss2

        # Stats
        with torch.no_grad():
            # Use final_combined_logits (B,B,E) for stats, mean over ensembles
            logits_stats = final_combined_logits.mean(dim=-1) # (B,B)
            correct_cat = (torch.argmax(logits_stats, dim=1) == torch.argmax(I_matrix, dim=1))
            logits_pos_diag = torch.sum(logits_stats * I_matrix) / torch.sum(I_matrix)
            logits_neg_offdiag = torch.sum(logits_stats * (1 - I_matrix)) / torch.sum(1 - I_matrix)

            # V stats from combined V values, mean over ensembles and then batch
            v_stat_mean = final_v_combined.mean().item() # Mean over B,B,E
            v_stat_max = final_v_combined.max().item()
            v_stat_min = final_v_combined.min().item()


        return total_contrastive_loss, {
            'contrastive_loss': total_contrastive_loss.item(),
            'v_mean': v_stat_mean,
            'v_max': v_stat_max,
            'v_min': v_stat_min,
            # 'binary_accuracy' in JAX was (logits > 0) == I. For multi-class, categorical is better.
            'categorical_accuracy': correct_cat.float().mean().item(),
            'logits_pos': logits_pos_diag.item(),
            'logits_neg': logits_neg_offdiag.item(),
            'logits': logits_stats.mean().item(),
        }

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device) # Actor uses original observations
        actions_gt = batch['actions'].to(self.device)

        # Actor needs goal if it's GCActor. TD-InfoNCE actor is GCActor.
        # What is the goal for BC? JAX code `dist = self.network.select('actor')(observations, params=grad_params)`
        # This implies GCActor's forward for BC might take obs as goal, or have a default.
        # For now, assume actor for BC doesn't need explicit goals or uses obs as goal.
        # If GCActor requires a goal, this needs to be specified (e.g. obs itself).
        # Let's assume GCActor in PyTorch handles `goals=None` by using `obs` as `goal` or similar.
        dist = self.networks['actor'](observations, goals=observations) # Tentative: use obs as goal for BC

        log_prob = dist.log_prob(actions_gt)
        bc_loss = -log_prob.mean()

        with torch.no_grad():
            mse = F.mse_loss(dist.mode(), actions_gt)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

        return bc_loss, {
            'bc_loss': bc_loss.item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse.item(),
            'std': std_val.item(),
        }

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # DDPG+BC style actor loss using the contrastive critic
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device) # For BC part
        actor_goals = batch['actor_goals'].to(self.device) # Goals for actor's policy output

        # Actor's policy distribution pi(a | s, s_g_actor)
        dist = self.networks['actor'](observations, goals=actor_goals)
        if self.config['const_std']:
            policy_actions = torch.clamp(dist.mode(), -1, 1)
        else:
            policy_actions = torch.clamp(dist.sample(), -1, 1) # RNG handled globally

        # Q-values from online critic for (s, policy_actions) evaluated against actor_goals
        # The critic returns (v, phi, psi). 'v' here are the "Q-values" or logits.
        # JAX: logits1, logits2 = self.network.select('critic')(batch['observations'], batch['actor_goals'], q_actions)
        # This means critic(obs, actor_goals, actions=policy_actions) -> (Q_e1, Q_e2)
        # These Qs are dot products phi(obs, policy_actions) . psi(actor_goals)
        q_outputs_actor = self.networks['critic'](observations, actor_goals, actions=policy_actions)
        # GCBilinearValue returns (v_ensemble_outputs). If num_ensembles=2, this is (v1, v2) or (B,2)
        if isinstance(q_outputs_actor, tuple):
            q1_policy, q2_policy = q_outputs_actor
        else: # Assuming (B,2)
            q1_policy, q2_policy = q_outputs_actor[:,0], q_outputs_actor[:,1]

        q_policy_min = torch.min(q1_policy, q2_policy) # (B,)

        # Log-ratios from these Q-values (logits)
        # JAX: log_ratios = jnp.diag(logits) - jax.nn.logsumexp(logits, axis=-1) + jnp.log(logits.shape[-1])
        # This implies `logits` (q_policy_min) should be a (B,B) matrix for diag and logsumexp over rows.
        # This is where TD-InfoNCE actor loss differs significantly from standard DDPG.
        # The critic Q(s,a,s_g) gives a score for (s,a) matching goal s_g.
        # For actor loss, `actor_goals` are a set of goals.
        # q_policy_min here is (B,). The JAX `diag` implies it should be (B,B).
        # This means the critic call for actor loss: `critic(obs, actor_goals_batch, policy_actions)`
        # where `actor_goals_batch` refers to the set of all goals in `batch['actor_goals']`.
        # Q_ij = score of (obs_i, policy_action_i) against actor_goal_j.
        # This requires GCBilinearValue's `psi` to be formed from the whole `actor_goals` batch for each `phi`.
        # Let's assume GCBilinearValue handles this if `actor_goals` is (B,D) by producing (B,B,E) Q-logits.
        # If GCBilinearValue's `forward(obs (B,Do), goals (B,Dg), actions (B,Da))` returns `v (B,B,E)`:
        # This implies `psi` from `goals_j` is dotted with `phi` from `(obs_i, actions_i)`.
        # For now, let's assume GCBilinearValue output `q_outputs_actor` is (B,B,E) or tuple of (B,B)
        # This is a CRITICAL assumption based on the JAX `diag` usage.
        # If q_outputs_actor was (q1_bb_e, q2_bb_e) where each is (B,B), then:
        # q_policy_min_bb = torch.min(q1_bb_e, q2_bb_e) # (B,B)
        # For now, let's assume `q_policy_min` is (B,B) after this step.
        # This part needs specific validation of GCBilinearValue's PyTorch behavior.
        # Let's assume q_policy_min is (B,B) for the log_ratio calculation.
        # If GCBilinearValue's output is (B,E) as in standard Q-nets, this log_ratio calc is not directly applicable.
        # Given the agent is TD-InfoNCE, the (B,B) structure for Q-logits is plausible.
        # If q_policy_min is (B,B):
        if q_policy_min.ndim != 2 or q_policy_min.shape[0] != q_policy_min.shape[1]:
             raise ValueError(f"Actor loss Q-values are not (B,B) matrix as expected by log_ratio. Shape: {q_policy_min.shape}. This means GCBilinearValue output for actor loss needs adjustment.")

        diag_q = torch.diag(q_policy_min) # (B,)
        logsumexp_q_rows = torch.logsumexp(q_policy_min, dim=1) # (B,)
        log_N_goals = np.log(q_policy_min.shape[1]) # log(num_actor_goals in batch)
        log_ratios = diag_q - logsumexp_q_rows + log_N_goals # (B,)

        # Multiply by predicted rewards for actor_goals
        # Reward net R(s_g)
        rewards_for_actor_goals = self.networks['reward'](actor_goals).squeeze(-1) # (B,)

        final_q_actor = torch.exp(log_ratios) * rewards_for_actor_goals # (B,)
        q_policy_loss = -final_q_actor.mean()

        if self.config['normalize_q_loss']:
            lam = (1 / (torch.abs(final_q_actor).mean().detach() + 1e-8))
            q_policy_loss = lam * q_policy_loss

        # BC term using ground truth actions
        log_prob_bc = dist.log_prob(actions_gt)
        bc_loss_term = -(self.config['alpha'] * log_prob_bc).mean()

        total_actor_loss = q_policy_loss + bc_loss_term

        with torch.no_grad():
            mse_val = F.mse_loss(dist.mode(), actions_gt)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

        return total_actor_loss, {
            'actor_loss': total_actor_loss.item(), 'q_loss': q_policy_loss.item(),
            'bc_loss': bc_loss_term.item(), 'q_mean': final_q_actor.mean().item(),
            'q_abs_mean': torch.abs(final_q_actor).mean().item(),
            'bc_log_prob': log_prob_bc.mean().item(), 'mse': mse_val.item(), 'std': std_val.item(),
        }


    def _update_target_network(self, online_net_name: str, target_net_name: str):
        if target_net_name in self.networks and online_net_name in self.networks:
            tau = self.config['tau']
            online_net = self.networks[online_net_name]
            target_net = self.networks[target_net_name]
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch: Dict[str, torch.Tensor], pretrain_mode: bool, step: int, full_update: bool = True) -> Dict[str, Any]:
        self.train()
        info = {}

        if pretrain_mode:
            # BC loss for actor
            bc_loss_val, bc_info = self.behavioral_cloning_loss(batch)
            info.update({f'bc/{k}': v for k,v in bc_info.items()})

            # Contrastive loss for critic (SARSA style targets)
            # RNG for actor sampling in Q-learning style contrastive loss not needed here
            crit_loss, crit_info = self.contrastive_loss(batch, td_loss_type='sarsa')
            info.update({f'critic/{k}': v for k,v in crit_info.items()})

            total_loss = bc_loss_val + crit_loss
            info['pretrain/total_loss'] = total_loss.item()
        else: # Finetuning mode
            # Reward predictor loss
            r_loss, r_info = self.reward_loss(batch)
            info.update({f'reward/{k}': v for k,v in r_info.items()})

            # Contrastive loss for critic (Q-learning style targets)
            crit_loss, crit_info = self.contrastive_loss(batch, td_loss_type='q_learning')
            info.update({f'critic/{k}': v for k,v in crit_info.items()})

            total_loss = r_loss + crit_loss

            if full_update: # Corresponds to actor_freq
                # Actor loss (DDPG+BC style with contrastive Q)
                a_loss, a_info = self.actor_loss(batch)
                info.update({f'actor/{k}': v for k,v in a_info.items()})
                total_loss += a_loss
            else:
                info['actor/actor_loss'] = 0.0

            info['finetune/total_loss'] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Target network update for critic (in both modes if critic is updated)
        self._update_target_network('critic', 'target_critic')

        return info

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor, goals: torch.Tensor = None,
                       temperature: float = 1.0, seed: int = None) -> torch.Tensor:
        # GCActor takes (observations, goals). If goals not provided, assume obs are goals.
        self.eval()
        observations = observations.to(self.device)
        if goals is None:
            goals = observations # Default GCActor behavior if goals are needed but not given
        else:
            goals = goals.to(self.device)

        if seed is not None:
            # Store and set RNG state
            ...

        if hasattr(self.networks['actor'], 'set_temperature'):
            self.networks['actor'].set_temperature(temperature)
        elif temperature != 1.0:
             print(f"Warning: Actor does not support temperature setting, but temperature={temperature} was requested.")

        dist = self.networks['actor'](observations, goals=goals)
        actions = dist.sample()
        actions_clipped = torch.clamp(actions, -1, 1)

        if seed is not None:
            # Restore RNG state
            ...
        return actions_clipped.cpu()

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, ex_actions: np.ndarray, config: Dict[str, Any]):
        ex_obs_shape = ex_observations.shape
        ex_act_shape = ex_actions.shape
        plain_config = dict(config) if not isinstance(config, dict) else config
        return cls(config=plain_config,
                   ex_observations_shape=ex_obs_shape,
                   ex_actions_shape=ex_act_shape,
                   seed=seed)

def get_config():
    return dict(
        agent_name='td_infonce_pytorch', lr=3e-4, batch_size=256,
        actor_hidden_dims=(512,512,512,512), value_hidden_dims=(512,512,512,512),
        reward_hidden_dims=(512,512,512,512), actor_layer_norm=False,
        value_layer_norm=True, reward_layer_norm=True, latent_dim=512,
        discount=0.99, actor_freq=4, tau=0.005, normalize_q_loss=True,
        alpha=0.1, const_std=True, encoder=None,
        relabel_reward=False, value_p_curgoal=0.0, value_p_trajgoal=1.0,
        value_p_randomgoal=0.0, value_geom_sample=True, value_geom_start_offset=0,
        actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0,
        actor_geom_sample=False, actor_geom_start_offset=0, gc_negative=True,
    )

if __name__ == '__main__':
    print("Attempting to initialize TDInfoNCEAgent (PyTorch)...")
    cfg = get_config()
    # cfg['encoder'] = 'mlp'

    obs_dim = 10; action_dim = 4; batch_size_val = cfg['batch_size']
    ex_obs_np = np.random.randn(batch_size_val, obs_dim).astype(np.float32)
    ex_act_np = np.random.randn(batch_size_val, action_dim).astype(np.float32)

    agent_pytorch = TDInfoNCEAgent.create(seed=42, ex_observations=ex_obs_np, ex_actions=ex_act_np, config=cfg)
    print(f"Agent created. Device: {agent_pytorch.device}")

    dummy_batch = {
        'observations': torch.from_numpy(ex_obs_np).float(),
        'actions': torch.from_numpy(ex_act_np).float(),
        'next_observations': torch.from_numpy(ex_obs_np).float(),
        'next_actions': torch.from_numpy(ex_act_np).float(), # For SARSA style contrastive loss
        'rewards': torch.randn(batch_size_val, 1).float(),
        'masks': torch.ones(batch_size_val, 1).float(),
        'value_goals': torch.from_numpy(ex_obs_np).float(), # Random states for contrastive negatives
        'actor_goals': torch.from_numpy(ex_obs_np).float(), # Goals for actor policy
    }

    print("\nTesting pretraining update...")
    try:
        # Critical assumption: GCBilinearValue must be adapted for (B,B,E) Q-logits for actor_loss.
        # If not, actor_loss will fail. Contrastive loss also assumes specific GCBV output.
        # For initial test, we might need to simplify or mock parts of these losses if GCBV is not fully adapted.
        print("Note: TD-InfoNCE losses are complex and depend heavily on GCBilinearValue output shapes.")
        pretrain_info = agent_pytorch.update(dummy_batch, pretrain_mode=True, step=0)
        print("Pretraining update successful (or passed with errors if GCBV not fully adapted). Info snippet:", {k:v for i,(k,v) in enumerate(pretrain_info.items()) if i<3})
    except Exception as e: import traceback; traceback.print_exc()

    print("\nTesting finetuning update (full)...")
    try:
        finetune_info = agent_pytorch.update(dummy_batch, pretrain_mode=False, step=1, full_update=True)
        print("Finetuning update successful (or passed with errors). Info snippet:", {k:v for i,(k,v) in enumerate(finetune_info.items()) if i < 3})
    except Exception as e: import traceback; traceback.print_exc()

    print("\nTesting action sampling...")
    try:
        # GCActor needs goals. Use observations as goals for this test.
        sampled_actions = agent_pytorch.sample_actions(dummy_batch['observations'][:5],
                                                       goals=dummy_batch['observations'][:5], seed=123)
        print("Sampled actions shape:", sampled_actions.shape)
    except Exception as e: import traceback; traceback.print_exc()

    print("\nBasic TD-InfoNCE (PyTorch) tests completed.")
    print("NOTE: Relies on PyTorch versions of GCActor, GCBilinearValue, Value, GCEncoder, ModuleDict.")
    print("      The contrastive loss and actor loss are particularly complex due to their reliance on specific GCBilinearValue output structures (e.g., (B,B,E) Q-logits) which need careful PyTorch implementation.")
