import copy
from functools import partial
from typing import Any, Dict, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.encoders import encoder_modules # Assuming PyTorch versions
from utils.networks import VectorField, Actor, IntentionEncoder, Value # Assuming PyTorch versions
from utils.datasets import _tree_map
from utils.torch_utils import to_torch

class InFOMAgent(nn.Module):
    """Intention-Conditioned Flow Occupancy Models (InFOM) agent. PyTorch version."""

    def __init__(self, config: Dict[str, Any], ex_observations_shape: Tuple, ex_actions_shape: Tuple, seed: int, device):
        super().__init__()
        self.config = config
        self.device = device

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # no apparent impact on speed
            torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed

        action_dim = ex_actions_shape[-1]
        obs_dim_for_vf = ex_observations_shape[-1] # Default to original obs_dim

        # Define encoders
        encoders = {}
        # If an encoder is used, it might change the effective observation dimension for downstream networks
        # The JAX code updates obs_dim based on encoder's output if specified.
        dummy_ex_obs_for_encoder_out = torch.randn(2, *ex_observations_shape[1:], device=self.device) # Batch=2

        if config['encoder'] is not None:
            encoder_module_class = encoder_modules[config['encoder']]
            # Instantiate a temporary encoder to find output dimension for VF
            temp_encoder = encoder_module_class().to(self.device)
            temp_encoder_output = temp_encoder(dummy_ex_obs_for_encoder_out)
            obs_dim_for_vf = temp_encoder_output.shape[-1] # This is the dim VF will see
            del temp_encoder # clean up

            # Actual encoders for networks
            encoders['critic'] = encoder_module_class() # For Q-value critic
            encoders['critic_vf_encoder'] = encoder_module_class() # Separate encoder instance for VF's input observations
            encoders['intention'] = encoder_module_class() # For intention encoder
            encoders['actor'] = encoder_module_class() # For actor

        # Define networks
        critic_def = Value( # Standard Q-value critic
            input_dim=obs_dim_for_vf+action_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2, # Clipped Double Q
            encoder=encoders.get('critic'), # Uses original observations if encoder is None
        )
        intention_encoder_def = IntentionEncoder(
            obs_dim=obs_dim_for_vf,
            action_dim=action_dim,
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
            encoder=encoders.get('intention'), # Uses original observations
        )
        # VectorField operates on (potentially encoded) observations
        critic_vf_def = VectorField(
            input_dim=obs_dim_for_vf,
            time_dim=1,
            vector_dim=obs_dim_for_vf, # Dimension of the space where flow occurs
            hidden_dims=config['value_hidden_dims'], # VF hidden dims
            obs_dim=obs_dim_for_vf,
            action_dim=action_dim, # Actions are used for conditioning
            latent_dim=config['latent_dim'],
            layer_norm=config['value_layer_norm'], # VF layer norm
            # Note: JAX VectorField does not take an encoder directly; it expects encoded obs if applicable.
            # The 'critic_vf_encoder' is handled upstream before calling critic_vf.
        )
        actor_def = Actor(
            input_dim=obs_dim_for_vf,
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std_init=config['const_std'],
            encoder=encoders.get('actor'), # Uses original observations
        )
        reward_def = Value( # Reward predictor
            input_dim=obs_dim_for_vf,
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
            num_ensembles=1, # Usually single reward predictor
            # Reward predictor in JAX takes encoded observations if encoder is present.
            # This implies its input dim should match obs_dim_for_vf if critic_vf_encoder is used.
            # For simplicity, if Value network takes an encoder, it should be critic_vf_encoder here.
            # Or, reward predictor takes raw obs and has its own internal encoding logic if needed.
            # JAX code: self.network.select('reward')(observations) where obs might be encoded by critic_vf_encoder
            # This is tricky. Let's assume reward_def will be called with appropriately encoded obs.
        )

        self.networks = nn.ModuleDict({})
        self.networks['critic'] = critic_def
        self.networks['critic_vf'] = critic_vf_def
        self.networks['target_critic_vf'] = copy.deepcopy(critic_vf_def)
        self.networks['intention_encoder'] = intention_encoder_def
        self.networks['actor'] = actor_def
        self.networks['reward'] = reward_def

        if config['encoder'] is not None:
            self.networks['critic_vf_encoder'] = encoders['critic_vf_encoder']
            self.networks['target_critic_vf_encoder'] = copy.deepcopy(encoders['critic_vf_encoder'])

        self.networks.to(self.device)

        # Initialize target networks
        self.networks['target_critic_vf'].load_state_dict(self.networks['critic_vf'].state_dict())
        if config['encoder'] is not None:
            self.networks['target_critic_vf_encoder'].load_state_dict(self.networks['critic_vf_encoder'].state_dict())

        # Optimizer
        online_params = []
        for name, net in self.networks.items():
            if 'target' not in name: # Don't optimize target nets
                online_params.extend(list(net.parameters()))
        self.optimizer = Adam(online_params, lr=config['lr'])

    @staticmethod
    def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def reward_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        rewards = batch['rewards'].to(self.device).squeeze(-1)

        # Encode observations if encoder is present (using critic_vf_encoder as per JAX logic)
        if self.config['encoder'] is not None and 'critic_vf_encoder' in self.networks:
            processed_obs = self.networks['critic_vf_encoder'](observations)
        else:
            processed_obs = observations

        reward_preds = self.networks['reward'](processed_obs).squeeze(-1) # Value net returns (B,1) or (B,)
        loss = F.mse_loss(reward_preds, rewards)
        return loss, {'reward_loss': loss.item()}

    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # This is the IQL-style critic loss for the standard Q-function
        observations = batch['observations'].to(self.device) # Original observations
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        next_actions = batch['next_actions'].to(self.device)

        # Process observations for VF and Reward net (consistent encoding)
        if self.config['encoder'] is not None and 'critic_vf_encoder' in self.networks:
            observations = self.networks['critic_vf_encoder'](observations)
            # next_observations will be used by intention encoder, which takes original next_obs

            # next_observations = next_observations (used by intention encoder)

        with torch.no_grad():
            # Sample noises for flow goal computation
            # Shape: (num_flow_goals, batch_size, obs_dim_for_vf)
            num_goals = self.config['num_flow_goals']
            batch_size = observations.shape[0]
            vf_obs_dim = observations.shape[-1]

            noises = torch.randn(num_goals, batch_size, vf_obs_dim,
                                 dtype=observations.dtype, device=self.device)

            # Sample latents z
            if self.config['critic_latent_type'] == 'prior':
                latents_z = torch.randn(num_goals, batch_size, self.config['latent_dim'], dtype=observations.dtype, device=self.device)
            elif self.config['critic_latent_type'] == 'encoding':
                # Intention encoder uses original next_observations and next_actions
                latent_dist = self.networks['intention_encoder'](next_observations, next_actions)
                latents_z = latent_dist.sample(sample_shape=(num_goals,)) # (num_goals, B, latent_dim)
            else:
                raise ValueError(f"Unknown critic_latent_type: {self.config['critic_latent_type']}")

            # Expand observations and actions for broadcasting with num_goals
            # observations: (B, D_vf) -> (1, B, D_vf) -> (N_goals, B, D_vf)
            # actions: (B, A_dim) -> (1, B, A_dim) -> (N_goals, B, A_dim)
            # Note: actions are used by VF, which expects original actions, not encoded.
            expanded_obs_for_vf = observations.unsqueeze(0).expand(num_goals, -1, -1)
            expanded_actions = actions.unsqueeze(0).expand(num_goals, -1, -1) # Original actions

            flow_goals = self._compute_fwd_flow_goals(
                noises, expanded_obs_for_vf, expanded_actions, latents_z, # latents_z is (N_goals, B, latent_dim)
                observation_min=batch.get('observation_min', None), 
                observation_max=batch.get('observation_max', None),
                use_target_network=False # Use online VF for critic's target Q computation based on flow
            ) # flow_goals will be (N_goals, B, D_vf)

            future_rewards = self.networks['reward'](flow_goals).squeeze(-1) # (N_goals, B)
            # Target Q is mean reward over flow goals, scaled by 1/(1-gamma)
            target_q_val = (1.0 / (1.0 - self.config['discount'])) * future_rewards.mean(dim=0) # (B,)

        # Online Q-critic estimates Q(s,a) using original observations
        q_online_outputs = self.networks['critic'](observations, actions)
        if isinstance(q_online_outputs, tuple): # If Value net returns tuple for ensembles
            q1_online, q2_online = q_online_outputs
        else: # If Value net returns (num_ensembles, B)
            q1_online, q2_online = q_online_outputs[0, :], q_online_outputs[1, :]

        # Expectile loss against the flow-based target Q
        # JAX code: expectile_loss(target_q - qs, target_q - qs, ...) where qs is a single tensor (mean or min of ensembles)
        # For PyTorch, let's assume qs for expectile loss is the mean of the online Q ensembles.
        q_online_combined = (q1_online + q2_online) / 2.0
        loss = self.expectile_loss(target_q_val - q_online_combined,
                                   target_q_val - q_online_combined,
                                   self.config['expectile']).mean()

        # For logging, use the configured aggregation
        with torch.no_grad():
            if self.config['q_agg'] == 'mean':
                q_log = q_online_combined
            else: # 'min'
                q_log = torch.min(q1_online, q2_online)

        return loss, {
            'critic_loss': loss.item(),
            'q_mean': q_log.mean().item(),
            'q_max': q_log.max().item(),
            'q_min': q_log.min().item(),
        }

    def flow_occupancy_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        next_actions = batch['next_actions'].to(self.device)
        batch_size = observations.shape[0]

        # Encode observations for VF
        if self.config['encoder'] is not None and 'critic_vf_encoder' in self.networks:
            obs_for_vf = self.networks['critic_vf_encoder'](observations)
            with torch.no_grad(): # Target encoder for next_obs
                next_obs_for_vf = self.networks['target_critic_vf_encoder'](next_observations)
        else:
            obs_for_vf = observations
            next_obs_for_vf = next_observations

        # Infer latents z from (s', a') using online intention encoder
        latent_dist = self.networks['intention_encoder'](next_observations, next_actions)
        latents_z = latent_dist.sample() # (B, latent_dim)

        # KL divergence loss for intention encoder
        kl_loss = -0.5 * (1 + 2 * latent_dist.stddev.log() - latent_dist.mean.pow(2) - latent_dist.stddev.pow(2)).mean()

        # SARSA^2 flow matching for occupancy models
        times = torch.rand(batch_size, device=self.device, dtype=obs_for_vf.dtype) # (B,)
        current_noises = torch.randn_like(obs_for_vf) # (B, D_vf)

        # Interpolated states for current VF prediction
        # x_t = t*s + (1-t)*noise
        s_interpolated_current = times.unsqueeze(-1) * obs_for_vf + (1 - times.unsqueeze(-1)) * current_noises

        # VF prediction v(x_t, t, s_0, a, z)
        # s_0 (obs_for_vf) and z (latents_z) are detached for VF input conditioning
        current_vf_pred = self.networks['critic_vf'](
            s_interpolated_current, times,
            obs_for_vf.detach(), actions, latents_z#.detach()
        )
        # Target for current flow matching: s_0 - noise
        current_flow_target = (obs_for_vf - current_noises).detach()
        current_flow_matching_loss = F.mse_loss(current_vf_pred, current_flow_target, reduction='none').mean(dim=-1) # (B,)

        # Future flow matching part
        future_noises = torch.randn_like(obs_for_vf) # (B, D_vf)
        # Compute s_T (flow goal) using online VF from next_obs_for_vf, next_actions, and detached latents_z
        # Note: JAX used use_target_network=True for compute_fwd_flow_goals here.
        # However, the VF call inside compute_fwd_flow_goals is target_critic_vf.
        # The outer VF calls (future_vf_target, future_vf_pred) use target/online respectively.
        # This means the s_T for future flow should be generated using target_critic_vf.
        flow_future_observations = self._compute_fwd_flow_goals(
            future_noises.unsqueeze(0), # (1, B, D_vf) for single "goal sample"
            next_obs_for_vf.unsqueeze(0), # (1, B, D_vf)
            next_actions.unsqueeze(0),    # (1, B, A_dim)
            latents_z.detach().unsqueeze(0), # (1, B, latent_dim)
            observation_min=batch.get('observation_min', None), 
            observation_max=batch.get('observation_max', None),
            use_target_network=True # Uses target_critic_vf inside
        ).squeeze(0) # (B, D_vf)

        s_interpolated_future = times.unsqueeze(-1) * flow_future_observations + (1 - times.unsqueeze(-1)) * future_noises # (B, D_vf)

        with torch.no_grad(): # Target VF for future part
            future_vf_target = self.networks['target_critic_vf'](
                s_interpolated_future, times,
                next_obs_for_vf, next_actions, latents_z.detach() # Conditioning on (s',a',z)
            )

        # Online VF for future part, conditioned on (s,a,z) from current step
        future_vf_pred = self.networks['critic_vf'](
            s_interpolated_future, times,
            obs_for_vf.detach(), actions, latents_z.detach() # Conditioning on (s,a,z)
        )
        future_flow_matching_loss = F.mse_loss(future_vf_pred, future_vf_target, reduction='none').mean(dim=-1) # (B,)

        flow_matching_loss = ((1 - self.config['discount']) * current_flow_matching_loss + self.config['discount'] * future_flow_matching_loss).mean()

        neg_elbo_loss = flow_matching_loss + self.config['kl_weight'] * kl_loss

        return neg_elbo_loss, {
            'neg_elbo_loss': neg_elbo_loss.item(),
            'flow_matching_loss': flow_matching_loss.item(),
            'kl_loss': kl_loss.item(),
            'flow_future_obs_max': flow_future_observations.max().item(),
            'flow_future_obs_min': flow_future_observations.min().item(),
            'current_flow_matching_loss': current_flow_matching_loss.mean().item(),
            'future_flow_matching_loss': future_flow_matching_loss.mean().item(),
        }

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device) # Actor uses original observations
        actions = batch['actions'].to(self.device)

        dist = self.networks['actor'](observations)
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        with torch.no_grad():
            mse = F.mse_loss(dist.mode, actions)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

        return bc_loss, {
            'bc_loss': bc_loss.item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse.item(),
            'std': std_val.item(),
        }

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the DDPG + BC actor loss."""
        observations = batch['observations'].to(self.device) # Actor and Critic use original observations
        actions = batch['actions'].to(self.device)

        dist = self.networks['actor'](observations)
        if self.config['const_std']:
            policy_actions = torch.clamp(dist.mode, -1, 1)
        else:
            policy_actions = torch.clamp(dist.sample(), -1, 1) # RNG handled globally

        with torch.no_grad(): # Q-values are detached for actor loss
            q_outputs = self.networks['critic'](observations, policy_actions)
            if isinstance(q_outputs, tuple):
                q1, q2 = q_outputs
            else: # (num_ensembles, B)
                q1, q2 = q_outputs[0, :], q_outputs[1, :]

            if self.config['q_agg'] == 'mean':
                q_val = (q1 + q2) / 2.0
            else: # 'min'
                q_val = torch.min(q1, q2)

        q_policy_loss = -q_val.mean()
        if self.config['normalize_q_loss']:
            lam = (1 / torch.abs(q_val).mean()).detach()
            q_policy_loss = lam * q_policy_loss

        log_prob_bc = dist.log_prob(actions) # BC against ground truth actions
        bc_loss = -(self.config['alpha'] * log_prob_bc).mean()

        total_actor_loss = q_policy_loss + bc_loss

        with torch.no_grad():
            mse_val = F.mse_loss(dist.mode, actions) # mse against gt actions
            std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)


        return total_actor_loss, {
            'actor_loss': total_actor_loss.item(),
            'q_loss': q_policy_loss.item(), # Q-value part of loss
            'bc_loss': bc_loss.item(), # BC part of loss
            'q_mean': q_val.mean().item(), # Q values for policy actions
            'q_abs_mean': torch.abs(q_val).mean().item(),
            'bc_log_prob': log_prob_bc.mean().item(), # Log prob of gt actions
            'mse': mse_val.item(), # Actor's mode MSE against gt actions
            'std': std_val.item(),
        }

    def _compute_fwd_flow_goals(self, noises_init: torch.Tensor,
                                observations_cond: torch.Tensor, actions_cond: torch.Tensor, latents_cond: torch.Tensor,
                                observation_min=None, observation_max=None,
                                init_times=None, end_times=None,
                                use_target_network=False) -> torch.Tensor:
        # noises_init, observations_cond, actions_cond, latents_cond expected to have leading dim for num_flow_goals if applicable
        # e.g. (N_goals, B, Dim) or (B, Dim) if N_goals=1 (implicitly by broadcasting later)

        vf_module_name = 'target_critic_vf' if use_target_network else 'critic_vf'

        current_flow_states = noises_init # s_t starts as noise

        if init_times is None: # (N_goals, B) or (B,)
            init_times = torch.zeros(current_flow_states.shape[:-1], dtype=current_flow_states.dtype, device=self.device)
        if end_times is None:
            end_times = torch.ones(current_flow_states.shape[:-1], dtype=current_flow_states.dtype, device=self.device)

        step_size = (end_times - init_times) / self.config['num_flow_steps'] # (N_goals, B) or (B,)
        if step_size.ndim < current_flow_states.ndim -1 : # Ensure step_size can broadcast with states (..., D)
            step_size = step_size.unsqueeze(-1) # (..., 1) for broadcasting with (..., D)

        for i in range(self.config['num_flow_steps']):
            current_times = i * step_size + init_times # (N_goals, B) or (B,)
            if current_times.ndim < current_flow_states.ndim -1:
                 current_times = current_times.unsqueeze(-1)


            # VF input shapes:
            # current_flow_states: (N_goals, B, D_vf) or (B, D_vf)
            # current_times: (N_goals, B) or (B,)
            # observations_cond, actions_cond, latents_cond: (N_goals, B, Dim) or (B, Dim)
            # VF expects times as (N_goals*B,) or (B,) if not per-goal
            # Let's ensure inputs to VF are flattened if N_goals > 1, then reshaped.

            original_shape_prefix = current_flow_states.shape[:-1] # (N_goals, B) or (B,)
            num_elements_prefix = np.prod(original_shape_prefix)

            vf_input_states = current_flow_states.reshape(num_elements_prefix, -1)
            vf_input_times = current_times.reshape(num_elements_prefix) # Ensure times is (Total,)

            # Conditioning variables might need broadcasting if they don't have N_goals dim
            # JAX code implies they are broadcasted.
            # obs_cond: (N_goals, B, D_vf) or (B, D_vf) -> reshape to (Total, D_vf)
            # act_cond: (N_goals, B, A_dim) or (B, A_dim) -> reshape to (Total, A_dim)
            # lat_cond: (N_goals, B, L_dim) or (B, L_dim) -> reshape to (Total, L_dim)

            def ensure_cond_shape(tensor_cond, target_shape_prefix):
                if tensor_cond.shape[:-1] == target_shape_prefix:
                    return tensor_cond.reshape(num_elements_prefix, -1)
                elif tensor_cond.ndim == 2 and len(target_shape_prefix)==2 : # (B,D) needs to become (N_goals*B, D)
                    return tensor_cond.unsqueeze(0).expand(target_shape_prefix[0],-1,-1).reshape(num_elements_prefix, -1)
                elif tensor_cond.ndim == 2 and len(target_shape_prefix)==1: # (B,D) matches (B,) prefix
                     return tensor_cond
                else:
                    raise ValueError(f"Shape mismatch for conditioning var: {tensor_cond.shape} vs prefix {target_shape_prefix}")

            vf_obs_cond = ensure_cond_shape(observations_cond, original_shape_prefix)
            vf_act_cond = ensure_cond_shape(actions_cond, original_shape_prefix)
            vf_lat_cond = ensure_cond_shape(latents_cond, original_shape_prefix)

            with torch.no_grad(): # VF calls during flow computation should not affect VF gradients for its own loss
                vf_output = self.networks[vf_module_name](
                    vf_input_states, vf_input_times,
                    vf_obs_cond, vf_act_cond, vf_lat_cond
                ) # (Total, D_vf)

            vf_output = vf_output.reshape(*original_shape_prefix, -1) # Reshape to (N_goals, B, D_vf) or (B, D_vf)

            current_flow_states = current_flow_states + vf_output * step_size.unsqueeze(-1) # Euler step

            if self.config['clip_flow_goals'] and observation_min is not None and observation_max is not None:
                # Ensure min/max are tensors on the correct device
                min_b = torch.as_tensor(observation_min, dtype=current_flow_states.dtype, device=self.device)
                max_b = torch.as_tensor(observation_max, dtype=current_flow_states.dtype, device=self.device)
                current_flow_states = torch.clamp(current_flow_states, min_b + 1e-5, max_b - 1e-5)

        return current_flow_states

    def _update_target_network(self, online_net_name: str):
        target_net_name = 'target_' + online_net_name
        if target_net_name in self.networks and online_net_name in self.networks:
            tau = self.config['tau']
            online_net = self.networks[online_net_name]
            target_net = self.networks[target_net_name]
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch: Dict[str, torch.Tensor], pretrain_mode: bool, step: int, full_update: bool = True) -> Dict[str, Any]:
        self.train()
        info = {}
        batch = _tree_map(lambda x: to_torch(x, self.device), batch)

        if pretrain_mode:
            # Flow occupancy loss (learns VF and Intention Encoder)
            flow_loss, flow_info = self.flow_occupancy_loss(batch)
            info.update({f'flow_occupancy/{k}': v for k,v in flow_info.items()})

            # Behavioral cloning loss for actor
            bc_loss_val, bc_info = self.behavioral_cloning_loss(batch)
            info.update({f'bc/{k}': v for k,v in bc_info.items()})

            total_loss = flow_loss + bc_loss_val
            info['pretrain/total_loss'] = total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Target updates for VF and its encoder
            if self.config['encoder'] is not None:
                self._update_target_network('critic_vf_encoder')
            self._update_target_network('critic_vf')

        else: # Finetuning mode
            # Reward predictor loss
            reward_loss, r_info = self.reward_loss(batch)
            info.update({f'reward/{k}': v for k,v in r_info.items()})

            # IQL-style Critic loss (for standard Q-function)
            critic_loss, critic_info = self.critic_loss(batch)
            info.update({f'critic/{k}': v for k,v in critic_info.items()})

            # Flow occupancy loss (continues to train VF and Intention Encoder)
            flow_loss, flow_info = self.flow_occupancy_loss(batch)
            info.update({f'flow_occupancy/{k}': v for k,v in flow_info.items()})

            total_loss = reward_loss + critic_loss + flow_loss

            if full_update: # Corresponds to actor_freq for main actor
                a_loss, a_info = self.actor_loss(batch)
                info.update({f'actor/{k}': v for k,v in a_info.items()})
                total_loss += a_loss
            else:
                info['actor/actor_loss'] = 0.0

            info['finetune/total_loss'] = total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Target updates for VF and its encoder (Q-critic has no target)
            if self.config['encoder'] is not None:
                self._update_target_network('critic_vf_encoder')
            self._update_target_network('critic_vf')
        return info

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor,
                       temperature: float = 1.0, seed: int = None, add_noise: bool = False, eval_mode=False) -> torch.Tensor:
        self.eval()
        observations = to_torch(observations, self.device) # Actor uses original observations
        if len(observations.shape) == 1:
            observations = observations.reshape((-1, observations.shape[-1]))

        dist = self.networks['actor'](observations, temperature=temperature)
        if eval_mode:
            actions = dist.mean
        else:
            actions = dist.sample()
        actions_clipped = torch.clamp(actions, -1, 1)

        return actions_clipped.cpu().numpy()

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, ex_actions: np.ndarray, config: Dict[str, Any], device):
        ex_obs_shape = ex_observations.shape
        ex_act_shape = ex_actions.shape
        plain_config = dict(config) if not isinstance(config, dict) else config
        return cls(config=plain_config,
                   ex_observations_shape=ex_obs_shape,
                   ex_actions_shape=ex_act_shape,
                   seed=seed,
                   device=device)

def get_config():
    # Returns a plain dict for PyTorch
    return dict(
        agent_name='infom', 
        lr=3e-4, 
        batch_size=256,
        intention_encoder_hidden_dims=(512, 512, 512, 512),
        actor_hidden_dims=(512, 512, 512, 512),
        value_hidden_dims=(512, 512, 512, 512), # For Q-critic and VF
        reward_hidden_dims=(512, 512, 512, 512),
        intention_encoder_layer_norm=True, 
        value_layer_norm=True, # For Q-critic and VF
        actor_layer_norm=False, 
        reward_layer_norm=True,
        latent_dim=512, 
        discount=0.99, 
        tau=0.005, 
        expectile=0.9,  # IQL style expectile.
        kl_weight=0.1,
        q_agg='min', 
        critic_latent_type='prior', # Type of critic latents. ('prior', 'encoding')
        num_flow_goals=16,  # Number of future flow goals for computing the target q.
        clip_flow_goals=True, 
        actor_freq=4,  # Actor update frequency.
        alpha=0.3,  # BC coefficient (need to be tuned for each environment). 
        const_std=True,
        num_flow_steps=10, 
        normalize_q_loss=False, 
        encoder=None, # Encoder name ('mlp', 'impala_small', etc.).
    )

if __name__ == '__main__':
    print("Attempting to initialize InFOMAgent (PyTorch)...")
    cfg = get_config()
    # cfg['encoder'] = 'mlp' # Test with and without encoder

    obs_dim = 10
    action_dim = 4
    batch_size_val = cfg['batch_size']
    example_obs_np = np.random.randn(batch_size_val, obs_dim).astype(np.float32)
    example_act_np = np.random.randn(batch_size_val, action_dim).astype(np.float32)

    # For clipping flow goals, if enabled
    obs_min_np = np.ones(obs_dim, dtype=np.float32) * -5.0
    obs_max_np = np.ones(obs_dim, dtype=np.float32) * 5.0

    agent_pytorch = InFOMAgent.create(seed=42,
                                     ex_observations=example_obs_np,
                                     ex_actions=example_act_np,
                                     config=cfg)
    print(f"Agent created. Device: {agent_pytorch.device}")

    dummy_batch = {
        'observations': torch.from_numpy(example_obs_np).float(),
        'actions': torch.from_numpy(example_act_np).float(),
        'next_observations': torch.from_numpy(example_obs_np).float(), # Simplicity
        'next_actions': torch.from_numpy(example_act_np).float(), # Simplicity
        'rewards': torch.randn(batch_size_val, 1).float(),
        'masks': torch.ones(batch_size_val, 1).float(), # Not directly used by InFOM losses but good to have
        'observation_min': obs_min_np if cfg['clip_flow_goals'] else None,
        'observation_max': obs_max_np if cfg['clip_flow_goals'] else None,
    }

    print("\nTesting pretraining update...")
    try:
        pretrain_info = agent_pytorch.update(dummy_batch, pretrain_mode=True, step=0)
        print("Pretraining update successful. Info snippet:", {k:v for i,(k,v) in enumerate(pretrain_info.items()) if i < 3})
    except Exception as e: 
        import traceback
        traceback.print_exc()

    print("\nTesting finetuning update (full)...")
    try:
        finetune_info = agent_pytorch.update(dummy_batch, pretrain_mode=False, step=1, full_update=True)
        print("Finetuning update successful. Info snippet:", {k:v for i,(k,v) in enumerate(finetune_info.items()) if i < 3})
    except Exception as e: 
        import traceback
        traceback.print_exc()

    print("\nTesting action sampling...")
    try:
        actions = agent_pytorch.sample_actions(dummy_batch['observations'][:5], seed=123)
        print("Sampled actions shape:", actions.shape)
    except Exception as e: 
        import traceback; 
        traceback.print_exc()

    print("\nBasic InFOM (PyTorch) tests completed.")
    print("NOTE: Relies on PyTorch versions of VectorField, Actor, IntentionEncoder, Value, ModuleDict and encoders.")
    print("      The _compute_fwd_flow_goals logic for handling shapes with num_flow_goals needs careful validation.")