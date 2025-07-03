import copy
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.encoders import encoder_modules # Assuming PyTorch versions
from utils.networks import Actor, Value # Assuming PyTorch versions
from utils.torch_utils import ModuleDict # Assuming PyTorch compatible utils


class MBPOReBRACAgent(nn.Module):
    """Model-based policy optimization + Revisited behavior-regularized actor-critic (ReBRAC) agent.
    PyTorch version.
    """

    def __init__(self, config: Dict[str, Any], ex_observations_shape: Tuple, ex_actions_shape: Tuple, seed: int):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        action_dim = ex_actions_shape[-1]
        obs_dim = ex_observations_shape[-1] # Original observation dimension
        
        # Define encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module_class = encoder_modules[config['encoder']]
            # Note: Encoders are passed to networks; networks handle input processing.
            encoders['transition'] = encoder_module_class()
            encoders['reward'] = encoder_module_class()
            encoders['critic'] = encoder_module_class()
            encoders['actor'] = encoder_module_class()

        # Define networks
        # Transition model T(s,a) -> s_next_residual
        transition_def = Value( # Using Value network as a generic MLP base
            hidden_dims=config['transition_hidden_dims'],
            output_dim=obs_dim, # Predicts residual, so output dim is obs_dim
            layer_norm=config['transition_layer_norm'],
            num_ensembles=1, # Typically single transition model unless ensembled
            encoder=encoders.get('transition'),
            # Value network's forward takes (observations, actions=None).
            # For transition model, actions are required. Assumed Value net can take actions.
        )
        # Reward model R(s) -> r  (or R(s,a) -> r if actions were passed)
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
            num_ensembles=1, # Single reward model
            encoder=encoders.get('reward'),
            # JAX version calls reward_def with (observations,).
        )
        # Critic Q(s,a) -> q
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2, # Clipped Double Q
            encoder=encoders.get('critic'),
        )
        # Actor pi(s) -> a_dist
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True, # As per JAX config
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        self.networks = ModuleDict({
            'transition': transition_def,
            'reward': reward_def,
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'actor': actor_def,
            'target_actor': copy.deepcopy(actor_def),
        }).to(self.device)

        # Initialize target networks
        self.networks['target_critic'].load_state_dict(self.networks['critic'].state_dict())
        self.networks['target_actor'].load_state_dict(self.networks['actor'].state_dict())

        # Optimizer: includes all parameters from non-target networks
        online_params = list(self.networks['transition'].parameters()) + \
                        list(self.networks['reward'].parameters()) + \
                        list(self.networks['critic'].parameters()) + \
                        list(self.networks['actor'].parameters())
        self.optimizer = Adam(online_params, lr=config['lr'])

    def transition_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)

        # Predict residual: s_next - s
        # Assumes Value network used for transition can take actions as input.
        obs_res_preds = self.networks['transition'](observations, actions=actions).squeeze(-1) # Value may output (B,1)

        target_obs_res = (next_observations - observations)
        loss = F.mse_loss(obs_res_preds, target_obs_res)

        return loss, {
            'transition_loss': loss.item(),
            'obs_res': target_obs_res.mean().item(), # Log mean of actual residuals
        }

    def reward_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        rewards_gt = batch['rewards'].to(self.device).squeeze(-1) # Ensure (B,)

        reward_preds = self.networks['reward'](observations).squeeze(-1) # Value may output (B,1)
        loss = F.mse_loss(reward_preds, rewards_gt)
        return loss, {'reward_loss': loss.item()}

    def critic_loss(self, model_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Uses data from model rollouts (model_batch)
        model_obs = model_batch['model_observations'].to(self.device)
        model_actions = model_batch['model_actions'].to(self.device)
        model_next_obs = model_batch['model_next_observations'].to(self.device)
        model_rewards = model_batch['model_rewards'].to(self.device).squeeze(-1)
        model_masks = model_batch['model_masks'].to(self.device).squeeze(-1)
        # model_next_actions are GT actions for next state in model rollout, for ReBRAC penalty
        model_next_actions_gt = model_batch['model_next_actions'].to(self.device)


        with torch.no_grad():
            next_dist = self.networks['target_actor'](model_next_obs)
            next_actions_policy = next_dist.mode() # TD3 uses deterministic target policy action

            # Add noise for target policy smoothing (TD3 style)
            noise = torch.randn_like(next_actions_policy) * self.config['actor_noise']
            noise = torch.clamp(noise, -self.config['actor_noise_clip'], self.config['actor_noise_clip'])
            next_actions_smoothed = torch.clamp(next_actions_policy + noise, -1, 1)

            # Target Q-values from target critic
            next_q_outputs = self.networks['target_critic'](model_next_obs, actions=next_actions_smoothed)
            if isinstance(next_q_outputs, tuple):
                next_q1, next_q2 = next_q_outputs
            else:
                next_q1, next_q2 = next_q_outputs[:,0], next_q_outputs[:,1]
            next_q = torch.min(next_q1, next_q2) # Clipped Double Q

            # ReBRAC specific penalty for critic (consistency with target actor's next actions)
            mse_penalty = F.mse_loss(next_actions_policy, model_next_actions_gt, reduction='none').sum(dim=-1) # Sum over action dim
            next_q_penalized = next_q - self.config['alpha_critic'] * mse_penalty

            target_q_val = model_rewards + self.config['discount'] * model_masks * next_q_penalized

        # Online Q-critic estimates for (s_model, a_model)
        q_online_outputs = self.networks['critic'](model_obs, actions=model_actions)
        if isinstance(q_online_outputs, tuple):
            q1_online, q2_online = q_online_outputs
        else:
            q1_online, q2_online = q_online_outputs[:,0], q_online_outputs[:,1]

        # MSE loss for each Q-function against the target
        loss = F.mse_loss(q1_online, target_q_val) + F.mse_loss(q2_online, target_q_val)

        # For logging, use mean of online Q estimates
        q_online_mean_log = (q1_online + q2_online).mean() / 2.0

        return loss, {
            'critic_loss': loss.item(),
            'q_mean': q_online_mean_log.item(), # JAX logs online Q mean
            'q_max': torch.max(q1_online, q2_online).max().item(), # Max of online Qs
            'q_min': torch.min(q1_online, q2_online).min().item(), # Min of online Qs
        }

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Uses real data from experience buffer
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device)

        dist = self.networks['actor'](observations) # Online actor
        log_prob = dist.log_prob(actions_gt)
        bc_loss = -log_prob.mean()

        with torch.no_grad():
            mse = F.mse_loss(dist.mode(), actions_gt)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

        return bc_loss, {
            'bc_loss': bc_loss.item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse.item(),
            'std': std_val.item() if torch.is_tensor(std_val) else std_val,
        }

    def actor_loss(self, model_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Uses data from model rollouts (model_batch)
        model_obs = model_batch['model_observations'].to(self.device)
        model_actions_gt = model_batch['model_actions'].to(self.device) # "Ground truth" from model rollout

        dist = self.networks['actor'](model_obs) # Online actor
        policy_actions = dist.mode() # ReBRAC uses deterministic actions for Q-value part

        # Q-loss part: Q-values from online critic for actor's policy_actions
        # These Q-values are part of actor's loss computation, so no detach needed here.
        q_policy_outputs = self.networks['critic'](model_obs, actions=policy_actions)
        if isinstance(q_policy_outputs, tuple):
            q1_policy, q2_policy = q_policy_outputs
        else:
            q1_policy, q2_policy = q_policy_outputs[:,0], q_policy_outputs[:,1]
        q_policy = torch.min(q1_policy, q2_policy) # Use min Q (TD3 style)

        # Normalize Q by its absolute mean (detached for stability)
        lam = (1 / (torch.abs(q_policy).mean().detach() + 1e-8))
        actor_q_loss = -(lam * q_policy).mean()

        # BC loss part: MSE between actor's policy_actions and model_actions_gt
        # JAX: mse = jnp.square(actions - batch['model_actions']).sum(axis=-1)
        mse_bc = F.mse_loss(policy_actions, model_actions_gt, reduction='none').sum(dim=-1) # Sum over action dim
        bc_loss_term = (self.config['alpha_actor'] * mse_bc).mean()

        total_loss = actor_q_loss + bc_loss_term

        with torch.no_grad():
            action_std = dist.stddev.mean() if hasattr(dist, 'stddev') and dist.stddev is not None else torch.tensor(0.0, device=self.device)
            # JAX code: if self.config['tanh_squash']: action_std = dist._distribution.stddev()
            # This needs Actor to expose base distribution's stddev if TanhBijector is used.
            # For now, assume Actor's stddev is directly accessible or approx.
            if self.config['tanh_squash'] and hasattr(dist, '_distribution') and hasattr(dist._distribution, 'stddev'):
                 action_std = dist._distribution.stddev.mean()


        return total_loss, {
            'total_loss': total_loss.item(),
            'actor_loss': actor_q_loss.item(), # Q-value part of loss
            'bc_loss': bc_loss_term.item(),    # BC part of loss
            'std': action_std.item() if torch.is_tensor(action_std) else action_std,
            'mse': mse_bc.mean().item(), # This is the BC MSE part for actor loss
        }

    def _update_target_network(self, online_net_name: str, target_net_name: str):
        if target_net_name in self.networks and online_net_name in self.networks:
            tau = self.config['tau']
            online_net = self.networks[online_net_name]
            target_net = self.networks[target_net_name]
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch: Dict[str, torch.Tensor], model_batch: Dict[str, torch.Tensor],
               pretrain_mode: bool, step: int, full_update: bool = True) -> Dict[str, Any]:
        # `batch` is from real replay buffer, `model_batch` is from model rollouts
        self.train()
        info = {}

        if pretrain_mode:
            # Transition model loss (uses real data)
            t_loss, t_info = self.transition_loss(batch)
            info.update({f'transition/{k}': v for k,v in t_info.items()})

            # Behavioral cloning loss for actor (uses real data)
            bc_loss_val, bc_info = self.behavioral_cloning_loss(batch)
            info.update({f'bc/{k}': v for k,v in bc_info.items()})

            total_loss = t_loss + bc_loss_val
            info['pretrain/total_loss'] = total_loss.item()
        else: # Finetuning mode
            # Reward model loss (uses real data)
            r_loss, r_info = self.reward_loss(batch)
            info.update({f'reward/{k}': v for k,v in r_info.items()})

            # Transition model loss (uses real data)
            t_loss, t_info = self.transition_loss(batch)
            info.update({f'transition/{k}': v for k,v in t_info.items()})

            # Critic loss (uses model-generated data)
            c_loss, c_info = self.critic_loss(model_batch)
            info.update({f'critic/{k}': v for k,v in c_info.items()})

            total_loss = r_loss + t_loss + c_loss

            if full_update: # Corresponds to actor_freq
                # Actor loss (uses model-generated data)
                a_loss, a_info = self.actor_loss(model_batch)
                info.update({f'actor/{k}': v for k,v in a_info.items()})
                total_loss += a_loss
            else:
                info['actor/total_loss'] = 0.0

            info['finetune/total_loss'] = total_loss.item()

        # Optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Target network updates (only in finetuning and if actor was updated)
        if not pretrain_mode and full_update:
            self._update_target_network('critic', 'target_critic')
            self._update_target_network('actor', 'target_actor')

        return info

    @torch.no_grad()
    def predict_rewards(self, observations: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        # `actions` argument is for API consistency if Value net could use it, but R(s) typically.
        self.eval()
        observations = observations.to(self.device)
        rewards = self.networks['reward'](observations) # Assumes reward net takes only obs
        return rewards.cpu()

    @torch.no_grad()
    def predict_next_observations(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.eval()
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # Assumes transition net (Value type) takes actions
        observation_residuals = self.networks['transition'](observations, actions=actions)
        next_observations = observation_residuals + observations
        return next_observations.cpu()

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor,
                       temperature: float = 1.0, seed: int = None) -> torch.Tensor:
        self.eval()
        observations = observations.to(self.device)

        if seed is not None:
            current_rng_state_cpu = torch.get_rng_state()
            current_rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        if hasattr(self.networks['actor'], 'set_temperature'):
            self.networks['actor'].set_temperature(temperature)
        elif temperature != 1.0:
             print(f"Warning: Actor does not support temperature setting, but temperature={temperature} was requested.")

        dist = self.networks['actor'](observations)
        actions_mode = dist.mode() # MBPO often uses deterministic action from mode + noise

        # Add noise as in JAX version's sample_actions
        action_noise = torch.randn_like(actions_mode) * self.config['actor_noise'] * temperature
        action_noise_clipped = torch.clamp(action_noise,
                                           -self.config['actor_noise_clip'],
                                           self.config['actor_noise_clip'])
        final_actions = torch.clamp(actions_mode + action_noise_clipped, -1, 1)

        if seed is not None:
            torch.set_rng_state(current_rng_state_cpu)
            if torch.cuda.is_available() and current_rng_state_cuda:
                torch.cuda.set_rng_state_all(current_rng_state_cuda)
        return final_actions.cpu()

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
        agent_name='mbpo_rebrac_pytorch', lr=3e-4, batch_size=256,
        transition_hidden_dims=(512, 512, 512, 512), reward_hidden_dims=(512, 512, 512, 512),
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        transition_layer_norm=True, reward_layer_norm=True,
        actor_layer_norm=False, value_layer_norm=True,
        discount=0.99, tau=0.005, tanh_squash=True, actor_fc_scale=0.01,
        num_model_rollouts=256, # Handled by training loop
        num_model_rollout_steps=1, # Handled by training loop
        alpha_actor=0.0, alpha_critic=0.0,
        actor_freq=4, actor_noise=0.2, actor_noise_clip=0.5,
        encoder=None,
    )

if __name__ == '__main__':
    print("Attempting to initialize MBPOReBRACAgent (PyTorch)...")
    cfg = get_config()
    # cfg['encoder'] = 'mlp'

    obs_dim = 10; action_dim = 4; batch_size_val = cfg['batch_size']
    example_obs_np = np.random.randn(batch_size_val, obs_dim).astype(np.float32)
    example_act_np = np.random.randn(batch_size_val, action_dim).astype(np.float32)

    agent_pytorch = MBPOReBRACAgent.create(seed=42,
                                           ex_observations=example_obs_np,
                                           ex_actions=example_act_np,
                                           config=cfg)
    print(f"Agent created. Device: {agent_pytorch.device}")

    # Dummy batch from real replay buffer
    dummy_real_batch = {
        'observations': torch.from_numpy(example_obs_np).float(),
        'actions': torch.from_numpy(example_act_np).float(),
        'next_observations': torch.from_numpy(example_obs_np).float(),
        'rewards': torch.randn(batch_size_val, 1).float(),
        'masks': torch.ones(batch_size_val, 1).float(), # Not used by this agent's losses directly
    }
    # Dummy batch from model rollouts
    dummy_model_batch = {
        'model_observations': torch.from_numpy(example_obs_np).float(),
        'model_actions': torch.from_numpy(example_act_np).float(),
        'model_next_observations': torch.from_numpy(example_obs_np).float(),
        'model_next_actions': torch.from_numpy(example_act_np).float(), # For ReBRAC critic penalty
        'model_rewards': torch.randn(batch_size_val, 1).float(),
        'model_masks': torch.ones(batch_size_val, 1).float(),
    }

    print("\nTesting pretraining update...")
    try: # Pretrain uses only real_batch
        pretrain_info = agent_pytorch.update(dummy_real_batch, dummy_model_batch, pretrain_mode=True, step=0)
        print("Pretraining update successful. Info:", {k:v for i,(k,v) in enumerate(pretrain_info.items()) if i < 3})
    except Exception as e: import traceback; traceback.print_exc()

    print("\nTesting finetuning update (full)...")
    try:
        finetune_info = agent_pytorch.update(dummy_real_batch, dummy_model_batch, pretrain_mode=False, step=1, full_update=True)
        print("Finetuning update successful. Info:", {k:v for i,(k,v) in enumerate(finetune_info.items()) if i < 3})
    except Exception as e: import traceback; traceback.print_exc()

    print("\nTesting model predictions...")
    try:
        pred_rewards = agent_pytorch.predict_rewards(dummy_real_batch['observations'][:5])
        print("Predicted rewards shape:", pred_rewards.shape)
        pred_next_obs = agent_pytorch.predict_next_observations(dummy_real_batch['observations'][:5], dummy_real_batch['actions'][:5])
        print("Predicted next_obs shape:", pred_next_obs.shape)
    except Exception as e: import traceback; traceback.print_exc()

    print("\nTesting action sampling...")
    try:
        actions = agent_pytorch.sample_actions(dummy_real_batch['observations'][:5], seed=123)
        print("Sampled actions shape:", actions.shape)
    except Exception as e: import traceback; traceback.print_exc()

    print("\nBasic MBPO-ReBRAC (PyTorch) tests completed.")
    print("NOTE: Relies on PyTorch versions of Actor, Value, ModuleDict, and encoders.")
    print("      Transition model (Value net) needs to correctly handle actions as input.")
