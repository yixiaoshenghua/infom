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


class IQLAgent(nn.Module):
    """Implicit Q-learning (IQL) agent. PyTorch version."""

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
            encoders['value'] = encoder_module_class()
            encoders['critic'] = encoder_module_class()
            encoders['actor'] = encoder_module_class()

        # Define networks
        value_def = Value( # V-function
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1, # Typically single V-function in IQL
            encoder=encoders.get('value'),
        )
        critic_def = Value( # Q-function (critic)
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2, # Clipped Double Q-learning uses two Q-functions
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False, # As per JAX config
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        self.networks = ModuleDict({
            'value': value_def,
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def), # Target network for Q-function
            'actor': actor_def,
        }).to(self.device)

        # Initialize target critic network
        self.networks['target_critic'].load_state_dict(self.networks['critic'].state_dict())

        # Optimizer: includes all parameters from non-target networks
        online_params = list(self.networks['value'].parameters()) + \
                        list(self.networks['critic'].parameters()) + \
                        list(self.networks['actor'].parameters())
        self.optimizer = Adam(online_params, lr=config['lr'])

    @staticmethod
    def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(adv >= 0, expectile, (1.0 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)

        with torch.no_grad():
            # Q-values from target critic network
            # Value.forward (as critic) returns (B, num_ensembles) or tuple of (B,)
            q_target_outputs = self.networks['target_critic'](observations, actions=actions)
            if isinstance(q_target_outputs, tuple): # If Value net returns tuple for ensembles
                 q1_target, q2_target = q_target_outputs
            else: # If Value net returns tensor (B, num_ensembles)
                 q1_target, q2_target = q_target_outputs[:, 0], q_target_outputs[:, 1]
            q_target = torch.min(q1_target, q2_target) # Clipped Double Q

        # Current V-value estimate from online value network
        # Value.forward (as V-function, num_ensembles=1) returns (B,) or (B,1)
        v_online = self.networks['value'](observations).squeeze(-1)

        loss = self.expectile_loss(q_target - v_online, q_target - v_online, self.config['expectile']).mean()

        return loss, {
            'value_loss': loss.item(),
            'v_mean': v_online.mean().item(),
            'v_max': v_online.max().item(),
            'v_min': v_online.min().item(),
        }

    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        rewards = batch['rewards'].to(self.device).squeeze(-1)
        masks = batch['masks'].to(self.device).squeeze(-1)

        with torch.no_grad():
            # Next state V-value from online value network (target for Q-learning)
            next_v = self.networks['value'](next_observations).squeeze(-1)
            target_q_val = rewards + self.config['discount'] * masks * next_v

        # Current Q-value estimates from online critic network
        q_online_outputs = self.networks['critic'](observations, actions=actions)
        if isinstance(q_online_outputs, tuple):
            q1_online, q2_online = q_online_outputs
        else:
            q1_online, q2_online = q_online_outputs[:,0], q_online_outputs[:,1]

        # MSE loss for each Q-function against the target
        critic_loss_val = F.mse_loss(q1_online, target_q_val) + F.mse_loss(q2_online, target_q_val)

        return critic_loss_val, {
            'critic_loss': critic_loss_val.item(),
            'q_mean': target_q_val.mean().item(), # JAX logs target_q_val stats
            'q_max': target_q_val.max().item(),
            'q_min': target_q_val.min().item(),
        }

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device) # Ground truth actions

        dist = self.networks['actor'](observations)
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
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device) # Ground truth actions for AWR/BC part

        actor_loss_type = self.config.get('actor_loss', 'awr') # Default to 'awr' if not specified

        if actor_loss_type == 'awr':
            with torch.no_grad(): # V and Q are targets for AWR weighting, no gradients through them
                v = self.networks['value'](observations).squeeze(-1)
                # Q-values from online critic (detached for adv computation)
                q_critic_outputs = self.networks['critic'](observations, actions=actions_gt)
                if isinstance(q_critic_outputs, tuple):
                    q1_critic, q2_critic = q_critic_outputs
                else:
                    q1_critic, q2_critic = q_critic_outputs[:,0], q_critic_outputs[:,1]
                q = torch.min(q1_critic, q2_critic) # Use min Q for conservative advantage
                adv = q - v

            exp_a = torch.exp(adv * self.config['alpha']) # Alpha is AWR temperature
            exp_a = torch.clamp(exp_a, max=100.0) # Clip as in JAX

            dist = self.networks['actor'](observations) # Online actor
            log_prob = dist.log_prob(actions_gt)

            actor_loss_val = -(exp_a * log_prob).mean()

            with torch.no_grad():
                mse_val = F.mse_loss(dist.mode(), actions_gt)
                std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

            actor_info = {
                'actor_loss': actor_loss_val.item(), 'adv': adv.mean().item(),
                'bc_log_prob': log_prob.mean().item(), 'mse': mse_val.item(), 'std': std_val.item(),
            }
            return actor_loss_val, actor_info

        elif actor_loss_type == 'ddpgbc':
            dist = self.networks['actor'](observations) # Online actor
            if self.config['const_std']:
                policy_actions = torch.clamp(dist.mode(), -1, 1)
            else:
                # PyTorch sample() doesn't take seed here, global RNG.
                policy_actions = torch.clamp(dist.sample(), -1, 1)

            # Q-values for policy_actions from online critic (these Qs are part of actor's loss)
            q_policy_outputs = self.networks['critic'](observations, actions=policy_actions)
            if isinstance(q_policy_outputs, tuple):
                q1_policy, q2_policy = q_policy_outputs
            else:
                q1_policy, q2_policy = q_policy_outputs[:,0], q_policy_outputs[:,1]
            q_policy = torch.min(q1_policy, q2_policy) # Use min Q

            # DDPG part: -Q(s, pi(s))
            # Normalize Q by its absolute mean (detached)
            q_policy_loss = -q_policy.mean() / (torch.abs(q_policy).mean().detach() + 1e-8)

            # BC part: log_prob of ground truth actions
            log_prob_bc = dist.log_prob(actions_gt)
            bc_loss_term = -(self.config['alpha'] * log_prob_bc).mean() # Alpha is BC coefficient here

            total_actor_loss = q_policy_loss + bc_loss_term

            with torch.no_grad():
                mse_val = F.mse_loss(dist.mode(), actions_gt) # MSE of mode against gt
                std_val = dist.scale.mean() if hasattr(dist, 'scale') and dist.scale is not None else torch.tensor(0.0, device=self.device)

            actor_info = {
                'actor_loss': total_actor_loss.item(), 'q_loss': q_policy_loss.item(),
                'bc_loss': bc_loss_term.item(), 'q_mean': q_policy.mean().item(),
                'q_abs_mean': torch.abs(q_policy).mean().item(),
                'bc_log_prob': log_prob_bc.mean().item(), 'mse': mse_val.item(), 'std': std_val.item(),
            }
            return total_actor_loss, actor_info
        else:
            raise ValueError(f"Unsupported actor_loss type: {actor_loss_type}")

    def _update_target_network(self, online_net_name: str, target_net_name: str):
        if target_net_name in self.networks and online_net_name in self.networks:
            tau = self.config['tau']
            online_net = self.networks[online_net_name]
            target_net = self.networks[target_net_name]
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch: Dict[str, torch.Tensor], pretrain_mode: bool, step: int, full_update: bool = True) -> Dict[str, Any]:
        self.train() # Set model to training mode
        info = {}

        if pretrain_mode:
            # Only behavioral cloning loss for pretraining actor
            bc_loss_val, bc_info = self.behavioral_cloning_loss(batch)
            info.update({f'bc/{k}': v for k,v in bc_info.items()})
            total_loss = bc_loss_val
            info['pretrain/total_loss'] = total_loss.item()
        else: # Finetuning mode (IQL)
            # Value function loss
            v_loss, v_info = self.value_loss(batch)
            info.update({f'value/{k}': v for k,v in v_info.items()})

            # Critic (Q-function) loss
            c_loss, c_info = self.critic_loss(batch)
            info.update({f'critic/{k}': v for k,v in c_info.items()})

            total_loss = v_loss + c_loss

            if full_update: # Corresponds to actor_freq logic from training loop
                # Actor loss (AWR or DDPG+BC)
                # RNG handling for actor (if DDPG+BC samples actions) is implicit via global torch RNG
                a_loss, a_info = self.actor_loss(batch)
                info.update({f'actor/{k}': v for k,v in a_info.items()})
                total_loss += a_loss
            else: # Actor not updated this step
                info['actor/actor_loss'] = 0.0 # Placeholder if actor not updated

            info['finetune/total_loss'] = total_loss.item()

        # Optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Target network update (only for critic in IQL during finetuning)
        if not pretrain_mode:
            self._update_target_network('critic', 'target_critic')

        return info

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor,
                       temperature: float = 1.0, seed: int = None) -> torch.Tensor:
        self.eval() # Set model to evaluation mode
        observations = observations.to(self.device)

        if seed is not None: # Affects global torch RNG
            current_rng_state_cpu = torch.get_rng_state()
            # Save CUDA RNG state if available and seed is used
            current_rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        # Actor might take temperature for scaling stddev in its distribution
        if hasattr(self.networks['actor'], 'set_temperature'): # Common pattern
            self.networks['actor'].set_temperature(temperature)
        elif temperature != 1.0: # Default temperature is 1.0
             # Optionally log a warning if temperature is requested but not supported
             print(f"Warning: Actor does not support temperature setting, but temperature={temperature} was requested.")

        dist = self.networks['actor'](observations) # Add temperature if actor forward supports it
        actions = dist.sample() # PyTorch sample() does not take seed here directly
        actions_clipped = torch.clamp(actions, -1, 1)

        if seed is not None: # Restore RNG state
            torch.set_rng_state(current_rng_state_cpu)
            if torch.cuda.is_available() and current_rng_state_cuda:
                torch.cuda.set_rng_state_all(current_rng_state_cuda)

        return actions_clipped.cpu() # Return actions on CPU

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, ex_actions: np.ndarray, config: Dict[str, Any]):
        ex_obs_shape = ex_observations.shape
        ex_act_shape = ex_actions.shape
        # Ensure config is a plain dict for PyTorch agent
        plain_config = dict(config) if not isinstance(config, dict) else config

        return cls(config=plain_config,
                   ex_observations_shape=ex_obs_shape,
                   ex_actions_shape=ex_act_shape,
                   seed=seed)

def get_config():
    # Returns a plain Python dict for PyTorch version
    config = dict(
        agent_name='iql_pytorch',
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        value_hidden_dims=(512, 512, 512, 512), # For V and Q functions
        actor_layer_norm=False,
        value_layer_norm=True, # For V and Q functions
        discount=0.99,
        tau=0.005, # Target network update rate for Q-critic
        actor_freq=4, # Typically handled by the training loop
        expectile=0.9, # IQL expectile for V-function loss
        actor_loss='awr', # 'awr' or 'ddpgbc'
        alpha=10.0, # Temperature for AWR or BC coefficient for ddpgbc
        const_std=True, # For actor's action distribution
        encoder=None, # Placeholder, e.g., 'mlp' or 'impala_small'
    )
    return config

if __name__ == '__main__':
    print("Attempting to initialize IQLAgent (PyTorch)...")
    cfg = get_config()
    # cfg['encoder'] = 'mlp' # Example: set encoder if needed
    # cfg['actor_loss'] = 'ddpgbc' # Example: test DDPG+BC actor loss

    obs_dim = 10
    action_dim = 4
    batch_size_val = cfg['batch_size']

    # Example data (NumPy)
    example_obs_np = np.random.randn(batch_size_val, obs_dim).astype(np.float32)
    example_act_np = np.random.randn(batch_size_val, action_dim).astype(np.float32)

    # Create agent
    agent_pytorch = IQLAgent.create(seed=42,
                                    ex_observations=example_obs_np,
                                    ex_actions=example_act_np,
                                    config=cfg)
    print(f"Agent created. Device: {agent_pytorch.device}. Actor loss type: {agent_pytorch.config.get('actor_loss')}")

    # Dummy batch for testing (PyTorch tensors)
    dummy_batch = {
        'observations': torch.from_numpy(example_obs_np).float(),
        'actions': torch.from_numpy(example_act_np).float(),
        'next_observations': torch.from_numpy(example_obs_np).float(), # Use same for simplicity
        'rewards': torch.randn(batch_size_val, 1).float(),
        'masks': torch.ones(batch_size_val, 1).float(), # 보통 done이 아닐 때 1
    }

    # Test pretraining update (BC loss only)
    print("\nTesting pretraining update...")
    try:
        pretrain_info = agent_pytorch.update(dummy_batch, pretrain_mode=True, step=0)
        print("Pretraining update successful. Info:", pretrain_info)
    except Exception as e:
        print(f"Error during pretraining update: {e}")
        import traceback
        traceback.print_exc()

    # Test finetuning update (full IQL losses)
    print("\nTesting finetuning update (full)...")
    try:
        finetune_info_full = agent_pytorch.update(dummy_batch, pretrain_mode=False, step=1, full_update=True)
        print("Finetuning update (full) successful. Info:", finetune_info_full)
    except Exception as e:
        print(f"Error during finetuning update (full): {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting finetuning update (critic/value only)...")
    try:
        # actor_freq equivalent: full_update=False means actor is not updated
        finetune_info_no_actor = agent_pytorch.update(dummy_batch, pretrain_mode=False, step=2, full_update=False)
        print("Finetuning update (critic/value only) successful. Info:", finetune_info_no_actor)
    except Exception as e:
        print(f"Error during finetuning update (critic/value only): {e}")
        import traceback
        traceback.print_exc()

    # Test action sampling
    print("\nTesting action sampling...")
    try:
        # Sample actions for a subset of observations
        sampled_actions = agent_pytorch.sample_actions(torch.from_numpy(example_obs_np[:5]).float(), seed=123)
        print("Sampled actions shape:", sampled_actions.shape)
        assert sampled_actions.shape == (5, action_dim)
    except Exception as e:
        print(f"Error during action sampling: {e}")
        import traceback
        traceback.print_exc()

    print("\nBasic IQLAgent (PyTorch) functionality tests completed.")
    print("NOTE: This test assumes that PyTorch versions of 'Actor', 'Value', 'ModuleDict', and 'encoder_modules' are correctly implemented.")