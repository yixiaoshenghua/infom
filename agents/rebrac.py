import copy
from functools import partial # Keep for partial if used, otherwise remove
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections # Keep if still used for config
import numpy as np # For random noise, can be replaced by torch.randn

# Assuming encoder_modules will be converted to PyTorch and available
# from utils.encoders import encoder_modules_torch as encoder_modules
# For now, let's assume encoders are passed if not None
from utils.torch_utils import to_torch, to_numpy, set_device # Assuming these exist
from utils.flax_utils import TrainState # This is the PyTorch version of TrainState
from utils.networks import Actor, Value # These are PyTorch versions


class ReBRACAgent: # Removed (flax.struct.PyTreeNode)
    """Revisited behavior-regularized actor-critic (ReBRAC) agent (PyTorch version).

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
    """

    # rng: Any -> PyTorch handles random seeding differently. Not stored directly in agent typically.
    # network: Any -> Will be replaced by actor, critic, target_actor, target_critic, and optimizers
    # config: Any = nonpytree_field() -> config can be a normal attribute

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 config: ml_collections.ConfigDict,
                 device: torch.device):
        self.config = config
        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim # Store obs_dim

        # Define encoders (assuming they are PyTorch nn.Modules)
        # This part needs to be adapted based on how encoders are defined and passed.
        # For now, let's assume they are None or nn.Identity() if not specified.
        actor_encoder_instance = None
        critic_encoder_instance = None
        # if config.encoder is not None:
        #     # This requires encoder_modules to be a dict of PyTorch module constructors
        #     # encoder_cls = encoder_modules[config.encoder]
        #     # actor_encoder_instance = encoder_cls() # Pass relevant args if any
        #     # critic_encoder_instance = encoder_cls()
        # else:
        actor_encoder_instance = nn.Identity().to(device)
        critic_encoder_instance = nn.Identity().to(device)


        # Actor Network
        # The input_dim for Actor/Value needs to be the output_dim of the encoder + potentially other things.
        # Let's assume for now obs_dim is the feature dim after encoder.
        # If encoder is nn.Identity, then obs_dim is the raw observation dim.
        # This needs careful handling based on actual encoder output shapes.
        actor_input_dim = obs_dim # Placeholder: adjust if encoder changes dim
        self.actor = Actor(
            input_dim=actor_input_dim,
            action_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            tanh_squash=config.tanh_squash,
            state_dependent_std=False, # ReBRAC seems to use const_std=True (log_stds=0) or learnable param
            const_std_init=0.0, # Corresponds to const_std=True in Flax (zeros_like means)
            final_fc_init_scale=config.actor_fc_scale,
            encoder=actor_encoder_instance # Pass the instantiated encoder
        ).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)

        # Critic Network
        # Input for critic: obs_features + action_dim
        critic_input_dim = actor_input_dim + action_dim # Placeholder
        self.critic = Value(
            input_dim=critic_input_dim,
            hidden_dims=config.value_hidden_dims,
            output_dim=1, # Critic outputs a single Q-value per ensemble member
            layer_norm=config.value_layer_norm,
            num_ensembles=2, # ReBRAC uses 2 critics
            encoder=critic_encoder_instance # Pass the instantiated encoder
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self.train_state_actor = TrainState(model=self.actor, optimizer=self.actor_optimizer)
        self.train_state_critic = TrainState(model=self.critic, optimizer=self.critic_optimizer)

        self.total_it = 0


    def _compute_critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        masks = batch['masks']

        with torch.no_grad():
            next_dist = self.target_actor(next_observations)
            next_actions_target = next_dist.mode # Get mode of the distribution

            # Add noise to target actions
            noise = (torch.randn_like(next_actions_target) * self.config.actor_noise).clamp(
                -self.config.actor_noise_clip, self.config.actor_noise_clip
            )
            next_actions_target = (next_actions_target + noise).clamp(-1.0, 1.0)

            # Get Q-values from target critic
            next_qs = self.target_critic(next_observations, next_actions_target) # Shape (num_ensembles, batch_size)
            next_q, _ = torch.min(next_qs, dim=0) # Min over ensembles

            # Critic BC regularization term (optional, if alpha_critic > 0)
            # Original ReBRAC paper and this code seem to use next_actions from policy, not from batch['next_actions']
            # mse_next_action = F.mse_loss(next_actions_target, batch['next_actions'], reduction='none').sum(axis=-1)
            # next_q = next_q - self.config.alpha_critic * mse_next_action
            # The provided JAX code uses batch['next_actions'] for mse calc with policy's next_actions
            if 'next_actions' in batch and self.config.alpha_critic > 0 : # Ensure next_actions is in batch
                 mse_penalty_critic = ((next_actions_target - batch['next_actions'])**2).sum(dim=-1)
                 next_q = next_q - self.config.alpha_critic * mse_penalty_critic


            target_q_val = rewards + self.config.discount * masks * next_q

        # Get current Q-values
        current_qs = self.critic(observations, actions) # Shape (num_ensembles, batch_size)

        # MSE loss for each critic against the target Q
        critic_loss = F.mse_loss(current_qs[0], target_q_val) + F.mse_loss(current_qs[1], target_q_val)

        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': current_qs.mean().item(),
            'q_min': current_qs.min().item(),
            'q_max': current_qs.max().item(),
        }
        return critic_loss, info

    def _compute_actor_loss_bc(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Behavioral cloning loss for pretraining."""
        observations = batch['observations']
        actions = batch['actions']

        dist = self.actor(observations)
        log_prob = dist.log_prob(actions) # Assumes dist has log_prob method
        bc_loss = -log_prob.mean()

        info = {
            'bc_loss': bc_loss.item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse_bc_actions': F.mse_loss(dist.mode, actions).item(),
        }
        return bc_loss, info

    def _compute_actor_loss_rl(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """RL actor loss for finetuning."""
        observations = batch['observations']

        dist = self.actor(observations)
        policy_actions = dist.mode # Use mode for actor loss calculation

        # Q-value loss component
        qs = self.critic(observations, policy_actions) # Use current critic (not target)
        q, _ = torch.min(qs, dim=0) # Min over ensembles

        # BC regularization component
        # mse_bc = F.mse_loss(policy_actions, batch['actions'], reduction='none').sum(dim=-1) # Per-sample MSE
        mse_bc = ((policy_actions - batch['actions'])**2).sum(dim=-1)


        # Normalize Q-values (TD3 trick)
        lam = 1.0 / (torch.abs(q).mean().detach() + 1e-8) # Add epsilon for stability

        actor_q_loss = -(lam * q).mean()
        actor_bc_loss = (self.config.alpha_actor * mse_bc).mean()

        total_actor_loss = actor_q_loss + actor_bc_loss

        std_metric = 0.0
        try:
            if self.config.tanh_squash:
                # For TanhTransformedNormal, base_dist is Independent(Normal)
                # Accessing stddev of the Normal distribution
                if hasattr(dist, 'base_dist') and hasattr(dist.base_dist, 'base_dist'): # TanhTransform -> Independent -> Normal
                     std_metric = dist.base_dist.base_dist.stddev.mean().item()
            elif hasattr(dist, 'stddev'): # If it's a simple Normal Independent
                std_metric = dist.stddev.mean().item()
        except Exception:
            pass # std calculation might fail for complex distributions

        info = {
            'actor_total_loss': total_actor_loss.item(),
            'actor_q_loss': actor_q_loss.item(),
            'actor_bc_loss': actor_bc_loss.item(),
            'actor_std': std_metric,
            'actor_mse_bc_actions': mse_bc.mean().item(), # Average MSE for logging
        }
        return total_actor_loss, info

    def _target_soft_update(self):
        """Soft update of target networks."""
        tau = self.config.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def pretrain(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Pre-train the agent using behavioral cloning."""
        self.total_it +=1
        # Convert numpy batch to torch tensors on the correct device
        torch_batch = {k: to_torch(v, self.device) for k, v in batch.items()}

        actor_loss, actor_info = self._compute_actor_loss_bc(torch_batch)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # self.train_state_actor.update(actor_loss) # Using TrainState's update

        # In pretraining, only actor is updated. Critic can be updated too if desired.
        # For ReBRAC pretraining, often only actor BC is done.
        # If critic pretraining is needed, it would involve a critic_loss calculation.
        # The JAX code's pretraining_loss only includes behavioral_cloning_loss for the actor.

        return actor_info


    def finetune(self, batch: Dict[str, np.ndarray], full_update: bool = True) -> Dict[str, float]:
        """Fine-tune the agent (both actor and critic)."""
        self.total_it +=1
        torch_batch = {k: to_torch(v, self.device) for k, v in batch.items()}
        all_info = {}

        # Critic update
        critic_loss, critic_info = self._compute_critic_loss(torch_batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # self.train_state_critic.update(critic_loss)
        all_info.update(critic_info)

        if full_update: # Corresponds to actor_freq in main loop
            # Actor update
            actor_loss, actor_info = self._compute_actor_loss_rl(torch_batch)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.train_state_actor.update(actor_loss)
            all_info.update(actor_info)

            # Target network soft updates
            self._target_soft_update()

        return all_info


    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0, add_noise: bool = True) -> np.ndarray:
        """Sample actions from the actor, with optional noise for exploration."""
        obs_torch = to_torch(observations, self.device)

        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad():
            dist = self.actor(obs_torch, temperature=temperature)
            actions_torch = dist.mode # Start with the mode of the distribution

            if add_noise: # Add exploration noise if required (e.g., during data collection)
                # This noise is different from target network noise. This is for exploration.
                # The original JAX sample_actions adds noise based on config.actor_noise
                # This is typical for TD3-style exploration.
                noise = (torch.randn_like(actions_torch) * self.config.actor_noise * temperature) # Scale noise by temp?
                noise = noise.clamp(-self.config.actor_noise_clip, self.config.actor_noise_clip)
                actions_torch = (actions_torch + noise).clamp(-1.0, 1.0)
        self.actor.train() # Set actor back to training mode

        return to_numpy(actions_torch)

    @classmethod
    def create(cls,
               seed: int, # For reproducibility
               ex_observations: np.ndarray,
               ex_actions: np.ndarray,
               config: ml_collections.ConfigDict,
              ) -> 'ReBRACAgent': # Return type hint
        """Create a new ReBRACAgent instance."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = set_device() # Get device from torch_utils

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        # The config from files like 'agents/rebrac.py' is an ml_collections.ConfigDict
        # We need to ensure all necessary fields are present or provide defaults.
        # The original JAX create method initializes networks and TrainState.
        # In PyTorch, __init__ does this. 'create' becomes a factory.

        return cls(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the agent for saving."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'config': self.config.to_dict() # Save config as dict
        }

    def load_state_dict(self, loaded_state: Dict[str, Any]):
        """Loads the agent's state."""
        self.actor.load_state_dict(loaded_state['actor_state_dict'])
        self.critic.load_state_dict(loaded_state['critic_state_dict'])
        self.target_actor.load_state_dict(loaded_state['target_actor_state_dict'])
        self.target_critic.load_state_dict(loaded_state['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(loaded_state['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(loaded_state['critic_optimizer_state_dict'])
        self.total_it = loaded_state.get('total_it', 0)
        # Config loading might be more complex if ml_collections is deeply nested
        # For now, assuming config passed at init is sufficient or can be updated if needed.
        # loaded_config = ml_collections.ConfigDict(loaded_state.get('config', {}))
        # self.config.update(loaded_config)


def get_config():
    # This config remains mostly the same, but its interpretation changes for PyTorch modules.
    # For example, encoder names would map to PyTorch encoder modules.
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rebrac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_critic=0.0,  # Critic BC coefficient (for policy eval part of critic loss)
            actor_freq=4,  # Actor update frequency.
            actor_noise=0.2,  # Actor noise scale (for exploration and target policy smoothing).
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name (e.g., 'mlp', 'impala_small').
        )
    )
    return config
