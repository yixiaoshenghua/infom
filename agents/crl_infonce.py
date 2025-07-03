import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ml_collections
from typing import Any, Dict, Tuple, Optional

# Assuming these utilities are in place and match the JAX versions' functionalities
from utils.torch_utils import to_torch, to_numpy # set_device can be called by main script
from utils.networks import Actor, Value, GCBilinearValue # PyTorch versions
from utils.encoders import encoder_modules_torch # PyTorch versions
# GCEncoder might be used if state_encoder/goal_encoder in GCBilinearValue are GCEncoders themselves.
# For now, assuming base encoders are directly passed.


class CRLInfoNCEAgentPytorch(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict, ex_observations: torch.Tensor, ex_actions: torch.Tensor, seed: int = 42):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        action_dim = ex_actions.shape[-1]

        # Determine feature dimension after encoding
        # This logic needs to be robust based on encoder type and observation shape
        obs_example_for_encoder = ex_observations[:1].to(self.device)
        if config.encoder is not None:
            # This is a simplified way to get encoder_output_dim.
            # Real implementation might need more sophisticated handling of encoder args (channels, etc.)
            if config.encoder == 'mlp':
                _obs_feat_dim = ex_observations.shape[-1]
                _encoder_fn_args = {'input_dim': _obs_feat_dim}
            elif 'impala' in config.encoder or 'resnet' in config.encoder:
                # For CNNs, input_dim is usually in_channels.
                # We need a robust way to get example output.
                # This part is highly dependent on specific encoder constructor.
                # For now, assume encoder factory can take a generic config or specific args are in config.encoder_kwargs
                # This is a placeholder section.
                _encoder_fn_args = config.get('encoder_kwargs', {})
                if 'in_channels' not in _encoder_fn_args and obs_example_for_encoder.ndim == 4 : # (B,C,H,W)
                     _encoder_fn_args['in_channels'] = obs_example_for_encoder.shape[1]
                if 'mlp_input_dim_after_conv' not in _encoder_fn_args and 'impala' in config.encoder: # Impala specific
                     _encoder_fn_args['mlp_input_dim_after_conv'] = 256 # A common default, should be configured
            else:
                _encoder_fn_args = {} # Default args for other encoders

            try:
                _temp_encoder = encoder_modules_torch[config.encoder](**_encoder_fn_args)
                _temp_encoder.to(self.device)
                encoder_output_dim = _temp_encoder(obs_example_for_encoder).shape[-1]
            except Exception as e:
                print(f"Warning: Could not infer encoder_output_dim for {config.encoder} due to: {e}. Defaulting or check config.")
                # Fallback or error. For now, let's assume a common dim if not inferable, or require it in config.
                encoder_output_dim = config.get('encoder_output_dim', 512) # Default if inference fails


            # Instantiate actual encoders
            encoders_dict = {
                name: encoder_modules_torch[config.encoder](**_encoder_fn_args).to(self.device)
                for name in ['reward_encoder', 'critic_state_encoder', 'critic_goal_encoder', 'actor_encoder']
            }
            current_feature_dim = encoder_output_dim
        else: # No encoder
            current_feature_dim = ex_observations.shape[-1]
            identity_encoder = nn.Identity().to(self.device)
            encoders_dict = {name: identity_encoder for name in ['reward_encoder', 'critic_state_encoder', 'critic_goal_encoder', 'actor_encoder']}

        reward_net = Value(
            input_dim=current_feature_dim,
            hidden_dims=config.reward_hidden_dims,
            layer_norm=config.reward_layer_norm,
            encoder=encoders_dict['reward_encoder']
        )

        critic_phi_input_dim = current_feature_dim # state_encoder output dim
        # GCBilinearValue's state_encoder handles obs. Actions are concatenated after state_encoding, if present.
        # So obs_action_dim for the MLP inside GCBilinearValue will be current_feature_dim + action_dim

        critic_net = GCBilinearValue(
            obs_action_dim=current_feature_dim + action_dim, # Dim for MLP after state_encoder + action concat
            goal_feat_dim=current_feature_dim, # Dim for MLP after goal_encoder
            hidden_dims=config.value_hidden_dims,
            latent_dim=config.latent_dim,
            layer_norm=config.value_layer_norm,
            num_ensembles=2,
            value_exp=True,
            state_encoder=encoders_dict['critic_state_encoder'],
            goal_encoder=encoders_dict['critic_goal_encoder']
        )

        actor_net_input_dim = current_feature_dim
        actor_net = Actor(
            input_dim=actor_net_input_dim,
            action_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            state_dependent_std=False,
            layer_norm=config.actor_layer_norm,
            const_std_init=0.0 if config.const_std else -1.0, # Placeholder, see below
            encoder=encoders_dict['actor_encoder']
        )
        if config.const_std:
            actor_net.log_stds = nn.Parameter(torch.zeros(action_dim, device=self.device), requires_grad=False)


        self.network = nn.ModuleDict({
            'reward': reward_net,
            'critic': critic_net,
            'actor': actor_net
        }).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

    def reward_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        rewards = batch['rewards']
        # Value network's encoder will process observations.
        reward_preds = self.network.reward(observations).squeeze(-1)
        reward_loss = F.mse_loss(reward_preds, rewards.squeeze(-1))
        return reward_loss, {'reward_loss': reward_loss.item()}

    def contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        actions = batch['actions']
        value_goals = batch['value_goals'] # These are goal observations
        batch_size = observations.shape[0]

        raw_v, phi, psi = self.network.critic(observations, value_goals, actions=actions, info=True)

        if self.network.critic.num_ensembles > 1:
            if isinstance(phi, list): phi = torch.stack(phi, dim=0) # (E, B, D)
            if isinstance(psi, list): psi = torch.stack(psi, dim=0) # (E, B, D)
        else:
            phi = phi.unsqueeze(0) # (1, B, D)
            psi = psi.unsqueeze(0) # (1, B, D)
            raw_v = raw_v.unsqueeze(0) # (1, B)

        # phi: (E, B_obs, D), psi: (E, B_goals, D) -> Assuming B_obs = B_goals = B (batch_size)
        # einsum 'ebk,egk->bge' gives (B, B, E)
        logits = torch.einsum('ebk,egk->bge', phi, psi) / torch.sqrt(torch.tensor(phi.shape[-1], dtype=torch.float32, device=self.device))

        I_labels = torch.arange(batch_size, device=self.device)

        total_contrastive_loss = 0.0
        num_ensembles = logits.shape[2]

        for e_idx in range(num_ensembles):
            ensemble_logits = logits[:, :, e_idx] # Shape (B_obs, B_goals)

            if self.config.contrastive_loss == 'forward_infonce':
                ce_loss = F.cross_entropy(ensemble_logits, I_labels)
                logsumexp_penalty = self.config.logsumexp_penalty_coeff * (torch.logsumexp(ensemble_logits, dim=1)**2).mean()
                current_loss = ce_loss + logsumexp_penalty
            elif self.config.contrastive_loss == 'symmetric_infonce':
                ce_loss1 = F.cross_entropy(ensemble_logits, I_labels)
                ce_loss2 = F.cross_entropy(ensemble_logits.transpose(0,1), I_labels)
                current_loss = ce_loss1 + ce_loss2
            else:
                raise NotImplementedError(f"Unknown contrastive loss: {self.config.contrastive_loss}")
            total_contrastive_loss += current_loss

        contrastive_loss_val = total_contrastive_loss / num_ensembles

        mean_logits = logits.mean(dim=-1)
        categorial_correct = (torch.argmax(mean_logits, dim=1) == I_labels)
        I_matrix_float = torch.eye(batch_size, device=self.device)
        logits_pos = (mean_logits * I_matrix_float).sum() / (I_matrix_float.sum() + 1e-8)
        logits_neg = (mean_logits * (1 - I_matrix_float)).sum() / ((1 - I_matrix_float).sum() + 1e-8)
        v_for_stats = raw_v.mean(dim=0)

        return contrastive_loss_val, {
            'contrastive_loss': contrastive_loss_val.item(),
            'v_mean': v_for_stats.mean().item(),
            'v_max': v_for_stats.max().item(),
            'v_min': v_for_stats.min().item(),
            'categorical_accuracy': categorial_correct.float().mean().item(),
            'logits_pos': logits_pos.item(),
            'logits_neg': logits_neg.item(),
            'logits': mean_logits.mean().item(),
        }

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        actions = batch['actions']

        dist = self.network.actor(observations) # Actor's encoder handles observations
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        with torch.no_grad():
            mse = F.mse_loss(dist.mode, actions)
            std_tensor = dist.base_dist.stddev if hasattr(dist, 'base_dist') else dist.stddev
            actor_std = std_tensor.mean()

        return bc_loss, {
            'bc_loss': bc_loss.item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse.item(),
            'std': actor_std.item(),
        }

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        actor_goals = batch.get('actor_goals', observations) # Default to observations if not present

        def value_transform(x): return torch.log(torch.clamp(x, min=1e-6))

        dist = self.network.actor(observations)

        with torch.no_grad():
            q_actions = torch.clamp(dist.mode if self.config.const_std else dist.sample(), -1, 1)

        # Critic output is (E,B) or (B,)
        q_values_exp = self.network.critic(observations, actor_goals, q_actions, info=False)
        if q_values_exp.ndim == 1: q_values_exp = q_values_exp.unsqueeze(0)

        transformed_q_values = value_transform(q_values_exp)

        if transformed_q_values.shape[0] >= 2: # Ensemble min
            q_critic_transformed = torch.min(transformed_q_values[0], transformed_q_values[1])
        else: # Single value
            q_critic_transformed = transformed_q_values.squeeze(0)

        q_sg_values = q_critic_transformed # Shape (B,)
        log_ratios = q_sg_values - torch.logsumexp(q_sg_values, dim=0) + \
                     torch.log(torch.tensor(q_sg_values.shape[0], device=self.device, dtype=torch.float32))

        reward_preds = self.network.reward(actor_goals).squeeze(-1)
        q = torch.exp(log_ratios) * reward_preds

        q_loss_val = -q.mean()
        if self.config.normalize_q_loss:
            lam = (1.0 / (torch.abs(q).mean() + 1e-8)).detach()
            q_loss_val = lam * q_loss_val

        log_prob_bc = dist.log_prob(batch['actions'])
        bc_loss_val = -(self.config.alpha * log_prob_bc).mean()
        actor_loss_val = q_loss_val + bc_loss_val

        with torch.no_grad():
            mse_val = F.mse_loss(dist.mode, batch['actions'])
            std_tensor = dist.base_dist.stddev if hasattr(dist, 'base_dist') else dist.stddev
            actor_std = std_tensor.mean()

        return actor_loss_val, {
            'actor_loss': actor_loss_val.item(), 'q_loss': q_loss_val.item(), 'bc_loss': bc_loss_val.item(),
            'q_mean': q.mean().item(), 'q_abs_mean': torch.abs(q).mean().item(),
            'bc_log_prob': log_prob_bc.mean().item(), 'mse': mse_val.item(), 'std': actor_std.item(),
        }

    def compute_pretraining_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        info = {}
        critic_loss, critic_info = self.contrastive_loss(batch)
        info.update({f'critic/{k}': v for k,v in critic_info.items()})
        bc_loss, bc_info = self.behavioral_cloning_loss(batch)
        info.update({f'bc/{k}': v for k,v in bc_info.items()})
        total_loss = critic_loss + bc_loss
        info['pretrain/total_loss'] = total_loss.item()
        return total_loss, info

    def pretrain_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        batch_torch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        total_loss, info = self.compute_pretraining_loss(batch_torch)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return info

    def compute_finetuning_loss(self, batch: Dict[str, torch.Tensor], full_update: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        info = {}
        reward_loss_val, reward_info = self.reward_loss(batch)
        info.update({f'reward/{k}': v for k,v in reward_info.items()})
        critic_loss_val, critic_info = self.contrastive_loss(batch)
        info.update({f'critic/{k}': v for k,v in critic_info.items()})

        if full_update:
            actor_loss_val, actor_info = self.actor_loss(batch)
            info.update({f'actor/{k}': v for k,v in actor_info.items()})
        else:
            actor_loss_val = torch.tensor(0.0, device=self.device)
            # Populate default actor_info if needed by logger
            default_actor_metrics = {'actor_loss':0.0, 'q_loss':0.0, 'bc_loss':0.0, 'q_mean':0.0,
                                     'q_abs_mean':0.0, 'bc_log_prob':0.0, 'mse':0.0, 'std':0.0}
            info.update({f'actor/{k}': v for k,v in default_actor_metrics.items()})

        total_loss = reward_loss_val + critic_loss_val + actor_loss_val
        info['finetune/total_loss'] = total_loss.item()
        return total_loss, info

    def finetune_step(self, batch: Dict[str, Any], full_update: bool = True) -> Dict[str, float]:
        batch_torch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        total_loss, info = self.compute_finetuning_loss(batch_torch, full_update)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return info

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        observations = observations.to(self.device)
        # Assuming Actor's forward method can accept temperature for scaling std
        if hasattr(self.network.actor, 'set_temperature'): # A possible way to handle temperature
            self.network.actor.set_temperature(temperature)

        dist = self.network.actor(observations) # Pass temperature if actor takes it in forward
        actions = dist.sample()
        actions = torch.clamp(actions, -1, 1)
        return actions.cpu()

def get_config():
    config = ml_collections.ConfigDict()
    config.agent_name = 'crl_infonce'
    config.lr = 3e-4
    config.batch_size = 256
    config.actor_hidden_dims = (512, 512, 512, 512)
    config.value_hidden_dims = (512, 512, 512, 512)
    config.reward_hidden_dims = (512, 512, 512, 512)
    config.latent_dim = 512
    config.actor_layer_norm = False
    config.value_layer_norm = True
    config.reward_layer_norm = True
    config.discount = 0.99
    config.actor_freq = 4
    config.contrastive_loss = 'forward_infonce'
    config.logsumexp_penalty_coeff = 0.01
    config.normalize_q_loss = True
    config.alpha = 0.1
    config.const_std = True
    config.encoder = ml_collections.config_dict.placeholder(str)
    # config.encoder_output_dim = 512 # Example: if known for non-MLP encoders
    # config.encoder_kwargs = {} # For passing specific args to encoder factories
    config.relabel_reward = False
    config.value_p_curgoal = 0.0
    config.value_p_trajgoal = 1.0
    config.value_p_randomgoal = 0.0
    config.value_geom_sample = True
    config.value_geom_start_offset = 0
    config.actor_p_curgoal = 0.0
    config.actor_p_trajgoal = 1.0
    config.actor_p_randomgoal = 0.0
    config.actor_geom_sample = False
    config.actor_geom_start_offset = 0
    config.gc_negative = True
    return config
