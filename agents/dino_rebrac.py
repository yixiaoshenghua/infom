import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ml_collections
import copy # For deepcopying networks
from typing import Any, Dict, Tuple, Optional

from utils.torch_utils import to_torch, to_numpy
from utils.networks import Actor, Value, Param # PyTorch versions
from utils.encoders import encoder_modules_torch # PyTorch versions

class DINOReBRACAgentPytorch(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict, ex_observations: torch.Tensor, ex_actions: torch.Tensor, seed: int = 42):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        action_dim = ex_actions.shape[-1]

        obs_example_for_encoder = ex_observations[:1].to(self.device)
        if config.encoder is not None:
            encoder_fn_args = config.get('encoder_kwargs', {})
            if isinstance(encoder_fn_args, ml_collections.ConfigDict): # Convert if it's ConfigDict
                encoder_fn_args = encoder_fn_args.to_dict()

            if config.encoder == 'mlp':
                if 'input_dim' not in encoder_fn_args: encoder_fn_args['input_dim'] = ex_observations.shape[-1]
            elif 'impala' in config.encoder or 'resnet' in config.encoder:
                if 'in_channels' not in encoder_fn_args and obs_example_for_encoder.ndim == 4:
                     encoder_fn_args['in_channels'] = obs_example_for_encoder.shape[1]
                if 'impala' in config.encoder: # Impala specific defaults if not in kwargs
                     encoder_fn_args.setdefault('mlp_input_dim_after_conv', config.get('impala_mlp_input_dim',256))
                     encoder_fn_args.setdefault('width', config.get('impala_width',1))
                     encoder_fn_args.setdefault('stack_sizes', config.get('impala_stack_sizes',(16,32,32)))
                     encoder_fn_args.setdefault('num_blocks', config.get('impala_num_blocks',2))


            try:
                _temp_encoder = encoder_modules_torch[config.encoder](**encoder_fn_args)
                _temp_encoder.to(self.device)
                encoder_output_dim = _temp_encoder(obs_example_for_encoder).shape[-1]
            except Exception as e:
                print(f"Warning: Could not infer encoder_output_dim for {config.encoder} (args: {encoder_fn_args}) due to: {e}. Defaulting.")
                encoder_output_dim = config.get('encoder_output_dim', 512)
        else:
            raise ValueError("DINOReBRAC requires an encoder.")

        online_encoder = encoder_modules_torch[config.encoder](**encoder_fn_args).to(self.device)

        critic_mlp_input_dim = encoder_output_dim + action_dim # Critic's MLP takes encoded_obs + actions
        actor_mlp_input_dim = encoder_output_dim # Actor's MLP takes encoded_obs

        online_critic = Value(
            input_dim=critic_mlp_input_dim,
            hidden_dims=config.value_hidden_dims,
            layer_norm=config.value_layer_norm,
            num_ensembles=2,
            encoder=None
        )
        online_actor = Actor(
            input_dim=actor_mlp_input_dim,
            action_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            tanh_squash=config.tanh_squash,
            state_dependent_std=False,
            const_std_init=0.0,
            final_fc_init_scale=config.actor_fc_scale,
            encoder=None
        )

        target_encoder = copy.deepcopy(online_encoder).to(self.device)
        target_critic = copy.deepcopy(online_critic).to(self.device)
        target_actor = copy.deepcopy(online_actor).to(self.device)
        target_repr_center_net = Param(shape=(encoder_output_dim,)).to(self.device)

        self.network = nn.ModuleDict({
            'encoder': online_encoder, 'critic': online_critic, 'actor': online_actor,
            'target_encoder': target_encoder, 'target_critic': target_critic, 'target_actor': target_actor,
            'target_repr_center': target_repr_center_net
        })

        self.network.target_encoder.load_state_dict(self.network.encoder.state_dict())
        self.network.target_critic.load_state_dict(self.network.critic.state_dict())
        self.network.target_actor.load_state_dict(self.network.actor.state_dict())

        online_params = list(self.network.encoder.parameters()) + \
                        list(self.network.critic.parameters()) + \
                        list(self.network.actor.parameters())
        self.optimizer = optim.Adam(online_params, lr=config.lr)

    @staticmethod
    def dino_loss(target_repr: torch.Tensor, repr_student: torch.Tensor,
                  target_repr_center: torch.Tensor,
                  target_temp: float = 0.04, temp: float = 0.1) -> torch.Tensor:
        target_repr = target_repr.detach()
        target_probs = F.softmax((target_repr - target_repr_center) / target_temp, dim=-1)
        student_log_probs = F.log_softmax(repr_student / temp, dim=-1)
        loss = -(target_probs * student_log_probs).sum(dim=-1).mean()
        return loss

    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            next_reprs = self.network.target_encoder(batch['next_observations'])
            next_dist = self.network.target_actor(next_reprs)
            next_actions_mode = next_dist.mode
            noise = torch.randn_like(next_actions_mode) * self.config.actor_noise
            noise = torch.clamp(noise, -self.config.actor_noise_clip, self.config.actor_noise_clip)
            next_actions_noisy = torch.clamp(next_actions_mode + noise, -1, 1)

            # Value net's input_dim is for encoded_obs + actions
            next_qs_input = torch.cat([next_reprs, next_actions_noisy], dim=-1)
            next_qs_raw = self.network.target_critic(next_qs_input)
            next_q = torch.min(next_qs_raw, dim=0)[0]

            mse_penalty = torch.sum((next_actions_noisy - batch['next_actions'])**2, dim=-1)
            next_q_penalized = next_q - self.config.alpha_critic * mse_penalty
            target_q = batch['rewards'].squeeze(-1) + self.config.discount * batch['masks'].squeeze(-1) * next_q_penalized

        current_reprs = self.network.encoder(batch['observations'])
        current_qs_input = torch.cat([current_reprs, batch['actions']], dim=-1)
        current_qs_raw = self.network.critic(current_qs_input)

        critic_loss_val = F.mse_loss(current_qs_raw[0], target_q) + F.mse_loss(current_qs_raw[1], target_q)
        q_for_stats = current_qs_raw.mean(dim=0)

        return critic_loss_val, {
            'critic_loss': critic_loss_val.item(), 'q_mean': q_for_stats.mean().item(),
            'q_max': q_for_stats.max().item(), 'q_min': q_for_stats.min().item(),
        }

    def representation_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations']
        if self.config.encoder == 'mlp':
            noise1 = torch.clamp(torch.randn_like(observations) * self.config.repr_noise, -self.config.repr_noise_clip, self.config.repr_noise_clip)
            aug1_obs, aug2_obs = observations + noise1, observations + torch.clamp(torch.randn_like(observations) * self.config.repr_noise, -self.config.repr_noise_clip, self.config.repr_noise_clip)
        elif 'impala' in self.config.encoder or 'resnet' in self.config.encoder:
            if 'aug1_observations' not in batch or 'aug2_observations' not in batch:
                 raise ValueError("Image augmentations missing for DINO.")
            aug1_obs, aug2_obs = batch['aug1_observations'], batch['aug2_observations']
        else: aug1_obs, aug2_obs = observations, observations

        repr1_student, repr2_student = self.network.encoder(aug1_obs), self.network.encoder(aug2_obs)
        with torch.no_grad():
            repr1_teacher, repr2_teacher = self.network.target_encoder(aug1_obs), self.network.target_encoder(aug2_obs)

        loss1 = self.dino_loss(repr1_teacher, repr1_student, self.network.target_repr_center(), self.config.target_repr_temp, self.config.repr_temp)
        loss2 = self.dino_loss(repr2_teacher, repr2_student, self.network.target_repr_center(), self.config.target_repr_temp, self.config.repr_temp)
        total_repr_loss = (loss1 + loss2) / 2.0
        return total_repr_loss, {'repr_loss': total_repr_loss.item()}

    def behavioral_cloning_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad(): reprs = self.network.encoder(batch['observations']).detach()
        dist = self.network.actor(reprs)
        log_prob = dist.log_prob(batch['actions'])
        bc_loss = -log_prob.mean()
        with torch.no_grad(): mse = F.mse_loss(dist.mode, batch['actions'])
        return bc_loss, {'bc_loss': bc_loss.item(), 'bc_log_prob': log_prob.mean().item(), 'mse': mse.item()}

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        reprs = self.network.encoder(batch['observations'])
        dist = self.network.actor(reprs)
        actions_mode = dist.mode

        # Critic's MLP takes encoded_obs + actions
        critic_input_for_actor_loss = torch.cat([reprs, actions_mode], dim=-1)
        qs_raw = self.network.critic(critic_input_for_actor_loss)
        q_min = torch.min(qs_raw, dim=0)[0]

        q_loss_val = -q_min.mean()
        if self.config.normalize_q_loss:
            lam = (1.0 / (torch.abs(q_min).mean() + 1e-8)).detach()
            q_loss_val = lam * q_loss_val

        mse_bc = torch.sum((actions_mode - batch['actions'])**2, dim=-1).mean()
        bc_loss_val = self.config.alpha_actor * mse_bc
        total_actor_loss = q_loss_val + bc_loss_val

        with torch.no_grad():
            std_tensor = dist.base_dist.stddev if hasattr(dist, 'base_dist') else dist.stddev
            actor_std = std_tensor.mean()
            mse_for_log = F.mse_loss(actions_mode, batch['actions'])

        return total_actor_loss, {
            'total_loss': total_actor_loss.item(), 'actor_loss': q_loss_val.item(),
            'bc_loss': bc_loss_val.item(), 'std': actor_std.item(), 'mse': mse_for_log.item(),
        }

    def _update_target_network_ema(self, online_net_key: str, target_net_key: str):
        tau = self.config.tau
        for po, pt in zip(self.network[online_net_key].parameters(), self.network[target_net_key].parameters()):
            pt.data.copy_(tau * po.data + (1.0 - tau) * pt.data)

    def target_center_update(self, batch: Dict[str, torch.Tensor]):
        observations = batch['observations']
        if self.config.encoder == 'mlp':
            noise1 = torch.clamp(torch.randn_like(observations) * self.config.repr_noise, -self.config.repr_noise_clip, self.config.repr_noise_clip)
            aug1_obs, aug2_obs = observations + noise1, observations + torch.clamp(torch.randn_like(observations) * self.config.repr_noise, -self.config.repr_noise_clip, self.config.repr_noise_clip)
        elif 'impala' in self.config.encoder or 'resnet' in self.config.encoder:
            aug1_obs, aug2_obs = batch['aug1_observations'], batch['aug2_observations']
        else: aug1_obs, aug2_obs = observations, observations

        with torch.no_grad():
            target_repr1, target_repr2 = self.network.target_encoder(aug1_obs), self.network.target_encoder(aug2_obs)

        all_target_features = torch.cat([target_repr1, target_repr2], dim=0)
        new_center_batch_mean = all_target_features.mean(dim=0)

        current_center = self.network.target_repr_center()
        updated_center = new_center_batch_mean * self.config.target_repr_center_tau + current_center * (1.0 - self.config.target_repr_center_tau)
        self.network.target_repr_center.value.data.copy_(updated_center.detach())

    def compute_pretraining_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        info = {}
        repr_loss, repr_info = self.representation_loss(batch); info.update({f'repr/{k}': v for k,v in repr_info.items()})
        bc_loss, bc_info = self.behavioral_cloning_loss(batch); info.update({f'bc/{k}': v for k,v in bc_info.items()})
        total_loss = repr_loss + bc_loss
        info['pretrain/total_loss'] = total_loss.item()
        return total_loss, info

    def pretrain_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        batch_torch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        total_loss, info = self.compute_pretraining_loss(batch_torch)
        self.optimizer.zero_grad(); total_loss.backward(); self.optimizer.step()
        self._update_target_network_ema('encoder', 'target_encoder')
        self.target_center_update(batch_torch)
        return info

    def compute_finetuning_loss(self, batch: Dict[str, torch.Tensor], full_update: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        info = {}
        critic_loss_val, critic_info = self.critic_loss(batch); info.update({f'critic/{k}': v for k,v in critic_info.items()})
        if full_update:
            actor_loss_val, actor_info = self.actor_loss(batch); info.update({f'actor/{k}': v for k,v in actor_info.items()})
        else:
            actor_loss_val = torch.tensor(0.0, device=self.device)
            info.update({f'actor/{k}': 0.0 for k in ['total_loss', 'actor_loss', 'bc_loss', 'std', 'mse']})
        total_loss = critic_loss_val + actor_loss_val
        info['finetune/total_loss'] = total_loss.item()
        return total_loss, info

    def finetune_step(self, batch: Dict[str, Any], full_update: bool = True) -> Dict[str, float]:
        batch_torch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        total_loss, info = self.compute_finetuning_loss(batch_torch, full_update)
        self.optimizer.zero_grad(); total_loss.backward(); self.optimizer.step()
        if full_update:
            self._update_target_network_ema('encoder', 'target_encoder')
            self._update_target_network_ema('critic', 'target_critic')
            self._update_target_network_ema('actor', 'target_actor')
            self.target_center_update(batch_torch)
        return info

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        observations = observations.to(self.device)
        reprs = self.network.encoder(observations)
        dist = self.network.actor(reprs)
        actions_mode = dist.mode
        noise = torch.randn_like(actions_mode) * self.config.actor_noise * temperature
        noise = torch.clamp(noise, -self.config.actor_noise_clip, self.config.actor_noise_clip)
        final_actions = torch.clamp(actions_mode + noise, -1, 1)
        return final_actions.cpu()

def get_config():
    config = ml_collections.ConfigDict()
    config.agent_name = 'dino_rebrac'; config.lr = 3e-4; config.batch_size = 256
    config.actor_hidden_dims = (512, 512, 512, 512); config.value_hidden_dims = (512, 512, 512, 512)
    config.actor_layer_norm = False; config.value_layer_norm = True
    config.discount = 0.99; config.tau = 0.005; config.tanh_squash = True
    config.actor_fc_scale = 0.01; config.alpha_actor = 0.0; config.alpha_critic = 0.0
    config.actor_freq = 4; config.normalize_q_loss = True
    config.actor_noise = 0.2; config.actor_noise_clip = 0.5
    config.repr_noise = 0.5; config.repr_noise_clip = 1.0
    config.target_repr_center_tau = 0.1; config.repr_temp = 0.1; config.target_repr_temp = 0.04
    config.encoder = ml_collections.config_dict.placeholder(str)
    # Example encoder_kwargs, to be set by user based on chosen encoder and obs_spec
    # config.encoder_kwargs = {'input_dim': None, 'in_channels': None, 'mlp_input_dim_after_conv': 256}
    # config.encoder_output_dim = 512 # Can be inferred or specified
    # config.impala_mlp_input_dim = 256 # Specific to impala if used
    # config.impala_width = 1
    # config.impala_stack_sizes = (16,32,32)
    # config.impala_num_blocks = 2
    return config
