import copy
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils.encoders import GCEncoder, encoder_modules # Assuming PyTorch versions
from utils.networks import GCActor, GCValue # Assuming PyTorch versions
from utils.torch_utils import ModuleDict # Assuming PyTorch compatible utils


class ForwardBackwardRepresentationAgent(nn.Module):
    """Forward Backward Representation + Implicit Q-Learning (FB + IQL) agent. PyTorch version."""

    def __init__(self, config: Dict[str, Any], ex_observations_shape: Tuple, ex_actions_shape: Tuple, seed: int):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        action_dim = ex_actions_shape[-1]
        # Dummy example latents for network initialization (if needed by PyTorch versions of GCActor/GCValue)
        # Shape: (batch_size_example, latent_dim)
        # Using a small batch size for dummy init, e.g., 1 or 2
        dummy_batch_size = 2
        ex_latents_shape = (dummy_batch_size, config['latent_dim'])
        dummy_latents = torch.randn(ex_latents_shape, device=self.device)
        dummy_obs = torch.randn(dummy_batch_size, *ex_observations_shape[1:], device=self.device)
        dummy_actions = torch.randn(dummy_batch_size, *ex_actions_shape[1:], device=self.device)


        # Define encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module_class = encoder_modules[config['encoder']]
            # For GCEncoder, it takes state_encoder as an argument.
            encoders['value'] = GCEncoder(state_encoder=encoder_module_class())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module_class())
            encoders['forward_repr'] = GCEncoder(state_encoder=encoder_module_class())
            encoders['backward_repr'] = GCEncoder(state_encoder=encoder_module_class()) # backward_repr uses GCEncoder for state
            encoders['actor'] = GCEncoder(state_encoder=encoder_module_class())
        else: # Handle case where no base encoder is specified if GCEncoders can work standalone
            pass

        # Define networks
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1, # As per JAX config
            gc_encoder=encoders.get('value'),
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2, # As per JAX config
            gc_encoder=encoders.get('critic'),
        )
        forward_repr_def = GCValue( # Using GCValue as a base for representations
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'], # Output dim is latent_dim
            layer_norm=config['fb_repr_layer_norm'],
            num_ensembles=2, # As per JAX config
            gc_encoder=encoders.get('forward_repr'),
        )
        backward_repr_def = GCValue( # Using GCValue as a base
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'], # Output dim is latent_dim
            layer_norm=config['fb_repr_layer_norm'],
            num_ensembles=1, # As per JAX config
            gc_encoder=encoders.get('backward_repr'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False, # As per JAX config
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )

        self.networks = ModuleDict({
            'value': value_def,
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
            'forward_repr': forward_repr_def,
            'target_forward_repr': copy.deepcopy(forward_repr_def),
            'backward_repr': backward_repr_def,
            'target_backward_repr': copy.deepcopy(backward_repr_def),
            'actor': actor_def,
        }).to(self.device)

        # Initialize target networks
        self.networks['target_critic'].load_state_dict(self.networks['critic'].state_dict())
        self.networks['target_forward_repr'].load_state_dict(self.networks['forward_repr'].state_dict())
        self.networks['target_backward_repr'].load_state_dict(self.networks['backward_repr'].state_dict())

        # Optimizer: includes all parameters from non-target networks
        online_params = list(self.networks['value'].parameters()) + \
                        list(self.networks['critic'].parameters()) + \
                        list(self.networks['forward_repr'].parameters()) + \
                        list(self.networks['backward_repr'].parameters()) + \
                        list(self.networks['actor'].parameters())
        self.optimizer = Adam(online_params, lr=config['lr'])

    @staticmethod
    def expectile_loss(adv: torch.Tensor, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        latents = batch['latents'].to(self.device) # Assumed to be (B, latent_dim)
        actions = batch['actions'].to(self.device)

        with torch.no_grad():
            # Target critic Q values. GCValue returns (B, num_ensembles) or (B,) if num_ensembles=1
            q1_target, q2_target = self.networks['target_critic'](
                observations, latents, actions, goal_encoded=True, return_ensemble=True
            ) # Assuming GCValue can return tuple for ensembles
            q_target = torch.min(q1_target, q2_target)

        # Current value estimate from online value network
        # GCValue with num_ensembles=1 should return (B,) or (B,1)
        v_online = self.networks['value'](observations, latents, goal_encoded=True).squeeze(-1)

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
        latents = batch['latents'].to(self.device) # (B, latent_dim)

        with torch.no_grad():
            next_v = self.networks['value'](next_observations, latents, goal_encoded=True).squeeze(-1)
            target_q_val = rewards + self.config['discount'] * masks * next_v

        # Current Q estimates from online critic
        q1_online, q2_online = self.networks['critic'](
            observations, latents, actions, goal_encoded=True, return_ensemble=True
        )

        critic_loss_val = F.mse_loss(q1_online, target_q_val) + F.mse_loss(q2_online, target_q_val)

        return critic_loss_val, {
            'critic_loss': critic_loss_val.item(),
            # JAX used q.mean() where q was target_q_val. Using target_q_val for consistency.
            'q_mean': target_q_val.mean().item(),
            'q_max': target_q_val.max().item(),
            'q_min': target_q_val.min().item(),
        }

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device) # Ground truth actions
        latents = batch['latents'].to(self.device)

        with torch.no_grad():
            v = self.networks['value'](observations, latents, goal_encoded=True).squeeze(-1)
            # Q values from online critic (but detached as per IQL for adv computation)
            q1_critic, q2_critic = self.networks['critic'](
                 observations, latents, actions_gt, goal_encoded=True, return_ensemble=True
            )
            q = torch.min(q1_critic, q2_critic)
            adv = q - v

        exp_a = torch.exp(adv * self.config['alpha_awr'])
        exp_a = torch.clamp(exp_a, max=100.0) # Clip as in JAX

        # Actor's policy distribution
        dist = self.networks['actor'](observations, latents, goal_encoded=True)
        log_prob = dist.log_prob(actions_gt)

        actor_loss_val = -(exp_a * log_prob).mean()

        with torch.no_grad():
            mse_val = F.mse_loss(dist.mode(), actions_gt)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') else torch.tensor(0.0, device=self.device) # dist.scale_diag -> dist.scale

        return actor_loss_val, {
            'actor_loss': actor_loss_val.item(),
            'adv': adv.mean().item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse_val.item(),
            'std': std_val.item(),
        }

    def forward_backward_repr_loss(self, batch: Dict[str, torch.Tensor], current_rng_state: Any = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        latents = batch['latents'].to(self.device) # (B, latent_dim)
        batch_size_eff = observations.shape[0] # Effective batch size

        with torch.no_grad():
            next_dist = self.networks['actor'](next_observations, latents, goal_encoded=True)
            if self.config['const_std']:
                next_actions_policy = torch.clamp(next_dist.mode(), -1, 1)
            else:
                # PyTorch sample() doesn't take seed directly. Global seeding or generator needed.
                # Assuming global seeding is handled if deterministic sampling is required.
                next_actions_policy = torch.clamp(next_dist.sample(), -1, 1)

            # Target forward and backward representations (detached)
            # GCValue returns (B, num_ensembles, D) or (B, D) if num_ensembles=1
            # forward_repr has num_ensembles=2, backward_repr has num_ensembles=1
            next_forward_reprs_target = self.networks['target_forward_repr'](
                next_observations, latents, next_actions_policy, goal_encoded=True
            ) # (B, E=2, D)
            next_backward_reprs_target = self.networks['target_backward_repr'](
                next_observations # backward_repr doesn't take latents/actions per JAX
            ).squeeze(1) # (B, D), squeeze if num_ensembles=1 adds a dim

            # Normalize backward representations
            norm_factor_bw_target = torch.sqrt(torch.tensor(self.config['latent_dim'], device=self.device).float())
            next_backward_reprs_target = F.normalize(next_backward_reprs_target, p=2, dim=-1) * norm_factor_bw_target

            # Target occupancy measures: einsum('esd,td->est', next_forward_reprs, next_backward_reprs)
            # next_forward_reprs_target: (B, E, D)
            # next_backward_reprs_target: (B, D)
            # einsum: (Batch, Ensemble, Dim), (Batch_goals, Dim) -> (Batch, Batch_goals, Ensemble)
            # Here, Batch_goals is also B. So, (B, B, E)
            target_occ_measures = torch.einsum('bed,gd->bge', next_forward_reprs_target, next_backward_reprs_target)

            if self.config['repr_agg'] == 'mean':
                target_occ_measures = target_occ_measures.mean(dim=-1) # (B, B), mean over ensembles
            else: # 'min'
                target_occ_measures = target_occ_measures.min(dim=-1)[0] # (B, B), min over ensembles

        # Online forward and backward representations
        forward_reprs_online = self.networks['forward_repr'](
            observations, latents, actions, goal_encoded=True
        ) # (B, E=2, D)
        backward_reprs_online = self.networks['backward_repr'](
            next_observations # Use next_observations for backward_reprs as in JAX
        ).squeeze(1) # (B, D)

        norm_factor_bw_online = torch.sqrt(torch.tensor(self.config['latent_dim'], device=self.device).float())
        backward_reprs_online = F.normalize(backward_reprs_online, p=2, dim=-1) * norm_factor_bw_online

        # Online occupancy measures: (B, B, E)
        occ_measures_online = torch.einsum('bed,gd->bge', forward_reprs_online, backward_reprs_online)

        # Loss calculation (JAX: vmap over ensembles)
        # PyTorch: can often operate on the ensemble dim directly.
        # JAX: I = jnp.eye(self.config['batch_size'])
        # JAX: repr_off_diag_loss = jax.vmap(lambda x: (x * (1 - I)) ** 2, 0, 0)(occ_measures - self.config['discount'] * target_occ_measures[None])
        # JAX: repr_diag_loss = jax.vmap(jnp.diag, 0, 0)(occ_measures)

        I_matrix = torch.eye(batch_size_eff, device=self.device) # (B,B)

        # occ_measures_online: (B,B,E), target_occ_measures: (B,B)
        # Need to align target_occ_measures for broadcasting: (B,B,1)
        delta_occ = occ_measures_online - self.config['discount'] * target_occ_measures.unsqueeze(-1)

        # Off-diagonal loss: (B, B, E)
        repr_off_diag_loss_elements = (delta_occ * (1 - I_matrix).unsqueeze(-1)) ** 2
        # Sum over non-diagonal elements (axis 1), then mean over batch (axis 0) and ensembles (axis 2)
        # JAX: jnp.sum(repr_off_diag_loss, axis=-1) / (self.config['batch_size'] - 1) -> sum over one of B dims
        # This means sum over columns for each row, for each ensemble.
        repr_off_diag_loss = repr_off_diag_loss_elements.sum(dim=1) / (batch_size_eff - 1 + 1e-8) # (B,E)
        repr_off_diag_loss = repr_off_diag_loss.mean() # Mean over B and E

        # Diagonal loss:
        # JAX: jax.vmap(jnp.diag, 0, 0)(occ_measures) -> extracts diagonal for each ensemble member
        # occ_measures_online is (B,B,E). We need diagonal of each (B,B) slice.
        # torch.diagonal operates on specified dims.
        # diag(occ_measures_online[e]) for each e.
        # Or view as (E, B, B) then take diagonal.
        repr_diag_loss_elements = torch.diagonal(occ_measures_online, dim1=0, dim2=1) # (E, B)
        repr_diag_loss = repr_diag_loss_elements.mean() # Mean over E and B. JAX sums then divides by (B-1)

        # The JAX `jnp.mean(repr_diag_loss + ...)` sums losses per ensemble then means.
        # Let's recalculate JAX way: loss per ensemble, then mean.

        # Loss per ensemble member
        losses_per_ensemble = []
        for e in range(occ_measures_online.shape[-1]): # Iterate over ensembles
            occ_e = occ_measures_online[:,:,e] # (B,B)
            # Target needs to match this ensemble. If target is already aggregated, use it.
            # If target_occ_measures was (B,B,E), use target_occ_measures[:,:,e]
            # Here target_occ_measures is (B,B) after aggregation.
            delta_occ_e = occ_e - self.config['discount'] * target_occ_measures

            off_diag_e = ((delta_occ_e * (1 - I_matrix))**2).sum(dim=1) / (batch_size_eff - 1 + 1e-8) # (B,)
            diag_e = torch.diag(occ_e) # (B,)
            losses_per_ensemble.append( (diag_e + off_diag_e).mean() ) # Mean over B

        repr_loss = sum(losses_per_ensemble) / len(losses_per_ensemble) # Mean over ensembles


        # Orthonormalization loss for backward_reprs_online (B, D)
        # JAX: covariance = jnp.matmul(backward_reprs, backward_reprs.T) -> (B,B)
        covariance = torch.matmul(backward_reprs_online, backward_reprs_online.T)
        ortho_diag = -2 * torch.diag(covariance) # (B,)
        ortho_off_diag = (covariance * (1 - I_matrix))**2 # (B,B), element-wise
        # Sum off-diagonal elements for each row, then divide by (B-1)
        ortho_off_diag_summed = ortho_off_diag.sum(dim=1) / (batch_size_eff - 1 + 1e-8) # (B,)
        ortho_loss = self.config['orthonorm_coeff'] * (ortho_diag + ortho_off_diag_summed).mean() # Mean over B

        fb_loss = repr_loss + ortho_loss

        # For logging, calculate some stats from one ensemble (e.g., first) or mean over ensembles
        occ_measures_for_stats = occ_measures_online.mean(-1) # Mean over ensembles (B,B)

        return fb_loss, {
            'repr_loss': repr_loss.item(),
            'repr_diag_loss': torch.diag(occ_measures_for_stats).mean().item(), # Mean of diagonal elements
            'repr_off_diag_loss': (occ_measures_for_stats * (1-I_matrix)).abs().mean().item(), # Mean of absolute off-diagonal
            'ortho_loss': ortho_loss.item(),
            'ortho_diag_loss': ortho_diag.mean().item(),
            'ortho_off_diag_loss': ortho_off_diag_summed.mean().item(),
            'occ_measure_mean': occ_measures_for_stats.mean().item(),
            'occ_measure_max': occ_measures_for_stats.max().item(),
            'occ_measure_min': occ_measures_for_stats.min().item(),
        }

    def forward_backward_actor_loss(self, batch: Dict[str, torch.Tensor], current_rng_state: Any = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        observations = batch['observations'].to(self.device)
        actions_gt = batch['actions'].to(self.device) # Ground truth actions for BC part
        latents = batch['latents'].to(self.device) # (B, latent_dim)

        dist = self.networks['actor'](observations, latents, goal_encoded=True)
        if self.config['const_std']:
            policy_actions = torch.clamp(dist.mode(), -1, 1)
        else:
            policy_actions = torch.clamp(dist.sample(), -1, 1) # RNG handling assumed global

        # Q-values from forward representations and latents (which act as goals)
        # forward_reprs: (B, E, D), latents: (B, D)
        # q = einsum('esd,td->est', forward_reprs, latents) -> (B, B, E)
        # This implies latents are a set of goals, and we match (obs,action)_i to latent_j
        forward_reprs_actor = self.networks['forward_repr'](
            observations, latents, policy_actions, goal_encoded=True
        ) # (B, E, D)

        # JAX: q1, q2 = jnp.einsum('esd,td->est', forward_reprs, latents) -> q1 from e=0, q2 from e=1
        # This means latents are used as the "psi" part of a bilinear product.
        # Each latent in the batch `batch['latents']` is a goal for its corresponding obs.
        # So we need `diag(einsum('bed,gd->bge'))`
        # q_bilinear = torch.einsum('bed,gd->bge', forward_reprs_actor, latents) # (B,B,E)
        # q_actor = torch.diagonal(q_bilinear, dim1=0, dim2=1) # (E,B) -> transpose to (B,E)
        # q_actor = q_actor.transpose(0,1) # (B,E)

        # Simpler: if latents are goals for their respective (obs,action) pairs in the batch:
        # einsum('bed,bd->be', forward_reprs_actor, latents)
        q_actor = torch.einsum('bed,bd->be', forward_reprs_actor, latents) # (B,E)

        q_actor_min = q_actor.min(dim=1)[0] # Min over ensembles (B,)

        q_loss = -q_actor_min.mean()
        if self.config['normalize_q_loss']:
            lam = (1 / torch.abs(q_actor_min).mean()).detach()
            q_loss = lam * q_loss

        log_prob = dist.log_prob(actions_gt) # BC against ground truth actions
        bc_loss = -(self.config['alpha_repr'] * log_prob).mean()

        total_actor_loss = q_loss + bc_loss

        with torch.no_grad():
            mse_val = F.mse_loss(dist.mode(), actions_gt)
            std_val = dist.scale.mean() if hasattr(dist, 'scale') else torch.tensor(0.0, device=self.device)

        return total_actor_loss, {
            'actor_loss': total_actor_loss.item(),
            'q_loss': q_loss.item(),
            'bc_loss': bc_loss.item(),
            'q_mean': q_actor_min.mean().item(),
            'q_abs_mean': torch.abs(q_actor_min).mean().item(),
            'bc_log_prob': log_prob.mean().item(),
            'mse': mse_val.item(),
            'std': std_val.item(),
        }


    def _update_target_network(self, online_net: nn.Module, target_net: nn.Module):
        tau = self.config['tau']
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    @torch.no_grad()
    def sample_latents(self, batch: Dict[str, torch.Tensor], current_rng_state: Any = None) -> torch.Tensor:
        # RNG for JAX `jax.random.split(rng, 3)` needs careful handling.
        # For PyTorch, if determinism is key, pass a torch.Generator. Here, assume global RNG.
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device) # For dtype matching
        batch_size_eff = observations.shape[0]

        # 1. Sample random Gaussian latents
        random_latents = torch.randn(batch_size_eff, self.config['latent_dim'],
                                     dtype=actions.dtype, device=self.device)
        norm_factor = torch.sqrt(torch.tensor(self.config['latent_dim'], device=self.device).float())
        random_latents = F.normalize(random_latents, p=2, dim=-1) * norm_factor

        # 2. Get backward representations from permuted observations
        perm = torch.randperm(batch_size_eff, device=self.device)
        permuted_obs = observations[perm]

        # Use online backward_repr network, consistent with JAX `self.network.select` (not target)
        backward_reprs_permuted = self.networks['backward_repr'](permuted_obs).squeeze(1) # (B,D)
        backward_reprs_permuted = F.normalize(backward_reprs_permuted, p=2, dim=-1) * norm_factor
        # No stop_gradient in PyTorch unless it's part of loss computation path, here it's for data generation.

        # 3. Mix random latents and backward representations
        mix_probs = torch.rand(batch_size_eff, 1, device=self.device)
        final_latents = torch.where(mix_probs < self.config['latent_mix_prob'],
                                    random_latents,
                                    backward_reprs_permuted)
        return final_latents


    def update(self, batch: Dict[str, torch.Tensor], pretrain_mode: bool, step: int, full_update: bool = True) -> Dict[str, Any]:
        self.train()
        info = {}
        # current_rng_state can be passed if fine-grained JAX-like RNG is needed, else None

        if pretrain_mode:
            # Sample latents and add to batch for this update step
            # Detach sampled latents as they are inputs, not to be trained through their generation process
            batch['latents'] = self.sample_latents(batch).detach()

            fb_repr_loss, fb_repr_info = self.forward_backward_repr_loss(batch)
            info.update({f'fb_repr/{k}': v for k,v in fb_repr_info.items()})

            fb_actor_loss, fb_actor_info = self.forward_backward_actor_loss(batch)
            info.update({f'fb_actor/{k}': v for k,v in fb_actor_info.items()})

            total_loss = fb_repr_loss + fb_actor_loss
            info['pretrain/total_loss'] = total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Target updates for FB representations
            self._update_target_network(self.networks['forward_repr'], self.networks['target_forward_repr'])
            self._update_target_network(self.networks['backward_repr'], self.networks['target_backward_repr'])

        else: # Finetuning mode (IQL)
            # Latents for finetuning are typically inferred once per task/episode, not per batch usually.
            # Assuming batch['latents'] is already populated correctly for finetuning.
            # If not, they might need to be inferred here.
            # The JAX code does not show latent sampling within finetuning_loss, implying latents are fixed or from data.
            # For this translation, we assume batch['latents'] is provided.
            if 'latents' not in batch or batch['latents'] is None:
                 # Fallback or error: For IQL part, latents should represent the 'goal' or 'task context'.
                 # If they are meant to be dynamic per batch like in pretraining, sample them:
                 # batch['latents'] = self.sample_latents(batch).detach()
                 # However, FB-IQL typically uses a fixed inferred latent for the task during finetuning.
                 # This part needs clarification based on how `infer_latent` is used in the overall training script.
                 # For now, let's assume they are in the batch.
                 pass


            value_loss_val, value_info = self.value_loss(batch)
            info.update({f'value/{k}': v for k,v in value_info.items()})

            critic_loss_val, critic_info = self.critic_loss(batch)
            info.update({f'critic/{k}': v for k,v in critic_info.items()})

            total_loss = value_loss_val + critic_loss_val

            if full_update: # Corresponds to actor_freq
                actor_loss_val, actor_info = self.actor_loss(batch)
                info.update({f'actor/{k}': v for k,v in actor_info.items()})
                total_loss += actor_loss_val
            else:
                info['actor/actor_loss'] = 0.0 # Placeholder

            info['finetune/total_loss'] = total_loss.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Target update for IQL critic
            self._update_target_network(self.networks['critic'], self.networks['target_critic'])
            # Note: JAX version only updates target_critic in finetune. Value net has no target.

        return info

    @torch.no_grad()
    def infer_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Infers a task-specific latent using reward-weighted average of backward representations."""
        self.eval()
        observations = batch['observations'].to(self.device)
        rewards = batch['rewards'].to(self.device) # Should be (N, 1) or (N,)

        # Use online backward_repr network
        backward_reprs = self.networks['backward_repr'](observations).squeeze(1) # (N, D)
        norm_factor = torch.sqrt(torch.tensor(self.config['latent_dim'], device=self.device).float())
        backward_reprs_normalized = F.normalize(backward_reprs, p=2, dim=-1) * norm_factor

        # Reward-weighted average: (1,N) @ (N,D) -> (1,D)
        # Ensure rewards are (1,N) for matmul, or handle sum differently
        if rewards.ndim == 1: rewards = rewards.unsqueeze(0) # (1,N)
        if rewards.shape[0] != 1 and rewards.shape[1] == 1: rewards = rewards.T # (1,N)

        # Weighted sum: sum(rewards_i * backward_reprs_i) / N
        # JAX: jnp.matmul(batch['rewards'].T, backward_reprs) / batch['rewards'].shape[0]
        # If rewards is (N,1), rewards.T is (1,N). Result (1,D).
        # Division by N implies rewards are not probabilities but raw rewards.
        latent_sum = torch.matmul(rewards, backward_reprs_normalized) # (1,D)
        latent_avg = latent_sum / observations.shape[0]

        inferred_latent = F.normalize(latent_avg, p=2, dim=-1) * norm_factor
        return inferred_latent.squeeze(0).cpu() # Return as (D,) on CPU

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor, latents: torch.Tensor,
                       temperature: float = 1.0, seed: int = None) -> torch.Tensor:
        self.eval()
        observations = observations.to(self.device)
        latents = latents.to(self.device) # Ensure latents are on device

        if seed is not None: # Affects global torch RNG
            current_rng_state_cpu = torch.get_rng_state()
            current_rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        # GCActor might take temperature for std scaling in its distribution
        if hasattr(self.networks['actor'], 'set_temperature'):
            self.networks['actor'].set_temperature(temperature)

        dist = self.networks['actor'](observations, latents, goal_encoded=True) # Add temp if actor takes it
        actions = dist.sample() # PyTorch sample() does not take seed directly
        actions_clipped = torch.clamp(actions, -1, 1)

        if seed is not None: # Restore RNG state
            torch.set_rng_state(current_rng_state_cpu)
            if torch.cuda.is_available() and current_rng_state_cuda:
                torch.cuda.set_rng_state_all(current_rng_state_cuda)

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
    config = dict(
        agent_name='fb_repr_pytorch',
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        value_hidden_dims=(512, 512, 512, 512),
        forward_repr_hidden_dims=(512, 512, 512, 512),
        backward_repr_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        value_layer_norm=True,
        fb_repr_layer_norm=True,
        latent_dim=512,
        discount=0.99,
        tau=0.005,
        expectile=0.9, # IQL expectile
        actor_freq=4, # Handled by training loop
        repr_agg='min', # 'min' or 'mean' for target FB repr aggregation
        orthonorm_coeff=1.0,
        latent_mix_prob=0.5, # For sampling latents during pretraining
        alpha_repr=10.0, # BC coefficient for FB actor loss
        alpha_awr=10.0, # Temperature for IQL actor loss (advantage weighting)
        const_std=True, # For actor's action distribution
        normalize_q_loss=False, # For FB actor loss
        num_latent_inference_samples=10_000, # For infer_latent method (size of batch)
        encoder=None, # Placeholder, e.g., 'mlp'
    )
    return config

if __name__ == '__main__':
    print("Attempting to initialize ForwardBackwardRepresentationAgent (PyTorch)...")
    cfg = get_config()
    cfg['encoder'] = 'mlp' # Specify encoder

    obs_dim = 10
    action_dim = 4
    latent_dim_val = cfg['latent_dim']
    batch_size_val = cfg['batch_size']

    # Example data (NumPy)
    example_obs_np = np.random.randn(batch_size_val, obs_dim).astype(np.float32)
    example_act_np = np.random.randn(batch_size_val, action_dim).astype(np.float32)
    example_latents_np = np.random.randn(batch_size_val, latent_dim_val).astype(np.float32)

    agent_pytorch = ForwardBackwardRepresentationAgent.create(seed=42,
                                                              ex_observations=example_obs_np,
                                                              ex_actions=example_act_np,
                                                              config=cfg)
    print(f"Agent created. Device: {agent_pytorch.device}")

    dummy_batch = {
        'observations': torch.from_numpy(example_obs_np).float(),
        'actions': torch.from_numpy(example_act_np).float(),
        'next_observations': torch.from_numpy(example_obs_np).float(), # Simplicity
        'rewards': torch.randn(batch_size_val, 1).float(),
        'masks': torch.ones(batch_size_val, 1).float(),
        'latents': torch.from_numpy(example_latents_np).float(), # Crucial for IQL part
    }

    # Test pretraining update
    print("\nTesting pretraining update...")
    try:
        pretrain_info = agent_pytorch.update(dummy_batch, pretrain_mode=True, step=0)
        print("Pretraining update successful. Info snippet:", {k:v for i,(k,v) in enumerate(pretrain_info.items()) if i < 5})
    except Exception as e:
        print(f"Error during pretraining update: {e}")
        import traceback; traceback.print_exc()

    # Test finetuning update
    print("\nTesting finetuning update (full)...")
    try:
        # Ensure latents are in batch for finetuning
        if 'latents' not in dummy_batch or dummy_batch['latents'] is None:
            print("Warning: Latents not found in batch for finetuning, using random ones.")
            dummy_batch['latents'] = torch.randn(batch_size_val, latent_dim_val).float()

        finetune_info_full = agent_pytorch.update(dummy_batch, pretrain_mode=False, step=1, full_update=True)
        print("Finetuning update (full) successful. Info snippet:", {k:v for i,(k,v) in enumerate(finetune_info_full.items()) if i < 5})
    except Exception as e:
        print(f"Error during finetuning update (full): {e}")
        import traceback; traceback.print_exc()

    # Test latent inference
    print("\nTesting latent inference...")
    try:
        # Create a batch for inference (can be larger)
        infer_obs_np = np.random.randn(cfg['num_latent_inference_samples'], obs_dim).astype(np.float32)
        infer_rewards_np = np.random.rand(cfg['num_latent_inference_samples'], 1).astype(np.float32)
        infer_batch = {
            'observations': torch.from_numpy(infer_obs_np).float(),
            'rewards': torch.from_numpy(infer_rewards_np).float(),
        }
        inferred_latent_tensor = agent_pytorch.infer_latent(infer_batch)
        print("Inferred latent shape:", inferred_latent_tensor.shape)
        assert inferred_latent_tensor.shape == (latent_dim_val,)
    except Exception as e:
        print(f"Error during latent inference: {e}")
        import traceback; traceback.print_exc()


    # Test action sampling
    print("\nTesting action sampling...")
    try:
        sample_obs_tensor = torch.from_numpy(example_obs_np[:5]).float()
        sample_latents_tensor = torch.from_numpy(example_latents_np[:5]).float()
        sampled_actions = agent_pytorch.sample_actions(sample_obs_tensor, sample_latents_tensor, seed=123)
        print("Sampled actions shape:", sampled_actions.shape)
    except Exception as e:
        print(f"Error during action sampling: {e}")
        import traceback; traceback.print_exc()

    print("\nBasic FB_Repr (PyTorch) functionality tests completed.")
    print("NOTE: Assumes PyTorch versions of GCEncoder, GCActor, GCValue, ModuleDict are correct.")
    print("The FB representation loss (esp. einsums and vmaps) and FB actor loss's Q calculation via einsum were complex to translate and may need further validation.")

'''
This is a complex agent. Here's a summary of the conversion approach for `agents/fb_repr.py`:

**Core Conversions:**

1.  **Class Structure**: `flax.struct.PyTreeNode` to `torch.nn.Module`.
2.  **Initialization (`__init__`)**:
    *   JAX PRNGKey to `torch.manual_seed`.
    *   Network definitions (`GCValue`, `GCActor`, `GCEncoder`) are assumed to be PyTorch `nn.Module` equivalents. `GCEncoder` is passed to `GCValue`/`GCActor` as in JAX.
    *   `ModuleDict` is assumed to be a PyTorch `nn.ModuleDict`.
    *   Target networks created using `copy.deepcopy` and initialized with online network state.
    *   Optimizer (`optax.adam` to `torch.optim.Adam`) created for online network parameters.
3.  **Loss Functions**:
    *   JAX numpy (`jnp`) operations to PyTorch tensor operations (`torch`).
    *   `jax.lax.stop_gradient` -> `tensor.detach()`.
    *   `jnp.linalg.norm` -> `torch.linalg.norm` or `F.normalize`.
    *   `jnp.einsum` -> `torch.einsum`. Careful attention to indices and operand shapes.
    *   **Expectile Loss**: Translated directly.
    *   **Value Loss (IQL)**: Uses target critic for Q-values (detached) and online value network.
    *   **Critic Loss (IQL)**: Uses online value network for next state value (detached) to form target Q, then MSE against online critic Q-values.
    *   **Actor Loss (IQL)**: Advantage-weighted regression (AWR) using detached Q and V from online networks.
    *   **Forward-Backward Representation Loss**: This is the most complex.
        *   Normalization of backward representations.
        *   `einsum` for occupancy measures. The original JAX `vmap` over ensemble dimension for losses is handled by either direct tensor operations on the ensemble dimension or by iterating if PyTorch's broadcasting is less direct for the specific JAX `vmap` pattern.
        *   Diagonal and off-diagonal components of the loss calculated.
        *   Orthonormalization loss for backward representations.
    *   **Forward-Backward Actor Loss**: DDPG+BC style. Q-values derived from `einsum` between forward representations and latents (goals). BC term uses log-probability of ground truth actions.
4.  **Target Updates (`_update_target_network`)**: Standard soft EMA update implemented by iterating through parameters.
5.  **Latent Sampling (`sample_latents`)**:
    *   Random Gaussian latents generated and normalized.
    *   Backward representations from permuted observations (using online `backward_repr` network) generated and normalized.
    *   These two sets of latents are mixed based on `latent_mix_prob`.
    *   RNG handling for permutations and random numbers uses PyTorch's global RNG.
6.  **Latent Inference (`infer_latent`)**:
    *   Calculates backward representations for a batch of observations.
    *   Computes a reward-weighted average of these representations.
    *   Normalizes the result to get the task-specific latent.
    *   Uses online `backward_repr` network and runs in `eval` mode with `torch.no_grad()`.
7.  **Action Sampling (`sample_actions`)**:
    *   Uses online actor with provided observations and latents.
    *   `temperature` is assumed to be handled by the PyTorch `GCActor` if it affects the distribution's properties (e.g., stddev).
    *   `dist.sample()` is used; JAX's per-sample `seed` is not directly available, relies on global PyTorch RNG.
8.  **Update Method (`update`)**:
    *   Combines `pretrain` and `finetune` logic from JAX.
    *   In `pretrain_mode`:
        *   Samples latents for the batch.
        *   Computes FB representation loss and FB actor loss.
        *   Backpropagates total loss and updates optimizer.
        *   Updates target forward and backward representation networks.
    *   In `finetune_mode` (IQL):
        *   Assumes `batch['latents']` are provided (e.g., inferred once for the task).
        *   Computes IQL value, critic, and (if `full_update`) actor losses.
        *   Backpropagates total loss and updates optimizer.
        *   Updates target critic network.
9.  **RNG Handling**: JAX's explicit PRNGKey splitting is replaced by PyTorch's global RNG, seeded at initialization. If specific per-operation determinism (like JAX `jax.random.split`) is critical, `torch.Generator` objects would need to be created and passed, which adds complexity. The current translation relies on global seeding for reproducibility of sequences of operations.
10. **`goal_encoded=True`**: This argument is passed to network calls as in JAX, assuming the PyTorch `GCValue`/`GCActor` modules handle it appropriately (likely by using the provided `latents` as goals).
11. **`select()` method**: Replaced by direct calls like `self.networks['module_name'](...)`.
12. **`TrainState`**: Not used directly. Parameters and optimizer are attributes of the agent module.
13. **Device Management**: Explicit `.to(self.device)` for tensors and model.

**Key Challenges & Assumptions:**

*   **`einsum` and `vmap` in FB Loss**: The JAX `forward_backward_repr_loss` uses `einsum` and `vmap` in ways that require careful translation to ensure dimensions and operations align correctly in PyTorch. The translation attempts to match the per-ensemble logic implied by `vmap`.
*   **FB Actor Q-Value Calculation**: The `einsum('esd,td->est', forward_reprs, latents)` for Q-values in `forward_backward_actor_loss` implies a specific bilinear interaction where `latents` act as goal embeddings. The PyTorch version uses `einsum('bed,bd->be', ...)` assuming each latent in the batch corresponds to its respective observation/action pair for Q-value calculation, effectively taking the "diagonal" of the more general batch-wise comparison.
*   **Latent Handling in Finetuning**: The JAX code doesn't explicitly show how `latents` are provided during IQL finetuning. The PyTorch version assumes they are present in the `batch` dictionary, likely inferred by `infer_latent` at a higher level (e.g., once per episode/task).
*   **Correctness of PyTorch Utilities**: The conversion heavily relies on the assumption that `utils.encoders_pytorch.GCEncoder`, `utils.networks_pytorch.GCActor`, `utils.networks_pytorch.GCValue`, and `utils.pytorch_utils.ModuleDict` are correctly implemented PyTorch equivalents of their JAX counterparts, especially regarding how they handle `goal_encoded`, ensembles, and latent inputs.

An example usage section (`if __name__ == '__main__':`) is included for basic smoke testing, but thorough validation with actual training loops and data is essential for such a complex agent.
'''
