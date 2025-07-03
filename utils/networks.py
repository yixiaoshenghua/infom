from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from torch.distributions.independent import Independent

from utils.torch_utils import init_weights # Assuming init_weights is in torch_utils


# Note: ensemblize function is removed. Ensembles can be handled by creating a list of modules in PyTorch.

class Identity(nn.Module):
    """Identity layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        input_dim: Input dimension.
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (if None, last hidden_dim is used).
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        layer_norm: Whether to apply layer normalization.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: Optional[int] = None,
                 activations: Any = F.gelu, # Changed from nn.gelu to F.gelu
                 activate_final: bool = False,
                 layer_norm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim if output_dim is not None else hidden_dims[-1]
        self.activations = activations
        self.activate_final = activate_final
        self.layer_norm = layer_norm

        layers = []
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
            if i < len(hidden_dims) - 1 or self.activate_final:
                if self.layer_norm:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(self.activations) # Directly use the function

        if output_dim is not None and len(hidden_dims)>0 and self.output_dim != hidden_dims[-1] :
             layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        elif output_dim is not None and len(hidden_dims) == 0: # MLP with no hidden layers
            layers.append(nn.Linear(input_dim, self.output_dim))


        self.net = nn.Sequential(*layers)
        self.apply(lambda m: init_weights(m)) # Apply custom initialization

        self.feature = None # For sow functionality if needed

    def forward(self, x):
        # sow functionality replacement:
        # We can store intermediate features if needed, but it's less common in PyTorch.
        # For simplicity, we'll skip direct sow replacement unless explicitly required later.
        # If a specific intermediate layer's output is needed, it should be handled explicitly.
        # Example: iterate through layers and store output of a specific one.
        # for i, layer in enumerate(self.net):
        #     x = layer(x)
        #     if i == len(self.net) - 3 and isinstance(self.net[i+1], nn.Linear): # Before last Linear and activation/norm
        #         self.feature = x
        # return x
        return self.net(x)


class Param(nn.Module):
    """Scalar parameter module."""
    def __init__(self, init_value: float = 0.0, shape: Sequence[int] = ()):
        super().__init__()
        self.value = nn.Parameter(torch.full(shape, init_value))

    def forward(self):
        return self.value


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        # Store log_value as parameter and exponentiate in forward
        self.log_value = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_value)))))

    def forward(self):
        return torch.exp(self.log_value)


class TanhNormal(Normal):
    """A Normal distribution transformed by a Tanh function.
    The mode is the mode of the base distribution transformed by tanh.
    """
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)
        self.tanh_transform = TanhTransform(cache_size=1)

    @property
    def mode(self):
        return self.tanh_transform(super().mean) # Use mean as mode for Normal

    def sample(self, sample_shape=torch.Size()):
        z = super().sample(sample_shape)
        return self.tanh_transform(z)

    def rsample(self, sample_shape=torch.Size()):
        z = super().rsample(sample_shape)
        # We need to apply the transform and account for it in the log_prob
        # The TanhTransform itself handles the log_abs_det_jacobian
        return self.tanh_transform(z)

    def log_prob(self, value):
        # value is x, we want to calculate log_prob for z = atanh(x)
        # log_prob(x) = log_prob_base(atanh(x)) - log_abs_det_jacobian(atanh(x))
        if self._validate_args:
            self.tanh_transform.domain.check(value)
        z = self.tanh_transform.inv(value)
        # log_prob of base distribution for z
        base_log_prob = super().log_prob(z)
        # log_abs_det_jacobian of the inverse transform
        ladj = self.tanh_transform.log_abs_det_jacobian(z, value) # Pass both z and value
        return base_log_prob - ladj


class Value(nn.Module):
    """Value/critic network."""
    def __init__(self,
                 input_dim: int, # Must be provided now
                 hidden_dims: Sequence[int],
                 output_dim: int = 1, # Renamed from value_dim for clarity
                 layer_norm: bool = True,
                 num_ensembles: int = 1,
                 encoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.num_ensembles = num_ensembles

        # If encoder is present, input_dim for MLP is encoder's output_dim
        # This needs to be known. For now, assume input_dim is for the state AFTER potential encoding.
        # If actions are also input, input_dim should account for that too.

        if self.num_ensembles > 1:
            self.value_nets = nn.ModuleList([
                MLP(input_dim=input_dim, # This input_dim needs to be correct
                    hidden_dims=list(hidden_dims) + [output_dim], # Add output_dim to hidden_dims for MLP
                    activate_final=False,
                    layer_norm=layer_norm) for _ in range(num_ensembles)
            ])
        else:
            self.value_net = MLP(input_dim=input_dim,
                                 hidden_dims=list(hidden_dims) + [output_dim],
                                 activate_final=False,
                                 layer_norm=layer_norm)
        self.output_dim = output_dim
        self.apply(lambda m: init_weights(m))


    def forward(self, observations, actions=None):
        if self.encoder is not None:
            # Assuming encoder takes observations and returns encoded_obs
            # The output dimension of encoder must match the expected input_dim for the MLP part
            # This needs careful handling if encoder is not None.
            # For now, let's assume observations are already what the MLP expects if encoder is None.
            obs_feat = self.encoder(observations)
        else:
            obs_feat = observations

        if actions is not None:
            inputs = torch.cat([obs_feat, actions], dim=-1)
        else:
            inputs = obs_feat

        # The input_dim for MLP should be inputs.shape[-1]

        if self.num_ensembles > 1:
            values = [net(inputs) for net in self.value_nets]
            v = torch.stack(values, dim=0) # Shape: (num_ensembles, batch_size, output_dim)
            if self.output_dim == 1:
                v = v.squeeze(-1) # Shape: (num_ensembles, batch_size)
        else:
            v = self.value_net(inputs)
            if self.output_dim == 1:
                v = v.squeeze(-1) # Shape: (batch_size,)
        return v


class Actor(nn.Module):
    """Gaussian actor network."""
    def __init__(self,
                 input_dim: int, # Must be provided
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 layer_norm: bool = False,
                 log_std_min: Optional[float] = -5.0,
                 log_std_max: Optional[float] = 2.0,
                 tanh_squash: bool = False,
                 state_dependent_std: bool = False,
                 const_std_init: float = 0.0, # Initial value for log_std if not state_dependent and not const_std=True (which means zero_like)
                 final_fc_init_scale: float = 1e-2, # Used for custom init if needed
                 encoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std

        # Determine actor_input_dim based on encoder
        # This needs to be known. For now, assume input_dim is for the state AFTER potential encoding.
        actor_input_dim = input_dim

        self.actor_net = MLP(input_dim=actor_input_dim,
                             hidden_dims=hidden_dims,
                             # Output of MLP is feature for mean/std heads
                             output_dim=hidden_dims[-1] if hidden_dims else actor_input_dim,
                             activate_final=True,
                             layer_norm=layer_norm)

        mlp_output_dim = hidden_dims[-1] if hidden_dims else actor_input_dim

        self.mean_net = nn.Linear(mlp_output_dim, action_dim)
        # Custom init for mean_net's final layer
        init_weights(self.mean_net, gain=final_fc_init_scale)


        if self.state_dependent_std:
            self.log_std_net = nn.Linear(mlp_output_dim, action_dim)
            init_weights(self.log_std_net, gain=final_fc_init_scale)
        else:
            # If const_std was True in JAX, it means log_stds = jnp.zeros_like(means)
            # If const_std was False, it used self.param('log_stds', nn.initializers.zeros, (self.action_dim,))
            # So, effectively, it's always initialized to zeros unless changed.
            # We use a nn.Parameter for this.
            self.log_stds = nn.Parameter(torch.full((action_dim,), float(const_std_init)))


        # Apply general init_weights to other layers if not specifically handled
        self.apply(lambda m: init_weights(m) if m not in [self.mean_net, getattr(self, 'log_std_net', None)] else None)


    def forward(self, observations, temperature=1.0):
        if self.encoder is not None:
            # Assuming encoder provides features of actor_input_dim
            inputs = self.encoder(observations)
        else:
            inputs = observations

        features = self.actor_net(inputs)
        means = self.mean_net(features)

        if self.state_dependent_std:
            log_stds = self.log_std_net(features)
        else:
            # Broadcast log_stds to match batch shape of means
            log_stds = self.log_stds.unsqueeze(0).expand_as(means)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds) * temperature

        # distrax.MultivariateNormalDiag is equivalent to Independent(Normal) in PyTorch
        # where Normal is used for each dimension and Independent combines them.
        base_dist = Normal(loc=means, scale=stds)
        # Making it Multivariate: treat the last dim as event_dim
        distribution = Independent(base_dist, 1)


        if self.tanh_squash:
            # Use our TanhNormal or PyTorch's TransformedDistribution with TanhTransform
            # PyTorch's TanhTransform is numerically more stable.
            # The distribution needs a .mode property and .rsample()
            # Option 1: Custom TanhNormal
            # distribution = TanhNormal(loc=means, scale=stds)
            # Option 2: PyTorch's TransformedDistribution
            # Need to ensure it has .mode. TanhTransform doesn't have .mode directly.
            # The mode of Tanh(Normal(mu, sigma)) is Tanh(mu).
            class TanhTransformedNormal(TransformedDistribution):
                def __init__(self, base_distribution, transforms, validate_args=None):
                    super().__init__(base_distribution, transforms, validate_args=validate_args)
                @property
                def mode(self):
                    # Mode of Normal is its mean. Apply transform to it.
                    # Assuming the first transform is TanhTransform
                    return self.transforms[0](self.base_dist.mean)

            distribution = TanhTransformedNormal(distribution, TanhTransform(cache_size=1))

        return distribution


class IntentionEncoder(nn.Module):
    """Transition encoder network."""
    def __init__(self,
                 obs_dim: int, # Dimension of observation
                 action_dim: int, # Dimension of action
                 hidden_dims: Sequence[int],
                 latent_dim: int,
                 layer_norm: bool = False,
                 encoder: nn.Module = None): # Encoder for observations
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim

        current_input_dim = obs_dim # This needs to be the dim after obs_encoder if present
        if self.encoder is not None:
            # This is tricky. The encoder's output dim needs to be known.
            # For now, assume obs_dim is ALREADY the output of self.encoder
            pass # current_input_dim remains obs_dim (feature dim)

        trunk_input_dim = current_input_dim + action_dim

        self.trunk_net = MLP(input_dim=trunk_input_dim,
                             hidden_dims=hidden_dims,
                             output_dim=hidden_dims[-1] if hidden_dims else trunk_input_dim, # Output features for mean/std heads
                             activate_final=True,
                             layer_norm=layer_norm)

        mlp_output_dim = hidden_dims[-1] if hidden_dims else trunk_input_dim

        self.mean_net = nn.Linear(mlp_output_dim, latent_dim)
        self.log_std_net = nn.Linear(mlp_output_dim, latent_dim)

        self.apply(init_weights)

    def forward(self, observations, actions):
        if self.encoder is not None:
            # Assume encoder processes observations
            # The output dim of encoder should be obs_dim used in __init__
            obs_feat = self.encoder(observations)
        else:
            obs_feat = observations

        inputs = torch.cat([obs_feat, actions], dim=-1)
        features = self.trunk_net(inputs)

        means = self.mean_net(features)
        log_stds = self.log_std_net(features)
        # Note: No clipping on log_stds here as in Actor, add if necessary

        # distrax.MultivariateNormalDiag
        base_dist = Normal(loc=means, scale=torch.exp(log_stds))
        distribution = Independent(base_dist, 1)

        return distribution


class VectorField(nn.Module):
    """Flow-matching vector field function."""
    def __init__(self,
                 input_dim: int, # Dimension of noisy_goals (after potential encoding)
                 time_dim: int, # Usually 1 for scalar time
                 vector_dim: int,
                 hidden_dims: Sequence[int],
                 obs_dim: Optional[int] = None, # Dimension of observations (after potential encoding)
                 action_dim: Optional[int] = None,
                 latent_dim: Optional[int] = None,
                 layer_norm: bool = True,
                 num_ensembles: int = 1,
                 encoder: nn.Module = None): # Encoder for noisy_goals and observations
        super().__init__()
        self.encoder = encoder # This encoder would apply to noisy_goals and observations
        self.num_ensembles = num_ensembles
        self.vector_dim = vector_dim

        # Calculate total input dimension for the MLP
        current_mlp_input_dim = input_dim + time_dim
        if obs_dim is not None:
            current_mlp_input_dim += obs_dim
        if action_dim is not None:
            current_mlp_input_dim += action_dim
        if latent_dim is not None:
            current_mlp_input_dim += latent_dim

        # Note: input_dim, obs_dim should be dimensions *after* self.encoder if it's used.

        if self.num_ensembles > 1:
            self.vector_field_nets = nn.ModuleList([
                MLP(input_dim=current_mlp_input_dim,
                    hidden_dims=list(hidden_dims) + [vector_dim],
                    activate_final=False,
                    layer_norm=layer_norm) for _ in range(num_ensembles)
            ])
        else:
            self.vector_field_net = MLP(input_dim=current_mlp_input_dim,
                                        hidden_dims=list(hidden_dims) + [vector_dim],
                                        activate_final=False,
                                        layer_norm=layer_norm)
        self.apply(init_weights)

    def forward(self, noisy_goals, times, observations=None, actions=None, latents=None):

        processed_noisy_goals = self.encoder(noisy_goals) if self.encoder else noisy_goals

        # Ensure times is [..., 1]
        if times.ndim == processed_noisy_goals.ndim -1 :
            times_expanded = times.unsqueeze(-1)
        else:
            times_expanded = times


        inputs_list = [processed_noisy_goals, times_expanded]

        if observations is not None:
            processed_obs = self.encoder(observations) if self.encoder else observations
            inputs_list.append(processed_obs)
        if actions is not None:
            inputs_list.append(actions)
        if latents is not None:
            inputs_list.append(latents)

        inputs = torch.cat(inputs_list, dim=-1)

        if self.num_ensembles > 1:
            vf_list = [net(inputs) for net in self.vector_field_nets]
            vf = torch.stack(vf_list, dim=0) # (num_ensembles, batch, vector_dim)
        else:
            vf = self.vector_field_net(inputs) # (batch, vector_dim)

        return vf


class GCActor(nn.Module):
    """Goal-conditioned actor."""
    def __init__(self,
                 obs_dim: int, # Dimension of observations
                 goal_dim: int, # Dimension of goals
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 layer_norm: bool = False,
                 log_std_min: Optional[float] = -5.0,
                 log_std_max: Optional[float] = 2.0,
                 tanh_squash: bool = False,
                 state_dependent_std: bool = False,
                 const_std_init: float = 0.0,
                 final_fc_init_scale: float = 1e-2,
                 gc_encoder: nn.Module = None): # Assumed to be a PyTorch module now
        super().__init__()
        self.gc_encoder = gc_encoder # This module should handle obs and goal encoding
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std

        # The input_dim to the actor_net MLP depends on gc_encoder's output.
        # This needs to be determined. For now, assume gc_encoder is responsible
        # for producing the correct feature dimension expected by the MLP.
        # Let's say gc_encoder_output_dim is that dimension.
        # If gc_encoder is None, then input is obs_dim + goal_dim.

        # This is a placeholder, should be derived from gc_encoder or (obs_dim + goal_dim)
        actor_input_dim_placeholder = obs_dim + goal_dim # Fallback if gc_encoder output dim is unknown
        if gc_encoder is not None and hasattr(gc_encoder, 'output_dim'): # Ideal
             actor_input_dim_placeholder = gc_encoder.output_dim


        self.actor_net = MLP(input_dim=actor_input_dim_placeholder, # This needs to be accurate
                             hidden_dims=hidden_dims,
                             output_dim=hidden_dims[-1] if hidden_dims else actor_input_dim_placeholder,
                             activate_final=True,
                             layer_norm=layer_norm)

        mlp_output_dim = hidden_dims[-1] if hidden_dims else actor_input_dim_placeholder

        self.mean_net = nn.Linear(mlp_output_dim, action_dim)
        init_weights(self.mean_net, gain=final_fc_init_scale)

        if self.state_dependent_std:
            self.log_std_net = nn.Linear(mlp_output_dim, action_dim)
            init_weights(self.log_std_net, gain=final_fc_init_scale)
        else:
            self.log_stds = nn.Parameter(torch.full((action_dim,), float(const_std_init)))

        self.apply(lambda m: init_weights(m) if m not in [self.mean_net, getattr(self, 'log_std_net', None)] else None)


    def forward(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            # gc_encoder handles combining observations and goals
            # It should accept goal_encoded flag
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            # Manual concatenation if no gc_encoder
            input_list = [observations]
            if goals is not None:
                # If goal_encoded is True but no gc_encoder, this is ambiguous.
                # Assuming goals are then features.
                input_list.append(goals)
            inputs = torch.cat(input_list, dim=-1)

        features = self.actor_net(inputs)
        means = self.mean_net(features)

        if self.state_dependent_std:
            log_stds = self.log_std_net(features)
        else:
            log_stds = self.log_stds.unsqueeze(0).expand_as(means)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds) * temperature

        base_dist = Normal(loc=means, scale=stds)
        distribution = Independent(base_dist, 1)

        if self.tanh_squash:
            class TanhTransformedNormal(TransformedDistribution):
                def __init__(self, base_distribution, transforms, validate_args=None):
                    super().__init__(base_distribution, transforms, validate_args=validate_args)
                @property
                def mode(self):
                    return self.transforms[0](self.base_dist.mean)
            distribution = TanhTransformedNormal(distribution, TanhTransform(cache_size=1))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function."""
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: Optional[int] = None, # For Q-value
                 hidden_dims: Sequence[int] = None,
                 output_dim: int = 1, # Renamed from value_dim
                 # num_residual_blocks: int = 1, # Not used in original MLP structure
                 layer_norm: bool = True,
                 num_ensembles: int = 1,
                 gc_encoder: nn.Module = None):
        super().__init__()
        self.gc_encoder = gc_encoder
        self.num_ensembles = num_ensembles
        self.output_dim = output_dim

        # Determine MLP input dimension
        # This is a placeholder, actual dim depends on gc_encoder output or manual concatenation
        mlp_input_dim_placeholder = obs_dim + goal_dim # Base for V(s,g)
        if gc_encoder is not None and hasattr(gc_encoder, 'output_dim'):
            mlp_input_dim_placeholder = gc_encoder.output_dim # If gc_encoder produces combined features

        if action_dim is not None: # For Q(s,a,g)
            if gc_encoder is None: # If no gc_encoder, actions are appended after obs,goal concat
                 mlp_input_dim_placeholder += action_dim
            # If gc_encoder exists, it might take actions too, or actions are appended after its output.
            # Assuming actions are appended after gc_encoder output if it doesn't take actions.
            # This needs clarification based on gc_encoder's design.
            # For now, let's assume actions are appended to gc_encoder's output if it exists.
            # Or appended to obs+goal if gc_encoder is None.
            # This part is tricky and depends on how gc_encoder is defined.
            # Let's assume for now: if gc_encoder is NOT None, its output_dim is the feature_dim.
            # if gc_encoder IS None, then feature_dim = obs_dim + goal_dim
            # And then action_dim is added to this feature_dim.

            # Simplified: mlp_input_dim_placeholder is for (obs, goal) features
            # action_dim will be added to it.
            if gc_encoder is None :
                feature_dim = obs_dim + goal_dim
            elif hasattr(gc_encoder, 'output_dim'):
                feature_dim = gc_encoder.output_dim
            else: # Fallback
                feature_dim = obs_dim + goal_dim # This is a guess

            actual_mlp_input_dim = feature_dim + (action_dim if action_dim is not None else 0)

        else: # V(s,g)
            if gc_encoder is None:
                actual_mlp_input_dim = obs_dim + goal_dim
            elif hasattr(gc_encoder, 'output_dim'):
                 actual_mlp_input_dim = gc_encoder.output_dim
            else: # Fallback
                actual_mlp_input_dim = obs_dim + goal_dim


        if self.num_ensembles > 1:
            self.value_nets = nn.ModuleList([
                MLP(input_dim=actual_mlp_input_dim,
                    hidden_dims=list(hidden_dims) + [output_dim],
                    activate_final=False,
                    layer_norm=layer_norm) for _ in range(num_ensembles)
            ])
        else:
            self.value_net = MLP(input_dim=actual_mlp_input_dim,
                                 hidden_dims=list(hidden_dims) + [output_dim],
                                 activate_final=False,
                                 layer_norm=layer_norm)
        self.apply(init_weights)

    def forward(self, observations, goals=None, actions=None, goal_encoded=False):
        if self.gc_encoder is not None:
            # gc_encoder produces features from (observations, goals)
            # It should accept goal_encoded
            features = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            # Manual concatenation if no gc_encoder
            input_list = [observations]
            if goals is not None:
                input_list.append(goals)
            features = torch.cat(input_list, dim=-1)

        if actions is not None:
            inputs = torch.cat([features, actions], dim=-1)
        else:
            inputs = features

        if self.num_ensembles > 1:
            values = [net(inputs) for net in self.value_nets]
            v = torch.stack(values, dim=0) # (num_ensembles, batch, output_dim)
            if self.output_dim == 1:
                v = v.squeeze(-1)
        else:
            v = self.value_net(inputs)
            if self.output_dim == 1:
                v = v.squeeze(-1)
        return v


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function."""
    def __init__(self,
                 obs_action_dim: int, # Dim of (obs) or (obs,action) after state_encoder
                 goal_feat_dim: int,  # Dim of (goal) after goal_encoder
                 hidden_dims: Sequence[int], # Hidden dims for phi and psi MLPs
                 latent_dim: int, # Output dim of phi and psi (the d in phi^T psi / sqrt(d))
                 layer_norm: bool = True,
                 num_ensembles: int = 1,
                 value_exp: bool = False,
                 state_encoder: nn.Module = None, # Encodes obs or (obs,action)
                 goal_encoder: nn.Module = None): # Encodes goal
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.latent_dim = latent_dim
        self.num_ensembles = num_ensembles
        self.value_exp = value_exp

        # Phi MLP processes (encoded) observations or (encoded obs, actions)
        # Psi MLP processes (encoded) goals
        # Their input_dims are obs_action_dim and goal_feat_dim respectively.
        # Both output latent_dim.

        if self.num_ensembles > 1:
            self.phi_nets = nn.ModuleList([
                MLP(input_dim=obs_action_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)
                for _ in range(num_ensembles)
            ])
            self.psi_nets = nn.ModuleList([
                MLP(input_dim=goal_feat_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)
                for _ in range(num_ensembles)
            ])
        else:
            self.phi_net = MLP(input_dim=obs_action_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)
            self.psi_net = MLP(input_dim=goal_feat_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)

        self.apply(init_weights)

    def forward(self, observations, goals, actions=None, info=False):

        obs_feat = self.state_encoder(observations) if self.state_encoder else observations
        goal_feat = self.goal_encoder(goals) if self.goal_encoder else goals

        if actions is not None:
            # Here, state_encoder should have processed (obs,actions) together, or obs_feat needs action concat
            # Assuming obs_action_dim was for the final input to phi_net.
            # If state_encoder only processes obs, then concat actions here.
            # This depends on how obs_action_dim was defined.
            # For now, assume if actions are present, obs_feat was just for obs, and we concat actions.
            # This might be wrong if state_encoder was meant to take (obs, actions).
            # Let's assume obs_feat is from obs, and we concat actions if they exist.
            # The input to phi_net should be obs_action_dim.
            # So, if state_encoder processes obs to get obs_feat, and then actions are concatenated,
            # obs_action_dim should be obs_feat.shape[-1] + actions.shape[-1]
            # This part requires careful definition of input dims.
            # Re-evaluating: state_encoder is for the "state" part.
            # If actions are given, phi_inputs = concat(state_encoder(obs), actions)
            # If no actions, phi_inputs = state_encoder(obs)
            # So, obs_action_dim for MLP must be flexible or defined for specific case.
            # Let's assume phi_input is constructed before passing to MLP.

            phi_input_features = torch.cat([obs_feat, actions], dim=-1)
        else:
            phi_input_features = obs_feat

        # psi_input_features is always goal_feat
        psi_input_features = goal_feat

        if self.num_ensembles > 1:
            phi_list = [net(phi_input_features) for net in self.phi_nets] # List of (batch, latent_dim)
            psi_list = [net(psi_input_features) for net in self.psi_nets] # List of (batch, latent_dim)

            v_list = []
            for phi_single, psi_single in zip(phi_list, psi_list):
                # phi_single: (batch, latent_dim), psi_single: (batch, latent_dim)
                # We need (batch_obs, batch_goals) -> (batch_obs, batch_goals) matrix if inputs are structured that way
                # Original was jnp.einsum('ik,jk->ij', phi, psi) -> phi is (num_obs, d), psi is (num_goals, d)
                # Here, inputs are batched. So phi/psi are (batch_size, latent_dim).
                # If observations and goals have same batch size and correspond element-wise:
                dot_product = (phi_single * psi_single).sum(dim=-1) # (batch_size)
                v_list.append(dot_product / (self.latent_dim**0.5))
            v = torch.stack(v_list, dim=0) # (num_ensembles, batch_size)

        else:
            phi = self.phi_net(phi_input_features) # (batch, latent_dim)
            psi = self.psi_net(psi_input_features) # (batch, latent_dim)

            # Assuming element-wise dot product for corresponding samples in batch
            # If a matrix of all-pairs is needed, dimensions of obs/goals would be different.
            # The JAX einsum 'ik,jk->ij' suggests phi has shape (I, K) and psi has (J, K), output (I, J).
            # This means if observations are (N, obs_dim) and goals are (M, goal_dim),
            # then phi is (N, latent_dim) and psi is (M, latent_dim).
            # The result v should be (N, M).
            # Current PyTorch setup assumes N=M (same batch size for obs and goals).
            # If N != M, then:
            # phi: (N, latent_dim), psi: (M, latent_dim)
            # v = torch.matmul(phi, psi.transpose(-2, -1)) / (self.latent_dim**0.5) # (N, M)
            # For now, stick to element-wise for batched inputs (N=M)
            v = (phi * psi).sum(dim=-1) / (self.latent_dim**0.5) # (batch_size)


        if self.value_exp:
            v = torch.exp(v)

        if info:
            # Need to return phi and psi that were used.
            # If ensemble, this would be lists of phi and psi.
            # For simplicity, if ensemble, return the lists.
            if self.num_ensembles > 1:
                 return v, phi_list, psi_list
            return v, phi, psi
        else:
            return v


class GCMetricValue(nn.Module):
    """Metric value function: -||phi(s) - phi(g)||_2."""
    def __init__(self,
                 input_dim: int, # Dim of s or g (after main encoder if used)
                 hidden_dims: Sequence[int], # Hidden dims for the phi MLP
                 latent_dim: int, # Output dim of phi MLP
                 layer_norm: bool = True,
                 num_ensembles: int = 1,
                 encoder: nn.Module = None): # Main encoder for s and g
        super().__init__()
        self.encoder = encoder # Applied to both observations and goals
        self.num_ensembles = num_ensembles

        # The phi network takes encoded input_dim and outputs latent_dim
        if self.num_ensembles > 1:
            self.phi_nets = nn.ModuleList([
                MLP(input_dim=input_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)
                for _ in range(num_ensembles)
            ])
        else:
            self.phi_net = MLP(input_dim=input_dim, hidden_dims=list(hidden_dims) + [latent_dim], activate_final=False, layer_norm=layer_norm)

        self.apply(init_weights)

    def forward(self, observations, goals, is_phi=False, info=False):
        if is_phi:
            phi_s_input = observations # Assumed to be list if ensemble
            phi_g_input = goals      # Assumed to be list if ensemble
        else:
            if self.encoder:
                obs_feat = self.encoder(observations)
                goal_feat = self.encoder(goals)
            else:
                obs_feat = observations
                goal_feat = goals

            if self.num_ensembles > 1:
                phi_s_input = [net(obs_feat) for net in self.phi_nets]
                phi_g_input = [net(goal_feat) for net in self.phi_nets]
            else:
                phi_s_input = self.phi_net(obs_feat)
                phi_g_input = self.phi_net(goal_feat)

        if self.num_ensembles > 1:
            v_list = []
            for phi_s_single, phi_g_single in zip(phi_s_input, phi_g_input):
                dist_sq = ((phi_s_single - phi_g_single)**2).sum(dim=-1)
                v_list.append(-torch.sqrt(torch.clamp(dist_sq, min=1e-12)))
            v = torch.stack(v_list, dim=0) # (num_ensembles, batch_size)
        else:
            dist_sq = ((phi_s_input - phi_g_input)**2).sum(dim=-1)
            v = -torch.sqrt(torch.clamp(dist_sq, min=1e-12)) # (batch_size)

        if info:
            return v, phi_s_input, phi_g_input # Return potentially lists of tensors
        else:
            return v
