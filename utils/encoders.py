import functools
from typing import Any, Callable, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For AddSpatialCoordinates, can be torch too

from utils.networks import MLP as NetworksMLP # Use the one from networks.py
from utils.torch_utils import init_weights


def _get_padding_value(padding_str: str, kernel_size: Tuple[int, int]) -> Union[str, int]:
    if padding_str.upper() == 'SAME':
        # For stride 1, padding=(kernel_size-1)/2
        # PyTorch Conv2d padding can be an int for symmetric padding
        return (kernel_size[0] - 1) // 2 # Assuming square kernel for simplicity in 'SAME'
    elif padding_str.upper() == 'VALID':
        return 0
    else:
        try:
            return int(padding_str)
        except ValueError:
            raise ValueError(f"Invalid padding string: {padding_str}")


class ResnetStack(nn.Module):
    """ResNet stack module for PyTorch."""
    def __init__(self, in_features: int, num_features: int, num_blocks: int, max_pooling: bool = True):
        super().__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        # Initial Convolution
        # Flax padding='SAME' needs to be handled for PyTorch
        # PyTorch padding: (padding_left, padding_right, padding_top, padding_bottom) or int
        self.initial_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=self.num_features,
            kernel_size=(3, 3),
            stride=1,
            padding=_get_padding_value('SAME', (3,3)) # Calculate padding for 'SAME'
        )

        if self.max_pooling:
            # Flax max_pool padding='SAME' with stride 2 often means PyTorch padding=1 for 3x3 kernel
            self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            # Each block consists of two conv layers with relu and residual connection
            block_layers = nn.ModuleList([
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=(3,3), stride=1, padding=_get_padding_value('SAME', (3,3))),
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=(3,3), stride=1, padding=_get_padding_value('SAME', (3,3))),
            ])
            self.blocks.append(block_layers)

        self.apply(init_weights)

    def forward(self, x):
        x = self.initial_conv(x)
        if self.max_pooling:
            x = self.pool(x)

        for block_module_list in self.blocks:
            residual = x
            current_x = x
            current_x = block_module_list[0](current_x) # ReLU
            current_x = block_module_list[1](current_x) # Conv1
            current_x = block_module_list[2](current_x) # ReLU
            current_x = block_module_list[3](current_x) # Conv2
            x = current_x + residual
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder for PyTorch."""
    def __init__(self,
                 in_channels: int, # e.g. 3 for RGB, or more for stacked frames
                 width: int = 1,
                 stack_sizes: tuple = (16, 32, 32),
                 num_blocks: int = 2,
                 dropout_p: float = 0.0, # Changed from rate to p for PyTorch convention
                 mlp_input_dim_after_conv: int = 0, # Must be calculated based on conv output
                 mlp_hidden_dims: Sequence[int] = (512,),
                 mlp_output_dim: Optional[int] = None, # If None, last hidden_dim is used
                 layer_norm_mlp: bool = False):
        super().__init__()
        self.dropout_p = dropout_p

        self.stack_blocks = nn.ModuleList()
        current_channels = in_channels
        for i in range(len(stack_sizes)):
            out_channels_stack = stack_sizes[i] * width
            self.stack_blocks.append(
                ResnetStack(
                    in_features=current_channels,
                    num_features=out_channels_stack,
                    num_blocks=num_blocks
                )
            )
            current_channels = out_channels_stack # Output of one stack is input to next

        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=self.dropout_p)

        self.final_relu = nn.ReLU()
        self.layer_norm_conv_output = None
        if layer_norm_mlp: # The original layer_norm was applied after reshape, so it's on flattened features
             # This mlp_input_dim_after_conv needs to be correct.
             # It's (last_stack_out_channels * H_out * W_out)
             # For now, assuming it's passed correctly or MLP handles it.
             # Let's assume the MLP will receive the flattened features.
             # The LayerNorm should be applied on this flattened dim.
             # So, if layer_norm_mlp is True, it will be part of the MLP.
             pass


        # The MLP input dimension needs to be calculated based on the output shape of the conv layers.
        # This is crucial. If mlp_input_dim_after_conv is not provided, it must be inferred.
        # For now, we pass mlp_input_dim_after_conv.
        if not mlp_input_dim_after_conv:
            raise ValueError("mlp_input_dim_after_conv must be specified for ImpalaEncoder's MLP part.")

        self.mlp = NetworksMLP( # Using the MLP from utils.networks
            input_dim=mlp_input_dim_after_conv,
            hidden_dims=mlp_hidden_dims,
            output_dim=mlp_output_dim if mlp_output_dim is not None else mlp_hidden_dims[-1],
            activate_final=True, # Original MLP had activate_final=True
            layer_norm=layer_norm_mlp # Pass layer_norm to MLP
        )
        self.apply(init_weights)


    def forward(self, x, train=True): # train flag for dropout
        # Input x is expected to be (B, C, H, W)
        x = x.float() / 255.0 # Normalize to [0, 1]

        conv_out = x
        for stack_block in self.stack_blocks:
            conv_out = stack_block(conv_out)
            if self.dropout_p > 0 and self.dropout is not None:
                # PyTorch dropout takes 'training' flag, not 'deterministic'
                self.dropout.train(train) # Set dropout mode
                conv_out = self.dropout(conv_out)

        conv_out = self.final_relu(conv_out)

        # Flatten the output of conv layers for MLP
        # Original: out = conv_out.reshape((*x.shape[:-3], -1))
        # This means (Batch, Channels_out * H_out * W_out)
        out_flattened = torch.flatten(conv_out, start_dim=1) # Flatten all dims except batch

        # if self.layer_norm_conv_output: # If LayerNorm was meant for conv output before MLP
        #     out_flattened = self.layer_norm_conv_output(out_flattened)

        out_mlp = self.mlp(out_flattened)
        return out_mlp


class ResNetBlock(nn.Module):
    """ResNet block for PyTorch."""
    def __init__(self, in_filters: int, filters: int, strides: Tuple[int, int] = (1, 1), conv_cls=nn.Conv2d, norm_cls=nn.BatchNorm2d, act_fn=nn.ReLU):
        super().__init__()
        self.act_fn = act_fn()

        self.conv1 = conv_cls(in_filters, filters, kernel_size=(3, 3), stride=strides, padding=1, bias=False) # padding=1 for 3x3 to keep size with stride 1
        self.norm1 = norm_cls(filters)
        self.conv2 = conv_cls(filters, filters, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.norm2 = norm_cls(filters)

        self.shortcut = nn.Sequential()
        if strides != (1, 1) or in_filters != filters:
            self.shortcut = nn.Sequential(
                conv_cls(in_filters, filters, kernel_size=(1, 1), stride=strides, bias=False),
                norm_cls(filters)
            )
        self.apply(init_weights)


    def forward(self, x):
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act_fn(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y += residual
        y = self.act_fn(y)
        return y


class AddSpatialCoordinates(nn.Module):
    """Adds spatial coordinates to the input tensor."""
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Create meshgrid
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=self.dtype)
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij') # H, W

        # Stack to form (2, H, W) and expand to (B, 2, H, W)
        coords = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        return torch.cat([x, coords], dim=1) # Concatenate along channel dim


class PyTorchGroupNorm(nn.GroupNorm):
    """Wrapper for GroupNorm to handle 3D input (B, C, L) by unsqueezing/squeezing spatial dim."""
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3: # (B, C, L)
            # Add a dummy spatial dimension H=1, W=L
            x_4d = x.unsqueeze(2) # (B, C, 1, L)
            out_4d = super().forward(x_4d)
            return out_4d.squeeze(2) # (B, C, L)
        elif x.ndim == 4: # (B, C, H, W)
            return super().forward(x)
        else:
            raise ValueError(f"Expected 3D (B,C,L) or 4D (B,C,H,W) input to GroupNorm, got {x.ndim}D")


class ResNetEncoder(nn.Module):
    """ResNetV1 for PyTorch."""
    def __init__(self,
                 in_channels: int, # e.g. 3 for RGB, or 5 if AddSpatialCoordinates
                 stage_sizes: Sequence[int],
                 block_cls: type = ResNetBlock, # PyTorch ResNetBlock
                 num_filters_init: int = 64,
                 act_fn_name: str = "relu", # Changed from 'swish' as PyTorch default is relu
                 norm_layer_type: str = "batch", # 'batch' or 'group' or 'layer'
                 add_spatial_coordinates: bool = True):
        super().__init__()
        self.add_spatial_coordinates = add_spatial_coordinates
        if self.add_spatial_coordinates:
            self.spatial_coords_adder = AddSpatialCoordinates()
            current_in_channels = in_channels + 2 # 2 for x,y coords
        else:
            current_in_channels = in_channels

        self.act_fn = getattr(F, act_fn_name) if hasattr(F, act_fn_name) else getattr(nn, act_fn_name, nn.ReLU)()


        if norm_layer_type == "batch":
            norm_constructor = lambda num_feats: nn.BatchNorm2d(num_feats, eps=1e-5) # PyTorch default eps for BN
        elif norm_layer_type == "group":
            # GroupNorm needs num_groups, default to 4 from original if not specified
            norm_constructor = lambda num_feats: PyTorchGroupNorm(num_groups=min(4, num_feats), num_channels=num_feats, eps=1e-5)
        elif norm_layer_type == "layer":
            # LayerNorm on (B, C, H, W) needs to be applied carefully.
            # nn.LayerNorm normalizes over the last D dimensions.
            # For image data, usually GroupNorm or BatchNorm are preferred over full LayerNorm.
            # If applied per-channel: LayerNorm([C, H, W]) -> not what nn.LayerNorm(num_feats) does.
            # Flax nn.LayerNorm is typically elementwise over the last axis after flatten, or per-channel if specified.
            # Here, let's assume it's a channel-wise LayerNorm if specified for conv features.
            # This is non-standard for typical ResNets. Using BatchNorm as a safer default.
            print("Warning: 'layer' norm for ResNetEncoder conv layers is non-standard. Consider 'batch' or 'group'.")
            norm_constructor = lambda num_feats: nn.LayerNorm([num_feats,1,1], eps=1e-5) # Placeholder, needs correct shape
        else:
            raise ValueError(f"Unsupported norm_layer_type: {norm_layer_type}")

        self.conv_init = nn.Conv2d(current_in_channels, num_filters_init, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm_init = norm_constructor(num_filters_init)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        current_filters = num_filters_init
        for i, num_blocks_in_stage in enumerate(stage_sizes):
            stage_layers = []
            for j in range(num_blocks_in_stage):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                stage_layers.append(
                    block_cls(
                        in_filters=current_filters if j==0 else num_filters_init * (2**i),
                        filters=num_filters_init * (2**i),
                        strides=strides,
                        conv_cls=nn.Conv2d, # Pass PyTorch Conv
                        norm_cls=norm_constructor, # Pass constructor
                        act_fn=self.act_fn
                    )
                )
            self.stages.append(nn.Sequential(*stage_layers))
            current_filters = num_filters_init * (2**i) # Update current_filters for the start of the next stage if it's different logic.
                                                        # Actually, block_cls should handle in_filters correctly.
                                                        # The input to the first block of a stage is current_filters.
                                                        # Output of all blocks in stage is num_filters_init * (2**i)
            current_filters = num_filters_init * (2**i)


        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        self.output_dim = current_filters # Output dim after global avg pooling is num_filters of last stage.
        self.apply(init_weights)


    def forward(self, observations: torch.Tensor, train: Optional[bool] = None): # train flag not used by default in PyTorch eval
        # Input observations: (B, C, H, W) or (B, H, W, C)
        # Assuming (B, C, H, W) for PyTorch Conv2d
        if observations.ndim == 4 and observations.shape[-1] <=3 : # Likely (B,H,W,C)
            x = observations.permute(0,3,1,2).contiguous() # Convert to (B,C,H,W)
        else:
            x = observations # Assume (B,C,H,W)

        x = x.float() / 127.5 - 1.0 # Normalize to [-1, 1]

        if self.add_spatial_coordinates:
            x = self.spatial_coords_adder(x)

        x = self.conv_init(x)
        x = self.norm_init(x)
        x = self.act_fn(x)
        x = self.max_pool(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avg_pool(x) # (B, C_last_stage, 1, 1)
        x = torch.flatten(x, 1) # (B, C_last_stage)
        return x


class GCEncoder(nn.Module):
    """Helper module for goal-conditioned networks in PyTorch."""
    def __init__(self,
                 state_encoder: Optional[nn.Module] = None,
                 goal_encoder: Optional[nn.Module] = None,
                 concat_encoder: Optional[nn.Module] = None,
                 output_dim : Optional[int] = None): # Optional: specify expected output_dim for downstream use
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.concat_encoder = concat_encoder
        self._output_dim = output_dim # Store for potential use by other modules

    def forward(self, observations: torch.Tensor, goals: Optional[torch.Tensor] = None, goal_encoded: bool = False) -> torch.Tensor:
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        else: # If no state_encoder, pass observations directly if they are part of concat
            if self.concat_encoder is None and self.goal_encoder is None : # only obs
                 reps.append(observations)


        if goals is not None:
            if goal_encoded:
                assert self.goal_encoder is None or self.concat_encoder is None, \
                    "If goal_encoded is True, either goal_encoder or concat_encoder (or both) must be None."
                reps.append(goals) # goals are already encoded features
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                # If only concat_encoder, it takes raw obs and raw goals
                if self.concat_encoder is not None:
                    # Need to decide what state_part is if state_encoder is also None
                    state_part_for_concat = observations # Default to raw observations
                    if self.state_encoder is not None: # If state_encoder produced the first element of reps
                        state_part_for_concat = reps[0] # Use already encoded state if available and concat_encoder is separate

                    # This logic for concat_encoder input needs care.
                    # If state_encoder and goal_encoder are present, their outputs are in reps.
                    # If concat_encoder is also present, what does it take?
                    # Original Flax: reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
                    # This implies concat_encoder always takes raw obs and raw goals.
                    reps.append(self.concat_encoder(torch.cat([observations, goals], dim=-1)))

        if not reps and observations is not None : # Case: only observations, no encoders
             final_reps = observations
        elif len(reps) == 1: # Single representation (e.g. only state_encoded, or only goal_encoded)
            final_reps = reps[0]
        elif len(reps) > 1: # Multiple representations to concatenate
            final_reps = torch.cat(reps, dim=-1)
        else: # No inputs or encoders, should not happen if called with observations
            final_reps = observations # Fallback, though likely an issue upstream
            if observations is None:
                 raise ValueError("GCEncoder called with no observations and no encoders produced representations.")

        return final_reps

    @property
    def output_dim(self):
        if self._output_dim is not None:
            return self._output_dim
        # Try to infer output_dim if not set. This is complex and error-prone.
        # Needs example inputs. For now, raise error or return a placeholder.
        # This is a common challenge in migrating from JAX (shape inference) to PyTorch (explicit shapes).
        # Best practice: calculate and set output_dim during GCEncoder initialization based on its components.
        print("Warning: output_dim of GCEncoder is not explicitly set. Inference is not implemented.")
        return -1 # Placeholder


# Update encoder_modules to use PyTorch versions
# This requires knowing input_channels for vision encoders, and input_dim for MLPs.
# These dimensions depend on the specific observation space of the environment.
# For now, I'll create placeholders or make assumptions.
# The user of these encoders (e.g., Agent creation) will need to ensure correct dims are passed.

def create_mlp_encoder(input_dim: int, hidden_dims: Sequence[int] = (512, 512, 512, 512), layer_norm: bool = True, output_dim: Optional[int]=None):
    # Using the MLP from utils.networks which is more general
    return NetworksMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim if output_dim else hidden_dims[-1], activate_final=True, layer_norm=layer_norm)

def create_impala_encoder(in_channels: int, width: int = 1, stack_sizes: tuple = (16, 32, 32), num_blocks: int = 2, dropout_p: float = 0.0, mlp_input_dim_after_conv: int = 256, mlp_hidden_dims: Sequence[int] = (512,), mlp_output_dim: Optional[int]=None, layer_norm_mlp: bool = False):
    # mlp_input_dim_after_conv needs to be calculated based on H, W of input and conv architecture.
    # Example: if input is 64x64, and conv output is 8x8 with last stack_size*width channels,
    # then mlp_input_dim_after_conv = (stack_sizes[-1]*width) * 8 * 8.
    # This must be correctly passed.
    if mlp_output_dim is None: mlp_output_dim = mlp_hidden_dims[-1]
    return ImpalaEncoder(in_channels=in_channels, width=width, stack_sizes=stack_sizes, num_blocks=num_blocks, dropout_p=dropout_p, mlp_input_dim_after_conv=mlp_input_dim_after_conv, mlp_hidden_dims=mlp_hidden_dims, mlp_output_dim=mlp_output_dim, layer_norm_mlp=layer_norm_mlp)

def create_resnet_encoder(in_channels: int, stage_sizes: Sequence[int]=(3,4,6,3), num_filters_init:int=64, add_spatial_coordinates:bool=True):
    return ResNetEncoder(in_channels=in_channels, stage_sizes=stage_sizes, num_filters_init=num_filters_init, add_spatial_coordinates=add_spatial_coordinates)


# The encoder_modules dictionary will now store functions that can be called
# with appropriate dimensions (e.g., in_channels, input_dim) to instantiate encoders.
encoder_modules = {
    'mlp': create_mlp_encoder, # User needs to call: encoder_modules['mlp'](input_dim=obs_spec.shape[0])
    'impala': functools.partial(create_impala_encoder), # User passes in_channels, mlp_input_dim_after_conv, etc.
    'impala_debug': functools.partial(create_impala_encoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(create_impala_encoder, num_blocks=1),
    'impala_large': functools.partial(create_impala_encoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'resnet_34': functools.partial(create_resnet_encoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock),
    # Add other ResNet variants if needed, e.g., resnet_18, resnet_50
}

# Example of how GCEncoder might be used (assuming PyTorch modules for sub-encoders)
# state_enc = encoder_modules['mlp'](input_dim=obs_dim)
# goal_enc = encoder_modules['mlp'](input_dim=goal_dim)
# gc_enc = GCEncoder(state_encoder=state_enc, goal_encoder=goal_enc)
# combined_features = gc_enc(observations, goals)
