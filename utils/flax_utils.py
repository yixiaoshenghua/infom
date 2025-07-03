import glob
import os
import pickle
from typing import Any, Dict, Optional # Sequence, Mapping removed as ModuleDict is removed

import torch
import torch.nn as nn
import torch.optim as optim
# Removed flax, jax, jax.numpy, optax, functools (nonpytree_field)


# ModuleDict is removed. PyTorch's nn.ModuleDict can be used directly if needed,
# or models can be structured with attributes for sub-modules.

class TrainState:
    """Custom train state for PyTorch models.

    Attributes:
        step: Counter to keep track of the training steps.
        model: The PyTorch model (or a dictionary of models).
        optimizer: The PyTorch optimizer (or a dictionary of optimizers).
        lr_scheduler: Optional PyTorch learning rate scheduler.
    """

    def __init__(self,
                 model: nn.Module, # Can be a single model or nn.ModuleDict
                 optimizer: optim.Optimizer, # Can be a single optimizer or a dict of optimizers
                 lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        self.step = 0
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def apply_gradients(self, loss: torch.Tensor, grad_clip_norm: Optional[float] = None) -> Dict[str, float]:
        """Computes gradients, applies them, and updates the step.

        Args:
            loss: The computed loss tensor.
            grad_clip_norm: Optional gradient clipping norm.

        Returns:
            A dictionary containing gradient statistics (e.g., grad_norm).
        """
        self.optimizer.zero_grad()
        loss.backward()

        grad_info = {}
        if grad_clip_norm is not None:
            # grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            # If model is a ModuleDict, need to iterate through its parameters or all optimizers' params
            params_to_clip = []
            if isinstance(self.model, nn.ModuleDict):
                for m_name in self.model:
                    params_to_clip.extend(self.model[m_name].parameters())
            else: # Single model
                params_to_clip.extend(self.model.parameters())

            if params_to_clip: # Ensure there are parameters to clip
                 grad_norm = nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
                 grad_info['grad_norm'] = grad_norm.item()


        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step() # Advance LR scheduler

        self.step += 1
        return grad_info

    def update(self, loss: torch.Tensor, grad_clip_norm: Optional[float] = None) -> Dict[str, float]:
        """Alias for apply_gradients for convenience if the agent structure uses 'update'."""
        return self.apply_gradients(loss, grad_clip_norm)

    def get_learning_rate(self) -> float:
        """Returns the current learning rate from the optimizer."""
        # Assumes a single optimizer or all param groups have the same LR
        return self.optimizer.param_groups[0]['lr']

    # The __call__ and select methods from Flax TrainState are not directly applicable
    # as model forward passes are done via model_instance(data).
    # If specific sub-modules of a larger nn.ModuleDict need to be called,
    # that logic would be handled by the agent itself (e.g., self.model['actor'](obs)).

    # apply_loss_fn is also removed as the gradient calculation and update
    # are more explicit in PyTorch (loss.backward(), optimizer.step()).
    # Gradient statistics can be computed manually if needed after .backward()
    # and before .step().


def save_agent_components(components: Dict[str, Any], save_dir: str, epoch: int, name_prefix="agent"):
    """Saves agent components (model state_dict, optimizer state_dict, etc.) to a file.

    Args:
        components: A dictionary where keys are component names (e.g., 'actor_model', 'actor_optimizer')
                    and values are the Pytorch objects (e.g., model.state_dict(), optimizer.state_dict()).
        save_dir: Directory to save the components.
        epoch: Epoch number.
        name_prefix: Prefix for the saved file name.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name_prefix}_epoch_{epoch}.pth')
    torch.save(components, save_path)
    print(f'Saved agent components to {save_path}')


def load_agent_components(restore_path: str, component_names: list[str], device: torch.device) -> Dict[str, Any]:
    """Restores agent components from a file.

    Args:
        restore_path: Exact path to the .pth file.
        component_names: A list of component names that are expected to be keys in the loaded dictionary.
                         (This is more for user verification, not strictly enforced here).
        device: The torch device to load tensors onto.

    Returns:
        A dictionary with the loaded components.
    """
    if not os.path.exists(restore_path):
        # Try to find it with glob if a directory was given (legacy behavior)
        if os.path.isdir(restore_path):
            # This part is a bit problematic if epoch is not passed.
            # The original function expected restore_path to be a directory and epoch to be separate.
            # For now, let's assume restore_path is the full file path.
            # If it's a directory, the user needs to specify the exact file.
            raise FileNotFoundError(f"Restore path is a directory, please provide the full path to the .pth file: {restore_path}")
        else:
            raise FileNotFoundError(f"Restore path not found: {restore_path}")

    print(f'Loading agent components from {restore_path}')
    # map_location ensures tensors are loaded to the specified device.
    loaded_components = torch.load(restore_path, map_location=device)

    # Basic check if expected components are present
    for name in component_names:
        if name not in loaded_components:
            print(f"Warning: Expected component '{name}' not found in the loaded file.")

    return loaded_components


# The old save_agent and restore_agent are specific to the Flax agent structure.
# The new functions save/load generic dictionaries of state_dicts.
# The agent itself will be responsible for populating these dictionaries
# and restoring its state from them.

# Example usage for new save/restore:
# To save:
# agent_data_to_save = {
#     'actor_model_state_dict': actor_model.state_dict(),
#     'critic_model_state_dict': critic_model.state_dict(),
#     'actor_optimizer_state_dict': actor_optimizer.state_dict(),
#     'critic_optimizer_state_dict': critic_optimizer.state_dict(),
#     'train_step': agent_train_state.step, # Or any other serializable data
# }
# save_agent_components(agent_data_to_save, "/path/to/save_dir", epoch_num)

# To restore:
# loaded_data = load_agent_components("/path/to/file.pth", ['actor_model_state_dict', ...], device)
# actor_model.load_state_dict(loaded_data['actor_model_state_dict'])
# critic_model.load_state_dict(loaded_data['critic_model_state_dict'])
# actor_optimizer.load_state_dict(loaded_data['actor_optimizer_state_dict'])
# ...
# agent_train_state.step = loaded_data['train_step']
