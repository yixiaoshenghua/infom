# PyTorch related utility functions will be added here.
import torch

def set_device():
    """Sets the device to CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m, gain=1.0):
    """Initializes weights of a PyTorch module."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

def to_torch(x, device=None):
    """Converts a NumPy array or JAX array to a PyTorch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device) if device else x
    return torch.from_numpy(x).float().to(device) if device else torch.from_numpy(x).float()

def to_numpy(x):
    """Converts a PyTorch tensor to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x
