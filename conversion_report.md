# JAX to PyTorch Conversion Report

This report details the conversion of JAX-based reinforcement learning agents to their PyTorch equivalents. The primary goal was to maintain model architecture, training logic, and output consistency while ensuring compatibility with the modern PyTorch ecosystem.

## Files Converted

The following agent files were converted from JAX to PyTorch:

1.  `agents/crl_infonce.py`
2.  `agents/dino_rebrac.py`
3.  `agents/fb_repr.py`
4.  `agents/hilp.py`
5.  `agents/infom.py`
6.  `agents/iql.py`
7.  `agents/mbpo_rebrac.py`
8.  `agents/td_infonce.py`

## Core Conversion Rules Applied

The conversion process adhered to the following general rules:

*   **Model Architecture**:
    *   `flax.linen.Module` or `haiku.Module` instances were converted to `torch.nn.Module`.
    *   Weight initialization is handled by PyTorch's default mechanisms for its layers (e.g., `torch.nn.Linear`, `torch.nn.Conv2d`). Where JAX used specific initialization schemes (e.g., `final_fc_init_scale`), corresponding PyTorch custom initialization would be needed in the network definitions (assumed to be handled within the PyTorch versions of `Actor`, `Value`, etc.).
*   **Automatic Differentiation**:
    *   `@jax.jit` decorators were removed. PyTorch's dynamic computation graph handles JIT compilation implicitly, and `torch.compile` (for PyTorch 2.0+) can be applied for further optimization at a higher level.
    *   `jax.grad()` was replaced by PyTorch's `autograd` mechanism (`loss.backward()`).
*   **Training Logic**:
    *   **Optimizers**: `optax` optimizers (e.g., `optax.adam`) were replaced with `torch.optim` equivalents (e.g., `torch.optim.Adam`).
    *   **Loss Functions**: Mathematical logic was preserved. JAX-specific operations like `jax.vmap` for per-example losses or ensemble operations were translated to PyTorch tensor operations, often involving explicit iteration or careful broadcasting.
    *   **Data Handling**: JAX batch processing was replaced by assuming batches are dictionaries of PyTorch tensors. `jax.tree_util.tree_map` for applying functions to batches/parameters was replaced by manual iteration or equivalent PyTorch patterns.
*   **Distributed Training**:
    *   `jax.pmap` considerations were noted but not implemented as part of this file-level conversion. PyTorch DDP/FSDP would be applied at the training script level.
*   **Random Number Generation**:
    *   `jax.random.PRNGKey` and its splitting mechanism were replaced by `torch.manual_seed` for global seeding and PyTorch's default RNG. For operations requiring JAX-like per-call determinism with explicit keys (e.g., sampling noise in some actor losses or action sampling), the PyTorch versions now rely on global seeding set before such operations if a seed is provided to the method. More fine-grained control would require `torch.Generator` objects.
*   **Device Management**:
    *   `jax.device_put()` was replaced by `tensor.to(device)`. A `self.device` attribute was added to agent classes.
*   **Array/Tensor Operations**:
    *   `jax.numpy` (jnp) operations were mapped to `torch.tensor` operations.
    *   `flax.struct.PyTreeNode` base class was replaced with `torch.nn.Module`.
    *   `chex` assertions were not directly translated but would correspond to `torch.testing` or standard Python asserts.

## Key Library/Component Mapping

| JAX Component                      | PyTorch Equivalent/Approach                                     | Notes                                                                                                                              |
| :--------------------------------- | :-------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `flax.struct.PyTreeNode`           | `torch.nn.Module`                                               | PyTorch modules manage parameters and submodules.                                                                                  |
| `flax.linen.Module` / `haiku.Module` | `torch.nn.Module`                                               | Custom network classes (`Actor`, `Value`, `GCValue`, etc.) are assumed to be reimplemented as `torch.nn.Module`s.                 |
| `optax` optimizers (e.g. `adam`)   | `torch.optim` (e.g. `Adam`)                                     | Optimizer state is managed by PyTorch optimizer objects.                                                                           |
| `jax.numpy` (jnp)                  | `torch` (PyTorch tensor operations)                             | Most operations have direct or near-direct equivalents.                                                                            |
| `jax.jit`                          | Removed (PyTorch default) / `torch.compile` (optional)          |                                                                                                                                    |
| `jax.grad`                         | `loss.backward()` / `torch.autograd.grad()`                     | Standard PyTorch autograd.                                                                                                         |
| `jax.vmap`                         | Manual iteration, broadcasting, or `torch.vmap` (experimental)  | Often handled by operating on an ensemble dimension or using broadcasting.                                                         |
| `jax.lax.scan`                     | Python `for` loop                                               | Used in `InFOMAgent` for iterative flow computation.                                                                               |
| `jax.lax.stop_gradient`            | `tensor.detach()`                                               |                                                                                                                                    |
| `jax.random.PRNGKey`, `jax.random.split` | `torch.manual_seed`, global RNG, (opt. `torch.Generator`)   | Simplification of RNG. JAX's explicit key system provides stronger guarantees for reproducibility of specific random operations. |
| `TrainState` (custom Flax util)    | Direct parameter management via `nn.Module`, `torch.optim`      | PyTorch agents store networks (which own their parameters) and optimizers as attributes. EMA updates are handled manually.         |
| `ModuleDict` (custom Flax util)    | `torch.nn.ModuleDict`                                           | Assumed to be available in `utils.pytorch_utils`.                                                                                  |
| `nonpytree_field`                  | Standard Python attribute                                       | Configs are typically plain dictionaries or dataclasses in PyTorch.                                                                |
| `ml_collections.ConfigDict`        | Python `dict`                                                   | Standard Python dictionaries used for configuration.                                                                               |

## General Challenges and Notes

1.  **Network Implementations**: This conversion assumes that PyTorch equivalents of specialized JAX networks (e.g., `Actor`, `Value`, `GCValue`, `GCMetricValue`, `GCBilinearValue`, `VectorField`, `IntentionEncoder`, and various `encoders`) are available in `utils_pytorch` directories and correctly mirror the JAX versions' interfaces and functionalities (including handling of ensembles, `goal_encoded` flags, `info=True` flags for extra outputs like `phi`/`psi`). The correctness of the agent logic heavily depends on these underlying network implementations.

2.  **`GCBilinearValue` and `GCMetricValue`**: These modules from some JAX agents have specific output requirements (like returning `phi`, `psi` features, or handling `info=True` flags). The PyTorch versions must replicate this. For instance, TD-InfoNCE and FB-Rep agents rely on `GCBilinearValue` producing (Batch x Batch x Ensemble) Q-logits for their actor loss calculations. HILP relies on `GCMetricValue` being able to provide `phi` features.

3.  **RNG**: The shift from JAX's explicit PRNGKey system to PyTorch's global RNG is a significant conceptual change. While global seeding ensures overall run-to-run reproducibility, achieving the same level of isolated reproducibility for specific stochastic operations (like action sampling with a given key in JAX) would require more elaborate use of `torch.Generator` instances in PyTorch, which was outside the scope of this direct translation.

4.  **`einsum` and `vmap`**: Complex JAX operations involving `einsum` (e.g., in contrastive losses) and `vmap` (often used for per-ensemble or per-example computations) were translated to PyTorch `einsum` and tensor operations. These translations require careful validation to ensure dimensional correctness and equivalent logic.

5.  **`TrainState` Pattern**: JAX agents often use a `TrainState` object to bundle parameters, gradients, and optimizer state. PyTorch handles this differently: `nn.Module`s own their parameters, and `torch.optim.Optimizer` objects manage optimizer state and apply gradients. Target network updates (EMA) are implemented as helper methods within the PyTorch agent classes.

6.  **Batch Data Structure**: The PyTorch agents assume that input `batch` (and `model_batch` where applicable) is a dictionary of PyTorch tensors, with keys matching those used in the JAX versions.

7.  **Configuration**: `ml_collections.ConfigDict` was replaced by standard Python dictionaries for configuration.

8.  **Debugging and Testing**: Each converted agent includes a basic `if __name__ == '__main__':` block for smoke testing initialization and core update/sampling methods. However, thorough testing with actual training loops, data, and comparison against the original JAX agent's behavior is crucial to verify correctness.

## Agent-Specific Noteworthy Points

*   **`crl_infonce.py`**: The actor loss's `log_ratios` calculation, which depends on a (Batch x Batch x Ensemble) Q-logit structure from `GCBilinearValue`, was a key complex part.
*   **`dino_rebrac.py`**: DINO loss and ReBRAC penalties. The DINO target center update logic was slightly re-interpreted for PyTorch common practices.
*   **`fb_repr.py`**: The forward-backward representation loss with its `einsum` operations and orthonormalization term was complex. The FB actor loss also uses `einsum` for Q-value calculation.
*   **`hilp.py`**: Relies on `GCMetricValue` providing `phi` features for latent sampling and inference. The HILP value loss (for learning phi) and IQL-style skill policy losses were translated.
*   **`infom.py`**: The flow occupancy loss (SARSA^2 flow matching) and the `_compute_fwd_flow_goals` method (Euler integration of VF) were major components. Handling of potentially encoded observations for the VF was important.
*   **`iql.py`**: Standard IQL conversion, relatively straightforward compared to others. Supports 'awr' and 'ddpgbc' actor losses.
*   **`mbpo_rebrac.py`**: Combines a learned world model (transition, reward) with ReBRAC for policy optimization using model-generated rollouts.
*   **`td_infonce.py`**: Highly complex contrastive loss involving importance sampling from a target critic. The actor loss relies on specific (Batch x Batch x Ensemble) Q-logit structures from `GCBilinearValue`. This agent's correctness is very sensitive to the `GCBilinearValue` PyTorch implementation.

## Conclusion

The conversion provides a functional PyTorch baseline for the listed JAX agents. The primary challenges revolved around replicating complex JAX patterns (like `vmap`, specific uses of `einsum`, and `PRNGKey` logic) in PyTorch, and the heavy dependency on the correct PyTorch reimplementation of shared utility modules and network architectures. Extensive testing and validation are recommended to ensure full behavioral parity with the original JAX implementations.
