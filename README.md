# Offline Reinforcement Learning with PyTorch

This repository provides a PyTorch implementation of various offline reinforcement learning algorithms, converted from an original JAX/Flax codebase. The goal is to replicate the behavior and performance of the original agents while leveraging the PyTorch ecosystem.

## Overview

The repository includes implementations of several offline RL agents, focusing on model-free and model-based approaches. The core training logic, network architectures, and agent-specific algorithms have been translated to PyTorch.

## Features

*   PyTorch implementations of agents such as ReBRAC, IQL, InfoMax, HILP, and others.
*   Modular network definitions for actors, critics, and encoders.
*   Dataset handling utilities for offline RL datasets.
*   Training and evaluation scripts.
*   Logging with Weights & Biases and CSV.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The primary dependencies are PyTorch, NumPy, and other libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have a version of PyTorch installed that is compatible with your CUDA version if you plan to use GPUs. You can find installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Dataset Preparation

This codebase is designed to work with datasets commonly used in offline reinforcement learning (e.g., D4RL, OGBench, custom datasets).

*   **D4RL:** If using D4RL datasets, ensure you have the `d4rl` library installed and the datasets downloaded.
*   **Custom Datasets:** Data is typically expected in a dictionary-like format where keys correspond to `observations`, `actions`, `rewards`, `next_observations`, `terminals` (or `dones`). The `utils/datasets.py` file handles loading and processing of this data. Refer to `envs/env_utils.py` for examples of how environments and datasets are registered and loaded.
*   The `data_gen_scripts/` directory may contain scripts for downloading or generating specific datasets used by the original codebase.

## Usage

### Training

The main script for training agents is `main.py`.

**Example training command:**

```bash
python main.py \
    --env_name <environment_name> \
    --agent agents/rebrac.py \
    --wandb_run_group <your_wandb_group> \
    --seed <random_seed> \
    --save_dir ./exp_pytorch \
    --pretraining_steps 100000 \
    --finetuning_steps 50000 \
    --log_interval 1000 \
    --eval_interval 10000
```

**Key command-line arguments for `main.py`:**

*   `--env_name`: Name of the environment/dataset (e.g., `halfcheetah-medium-v2`, `pen-human-v1` from D4RL, or custom names).
*   `--agent`: Path to the agent's configuration file (e.g., `agents/rebrac.py`, `agents/iql.py`). This file typically defines agent-specific hyperparameters using `ml_collections.ConfigDict`.
*   `--seed`: Random seed for reproducibility.
*   `--save_dir`: Directory to save experiment outputs (logs, model checkpoints).
*   `--restore_path`: Path to a saved agent checkpoint (`.pth` file) to resume training or for evaluation.
*   `--pretraining_steps`: Number of offline pre-training steps.
*   `--finetuning_steps`: Number of online fine-tuning steps (if applicable).
*   `--batch_size`: Batch size for training (usually defined in the agent's config file, but can be overridden if `main.py` supports it).
*   `--lr`: Learning rate (usually defined in the agent's config file).
*   `--enable_wandb`: Set to `1` to enable Weights & Biases logging (default), `0` to disable.
*   `--wandb_run_group`: Wandb run group name.
*   `--wandb_mode`: Wandb mode (`online`, `offline`, `disabled`).
*   ... and other flags defined in `main.py`.

Refer to the agent-specific configuration files in the `agents/` directory for hyperparameters related to each algorithm.

### Evaluation

Evaluation is typically performed periodically during training, controlled by `--eval_interval` and `--eval_episodes`.

To run evaluation on a pre-trained model, you can often use `main.py` with `--finetuning_steps 0` (or a very small number of total steps just past pretraining) and by providing `--restore_path` pointing to your saved model checkpoint. Ensure `--eval_interval` is set appropriately (e.g., to 1 or a low number if you only want evaluation).

The `utils/evaluation.py` script contains the core evaluation logic.

## Code Structure

*   `main.py`: Main script for training and evaluation.
*   `agents/`: Contains implementations of different offline RL agents.
    *   `crl_infonce_pytorch.py`: PyTorch implementation of Contrastive RL with InfoNCE.
    *   `dino_rebrac_pytorch.py`: PyTorch implementation of DINO with ReBRAC.
    *   Other files may be PyTorch versions (e.g., `fb_repr.py`) or original JAX versions pending conversion.
    *   Each agent file typically defines the agent class and a `get_config()` function for its default hyperparameters.
*   `utils/`: Utility functions.
    *   `networks.py`: Core PyTorch network architectures (MLP, Actor, Critic, etc.).
    *   `encoders.py`: PyTorch implementations of various encoders (CNNs like Impala, ResNet).
    *   `datasets.py`: NumPy-based dataset handling, replay buffers, and goal-conditioned dataset utilities.
    *   `torch_utils.py`: PyTorch-specific helper functions (device management, initialization).
    *   `flax_utils.py`: Contains the PyTorch `TrainState` and new `save/load_agent_components` functions (name kept for minimal diff during transition, but content is PyTorch-specific).
    *   `evaluation.py`: Evaluation loops.
    *   `log_utils.py`: Logging utilities (CSV, Wandb).
*   `envs/`: Environment wrappers and dataset registration.
    *   `env_utils.py`: Utility to create environments and associated datasets.
*   `custom_dmc_tasks/`: Custom DeepMind Control Suite tasks (if used).
*   `data_gen_scripts/`: Scripts for dataset generation or downloading.
*   `requirements.txt`: Python dependencies.

## Notes on JAX to PyTorch Conversion

This codebase is a result of converting a JAX/Flax/Optax project to PyTorch. Key aspects of the conversion include:
*   Replacing JAX array operations with PyTorch tensor operations.
*   Converting Flax `nn.Module`s to `torch.nn.Module`s.
*   Replacing Optax optimizers with `torch.optim` equivalents.
*   Managing randomness with `torch.manual_seed`.
*   Handling device placement explicitly using `tensor.to(device)`.
*   Removing JAX-specific transformations like `@jit` and `@vmap`, relying on PyTorch's eager execution, computation graph, and batched operations. `torch.compile` can be explored for performance optimization.

While efforts were made to maintain functional equivalence, subtle differences in numerical precision, default library behaviors, or random number generation between JAX and PyTorch might lead to variations in results.
As of the latest update, `CRLInfoNCEAgent` (as `crl_infonce_pytorch.py`) and `DINOReBRACAgent` (as `dino_rebrac_pytorch.py`) have been converted to PyTorch. Other agents are progressively being converted.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE). (Assuming Apache License from original repo, please update if different).
