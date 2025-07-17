# Offline Reinforcement Learning with PyTorch

This repository provides a PyTorch implementation of various offline reinforcement learning algorithms, converted from an original JAX/Flax codebase. The goal is to replicate the behavior and performance of the original agents while leveraging the PyTorch ecosystem.

## Overview

The repository includes implementations of several offline RL agents, focusing on model-free and model-based approaches. The core training logic, network architectures, and agent-specific algorithms have been translated to PyTorch.

## Features

*   PyTorch implementations of agents such as ReBRAC, IQL, InfoMax, HILP, and others.
*   Modular network definitions for actors, critics, and encoders.
*   Dataset handling utilities for offline RL datasets.
*   Training and evaluation scripts.
*   Logging with TensorBoard and CSV.

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

We provide scripts to generate datasets for pre-training and fine-tuning in the `data_gen_scripts` folder.

### ExORL

<details>
<summary><b>Click to expand the commands to generate ExORL datasets</b></summary>

The default directory to store the datasets is `~/.exorl`.

1. Download exploratory datasets
   ```
   ./data_gen_scripts/download.sh cheetah rnd
   ./data_gen_scripts/download.sh walker rnd
   ./data_gen_scripts/download.sh quadruped rnd
   ./data_gen_scripts/download.sh jaco rnd
   ```
2. Generate pre-training datasets
   ```
   # cheetah
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run --save_path=~/.exorl/data/rnd_reward_free_cheetah.hdf5 --skip_size=0 --dataset_size=5_000_000 --relabel_reward=0
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run --save_path=~/.exorl/data/rnd_reward_free_cheetah_val.hdf5 --skip_size=5_000_000 --dataset_size=50_000 --relabel_reward=0
   
   # walker
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_walk --save_path=/root/RL/infom/datasets/rnd_reward_free_walker.hdf5 --skip_size=0 --dataset_size=5_000_000 --relabel_reward=0 --dataset_dir=/root/RL/infom/datasets
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_walk --save_path=/root/RL/infom/datasets/rnd_reward_free_walker_val.hdf5 --skip_size=5_000_000 --dataset_size=50_000 --relabel_reward=0 --dataset_dir=/root/RL/infom/datasets

   # quadruped
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_run --dataset_dir=~/.exorl/expl_datasets --save_path=~/.exorl/data/rnd_reward_free_quadruped.hdf5 --skip_size=0 --dataset_size=5_000_000 --relabel_reward=0
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_run --dataset_dir=~/.exorl/expl_datasets --save_path=~/.exorl/data/rnd_reward_free_quadruped_val.hdf5 --skip_size=50_000 --dataset_size=500_000 --relabel_reward=0
   
   # jaco
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_left --dataset_dir=~/.exorl/expl_datasets --save_path=~/.exorl/data/rnd_reward_free_jaco.hdf5 --skip_size=0 --dataset_size=5_000_000 --relabel_reward=0
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_left --dataset_dir=~/.exorl/expl_datasets --save_path=~/.exorl/data/rnd_reward_free_jaco_val.hdf5 --skip_size=50_000 --dataset_size=500_000 --relabel_reward=0
   ```
3. Generate fine-tuning datasets
   ```
   # cheetah {run, run backward, walk, walk backward}
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_run.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_run_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run_backward --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_run_backward.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1 --dataset_dir=/root/RL/infom/datasets
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_run_backward --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_run_backward_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1 --dataset_dir=/root/RL/infom/datasets
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_walk --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_walk.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_walk --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_walk_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_walk_backward --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_walk_backward.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=cheetah_walk_backward --save_path=~/.exorl/data/rnd_reward_labeled_cheetah_walk_backward_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   # walker {walk, run, stand, flip}
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_walk --save_path=/root/RL/infom/datasets/rnd_reward_labeled_walker_walk.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1 --dataset_dir=/root/RL/infom/datasets
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_walk --save_path=/root/RL/infom/datasets/rnd_reward_labeled_walker_walk_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1 --dataset_dir=/root/RL/infom/datasets
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_run --save_path=~/.exorl/data/rnd_reward_labeled_walker_run.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_run --save_path=~/.exorl/data/rnd_reward_labeled_walker_run_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_stand --save_path=~/.exorl/data/rnd_reward_labeled_walker_stand.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_stand --save_path=~/.exorl/data/rnd_reward_labeled_walker_stand_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_flip --save_path=~/.exorl/data/rnd_reward_labeled_walker_flip.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=walker_flip --save_path=~/.exorl/data/rnd_reward_labeled_walker_flip_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   # quadruped {run, jump, stand, walk}
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_run --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_run.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_run --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_run_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_jump --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_jump.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_jump --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_jump_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_stand --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_stand.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_stand --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_stand_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_walk --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_walk.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=quadruped_walk --save_path=~/.exorl/data/rnd_reward_labeled_quadruped_walk_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   # jaco {reach top left, reach top right, reach bottom left, reach bottom right}
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_left --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_top_left.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_left --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_top_left_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_right --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_top_right.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_top_right --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_top_right_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_bottom_left --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_bottom_left.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_bottom_left --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_bottom_left_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_bottom_right --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_bottom_right.hdf5 --skip_size=5_000_000 --dataset_size=500_000 --relabel_reward=1
   python data_gen_scripts/generate_exorl_dataset.py --env_name=jaco_reach_bottom_right --save_path=~/.exorl/data/rnd_reward_labeled_jaco_reach_bottom_right_val.hdf5 --skip_size=5_500_000 --dataset_size=50_000 --relabel_reward=1
   ```

</details>

### OGBench

<details>
<summary><b>Click to expand the commands to generate OGBench datasets</b></summary>

We use the default datasets in OGBench for pre-training. The following datasets will be downloaded automatically to `~/.ogbench/data` when executing the code:
- cube single 
  - cube-single-play-v0
  - cube-single-play-v0-val
- cube double 
  - cube-double-play-v0
  - cube-double-play-v0-val
- scene
  - scene-play-v0
  - scene-play-v0-val
- puzzle 4x4
  - puzzle-4x4-play-v0
  - puzzle-4x4-play-v0-val 
- visual cube single task 1
  - visual-cube-single-play-v0
  - visual-cube-single-play-v0-val
- visual cube double task 1
  - visual-cube-double-play-v0
  - visual-cube-double-play-v0-val
- visual scene task 1
  - visual-scene-play-v0
  - visual-scene-play-v0-val
- visual puzzle 4x4 task 1
  - visual-puzzle-4x4-play-v0
  - visual-puzzle-4x4-play-v0-val

We generate fine-tuning datasets using following commands

```
# cube single
python data_gen_scripts/generate_manipspace.py --env_name=cube-single-v0 --save_path=~/.ogbench/data/cube-single-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=cube-single-v0 --save_path=~/.ogbench/data/cube-single-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# cube double
python data_gen_scripts/generate_manipspace.py --env_name=cube-double-v0 --save_path=~/.ogbench/data/cube-double-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=cube-double-v0 --save_path=~/.ogbench/data/cube-double-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# scene
python data_gen_scripts/generate_manipspace.py --env_name=scene-v0 --save_path=~/.ogbench/data/scene-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=scene-v0 --save_path=~/.ogbench/data/scene-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# puzzle 4x4
python data_gen_scripts/generate_manipspace.py --env_name=puzzle-4x4-v0 --save_path=~/.ogbench/data/puzzle-4x4-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=puzzle-4x4-v0 --save_path=~/.ogbench/data/puzzle-4x4-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# visual cube single task 1
python data_gen_scripts/generate_manipspace.py --env_name=visual-cube-single-v0 --save_path=~/.ogbench/data/visual-cube-single-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=visual-cube-single-v0 --save_path=~/.ogbench/data/visual-cube-single-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# visual cube double task 1
python data_gen_scripts/generate_manipspace.py --env_name=visual-cube-double-v0 --save_path=~/.ogbench/data/visual-cube-double-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=visual-cube-double-v0 --save_path=~/.ogbench/data/visual-cube-double-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# visual scene task 1
python data_gen_scripts/generate_manipspace.py --env_name=visual-scene-v0 --save_path=~/.ogbench/data/visual-scene-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=visual-scene-v0 --save_path=~/.ogbench/data/visual-scene-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

# visual puzzle 4x4 task 1
python data_gen_scripts/generate_manipspace.py --env_name=visual-puzzle-4x4-v0 --save_path=~/.ogbench/data/visual-puzzle-4x4-play-ft-v0.npz --num_episodes=500 --max_episode_steps=1001 --dataset_type=play
python data_gen_scripts/generate_manipspace.py --env_name=visual-puzzle-4x4-v0 --save_path=~/.ogbench/data/visual-puzzle-4x4-play-ft-v0-val.npz --num_episodes=50 --max_episode_steps=1001 --dataset_type=play
```

</details>

## Usage

### Training

The main script for training agents is `main.py`.

**Example training command:**

```bash
python main.py \
    --env_name <environment_name> \
    --seed <random_seed> \
    --save_dir ./exp_pytorch \
    --pretraining_steps 100000 \
    --finetuning_steps 50000 \
    --log_interval 1000 \
    --eval_interval 10000

MUJOCO_GL=osmesa PYOPENGL_PLATFORM="osmesa" python main.py --env-name walker_walk --seed 1 --save-dir ./logdir --pretraining-steps 1000000 --finetuning-steps 500000 --log-interval 100 --eval-interval 10000 --save-interval 500000 --video-episodes 0
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
