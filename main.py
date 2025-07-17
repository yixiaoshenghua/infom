import os
import json
import random
import time
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import warnings
import logging

import torch # Added
import numpy as np
import tqdm
from collections import defaultdict
import argparse
from agents import agents # Should now import PyTorch agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import GCDataset, Dataset, ReplayBuffer # May need PyTorch compatibility adjustments
from utils.evaluation import evaluate # Needs to be PyTorch compatible
# Changed from flax_utils to torch_utils for save/load
from utils.flax_utils import load_agent_components, save_agent_components
from utils.log_utils import Logger, get_exp_name, get_render_video, TerminalOutput, TensorBoardOutputPytorch, JSONLOutput
from utils.torch_utils import set_device, to_torch, to_numpy # Added
try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorboard
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*", category=FutureWarning)
warnings.filterwarnings("ignore", ".*")
warnings.filterwarnings("ignore", "Conversion of an array", category=DeprecationWarning)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'

from agents.infom import get_config # Agent specific config

def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')

    # base configure
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--env-name', type=str, default='walker_walk',#'cube-single-play-singletask-v0',
                        dest='env_name', help='Environment (dataset) name')
    parser.add_argument('--save-dir', type=str, default='logdir/',
                        dest='save_dir', help='Save directory')
    parser.add_argument('--restore-path', type=str, default=None,
                        dest='restore_path', help='Restore path (exact file path to .pth)')
    parser.add_argument('--restore-epoch', type=int, default=None,
                        dest='restore_epoch', help='Restore epoch (used if restore_path is a dir pattern)')

    # train
    parser.add_argument('--pretraining-steps', type=int, default=1_000_000,
                        dest='pretraining_steps', help='Number of offline steps')
    parser.add_argument('--pretraining-size', type=int, default=1_000_000,
                        dest='pretraining_size', help='Size of the dataset for pre-training')
    parser.add_argument('--finetuning-steps', type=int, default=500_000,
                        dest='finetuning_steps', help='Number of online steps')
    parser.add_argument('--finetuning-size', type=int, default=500_000,
                        dest='finetuning_size', help='Size of the dataset for fine-tuning')

    # logging
    parser.add_argument('--log-interval', type=int, default=5_000,
                        dest='log_interval', help='Logging interval')
    parser.add_argument('--eval-interval', type=int, default=50_000,
                        dest='eval_interval', help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=1_500_000,
                        dest='save_interval', help='Saving interval')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        dest='eval_episodes', help='Number of evaluation episodes')
    parser.add_argument('--video-episodes', type=int, default=0,
                        dest='video_episodes', help='Number of video episodes for each task')
    parser.add_argument('--video-frame-skip', type=int, default=3,
                        dest='video_frame_skip', help='Frame skip for videos')

    # data processing
    parser.add_argument('--obs-norm-type', type=str, default='normal',
                        dest='obs_norm_type', help='Observation normalization type (none, normal, bounded)')
    parser.add_argument('--p-aug', type=float, default=None,
                        dest='p_aug', help='Probability of applying image augmentation')
    parser.add_argument('--num-aug', type=int, default=1,
                        dest='num_aug', help='Number of image augmentations')
    parser.add_argument('--inplace-aug', type=int, default=1,
                        dest='inplace_aug', help='Whether to replace the original image after augmentations')

    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use tensorboard for logging')

    parser.add_argument('--frame-stack', type=int, default=None,
                        dest='frame_stack', help='Number of frames to stack')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = get_config() # Agent specific config
    # Set up logger and device
    exp_name = get_exp_name(args.seed)
    args.save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    device = set_device()
    print(f"Using device: {device}")

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        saved_configs = {}
        saved_configs.update(config)
        saved_configs.update(vars(args))
        json.dump(saved_configs, f, sort_keys=True, indent=4)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Make environment and datasets
    # make_env_and_datasets might need adjustment if it returns JAX arrays
    # Assuming it returns NumPy arrays which are then handled by Dataset class
    _, _, pretraining_train_dataset_np, pretraining_val_dataset_np = make_env_and_datasets(
        args.env_name, frame_stack=args.frame_stack, max_size=args.pretraining_size, reward_free=True)
    _, eval_env, finetuning_train_dataset_np, finetuning_val_dataset_np = make_env_and_datasets(
        args.env_name, frame_stack=args.frame_stack, max_size=args.finetuning_size, reward_free=False)

    if args.video_episodes > 0:
        assert 'singletask' in args.env_name, 'Rendering is currently only supported for OGBench environments.'


    # Set up datasets (Dataset class might need to handle PyTorch tensors or ensure conversion)
    # Assuming Dataset.create and GCDataset can work with NumPy data and agent will handle torch conversion.
    pretraining_train_dataset = Dataset.create(**pretraining_train_dataset_np)
    finetuning_train_dataset = Dataset.create(**finetuning_train_dataset_np)

    pretraining_val_dataset = None
    if pretraining_val_dataset_np is not None:
        pretraining_val_dataset = Dataset.create(**pretraining_val_dataset_np)

    finetuning_val_dataset = None
    if finetuning_val_dataset_np is not None:
        finetuning_val_dataset = Dataset.create(**finetuning_val_dataset_np)


    if config['agent_name'] == 'mbpo_rebrac':
        example_transition = {k: v[0] for k, v in finetuning_train_dataset.items()}
        finetuning_replay_buffer = ReplayBuffer.create(example_transition, size=100) # Check ReplayBuffer compatibility
        finetuning_replay_buffer.return_next_actions = True

    for dataset_obj in [pretraining_train_dataset, pretraining_val_dataset,
                        finetuning_train_dataset, finetuning_val_dataset]:
        if dataset_obj is not None:
            dataset_obj.obs_norm_type = args.obs_norm_type
            dataset_obj.p_aug = args.p_aug
            dataset_obj.num_aug = args.num_aug
            dataset_obj.inplace_aug = args.inplace_aug
            dataset_obj.frame_stack = args.frame_stack
            if config['agent_name'] in ['infom', 'rebrac', 'dino_rebrac', 'mbpo_rebrac',
                                        'td_infonce', 'fb_repr_fom', 'hilp_fom']:
                dataset_obj.return_next_actions = True
            dataset_obj.normalize_observations() # Ensure this works with NumPy/Torch as needed

    if config['agent_name'] in ['crl_infonce', 'td_infonce', 'hilp']:
        config['p_aug'] = args.p_aug # Pass along to agent config
        config['frame_stack'] = args.frame_stack
        # GCDataset might need to be PyTorch compatible or return data agent can convert
        pretraining_train_dataset = GCDataset(pretraining_train_dataset, config)
        finetuning_train_dataset = GCDataset(finetuning_train_dataset, config)
        if pretraining_val_dataset is not None:
            pretraining_val_dataset = GCDataset(pretraining_val_dataset, config)
        if finetuning_val_dataset is not None:
            finetuning_val_dataset = GCDataset(finetuning_val_dataset, config)

    # Create agent
    # Example batch for agent creation (obs and action shapes)
    # Ensure example_batch values are NumPy arrays as PyTorch agent's create might expect that for shape inference
    example_batch_np = pretraining_train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    # Agent's create method now expects, config, obs_np, actions_np, seed
    # and returns a PyTorch agent instance.
    agent = agent_class.create(
        args.seed,
        example_batch_np['observations'],
        example_batch_np['actions'],
        config, # Pass the agent specific config directly
        device,
    )
    # agent.to(device) # Agent's internal components should be moved to device in its __init__

    # Restore agent
    if args.restore_path is not None:
        # Construct the full path if restore_epoch is given and restore_path is a directory pattern
        # For now, assume args.restore_path is the exact file path for .pth
        restore_file_path = args.restore_path
        if os.path.isdir(args.restore_path) and args.restore_epoch is not None:
             # This logic might need adjustment based on how files are named by save_agent_components
            restore_file_path = os.path.join(args.restore_path, f'agent_epoch_{args.restore_epoch}.pth')

        if os.path.isfile(restore_file_path):
            print(f"Restoring agent from {restore_file_path}")
            # Define component names that the agent's load_state_dict expects
            # This is a placeholder, actual names depend on agent's state_dict keys
            component_keys = list(agent.state_dict().keys()) # Get keys from current agent state_dict
            loaded_state = load_agent_components(restore_file_path, component_keys, device)
            agent.load_state_dict(loaded_state)
        else:
            print(f"Warning: Restore path {restore_file_path} not found. Training from scratch.")

    outputs = [
        TerminalOutput(),
        JSONLOutput(args.save_dir),
    ]
    if args.tensorboard:
        outputs += [TensorBoardOutputPytorch(args.save_dir)]
    logger = Logger(outputs)

    # Train agent
    first_time = time.time()
    last_time = time.time()

    inferred_latent = None  # Only for HILP and FB.

    for i in tqdm.tqdm(range(1, args.pretraining_steps + args.finetuning_steps + 1), smoothing=0.1, dynamic_ncols=True):
        update_info = {}
        if i <= args.pretraining_steps:
            logger_prefix = 'pretraining_train'
            batch_np = pretraining_train_dataset.sample(config['batch_size'])
            # Agent's pretrain method should handle numpy to torch conversion internally or accept torch tensors
            update_info = agent.update(batch_np, pretrain_mode=True, step=i)
            
        else:
            logger_prefix = 'finetuning_train'
            if i == (args.pretraining_steps + 1):
                if hasattr(agent, 'target_reset'): # Check if agent has target_reset
                    agent.target_reset()

                if config['agent_name'] in ['hilp', 'fb_repr'] and hasattr(agent, 'infer_latent'):
                    num_samples = 0
                    inference_batch_list = defaultdict(list)
                    # Sample NumPy data for inference
                    while num_samples < config['num_latent_inference_samples']:
                        batch_np_infer = finetuning_train_dataset.sample(config['batch_size'])
                        for k, v_np in batch_np_infer.items():
                            inference_batch_list[k].append(v_np)
                        num_samples += config['batch_size']

                    inference_batch_np_concat = {}
                    for k, v_list_np in inference_batch_list.items():
                        if k not in ['observation_min', 'observation_max']: # From original code
                            inference_batch_np_concat[k] = np.concatenate(v_list_np, axis=0)[:config['num_latent_inference_samples']]
                        else:
                            inference_batch_np_concat[k] = v_list_np[0] # Assuming these don't need concat

                    # agent.infer_latent should handle numpy to torch conversion
                    inferred_latent_np = agent.infer_latent(inference_batch_np_concat)
                    inferred_latent = to_torch(inferred_latent_np, device) # Store as torch tensor on device

            # Prepare batch for finetuning
            current_batch_np = {}
            if (config['agent_name'] == 'mbpo_rebrac') and (finetuning_replay_buffer.size > config['batch_size']):
                current_batch_np = finetuning_train_dataset.sample(config['batch_size'])
                replay_batch_np = finetuning_replay_buffer.sample(config['batch_size'])
                for k_rb, v_rb_np in replay_batch_np.items():
                    current_batch_np[f'model_{k_rb}'] = v_rb_np
            else:
                current_batch_np = finetuning_train_dataset.sample(config['batch_size'])

            if config['agent_name'] in ['hilp', 'fb_repr'] and inferred_latent is not None:
                # Batch needs to be torch tensor here for agent.finetune
                # This assumes agent.finetune will convert numpy parts of batch to torch
                # Or, convert current_batch_np to torch_batch here first.
                # For consistency, let's assume agent methods take numpy and convert internally.
                current_batch_np['latents'] = np.tile(to_numpy(inferred_latent), (current_batch_np['observations'].shape[0], 1))

            update_info = agent.update(current_batch_np, pretrain_mode=False, step=i, full_update=(i % config['actor_freq'] == 0))

        # MBPO imaginary rollouts (MBPO agent needs methods like sample_actions, predict_rewards, predict_next_observations)
        if config['agent_name'] in ['mbpo_rebrac'] and (i > args.pretraining_steps) and hasattr(agent, 'predict_rewards'):
            rollout_batch_np = finetuning_train_dataset.sample(config['num_model_rollouts'])
            observations_np = rollout_batch_np['observations']
            for _ in range(config['num_model_rollout_steps']):
                # agent.sample_actions now takes numpy, returns numpy. No seed needed at call site.
                actions_np = agent.sample_actions(observations=observations_np, temperature=1.0, add_noise=True) # add_noise for rollout?
                # predict_rewards and predict_next_observations should also take numpy
                rewards_np = agent.predict_rewards(observations=observations_np, actions=actions_np)
                next_observations_np = agent.predict_next_observations(observations=observations_np, actions=actions_np)

                finetuning_replay_buffer.add_transitions(
                    dict(
                        observations=observations_np,
                        actions=actions_np,
                        rewards=rewards_np,
                        terminals=np.zeros_like(rewards_np), # Assuming numpy here
                        masks=np.ones_like(rewards_np),     # Assuming numpy here
                        next_observations=next_observations_np,
                    )
                )
                observations_np = next_observations_np # Update for next step of rollout

        # Log metrics
        if i % args.log_interval == 0 and update_info: # Ensure update_info is not empty
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            # Validation loss calculation
            # This part needs significant change as agent loss functions are not exposed directly
            # and grad_params is not used.
            # We need a dedicated evaluation method in the agent or skip this if too complex for now.
            # For now, commenting out the direct validation loss computation.
            # if i <= args.pretraining_steps:
            #     val_dataset_current = pretraining_val_dataset
            #     # loss_fn_current = agent.pretraining_loss_eval_mode # Needs new method in agent
            # else:
            #     val_dataset_current = finetuning_val_dataset
            #     # loss_fn_current = agent.finetuning_loss_eval_mode # Needs new method in agent
            #
            # if val_dataset_current is not None and hasattr(agent, 'evaluate_loss'): # Check for new method
            #     val_batch_np = val_dataset_current.sample(config['batch_size'])
            #     if config['agent_name'] in ['hilp', 'fb_repr'] and inferred_latent is not None:
            #         val_batch_np['latents'] = np.tile(to_numpy(inferred_latent), (val_batch_np['observations'].shape[0], 1))
            #
            #     val_info = agent.evaluate_loss(val_batch_np) # Agent method returns dict of val losses
            #     train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()

            logger.add(train_metrics, step=i, prefix=logger_prefix)
            logger.write(step=i, fps=True)

        # Evaluate agent
        if (args.eval_interval != 0 and (i > args.pretraining_steps) and (i == (args.pretraining_steps + 1) or i % args.eval_interval == 0)):
            renders = []
            eval_metrics = {}
            # evaluate function needs to be PyTorch compatible
            # It should take PyTorch agent and handle device internally or accept device
            eval_info, trajs, cur_renders = evaluate(
                agent=agent, # PyTorch agent
                env=eval_env,
                dataset=finetuning_train_dataset, # Pass dataset for normalization info
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=args.video_episodes,
                video_frame_skip=args.video_frame_skip,
                inferred_latent=to_numpy(inferred_latent) if inferred_latent is not None else None, # Pass numpy latent
                should_render=args.video_episodes>0,
                device=device # Pass device if evaluate needs it
            )
            renders.extend(cur_renders)
            for k, v_eval in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v_eval

            if args.video_episodes > 0 and renders: # Ensure renders is not empty
                video = get_render_video(renders=renders)
                eval_metrics['video'] = video

            logger.add(eval_metrics, step=i, prefix="finetuning_eval")
            logger.write(step=i, fps=True)

        # Save agent
        if i % args.save_interval == 0:
            agent_state_to_save = agent.state_dict() # Get state dict from PyTorch agent
            save_agent_components(agent_state_to_save, args.save_dir, i, name_prefix=config['agent_name'])


if __name__ == '__main__':
    main()
