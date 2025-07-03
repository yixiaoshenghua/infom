import os
import json
import random
import time

import torch # Added
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents # Should now import PyTorch agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import GCDataset, Dataset, ReplayBuffer # May need PyTorch compatibility adjustments
from utils.evaluation import evaluate # Needs to be PyTorch compatible
# Changed from flax_utils to torch_utils for save/load
from utils.flax_utils import load_agent_components, save_agent_components
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.torch_utils import set_device, to_torch, to_numpy # Added

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'debug', 'Run group.')
flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path (exact file path to .pth).') # Clarified
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch (used if restore_path is a dir pattern, less common now).') # Kept for now

flags.DEFINE_integer('pretraining_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('pretraining_size', 1_000_000, 'Size of the dataset for pre-training.')
flags.DEFINE_integer('finetuning_steps', 500_000, 'Number of online steps.')
flags.DEFINE_integer('finetuning_size', 500_000, 'Size of the dataset for fine-tuning.')
flags.DEFINE_integer('log_interval', 5_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1_500_000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_string('obs_norm_type', 'normal',
                    'Type of observation normalization. (none, normal, bounded)')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('num_aug', 1, 'Number of image augmentations.')
flags.DEFINE_integer('inplace_aug', 1, 'Whether to replace the original image after applying augmentations.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/infom.py', lock_config=False) # Agent config file


def main(_):
    # Set up logger and device
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    device = set_device()
    print(f"Using device: {device}")

    if FLAGS.enable_wandb:
        _, trigger_sync = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='infom_pytorch', # Consider changing project name
            group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Make environment and datasets
    config = FLAGS.agent # Agent specific config
    # make_env_and_datasets might need adjustment if it returns JAX arrays
    # Assuming it returns NumPy arrays which are then handled by Dataset class
    _, _, pretraining_train_dataset_np, pretraining_val_dataset_np = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.pretraining_size, reward_free=True)
    _, eval_env, finetuning_train_dataset_np, finetuning_val_dataset_np = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, max_size=FLAGS.finetuning_size, reward_free=False)

    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'


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
            dataset_obj.obs_norm_type = FLAGS.obs_norm_type
            dataset_obj.p_aug = FLAGS.p_aug
            dataset_obj.num_aug = FLAGS.num_aug
            dataset_obj.inplace_aug = FLAGS.inplace_aug
            dataset_obj.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['infom', 'rebrac', 'dino_rebrac', 'mbpo_rebrac',
                                        'td_infonce', 'fb_repr_fom', 'hilp_fom']:
                dataset_obj.return_next_actions = True
            dataset_obj.normalize_observations() # Ensure this works with NumPy/Torch as needed

    if config['agent_name'] in ['crl_infonce', 'td_infonce', 'hilp']:
        config['p_aug'] = FLAGS.p_aug # Pass along to agent config
        config['frame_stack'] = FLAGS.frame_stack
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
    # Agent's create method now expects seed, obs_np, actions_np, config
    # and returns a PyTorch agent instance.
    agent = agent_class.create(
        FLAGS.seed,
        example_batch_np['observations'],
        example_batch_np['actions'],
        config, # Pass the agent specific config directly
    )
    # agent.to(device) # Agent's internal components should be moved to device in its __init__

    # Restore agent
    if FLAGS.restore_path is not None:
        # Construct the full path if restore_epoch is given and restore_path is a directory pattern
        # For now, assume FLAGS.restore_path is the exact file path for .pth
        restore_file_path = FLAGS.restore_path
        if os.path.isdir(FLAGS.restore_path) and FLAGS.restore_epoch is not None:
             # This logic might need adjustment based on how files are named by save_agent_components
            restore_file_path = os.path.join(FLAGS.restore_path, f'agent_epoch_{FLAGS.restore_epoch}.pth')

        if os.path.isfile(restore_file_path):
            print(f"Restoring agent from {restore_file_path}")
            # Define component names that the agent's load_state_dict expects
            # This is a placeholder, actual names depend on agent's state_dict keys
            component_keys = list(agent.state_dict().keys()) # Get keys from current agent state_dict
            loaded_state = load_agent_components(restore_file_path, component_keys, device)
            agent.load_state_dict(loaded_state)
        else:
            print(f"Warning: Restore path {restore_file_path} not found. Training from scratch.")


    # Train agent
    pretraining_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_train.csv'))
    pretraining_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'pretraining_eval.csv'))
    finetuning_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_train.csv'))
    finetuning_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'finetuning_eval.csv'))

    first_time = time.time()
    last_time = time.time()

    inferred_latent = None  # Only for HILP and FB.
    # rng = jax.random.PRNGKey(FLAGS.seed)  # Removed, PyTorch uses global random state

    for i in tqdm.tqdm(range(1, FLAGS.pretraining_steps + FLAGS.finetuning_steps + 1), smoothing=0.1, dynamic_ncols=True):
        update_info = {}
        if i <= FLAGS.pretraining_steps:
            batch_np = pretraining_train_dataset.sample(config['batch_size'])
            # Agent's pretrain method should handle numpy to torch conversion internally or accept torch tensors
            update_info = agent.pretrain(batch_np)
            train_logger = pretraining_train_logger
            # eval_logger = pretraining_eval_logger # Eval logger not used in pretrain loop body
        else:
            if i == (FLAGS.pretraining_steps + 1):
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

            train_logger = finetuning_train_logger
            # eval_logger = finetuning_eval_logger # Eval logger not used in finetune loop body

            if config['agent_name'] in ['hilp', 'fb_repr'] and inferred_latent is not None:
                # Batch needs to be torch tensor here for agent.finetune
                # This assumes agent.finetune will convert numpy parts of batch to torch
                # Or, convert current_batch_np to torch_batch here first.
                # For consistency, let's assume agent methods take numpy and convert internally.
                current_batch_np['latents'] = np.tile(to_numpy(inferred_latent), (current_batch_np['observations'].shape[0], 1))

            update_info = agent.finetune(current_batch_np, full_update=(i % config['actor_freq'] == 0))

        # MBPO imaginary rollouts (MBPO agent needs methods like sample_actions, predict_rewards, predict_next_observations)
        if config['agent_name'] in ['mbpo_rebrac'] and (i > FLAGS.pretraining_steps) and hasattr(agent, 'predict_rewards'):
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
        if i % FLAGS.log_interval == 0 and update_info: # Ensure update_info is not empty
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            # Validation loss calculation
            # This part needs significant change as agent loss functions are not exposed directly
            # and grad_params is not used.
            # We need a dedicated evaluation method in the agent or skip this if too complex for now.
            # For now, commenting out the direct validation loss computation.
            # if i <= FLAGS.pretraining_steps:
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

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)
                if FLAGS.wandb_mode == 'offline' and trigger_sync:
                    trigger_sync()
            train_logger.log(train_metrics, step=i)

        # Evaluate agent
        if (FLAGS.eval_interval != 0 and (i > FLAGS.pretraining_steps)
            and (i == (FLAGS.pretraining_steps + 1) or i % FLAGS.eval_interval == 0)):
            renders = []
            eval_metrics = {}
            # evaluate function needs to be PyTorch compatible
            # It should take PyTorch agent and handle device internally or accept device
            eval_info, trajs, cur_renders = evaluate(
                agent=agent, # PyTorch agent
                env=eval_env,
                dataset=finetuning_train_dataset, # Pass dataset for normalization info
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                inferred_latent=to_numpy(inferred_latent) if inferred_latent is not None else None, # Pass numpy latent
                # device=device # Pass device if evaluate needs it
            )
            renders.extend(cur_renders)
            for k, v_eval in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v_eval

            if FLAGS.video_episodes > 0 and renders: # Ensure renders is not empty
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
                if FLAGS.wandb_mode == 'offline' and trigger_sync:
                    trigger_sync()
            eval_logger.log(eval_metrics, step=i)

        # Save agent
        if i % FLAGS.save_interval == 0:
            agent_state_to_save = agent.state_dict() # Get state dict from PyTorch agent
            save_agent_components(agent_state_to_save, FLAGS.save_dir, i, name_prefix=config['agent_name'])

    pretraining_train_logger.close()
    pretraining_eval_logger.close()
    finetuning_train_logger.close()
    finetuning_eval_logger.close()


if __name__ == '__main__':
    app.run(main)
