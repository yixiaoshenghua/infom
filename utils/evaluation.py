from collections import defaultdict
import xml.etree.ElementTree as ET # Keep for evaluate_heatmaps if used

import numpy as np
from tqdm import trange
import torch # Added for no_grad

# Removed jax and jax.random

# def supply_rng(f, rng=jax.random.PRNGKey(0)): # JAX specific, removed
#     """Helper function to split the random number generator key before each call to the function."""
#
#     def wrapped(*args, **kwargs):
#         nonlocal rng
#         rng, key = jax.random.split(rng)
#         return f(*args, seed=key, **kwargs)
#
#     return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'): # Check if it's dict-like
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent, # Expected to be a PyTorch agent
    env,
    dataset=None, # For observation normalization
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    inferred_latent=None, # Expected to be NumPy array if provided
    eval_temperature=0.0, # Changed default to float
    # device=None, # Optional: device for agent if not handled internally by agent.sample_actions
):
    """Evaluate the PyTorch agent in the environment."""
    # actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))) # Removed
    # PyTorch agent.sample_actions should handle its own device placement and random state if necessary for exploration noise.
    # For deterministic eval (temp=0, no exploration noise), it should be straightforward.

    trajs = []
    stats = defaultdict(list)
    renders = []

    # Ensure agent is in eval mode if it's a PyTorch nn.Module
    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module): # Common pattern
        agent.actor.eval()
    elif isinstance(agent, torch.nn.Module): # If agent itself is the module
         agent.eval()


    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        if dataset is not None and hasattr(dataset, 'normalize_observations'):
            observation = dataset.normalize_observations(observation) # Assume this returns NumPy

        done = False
        step = 0
        current_episode_render = [] # Changed name from render to avoid conflict

        while not done:
            # agent.sample_actions is expected to take NumPy obs and return NumPy action
            # It should handle torch.no_grad() and device internally.
            action_args = {"observations": observation, "temperature": eval_temperature, "add_noise": False} # No exploration noise for eval
            if inferred_latent is not None:
                action_args["latents"] = inferred_latent # Pass NumPy latent

            action = agent.sample_actions(**action_args)
            action = np.clip(action, -1.0, 1.0) # Clip action from agent

            next_observation, reward, terminated, truncated, info = env.step(action)
            if dataset is not None and hasattr(dataset, 'normalize_observations'):
                next_observation = dataset.normalize_observations(next_observation)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                if hasattr(env, 'render') and callable(env.render):
                    try:
                        frame = env.render() # mode='rgb_array' often implicit or default
                        if isinstance(frame, list) and len(frame) > 0 : # Handle cases like RecordVideo wrapper
                            frame = frame[0]
                        if frame is not None:
                             current_episode_render.append(frame.copy())
                    except Exception as e:
                        print(f"Warning: env.render() failed: {e}")
                        pass # Skip frame if render fails


            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done, # Store combined done signal
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        if i < num_eval_episodes:
            # Make sure info is flattened correctly
            if isinstance(info, dict):
                 add_to(stats, flatten(info))
            elif isinstance(info, tuple) and len(info) > 0 and isinstance(info[0], dict) : # Handle envs that return (obs, info)
                 add_to(stats, flatten(info[0]))

            trajs.append(traj)
        else:
            if current_episode_render: # Only if frames were captured
                renders.append(np.array(current_episode_render))

    # Set agent back to train mode if it was changed
    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module):
        agent.actor.train()
    elif isinstance(agent, torch.nn.Module):
        agent.train()

    # Calculate mean of stats
    final_stats = {}
    for k, v_list in stats.items():
        if v_list: # Ensure list is not empty
            try:
                final_stats[k] = np.mean([item for item in v_list if isinstance(item, (int, float, np.number))])
            except Exception as e:
                print(f"Warning: Could not compute mean for stat {k}: {e}")
                final_stats[k] = "Error" # Or some other placeholder
        else:
            final_stats[k] = 0 # Or np.nan or skip

    return final_stats, trajs, renders


def evaluate_gc( # This function will need similar PyTorch conversion if used
    agent,
    env,
    task_id=None,
    config=None, # Assumed to be ml_collections.ConfigDict for agent/eval params
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0.0,
    eval_gaussian=None, # Noise std for actions
):
    """Evaluate the goal-conditioned PyTorch agent."""
    trajs = []
    stats = defaultdict(list)
    renders = []

    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module):
        agent.actor.eval()
    elif isinstance(agent, torch.nn.Module):
        agent.eval()

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        # Assuming observation is NumPy, goal might be part of obs or info
        goal = info.get('goal') # Assuming goal is NumPy
        if goal is None and 'achieved_goal' in observation and 'desired_goal' in observation: # HER style
            goal = observation['desired_goal']
            observation = observation['observation']


        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        current_episode_render = []
        while not done:
            action_args = {"observations": observation, "goals": goal, "temperature": eval_temperature, "add_noise": False}
            action = agent.sample_actions(**action_args) # Expected to take NumPy, return NumPy

            if not config.get('discrete_action', False): # Check if action space is discrete from config
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1.0, 1.0)

            next_observation, reward, terminated, truncated, info = env.step(action)
            if isinstance(next_observation, dict) and 'observation' in next_observation: # HER style
                 next_observation_processed = next_observation['observation']
            else:
                 next_observation_processed = next_observation

            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                if hasattr(env, 'render') and callable(env.render):
                    try:
                        frame = env.render()
                        if isinstance(frame, list) and len(frame) > 0 : frame = frame[0]
                        if frame is not None:
                            if goal_frame is not None:
                                current_episode_render.append(np.concatenate([goal_frame, frame], axis=0))
                            else:
                                current_episode_render.append(frame.copy())
                    except Exception as e:
                        print(f"Warning: env.render() failed in evaluate_gc: {e}")
                        pass


            transition = dict(
                observation=observation, # Store original observation
                next_observation=next_observation_processed,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation_processed # Update observation for next step

        if i < num_eval_episodes:
            if isinstance(info, dict): add_to(stats, flatten(info))
            elif isinstance(info, tuple) and len(info) > 0 and isinstance(info[0], dict) : add_to(stats, flatten(info[0]))
            trajs.append(traj)
        else:
            if current_episode_render: renders.append(np.array(current_episode_render))

    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module):
        agent.actor.train()
    elif isinstance(agent, torch.nn.Module):
        agent.train()

    final_stats = {}
    for k, v_list in stats.items():
        if v_list:
            try: final_stats[k] = np.mean([item for item in v_list if isinstance(item, (int, float, np.number))])
            except Exception as e: print(f"Warning: Could not compute mean for stat {k} in GC: {e}"); final_stats[k] = "Error"
        else: final_stats[k] = 0
    return final_stats, trajs, renders


def evaluate_octo( # PyTorch conversion needed if used
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0.0,
):
    """Evaluate the Octo-style PyTorch agent."""
    trajs = []
    stats = defaultdict(list)
    renders = []

    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module): agent.actor.eval()
    elif isinstance(agent, torch.nn.Module): agent.eval()

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        observation, info = env.reset() # Obs likely a dict for Octo
        done = False
        step = 0
        current_episode_render = []
        while not done:
            # Octo sample_actions might take full observation dict and task
            action_args = {"observations": observation, "tasks": env.task, "temperature": eval_temperature, "add_noise": False}
            action = agent.sample_actions(**action_args) # Expected to take NumPy/dict, return NumPy

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                # Assuming primary image is for rendering in Octo
                if 'image_primary' in observation and observation['image_primary'] is not None:
                    frame_data = observation['image_primary']
                    # Handle potential batch and channel dimensions if they exist (e.g. B, T, H, W, C or H, W, C)
                    if isinstance(frame_data, torch.Tensor): frame_data = frame_data.cpu().numpy()
                    while frame_data.ndim > 3 and frame_data.shape[0] == 1: frame_data = frame_data[0] # Remove batch
                    while frame_data.ndim > 3 and frame_data.shape[0] <= 4: frame_data = frame_data[-1] # Take last time step if T dim

                    if frame_data.ndim == 3: current_episode_render.append(frame_data.copy())
                    else: print(f"Warning: Frame data has unexpected shape {frame_data.shape} for rendering.")
                else:
                    print("Warning: 'image_primary' not found in observation for rendering.")


            transition = dict(observation=observation, next_observation=next_observation, action=action, reward=reward, done=done, info=info)
            add_to(traj, transition)
            observation = next_observation

        if i < num_eval_episodes:
            if isinstance(info, dict): add_to(stats, flatten(info))
            elif isinstance(info, tuple) and len(info) > 0 and isinstance(info[0], dict) : add_to(stats, flatten(info[0]))
            trajs.append(traj)
        else:
            if current_episode_render: renders.append(np.array(current_episode_render))

    if hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module): agent.actor.train()
    elif isinstance(agent, torch.nn.Module): agent.train()

    final_stats = {}
    for k, v_list in stats.items():
        if v_list:
            try: final_stats[k] = np.mean([item for item in v_list if isinstance(item, (int, float, np.number))])
            except Exception as e: print(f"Warning: Could not compute mean for stat {k} in Octo: {e}"); final_stats[k] = "Error"
        else: final_stats[k] = 0
    return final_stats, trajs, renders


def evaluate_policy_evaluation( # PyTorch conversion needed if used; JAX code was incomplete
    estimator, # This would be a PyTorch module/class
    dataset,   # Assumed to provide NumPy data
    num_eval_transitions=10_000,
):
    stats = defaultdict(list)
    batch_np = dataset.sample(num_eval_transitions) # Sample NumPy data

    # Estimator would be a PyTorch based one.
    # It needs an 'evaluate_estimation' method that takes NumPy batch
    # and handles torch conversion / device placement internally.
    if hasattr(estimator, 'evaluate_estimation'):
        # Seed handling for PyTorch needs to be via torch.manual_seed if estimator uses random ops
        # The original JAX seed passing won't work directly.
        # For now, assume estimator handles its own randomness or is deterministic for eval.
        current_stats = estimator.evaluate_estimation(batch_np) # No seed passed
        if isinstance(current_stats, dict):
            for k, v in current_stats.items():
                stats[k] = np.asarray(v) # Ensure stored as array
    else:
        print("Warning: Estimator does not have 'evaluate_estimation' method.")
        return {"error": "Estimator missing evaluate_estimation"}


    # Original JAX code for this function was mostly commented out.
    # The key is that 'estimator.evaluate_estimation' needs to be implemented
    # in a PyTorch-compatible way.

    final_stats = {}
    for k, v_val in stats.items(): # v_val is already an array here
        # The original code did np.asarray(v), but then np.mean(v) in other eval functions.
        # If evaluate_estimation returns mean values directly, this is fine.
        # If it returns lists of values, then mean should be taken.
        # For now, assume v_val is the final stat value (scalar or array).
        final_stats[k] = v_val
    return final_stats


def evaluate_heatmaps( # PyTorch conversion needed if used
        estimator, # PyTorch version of the estimator
        dataset,
        env,
        num_heatmaps,
        num_grid_x=50,
        num_grid_y=50,
):
    # XML parsing and env_info setup can remain the same
    unwrapped_env = env.unwrapped
    # This might fail if fullpath is not available or env is not a file-based MuJoCo env
    tree_path = getattr(unwrapped_env, 'fullpath', None)
    if tree_path is None and hasattr(unwrapped_env, 'model') and hasattr(unwrapped_env.model, 'opt') and hasattr(unwrapped_env.model.opt, 'vis') and hasattr(unwrapped_env.model.opt.vis, 'map') and hasattr(unwrapped_env.model.opt.vis.map, 'file'):
        # Try to get path from dm_control model if possible (very specific)
        tree_path = unwrapped_env.model.opt.vis.map.file
        if not tree_path and hasattr(unwrapped_env, 'name'): # Fallback for dmc
            # This is a heuristic and might not work for all dmc envs
            try:
                from dm_control import suite
                domain_name, task_name = unwrapped_env.name.split('_',1)
                xml_path = suite.get_model_xml_path(domain_name, task_name)
                if xml_path : tree_path = xml_path
            except Exception:
                 pass # Could not get xml path

    if not tree_path or not os.path.exists(tree_path):
        print(f"Warning: Could not find or parse XML for heatmap generation (path: {tree_path}). Skipping heatmap.")
        return {}, {}

    tree = ET.parse(tree_path)
    worldbody = tree.find('.//worldbody')
    if worldbody is None: # Try another common tag for bodies
        worldbody = tree.find('.//worldbody') # often this is it, but sometimes assets are elsewhere

    wall_positions = []
    wall_sizes = []

    if worldbody is not None:
        for element in worldbody.findall(".//geom[@material='wall']"): # material might be different
            pos_str = element.get('pos')
            size_str = element.get('size')
            if pos_str and size_str:
                try:
                    pos = [float(s) for s in pos_str.split(' ')][:2]
                    size = [float(s) for s in size_str.split(' ')][:2]
                    wall_positions.append(pos)
                    wall_sizes.append(size)
                except ValueError:
                    print(f"Warning: Could not parse pos/size for wall geom: {pos_str}, {size_str}")

    if not wall_positions: # Fallback if no walls found or parsing failed
        print("Warning: No walls found or failed to parse wall positions/sizes. Using default world ranges for heatmap.")
        world_ranges = np.array([[-1.0, 1.0], [-1.0, 1.0]]) # Default range
    else:
        wall_positions = np.array(wall_positions)
        wall_sizes = np.array(wall_sizes)
        # Calculate world ranges based on wall positions and sizes for safety
        min_coords = np.min(wall_positions - wall_sizes, axis=0)
        max_coords = np.max(wall_positions + wall_sizes, axis=0)
        world_ranges = np.array([min_coords, max_coords]).T


    env_info = {'wall_positions': wall_positions, 'wall_sizes': wall_sizes, 'world_ranges': world_ranges}

    grid_x = np.linspace(world_ranges[0,0], world_ranges[0,1], num_grid_x)
    grid_y = np.linspace(world_ranges[1,0], world_ranges[1,1], num_grid_y)
    mesh_x, mesh_y = np.array(np.meshgrid(grid_x, grid_y))
    mesh_grid_xys = np.stack([mesh_x, mesh_y], axis=-1)

    batch_np = dataset.sample(num_heatmaps) # NumPy batch

    # Estimator's compute_values method should take NumPy arrays and handle torch conversion
    # observations_np = batch_np['observations'][:, None, :].repeat(num_grid_x * num_grid_y, axis=1).reshape([-1, env.observation_space.shape[-1]])
    # Ensure observation space shape is correct, might be nested for dict obs space
    obs_shape = env.observation_space.shape if hasattr(env.observation_space, 'shape') else env.observation_space['observation'].shape # Basic handling

    observations_np_expanded = np.repeat(batch_np['observations'][:, np.newaxis, :], num_grid_x * num_grid_y, axis=1)
    observations_np_flat = observations_np_expanded.reshape([-1, obs_shape[-1]])

    goals_np_expanded = np.repeat(mesh_grid_xys[np.newaxis, :, :, :], num_heatmaps, axis=0)
    goals_np_flat = goals_np_expanded.reshape([-1, obs_shape[-1]]) # Assuming goal dim matches obs dim

    # Estimator.compute_values should handle device and torch.no_grad internally
    # It should accept numpy arrays for observations and goals
    if hasattr(estimator, 'compute_values'):
        values_flat = estimator.compute_values(observations_np_flat, goals_np_flat) # No seed needed for PyTorch typically
        values = values_flat.reshape(num_heatmaps, num_grid_x, num_grid_y) # Reshape to (num_heatmaps, grid_x, grid_y)
    else:
        print("Warning: Estimator does not have 'compute_values' method.")
        values = np.zeros((num_heatmaps, num_grid_x, num_grid_y))


    value_info = {
        'observations': batch_np['observations'], # Original sampled observations
        'mesh_grid_xys': mesh_grid_xys, # The grid itself
        'mesh_x': mesh_x, # X coordinates of the grid
        'mesh_y': mesh_y, # Y coordinates of the grid
        'values': values  # Computed values on the grid
    }

    return value_info, env_info
