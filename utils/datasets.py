import dataclasses
from typing import Any, Dict, List, Union, Callable, Optional # Added Dict, List, Union, Callable, Optional
import numpy as np
# Removed jax, jax.numpy, flax.core.frozen_dict


def _tree_map(func: Callable, tree: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """A simplified tree_map that works for nested dicts/lists of arrays."""
    if isinstance(tree, dict):
        return {k: _tree_map(func, v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map(func, v) for v in tree]
    # Check if it's a NumPy array before applying func, otherwise return as is
    elif isinstance(tree, np.ndarray):
        return func(tree)
    else: # Not a dict, list, or ndarray, return as is
        return tree

def _tree_leaves(tree: Union[Dict, List, Any]) -> List[Any]:
    """A simplified tree_leaves that works for nested dicts/lists."""
    leaves = []
    if isinstance(tree, dict):
        for _, v in tree.items(): # Iterate over values only for leaves
            leaves.extend(_tree_leaves(v))
    elif isinstance(tree, list):
        for v in tree:
            leaves.extend(_tree_leaves(v))
    else: # Assumed to be a leaf node (e.g. np.ndarray or other type)
        leaves.append(tree)
    return leaves


def get_size(data: Dict[str, np.ndarray]) -> int:
    """Return the size of the dataset (max length of arrays in the dict)."""
    if not data:
        return 0
    # Filter for ndarray leaves before calling len
    sizes = [len(arr) for arr in _tree_leaves(data) if isinstance(arr, np.ndarray) and hasattr(arr, '__len__')]
    return max(sizes) if sizes else 0


def random_crop_numpy(img: np.ndarray, crop_from_yx: np.ndarray, final_h: int, final_w: int, padding: int) -> np.ndarray:
    """Randomly crop an image using NumPy.
    Args:
        img: Image to crop (H, W, C) or (B, H, W, C).
        crop_from_yx: Coordinates to crop from (y_start, x_start). For batched, (B, 2).
        final_h, final_w: Final height and width of the cropped image.
        padding: Padding size.
    """
    is_batched = img.ndim == 4
    if not is_batched: # Add batch dim if single image
        img = img[np.newaxis, ...]
        crop_from_yx = crop_from_yx[np.newaxis, ...]

    B, H, W, C = img.shape
    padded_img = np.pad(img, ((0,0), (padding, padding), (padding, padding), (0, 0)), mode='edge')

    cropped_imgs = np.empty((B, final_h, final_w, C), dtype=img.dtype)
    for i in range(B):
        y_start, x_start = crop_from_yx[i, 0], crop_from_yx[i, 1]
        cropped_imgs[i] = padded_img[i, y_start : y_start + final_h, x_start : x_start + final_w, :]

    return cropped_imgs if is_batched else cropped_imgs[0]


def batched_random_crop_numpy(imgs: np.ndarray, crop_from_yx_batch: np.ndarray, final_h: int, final_w: int, padding: int) -> np.ndarray:
    """Batched version of random_crop_numpy.
    Args:
        imgs: Images to crop (B, H, W, C).
        crop_from_yx_batch: Coordinates to crop from (B, 2) for (y,x) offsets.
        final_h, final_w: Final height and width.
        padding: Padding size.
    """
    return random_crop_numpy(imgs, crop_from_yx_batch, final_h, final_w, padding)


def augment(batch: Dict[str, Any], keys: List[str], new_key_prefix: str = ''):
    """Apply image augmentation (random crop) to the given keys in a batch (dict of NumPy arrays)."""
    if not keys or not batch: return

    first_image_arr = None
    for key_to_check in keys:
        item = batch.get(key_to_check)
        # Check if item is a dict (nested observations) or a direct ndarray
        if isinstance(item, dict): # Nested case, e.g. observations: {'pixels': ndarray}
            for sub_item_key in item:
                sub_item = item[sub_item_key]
                if isinstance(sub_item, np.ndarray) and sub_item.ndim == 4: # B, H, W, C
                    first_image_arr = sub_item
                    break
        elif isinstance(item, np.ndarray) and item.ndim == 4: # Direct ndarray B, H, W, C
            first_image_arr = item
        if first_image_arr is not None: break

    if first_image_arr is None: return # No suitable image array found

    batch_size, H, W, _ = first_image_arr.shape
    padding = 4 # Original padding in JAX code was 3, but often image aug uses padding=4 for 84x84 -> 84x84 random crop. Let's use 4.
                # The old code used padding=3, and crop_froms were (0, 2*padding+1).
                # If input H,W is e.g. 100x100, padding=4. Padded size 108x108.
                # To get 100x100 output, crop_from range is [0, 8]. So max_offset is 2*padding.

    # Generate crop coordinates for the entire batch once
    # crop_froms should be (batch_size, 2) for (y,x) offsets
    # Max offset for padded image to produce original size crop is 2 * padding.
    crop_offsets_yx = np.random.randint(0, 2 * padding + 1, (batch_size, 2))

    for key in keys:
        if key in batch:
            current_item = batch[key]

            def apply_aug_if_image(arr_leaf: np.ndarray) -> np.ndarray:
                if isinstance(arr_leaf, np.ndarray) and arr_leaf.ndim == 4: # B, H, W, C
                    # Assuming the crop should result in an image of the same H, W as original
                    return batched_random_crop_numpy(arr_leaf, crop_offsets_yx, arr_leaf.shape[1], arr_leaf.shape[2], padding)
                return arr_leaf

            batch[new_key_prefix + key] = _tree_map(apply_aug_if_image, current_item)


class Dataset(dict):
    @classmethod
    def create(cls, freeze: bool = True, **fields: np.ndarray) -> 'Dataset':
        data = fields
        assert 'observations' in data, "'observations' field is required."
        if freeze:
            def set_read_only(arr):
                if isinstance(arr, np.ndarray): arr.flags.writeable = False
                return arr
            data = _tree_map(set_read_only, data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self)
        self.obs_norm_type = 'none'
        self.frame_stack = None
        self.p_aug = None
        self.num_aug = 1
        self.inplace_aug = False
        self.return_next_actions = False
        self._prestacked = False
        self.obs_mean, self.obs_var, self.obs_max, self.obs_min = None, None, None, None
        self.normalized_obs_max, self.normalized_obs_min = None, None
        self.epsilon = 1e-8

        if self.size > 0 and 'terminals' in self and isinstance(self['terminals'], np.ndarray):
            terminals_data = self['terminals']
            # Ensure terminals_data is 1D for np.nonzero
            if terminals_data.ndim > 1 and terminals_data.shape[0] == self.size :
                if terminals_data.shape[1] == 1: terminals_data = terminals_data.squeeze(1)
                else: raise ValueError("Terminals should be 1D or (N,1) array")

            self.terminal_locs = np.nonzero(terminals_data > 0)[0]

            temp_initial_locs = [0]
            if len(self.terminal_locs) > 0:
                temp_initial_locs.extend(self.terminal_locs[:-1] + 1)
            self.initial_locs = np.unique(np.array(temp_initial_locs, dtype=np.int64)) # Ensure sorted and unique
            self.initial_locs = self.initial_locs[self.initial_locs < self.size]
            if len(self.initial_locs) == 0 and self.size > 0: self.initial_locs = np.array([0])

            # Ensure terminal_locs includes the very last index if dataset doesn't end on a terminal
            if len(self.terminal_locs) == 0 or self.terminal_locs[-1] != self.size - 1:
                self.terminal_locs = np.unique(np.append(self.terminal_locs, self.size - 1))
        else:
            self.terminal_locs = np.array([self.size - 1], dtype=np.int64) if self.size > 0 else np.array([], dtype=np.int64)
            self.initial_locs = np.array([0], dtype=np.int64) if self.size > 0 else np.array([], dtype=np.int64)

    @staticmethod
    def normalize(observations: np.ndarray, obs_mean: np.ndarray, obs_var: np.ndarray,
                  obs_max: np.ndarray, obs_min: np.ndarray,
                  normalizer_type: str = 'none', epsilon: float = 1e-8) -> np.ndarray:
        if normalizer_type == 'normal':
            return (observations - obs_mean) / (np.sqrt(obs_var) + epsilon) # Use stddev directly
        elif normalizer_type == 'bounded':
            range_val = obs_max - obs_min
            return 2 * (observations - obs_min) / (range_val + epsilon) - 1.0
        elif normalizer_type == 'none':
            return observations
        else:
            raise TypeError(f"Unsupported normalizer type: {normalizer_type}")

    def normalize_observations(self, observations: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self.size == 0: return observations

        # Determine which observations to use for calculating stats (tree map for nested dicts)
        obs_for_stats = self.get('observations')
        if not isinstance(obs_for_stats, np.ndarray) and isinstance(obs_for_stats, dict):
            # Heuristic: find the first ndarray if obs is a dict (e.g. pixels, state)
            for k_obs in obs_for_stats:
                if isinstance(obs_for_stats[k_obs], np.ndarray):
                    obs_for_stats = obs_for_stats[k_obs]
                    break
            if not isinstance(obs_for_stats, np.ndarray): # Still not an array
                 print("Warning: Could not determine main observation array for statistics.")
                 return observations


        if observations is None: # Calculate stats from self and normalize self
            if not isinstance(obs_for_stats, np.ndarray):
                 raise ValueError("Dataset 'observations' not found or not a NumPy array for stat calculation.")

            self.obs_mean = np.mean(obs_for_stats, axis=0, dtype=np.float32)
            self.obs_var = np.var(obs_for_stats, axis=0, dtype=np.float32)
            self.obs_max = np.max(obs_for_stats, axis=0)
            self.obs_min = np.min(obs_for_stats, axis=0)

            self.normalized_obs_max = self.normalize(self.obs_max, self.obs_mean, self.obs_var, self.obs_max, self.obs_min, self.obs_norm_type, self.epsilon)
            self.normalized_obs_min = self.normalize(self.obs_min, self.obs_mean, self.obs_var, self.obs_max, self.obs_min, self.obs_norm_type, self.epsilon)

            def _normalize_leaf(leaf_arr):
                if isinstance(leaf_arr, np.ndarray) and np.issubdtype(leaf_arr.dtype, np.floating): # Only normalize float arrays
                    return self.normalize(leaf_arr, self.obs_mean, self.obs_var, self.obs_max, self.obs_min, self.obs_norm_type, self.epsilon)
                return leaf_arr

            self['observations'] = _tree_map(_normalize_leaf, self['observations'])
            if 'next_observations' in self:
                self['next_observations'] = _tree_map(_normalize_leaf, self['next_observations'])
            return self['observations']
        else: # Normalize provided observations using stored stats
            if self.obs_mean is None: self.normalize_observations() # Compute stats if not done
            # If observations is a dict, apply normalize to each appropriate leaf
            return _tree_map(lambda arr: self.normalize(arr, self.obs_mean, self.obs_var, self.obs_max, self.obs_min, self.obs_norm_type, self.epsilon) if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating) else arr, observations)


    def get_random_idxs(self, num_idxs: int) -> np.ndarray:
        if self.size == 0: return np.array([], dtype=int)
        return np.random.randint(0, self.size, size=num_idxs)

    def _prestack_frames(self):
        if self.size == 0 or self.frame_stack is None or self.frame_stack <= 1:
            self._prestacked = True; return

        if len(self.initial_locs) == 0:
            print("Warning: _prestack_frames called with no initial_locs. Frame stacking might be incorrect.")
            initial_state_idxs_base = np.zeros(self.size, dtype=int)
        else:
            initial_state_idxs_base = self.initial_locs[np.searchsorted(self.initial_locs, np.arange(self.size), side='right') - 1]

        def stack_field(field_data):
            if not isinstance(field_data, np.ndarray): return field_data # Skip non-array fields

            frames_list = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(np.arange(self.size) - i, initial_state_idxs_base)
                cur_idxs = np.clip(cur_idxs, 0, self.size - 1)
                frames_list.append(field_data[cur_idxs])
            return np.concatenate(frames_list, axis=-1) # Concatenate along the last (feature) axis

        self['observations'] = _tree_map(stack_field, self['observations'])
        # For next_observations, stack T-k+1..T from current obs, and then next_obs[T]
        def stack_next_obs_field(field_data_obs, field_data_next_obs):
            if not isinstance(field_data_obs, np.ndarray): return field_data_next_obs

            frames_list_next = []
            for i in reversed(range(1, self.frame_stack)): # obs[t-frame_stack+2]...obs[t]
                cur_idxs = np.maximum(np.arange(self.size) - (i-1), initial_state_idxs_base) # i-1 because we want one step forward
                cur_idxs = np.clip(cur_idxs, 0, self.size -1)
                frames_list_next.append(field_data_obs[cur_idxs])
            frames_list_next.append(field_data_next_obs) # actual next_obs[t]
            return np.concatenate(frames_list_next, axis=-1)

        if isinstance(self['observations'], dict) and isinstance(self['next_observations'], dict):
            stacked_next_obs_dict = {}
            for k in self['observations']: # Assuming same keys
                 if k in self['next_observations']:
                     stacked_next_obs_dict[k] = stack_next_obs_field(self['observations'][k], self['next_observations'][k])
                 else: # Key in obs but not next_obs (e.g. 'state' vs 'pixels')
                     stacked_next_obs_dict[k] = _tree_map(stack_field, self['next_observations'])[k] # Stack next_obs independently
            self['next_observations'] = stacked_next_obs_dict
        elif isinstance(self['observations'], np.ndarray) and isinstance(self['next_observations'], np.ndarray):
            self['next_observations'] = stack_next_obs_field(self['observations'], self['next_observations'])
        else:
            print("Warning: Mismatch in observation/next_observation structure for frame stacking.")

        self._prestacked = True

    def sample(self, batch_size: int, idxs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.size == 0: return {}
        if (self.frame_stack is not None and self.frame_stack > 1) and (not self._prestacked):
            self._prestack_frames()
        if idxs is None: idxs = self.get_random_idxs(batch_size)
        if len(idxs) == 0: return {}

        batch = self.get_subset(idxs)
        if not batch : return {} # if get_subset returns empty

        if self.normalized_obs_min is not None: batch['observation_min'] = self.normalized_obs_min
        if self.normalized_obs_max is not None: batch['observation_max'] = self.normalized_obs_max

        if self.p_aug is not None and self.p_aug > 0:
            if np.random.rand() < self.p_aug:
                aug_keys = [k for k in ['observations', 'next_observations', 'value_goals', 'actor_goals'] if k in batch]
                if aug_keys:
                    if self.inplace_aug: augment(batch, aug_keys)
                    else:
                        for i_aug in range(self.num_aug): augment(batch, aug_keys, f'aug{i_aug + 1}_')
        return batch

    def get_subset(self, idxs: np.ndarray) -> Dict[str, Any]:
        if len(idxs) == 0: return {k: np.array([]) for k in self.keys()}
        # Ensure idxs are within bounds
        idxs = np.clip(idxs, 0, self.size - 1)
        result = _tree_map(lambda arr: arr[idxs] if isinstance(arr, np.ndarray) and len(arr)>0 else arr, self)
        if self.return_next_actions and 'actions' in self and self.size > 0:
            next_indices = np.minimum(idxs + 1, self.size - 1)
            result['next_actions'] = self['actions'][next_indices]
        return result


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition: Dict[str, Any], size: int) -> 'ReplayBuffer':
        def create_buffer_arr(example_arr: Any) -> np.ndarray:
            example_arr_np = np.array(example_arr)
            return np.zeros((size, *example_arr_np.shape), dtype=example_arr_np.dtype)
        buffer_dict = _tree_map(create_buffer_arr, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: Dataset, size: int) -> 'ReplayBuffer':
        def create_buffer_from_existing(init_arr: np.ndarray) -> np.ndarray:
            new_buffer = np.zeros((size, *init_arr.shape[1:]), dtype=init_arr.dtype)
            len_init = len(init_arr)
            if len_init > 0: new_buffer[:min(len_init, size)] = init_arr[:min(len_init, size)]
            return new_buffer
        buffer_dict = _tree_map(create_buffer_from_existing, init_dataset)
        replay_buffer_instance = cls(buffer_dict)
        initial_content_size = get_size(init_dataset)
        replay_buffer_instance.size = min(initial_content_size, size)
        replay_buffer_instance.pointer = min(initial_content_size, size) % size if size > 0 else 0
        replay_buffer_instance._update_terminal_initial_locs() # Update based on initial content
        return replay_buffer_instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Calls Dataset.__init__
        self.max_size = get_size(self)
        if not hasattr(self, '_replay_initialized'):
            self.size = 0
            self.pointer = 0
            self._replay_initialized = True
            self._update_terminal_initial_locs() # Ensure locs are initialized for empty buffer

    def add_transition(self, transition: Dict[str, Any]):
        if self.max_size == 0: return # Cannot add to zero-capacity buffer
        _tree_map(lambda buf, elem: self._set_idx_val(buf, self.pointer, elem), self, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)
        self._update_terminal_initial_locs()

    def _set_idx_val(self, buffer_array: np.ndarray, index: int, element: Any):
        if isinstance(buffer_array, np.ndarray) and index < len(buffer_array):
            try: buffer_array[index] = element
            except (ValueError, TypeError) as e: print(f"Warning: could not set value in replay buffer. Idx {index}, Buf shape {buffer_array.shape}, Elem type {type(element)}, Elem shape {getattr(element, 'shape', 'N/A')}. Error: {e}")


    def _update_terminal_initial_locs(self):
        if self.size > 0 and 'terminals' in self and isinstance(self['terminals'], np.ndarray):
            valid_terminals_data = self['terminals'][:self.size]
            if valid_terminals_data.ndim > 1 and valid_terminals_data.shape[1] == 1: valid_terminals_data = valid_terminals_data.squeeze(1)

            self.terminal_locs = np.nonzero(valid_terminals_data > 0)[0]
            current_initial_locs = [0]
            if len(self.terminal_locs) > 0:
                current_initial_locs.extend(self.terminal_locs[:-1] + 1)
            self.initial_locs = np.unique(np.array(current_initial_locs, dtype=np.int64))
            self.initial_locs = self.initial_locs[self.initial_locs < self.size]
            if len(self.initial_locs) == 0 and self.size > 0: self.initial_locs = np.array([0])

            if self.size > 0 and (len(self.terminal_locs) == 0 or self.terminal_locs[-1] != self.size -1):
                 self.terminal_locs = np.unique(np.append(self.terminal_locs, self.size-1))
        else:
            self.terminal_locs = np.array([self.size -1], dtype=np.int64) if self.size > 0 else np.array([],dtype=np.int64)
            self.initial_locs = np.array([0], dtype=np.int64) if self.size > 0 else np.array([], dtype=np.int64)


    def add_transitions(self, transitions: Dict[str, np.ndarray]):
        batch_size = get_size(transitions)
        if batch_size == 0 or self.max_size == 0: return
        idxs_to_write = (np.arange(batch_size) + self.pointer) % self.max_size
        _tree_map(lambda buf, elems: self._set_idxs_vals(buf, idxs_to_write, elems), self, transitions)
        self.pointer = (self.pointer + batch_size) % self.max_size
        self.size = min(self.max_size, self.size + batch_size)
        self._update_terminal_initial_locs()

    def _set_idxs_vals(self, buffer_array: np.ndarray, indices: np.ndarray, elements: np.ndarray):
        if isinstance(buffer_array, np.ndarray) and isinstance(elements, np.ndarray):
            try: buffer_array[indices] = elements
            except (ValueError, TypeError) as e: print(f"Warning: could not set values in replay buffer. Error: {e}")


    def clear(self):
        self.size = 0; self.pointer = 0
        self._update_terminal_initial_locs()


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        if self.size > 0:
            terminals_data = self.dataset.get('terminals')
            if terminals_data is None or not isinstance(terminals_data, np.ndarray):
                raise ValueError("GCDataset requires 'terminals' in the underlying dataset.")
            if terminals_data.ndim > 1 and terminals_data.shape[1] == 1: terminals_data = terminals_data.squeeze(1)

            self.terminal_locs = np.nonzero(terminals_data[:self.dataset.size] > 0)[0]
            current_initial_locs = [0]
            if len(self.terminal_locs) > 0:
                current_initial_locs.extend(self.terminal_locs[:-1] + 1)
            self.initial_locs = np.unique(np.array(current_initial_locs, dtype=np.int64))
            self.initial_locs = self.initial_locs[self.initial_locs < self.dataset.size]
            if len(self.initial_locs) == 0 and self.dataset.size > 0: self.initial_locs = np.array([0])

            if len(self.terminal_locs) == 0 or self.terminal_locs[-1] != self.dataset.size - 1:
                 self.terminal_locs = np.unique(np.append(self.terminal_locs, self.dataset.size -1))
            assert self.terminal_locs[-1] == self.dataset.size - 1, f"Last terminal {self.terminal_locs[-1]} != size-1 {self.dataset.size-1}"
        else:
            self.terminal_locs = np.array([], dtype=np.int64)
            self.initial_locs = np.array([], dtype=np.int64)

        prob_sum_value = self.config.get('value_p_curgoal',0) + self.config.get('value_p_trajgoal',0) + self.config.get('value_p_randomgoal',0)
        prob_sum_actor = self.config.get('actor_p_curgoal',0) + self.config.get('actor_p_trajgoal',0) + self.config.get('actor_p_randomgoal',0)
        assert np.isclose(prob_sum_value, 1.0), f"Value goal probabilities sum to {prob_sum_value}, not 1.0"
        assert np.isclose(prob_sum_actor, 1.0), f"Actor goal probabilities sum to {prob_sum_actor}, not 1.0"

    def sample(self, batch_size: int, idxs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.dataset.size == 0: return {}
        if idxs is None: idxs = self.dataset.get_random_idxs(batch_size)
        if len(idxs) == 0: return {}

        batch = self.dataset.sample(batch_size, idxs)
        if not batch: return {}

        value_start = self.config.get('value_geom_start_offset', 1)
        actor_start = self.config.get('actor_geom_start_offset', 1)

        value_goal_idxs = self.sample_goals(idxs, self.config.get('value_p_curgoal',0), self.config.get('value_p_trajgoal',0), self.config.get('value_p_randomgoal',0), self.config.get('value_geom_sample',False), value_start)
        actor_goal_idxs = self.sample_goals(idxs, self.config.get('actor_p_curgoal',0), self.config.get('actor_p_trajgoal',0), self.config.get('actor_p_randomgoal',0), self.config.get('actor_geom_sample',False), actor_start)

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        # Ensure value_goal_idxs has same length as idxs for comparison
        if len(value_goal_idxs) == len(idxs):
            successes = (idxs == value_goal_idxs).astype(float)
            batch['relabeled_masks'] = 1.0 - successes
            batch['relabeled_rewards'] = successes - (1.0 if self.config.get('gc_negative', False) else 0.0)
            if self.config.get('relabel_reward', False):
                batch['masks'] = batch['relabeled_masks']
                batch['rewards'] = batch['relabeled_rewards']
        else: # Should not happen if sample_goals is correct
            print(f"Warning: Mismatch in length of idxs ({len(idxs)}) and value_goal_idxs ({len(value_goal_idxs)})")
            batch['relabeled_masks'] = np.ones(len(idxs), dtype=float) # Default to no success
            batch['relabeled_rewards'] = np.zeros(len(idxs), dtype=float) - (1.0 if self.config.get('gc_negative', False) else 0.0)


        p_aug = self.config.get('p_aug', None)
        if p_aug is not None and p_aug > 0:
            if np.random.rand() < p_aug:
                aug_keys = [k for k in ['observations', 'next_observations', 'value_goals', 'actor_goals'] if k in batch]
                if aug_keys: augment(batch, aug_keys)
        return batch

    def sample_goals(self, idxs: np.ndarray, p_curgoal: float, p_trajgoal: float, p_randomgoal: float,
                     geom_sample: bool, geom_start_offset: int = 1) -> np.ndarray:
        batch_size = len(idxs)
        if batch_size == 0: return np.array([], dtype=int)
        if len(self.terminal_locs) == 0: return idxs.copy() # Cannot sample future if no terminals known

        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        traj_end_indices = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs, side='left')]
        traj_end_indices = np.minimum(traj_end_indices, self.dataset.size - 1)

        if geom_sample:
            discount = self.config.get('discount', 0.99)
            if not (0 < discount < 1): discount = 0.99
            offsets = np.random.geometric(p=1.0 - discount, size=batch_size) + (geom_start_offset - 1)
            middle_goal_idxs = np.minimum(idxs + offsets, traj_end_indices)
        else:
            min_offsets = np.minimum(idxs + geom_start_offset, traj_end_indices)
            max_offsets = np.maximum(traj_end_indices, min_offsets)
            middle_goal_idxs = np.array([np.random.randint(min_o, max_o + 1) if max_o >= min_o else min_o for min_o, max_o in zip(min_offsets, max_offsets)], dtype=int)

        prob_traj_vs_random = p_trajgoal / (1.0 - p_curgoal + 1e-8)
        choice_traj = np.random.rand(batch_size) < prob_traj_vs_random
        interim_goal_idxs = np.where(choice_traj, middle_goal_idxs, random_goal_idxs)
        choice_cur = np.random.rand(batch_size) < p_curgoal
        final_goal_idxs = np.where(choice_cur, idxs, interim_goal_idxs)
        return final_goal_idxs.astype(int)

    def normalize_observations(self, observations: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self.dataset.normalize_observations(observations)

    def get_observations(self, idxs: np.ndarray) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if len(idxs) == 0:
            obs_structure = self.dataset.get('observations')
            if isinstance(obs_structure, dict): return {k: np.array([]) for k in obs_structure}
            return np.array([])

        # Ensure idxs are valid before use
        idxs = np.clip(idxs, 0, self.dataset.size - 1 if self.dataset.size > 0 else 0)


        frame_stack_val = self.config.get('frame_stack', None)
        if frame_stack_val is None or frame_stack_val <= 1:
            return _tree_map(lambda arr: arr[idxs] if isinstance(arr, np.ndarray) and len(arr)>0 else arr, self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs, frame_stack_val)

    def get_stacked_observations(self, idxs: np.ndarray, frame_stack: int) -> Union[np.ndarray, Dict[str,np.ndarray]]:
        if self.dataset.size == 0 or len(self.initial_locs) == 0 :
            print("Warning: get_stacked_observations called on empty dataset or no initial_locs.")
            return _tree_map(lambda arr: arr[idxs] if isinstance(arr, np.ndarray) and len(arr)>0 else arr, self.dataset['observations']) # Fallback


        initial_state_indices_for_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]

        def stack_leaf(leaf_data: np.ndarray) -> np.ndarray:
            if not isinstance(leaf_data, np.ndarray) or leaf_data.ndim == 0 : return leaf_data

            frames_for_leaf = []
            for i in reversed(range(frame_stack)):
                # Offset from current idxs: idxs - i
                # Ensure we don't go before the start of the trajectory for that idx
                current_frame_indices = np.maximum(idxs - i, initial_state_indices_for_idxs)
                current_frame_indices = np.clip(current_frame_indices, 0, self.dataset.size - 1)
                frames_for_leaf.append(leaf_data[current_frame_indices])
            # Stack along the last axis (feature axis)
            return np.concatenate(frames_for_leaf, axis=-1)

        return _tree_map(stack_leaf, self.dataset['observations'])
