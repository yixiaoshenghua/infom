import os
import tempfile
from datetime import datetime
import collections
import absl.flags as flags
import ml_collections
import json
import pathlib
import numpy as np
from PIL import Image, ImageEnhance
import torch
import time

# class Logger:
#     """logger for logging metrics to a CSV file and tensorboard."""

#     def __init__(self, path):
#         self.path = path
#         self.header = None
#         self.file = None
#         self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

#     def log(self, row, step):
#         row['step'] = step
#         if self.file is None:
#             self.file = open(self.path, 'w')
#             if self.header is None:
#                 self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
#                 self.file.write(','.join(self.header) + '\n')
#             filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
#             self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
#         else:
#             filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
#             self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
#         self.file.flush()

#     def close(self):
#         if self.file is not None:
#             self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')

class Logger:

    def __init__(self, outputs, multiplier=1):
        # self._step = step
        self._outputs = outputs
        self._multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, step, prefix=None):
        # step = int(self._step) * self._multiplier
        for name, value in dict(mapping).items():
            name = f"{prefix}_{name}" if prefix else name
            value = np.array(value)
            if len(value.shape) not in (0, 2, 3, 4):
                raise ValueError(
                    f"Shape {value.shape} for name '{name}' cannot be "
                    "interpreted as scalar, image, or video."
                )
            self._metrics.append((step, name, value))

    def scalar(self, name, value, step):
        self.add({name: value}, step)

    def image(self, name, value, step):
        self.add({name: value}, step)

    def video(self, name, value, step):
        self.add({name: value}, step)

    def write(self, step, fps=False):
        fps and self.scalar("fps", self._compute_fps(step))
        if not self._metrics:
            return
        for output in self._outputs:
            output(self._metrics)
        self._metrics.clear()

    def _compute_fps(self, step):
        # step = int(self._step) * self._multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

class TensorBoardOutputPytorch:

    # FIXME image dataformats='CHW' by default

    def __init__(self, logdir, fps=20):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(logdir), max_queue=1000)
        self._fps = fps

    def __call__(self, summaries):
        # NOTE: to aggregate values in the same step
        scalar_summaries = collections.defaultdict(list)
        for step, name, value in summaries:
            if len(value.shape) == 0:
                scalar_summaries[(step, name)].append(value.item())
        for (step, name), value in scalar_summaries.items():
            self._writer.add_scalar("scalars/" + name, np.mean(value), step)

        for step, name, value in summaries:
            # if len(value.shape) == 0:
            #     self._writer.add_scalar("scalars/" + name, value, step)
            # elif len(value.shape) == 2:
            if len(value.shape) == 2:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 3:
                self._writer.add_image(name, value, step)
            elif len(value.shape) == 4:
                self._video_summary(name, value, step)
                # self._writer.add_video(name,value[None], step,fps=self._fps)
                # vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
                # batch, time, channels, height and width
        self._writer.flush()

    def _video_summary(self, name, video, step):
        # import tensorflow as tf
        # import tensorflow.compat.v1 as tf1
        from tensorboard.compat.proto.summary_pb2 import Summary

        name = name if isinstance(name, str) else name.decode("utf-8")
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            # summary = tb.RecordWriter()
            image = Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = encode_gif(video, self._fps)
            self._writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag=name, image=image)]), step)
            # tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print("GIF summaries require ffmpeg in $PATH.", e)
            # self._writer.add_image(name, video, step)
            self._writer.add_video(name, torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0), step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out

class TerminalOutput:
    def __call__(self, summaries):
        # TODO aggregate
        step = max(s for s, _, _, in summaries)
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        print(f"[{step}]", " / ".join(f"{k} {v}" for k, v in formatted.items()))

    def _format_value(self, value):
        if value == 0:
            return "0"
        elif 0.01 < abs(value) < 10000:
            value = f"{value:.2f}"
            value = value.rstrip("0")
            value = value.rstrip("0")
            value = value.rstrip(".")
            return value
        else:
            value = f"{value:.1e}"
            value = value.replace(".0e", "e")
            value = value.replace("+0", "")
            value = value.replace("+", "")
            value = value.replace("-0", "-")
        return value


class JSONLOutput:

    def __init__(self, logdir):
        self._logdir = pathlib.Path(logdir).expanduser()

    def __call__(self, summaries):
        # NOTE: to aggregate values in the same step
        scalar_summaries = collections.defaultdict(lambda: collections.defaultdict(list))
        for step, name, value in summaries:
            if len(value.shape) == 0:
                scalar_summaries[step][name].append(value.item())
        for step in scalar_summaries:
            scalars = {k: np.mean(v) for k, v in scalar_summaries[step].items()}
            with (self._logdir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **scalars}) + "\n")