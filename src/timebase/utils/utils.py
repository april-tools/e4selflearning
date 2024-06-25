import collections
import copy
import csv
import itertools
import random
import subprocess
import typing as t
from copy import deepcopy

import pandas as pd
import torch
import wandb
from torch import nn

from timebase.data.static import *
from timebase.models.models import Classifier
from timebase.utils import yaml


def set_random_seed(seed: int, deterministic: bool = False, verbose: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if verbose > 2:
        print(f"set random seed: {seed}")


def get_device(args):
    """Get the appropriate torch.device from args.device argument"""
    device = args.device
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            # allow TensorFloat32 computation
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = "mps"
    args.device = torch.device(device)


def wandb_init(args, wandb_sweep: bool):
    """initialize wandb and strip information from args"""
    os.environ["WANDB_SILENT"] = "true"
    if not wandb_sweep:
        try:
            config = deepcopy(args.__dict__)
            config.pop("ds_info", None)
            config.pop("input_shapes", None)
            config.pop("output_dir", None)
            config.pop("device", None)
            config.pop("format", None)
            config.pop("dpi", None)
            config.pop("save_plots", None)
            config.pop("plot_mode", None)
            config.pop("save_predictions", None)
            config.pop("verbose", None)
            config.pop("use_wandb", None)
            config.pop("wandb_group", None)
            config.pop("reuse_stats", None)
            config.pop("clear_output_dir", None)
            wandb.init(
                config=config,
                dir=os.path.join(args.output_dir, "wandb"),
                project="ssl",
                entity="filippo-corponi",
                group=args.wandb_group,
                name=os.path.basename(args.output_dir),
            )
        except AssertionError as e:
            print(f"wandb.init error: {e}\n")
            args.use_wandb = False


def update_dict(target: t.Dict, source: t.Dict, replace: bool = False):
    """add or update items in source to target"""
    for key, value in source.items():
        if replace:
            target[key] = value
        else:
            if key not in target:
                target[key] = []
            target[key].append(value)


def check_output(command: list):
    """Execute command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to output_dir/args.yaml"""
    """Save args object as dictionary to args.output_dir/args.yaml"""
    arguments = copy.deepcopy(args.__dict__)
    try:
        arguments["git_hash"] = check_output(["git", "describe", "--always"])
        arguments["hostname"] = check_output(["hostname"])
    except subprocess.CalledProcessError as e:
        if args.verbose:
            print(f"Unable to call subprocess: {e}")
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args, dir: str, replace: bool = False, visualization: bool = False):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(dir, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    fine_tuning_args = [
        "dataset",
        "filter_collections",
        "unlabelled_data_resampling_percentage",
        "exclude_anomalies",
        "disable_bias",
        "emb_dim",
        "emb_num_filters",
        "emb_type",
        "mlp_dim",
        "num_blocks",
        "num_heads",
        "num_units",
        "pretext_task",
        "representation_module",
        "scaling_mode",
        "split_mode",
        "include_hr",
    ]
    read_out_args = fine_tuning_args + [
        "a_dropout",
        "drop_path",
        "dropout",
        "m_dropout",
    ]
    visualization_args = [
        "weight_decay",
        "task_mode",
        "split_mode",
        "scaling_mode",
        "seed",
        "reuse_stats",
        "representation_module",
        "precision",
        "overwrite_masks",
        "mlp_dim",
        "num_blocks",
        "num_heads",
        "num_units",
        "mixed_precision",
        "lr",
        "lr_patience",
        "input_shapes",
        "m_dropout",
        "emb_dim",
        "emb_num_filters",
        "emb_type",
        "a_dropout",
        "batch_size",
        "dataset",
        "disable_bias",
        "drop_path",
        "ds_info",
        "exclude_anomalies",
        "filter_collections",
        "unlabelled_data_resampling_percentage",
        "num_train_subjects",
        "pretext_task",
        "masking_ratio",
        "lm",
        "snr",
        "num_sub_segments",
        "stretch_factor",
        "temperature",
        "include_hr",
    ]

    if hasattr(args, "task_mode") and not visualization:
        for key, value in arguments.items():
            match args.task_mode:
                case 1:
                    if key in fine_tuning_args:  # fine-tuning
                        setattr(args, key, value)
                case 2 | 4:
                    if key in read_out_args:  # read_out
                        setattr(args, key, value)
    elif visualization:
        for key, value in arguments.items():
            if key in visualization_args:
                setattr(args, key, value)
    else:  # TODO not clear
        args.task_mode = 2
        args.num_train_subjects = 10
        for key, value in arguments.items():
            if (replace == True) or (not hasattr(args, key)):
                setattr(args, key, value)


def load_args_oos(args):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(args.output_dir, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if not hasattr(args, key) and key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
        ]:
            setattr(args, key, value)


def write_csv(output_dir, content: list):
    with open(os.path.join(output_dir, "results.csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow(content)


def to_numpy(a: t.Union[torch.Tensor, np.ndarray]):
    return a.cpu().numpy() if torch.is_tensor(a) else a


class BufferDict(nn.Module):
    """Holds buffers in a dictionary.

    Reference: https://botorch.org/api/utils.html#botorch.utils.torch.BufferDict

    BufferDict can be indexed like a regular Python dictionary, but buffers it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.BufferDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.BufferDict` (the argument to
      :meth:`~torch.nn.BufferDict.update`).

    Note that :meth:`~torch.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Args:
        buffers (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = nn.BufferDict({
                        'left': torch.randn(5, 10),
                        'right': torch.randn(5, 10)
                })

            def forward(self, x, choice):
                x = self.buffers[choice].mm(x)
                return x
    """

    def __init__(self, buffers=None):
        r"""
        Args:
            buffers: A mapping (dictionary) from string to :class:`~torch.Tensor`, or
                an iterable of key-value pairs of type (string, :class:`~torch.Tensor`).
        """
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""Update the :class:`~torch.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`buffers` is an ``OrderedDict``, a :class:`~torch.nn.BufferDict`,
            or an iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable): a mapping (dictionary) from string to
                :class:`~torch.Tensor`, or an iterable of
                key-value pairs of type (string, :class:`~torch.Tensor`)
        """
        if not isinstance(buffers, collections.abc.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, collections.abc.Mapping):
            if isinstance(buffers, (collections.OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, collections.abc.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Buffer containing: [{} of size {}{}]".format(
                torch.typename(p), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")


def create_young_hamilton_labels(args, y: t.Dict[str, np.ndarray]):
    ymrs_sum_binned = pd.cut(
        np.sum(
            np.concatenate(
                [
                    np.expand_dims(y[col], axis=1)
                    for col in args.selected_items
                    if "YMRS" in col
                ],
                axis=1,
            ),
            axis=1,
        ),
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        bins=[
            0,
            7,
            14,
            25,
            60,
        ],
        include_lowest=True,
        labels=False,
    )
    hdrs_sum_binned = pd.cut(
        np.sum(
            np.concatenate(
                [
                    np.expand_dims(y[col], axis=1)
                    for col in args.selected_items
                    if "HDRS" in col
                ],
                axis=1,
            ),
            axis=1,
        ),
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        bins=[
            0,
            7,
            14,
            23,
            52,
        ],  # [0, 7, 14, 23, 52] <- https://en.wikipedia.org/wiki/Hamilton_Rating_Scale_for_Depression,
        include_lowest=True,
        labels=False,
    )
    return (
        np.array(
            pd.Series(
                [
                    f"young{str(young)}_hamilton{str(ham)}"
                    for young, ham in zip(ymrs_sum_binned, hdrs_sum_binned)
                ]
            ).replace(YOUNG_HAMILTON_DICT)
        ),
        ymrs_sum_binned,
        hdrs_sum_binned,
    )


def get_sequences_boundaries_index(arr, value):
    """
    Given a 1D array-like object return a list of lists where each sub-list
    contains two elements, the former being the index where a given sequence of
    array entries equal to value starts and the letter being the index where
    the same sequence ends
    For example:
    arr = np.array([0,0,1,1,1,0,0,1,0,1])
    get_indexes(arr=arr, value=1) -> [[2, 4], [7, 7], [9, 9]]
    get_indexes(arr=arr, value=0) -> [[0, 1], [5, 6], [8, 8]]
    """
    seqs = [(key, len(list(val))) for key, val in itertools.groupby(arr)]
    seqs = [
        (key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)
    ]
    return [[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == value]


def load_encoder_ckp(args, classifier: Classifier, path2pretraining_res: str):
    if os.path.exists(
        os.path.join(path2pretraining_res, "ckpt_classifier", "model_state.pt")
    ):
        filename = os.path.join(
            path2pretraining_res, "ckpt_classifier", "model_state.pt"
        )
    elif os.path.exists(
        os.path.join(path2pretraining_res, "ckpt_sslearner", "model_state.pt")
    ):
        filename = os.path.join(
            path2pretraining_res, "ckpt_sslearner", "model_state.pt"
        )
    else:
        raise FileNotFoundError(f"checkpoint not found at {path2pretraining_res}")
    ckpt = torch.load(filename, map_location=args.device)
    if args.task_mode in (0, 3):
        state_dict = classifier.state_dict()
        state_dict.update(ckpt["model"])
        classifier.load_state_dict(state_dict)
    else:
        state_dict = classifier.sslearner.state_dict()
        state_dict.update(
            {
                # the transform head is discarded
                k.replace("sslearner.", ""): v
                for k, v in ckpt["model"].items()
                if any(
                    module in k for module in ["channel_embedding", "feature_encoder"]
                )
            }
        )
        classifier.sslearner.load_state_dict(state_dict)
    if args.verbose:
        print(f"Encoder's parameters loaded from epoch {ckpt['epoch']}")
