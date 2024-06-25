import pickle
import random
import typing as t
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from timebase.data.static import *
from timebase.utils import h5


def process_features(data: t.Dict[str, np.ndarray], scaling_mode: int, stats: t.Dict):
    """Process data based on args.scaling_mode
    0 - no scaling
    1 - normalize features by the overall min and max values from the
        training set
    2 - standardize features by the overall mean and standard deviation
        from the training set
    """
    eps = np.finfo(np.float32).eps
    match scaling_mode:
        case 0:
            pass
        case 1:
            for k, v in data.items():
                # min-max scale
                data[k] = (
                    np.clip(a=v, a_min=stats[k]["min"], a_max=stats[k]["max"])
                    - stats[k]["min"]
                ) / (stats[k]["max"] - stats[k]["min"] + eps)
        case 2:
            for k, v in data.items():
                # standardize
                data[k] = (
                    np.clip(a=v, a_min=stats[k]["min"], a_max=stats[k]["max"])
                    - stats[k]["mean"]
                ) / (stats[k]["std"] + eps)
        case 3:
            for k, v in data.items():
                # robust standardize
                data[k] = (
                    np.clip(a=v, a_min=stats[k]["min"], a_max=stats[k]["max"])
                    - stats[k]["median"]
                ) / (stats[k]["iqr"] + eps)

        case _:
            raise NotImplementedError(f"scaling_mode {scaling_mode} not implemented.")
    return data


def magnitude_warp(args, x):
    # Generate a random smooth curve using cubic spline interpolation
    timestamps = np.arange(len(x))
    spline = CubicSpline(timestamps, x)
    curve = spline(timestamps)
    # Apply the transformation to the signal
    transformed_signal = x * curve
    return transformed_signal


def permutation(args, x):
    # Split the signal into non-overlapping segments
    segments = np.array_split(x, args.num_sub_segments)
    # Shuffle the order of the segments
    np.random.shuffle(segments)
    # Recombine the permuted segments to form the permuted signal
    permuted_signal = np.concatenate(segments)
    return permuted_signal


def time_warp(args, x):
    def _F(segment, k):
        # Create a new array with the same length as the segment.
        interpolated_segment = np.zeros(len(segment))
        # Interpolate the values in the segment.
        for i in range(len(segment)):
            interpolated_segment[i] = segment[i] * k ** (i / len(segment))
        return interpolated_segment

    # Divide the signal into n non-overlapping segments.
    segments = np.array_split(x, args.num_sub_segments)
    # Randomly select half of the segments to be stretched.
    stretched_segments = random.sample(segments, args.num_sub_segments // 2)
    # Stretch the stretched segments by the factor k.
    stretched_segments = np.array(
        [_F(segment=segment, k=args.stretch_factor) for segment in stretched_segments]
    )
    # Squeeze the remaining segments by the factor 1/k.
    squeezed_segments = np.array(
        [
            _F(segment=segment, k=args.stretch_factor**-1)
            for segment in segments[args.num_sub_segments // 2 :]
        ]
    )
    # Concatenate the stretched and squeezed segments.
    time_warped_signal = np.concatenate([stretched_segments, squeezed_segments])
    # Resize the time-warped signal to the original length.
    time_warped_signal = np.resize(time_warped_signal, len(x))
    return time_warped_signal


def cropping(args, x):
    # Split the signal into non-overlapping segments
    segments = np.array_split(x, args.num_sub_segments)
    # Randomly select a segment for resampling
    selected_segment = random.sample(segments, 1)[0]
    # Resample the selected segment to the original length
    resampled_segment = np.resize(selected_segment, len(x))
    return resampled_segment


def gaussian_noise(args, x):
    # Average power of the signal
    signal_power = np.mean(np.abs(x) ** 2)
    # Power of the noise based on SNR
    noise_power = 10 ** (-(args.snr / 10)) * signal_power
    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(x))
    # Add the noise to the signal
    noisy_signal = x + noise

    return noisy_signal


def identity(args, x):
    return x


class ClassificationDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        labels: t.Dict[str, np.ndarray],
        rec_ids: np.ndarray,
        stats: t.Dict[str, t.Dict[str, float]],
        recording_id_str_to_num: t.Dict[str, int],
        subject_ids: np.array = None,
    ):
        self.filenames = filenames
        self.labels = labels
        self.subject_ids = subject_ids
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2, 3)
        self.scaling_mode = args.scaling_mode
        self.positive_classes = [
            v for k, v in DICT_STATE.items() if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
        ]
        self.recording_id = rec_ids
        self.recording_id_str_to_num = recording_id_str_to_num

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def segment_id(filename: str):
        return int(os.path.basename(filename).replace(".h5", ""))

    @staticmethod
    def session_id(filename: str):
        return int(os.path.basename(os.path.dirname(filename)))

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = process_features(
            data=data, scaling_mode=self.scaling_mode, stats=self.stats
        )
        label = {k: v[idx] for k, v in self.labels.items()}
        target = np.array(
            1 if label["status"] in self.positive_classes else 0,
            dtype=np.float32,
            ndmin=1,
        )
        metadata = {
            "session_id": self.session_id(filename),
            "segment_id": self.segment_id(filename),
            "recording_id": self.recording_id_str_to_num[self.recording_id[idx]],
        }
        sample = {"data": data, "label": label, "target": target, "metadata": metadata}
        if self.subject_ids is not None:
            sample["subject_id"] = self.subject_ids[idx]
        return sample


class ImputationDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        stats: t.Dict[str, t.Dict[str, float]],
        segment_length: int,
        channel_freq: t.Dict[str, int],
    ):
        self.filenames = filenames
        self.dataset_path = args.dataset
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2, 3)
        self.scaling_mode = args.scaling_mode
        self.channel_freq = channel_freq
        self.segment_length = segment_length
        self.masking_ratio = args.masking_ratio
        self.lm = args.lm
        loaded_masks = np.load(os.path.join(args.dataset, "masks.npz"))
        self.masks = loaded_masks["data"]
        self.scaling_mode = args.scaling_mode
        assert self.scaling_mode in (1, 2, 3)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def segment_id(filename: str):
        return int(os.path.basename(filename).replace(".h5", ""))

    @staticmethod
    def session_id(filename: str):
        return int(os.path.basename(os.path.dirname(filename)))

    @staticmethod
    def get_collection_name(filename: str, dataset_path: str):
        return filename.replace(f"{dataset_path}/", "").split("/")[0]

    def create_noise(self, mask):
        if self.scaling_mode in (2, 3):
            noise = np.random.normal(0, 1, size=mask.shape).astype(np.float32)
        else:
            noise = np.random.randn(*mask.shape)
        return mask * noise

    def inject_noise(self, data):
        masks_indices = np.random.randint(
            low=0, high=NUM_MASKS, size=len(self.channel_freq)
        )
        mask, data2corrupt = {}, {}
        for (channel_name, arr), idx in zip(data.items(), masks_indices):
            channel_mask = np.repeat(
                a=self.masks[idx], repeats=self.channel_freq[channel_name]
            )
            mask[channel_name] = channel_mask
            # noise = self.create_noise(mask=channel_mask)
            # data2corrupt[channel_name] = arr + noise
            data2corrupt[channel_name] = arr * ~channel_mask

        return mask, data2corrupt

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = process_features(
            data=data, scaling_mode=self.scaling_mode, stats=self.stats
        )
        mask, corrupted_data = self.inject_noise(data=data)
        sample = {
            "original": data,
            "corrupted": corrupted_data,
            "mask": mask,
            "collection": COLLECTIONS_DICT[
                self.get_collection_name(
                    filename=filename, dataset_path=self.dataset_path
                )
            ],
        }

        return sample


class TransformationDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        stats: t.Dict[str, t.Dict[str, float]],
        segment_length: int,
        channel_freq: t.Dict[str, int],
    ):
        self.filenames = filenames
        self.dataset_path = args.dataset
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2)
        self.scaling_mode = args.scaling_mode
        self.channel_freq = channel_freq
        self.segment_length = segment_length
        self.TRANSFORMATION_FUNC_DICT = {
            0: partial(identity, args),
            1: partial(magnitude_warp, args),
            2: partial(time_warp, args),
            3: partial(permutation, args),
            4: partial(cropping, args),
            5: partial(gaussian_noise, args),
        }

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def segment_id(filename: str):
        return int(os.path.basename(filename).replace(".h5", ""))

    @staticmethod
    def session_id(filename: str):
        return int(os.path.basename(os.path.dirname(filename)))

    @staticmethod
    def get_collection_name(filename: str, dataset_path: str):
        return filename.replace(f"{dataset_path}/", "").split("/")[0]

    def sample_transformations(self):
        choices = np.arange(len(self.TRANSFORMATION_FUNC_DICT))
        probs = [1 / len(choices)] * len(choices)
        return np.random.choice(choices, size=len(self.channels), p=probs)

    def apply_transformation(self, data: t.Dict, transformations: np.ndarray):
        transformed_data = deepcopy(data)
        for i, (c, v) in enumerate(transformed_data.items()):
            transformed_data[c] = self.TRANSFORMATION_FUNC_DICT[transformations[i]](
                x=v
            ).astype(np.float32)
        return transformed_data

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = process_features(
            data=data, scaling_mode=self.scaling_mode, stats=self.stats
        )
        transformations = self.sample_transformations()
        transformed_data = self.apply_transformation(
            data=data,
            transformations=transformations,
        )

        sample = {
            "transformed_data": transformed_data,
            "transformation": transformations.astype(np.int32),
            "collection": COLLECTIONS_DICT[
                self.get_collection_name(
                    filename=filename, dataset_path=self.dataset_path
                )
            ],
        }
        return sample


class RecIDSampler(Sampler):
    def __init__(self, dataset: np.ndarray, batch_size: int, drop_last=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        assert len(np.unique(self.dataset)) >= self.batch_size

    def batch_dataset(self, rec_id_indexes, columns_of_batches):
        cut_point = np.min([len(v) for v in rec_id_indexes.values()][: self.batch_size])
        batches = np.ones(shape=(self.batch_size, cut_point), dtype=np.int32)
        for i in range(self.batch_size):
            batches[i] = list(rec_id_indexes.values())[i][:cut_point]
        columns_of_batches.append(batches)
        rec_id_indexes_updated = {
            key: list(set(value).difference(set(batches.flatten().tolist())))
            for key, value in rec_id_indexes.items()
            if list(set(value).difference(set(batches.flatten().tolist())))
        }
        rec_id_indexes_updated = dict(
            sorted(
                rec_id_indexes_updated.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )
        for key, value in rec_id_indexes_updated.items():
            random.shuffle(value)
        return rec_id_indexes_updated

    def __iter__(self):
        rec_id_indexes = {}
        for rec_id in np.unique(self.dataset):
            indexes = list(np.where(self.dataset == rec_id)[0])
            np.random.shuffle(indexes)
            rec_id_indexes[rec_id] = indexes
        rec_id_indexes = dict(
            sorted(rec_id_indexes.items(), key=lambda item: len(item[1]), reverse=True)
        )
        columns_of_batches = []
        while len(rec_id_indexes) >= self.batch_size:
            rec_id_indexes = self.batch_dataset(
                rec_id_indexes=rec_id_indexes, columns_of_batches=columns_of_batches
            )
        batches = np.concatenate(columns_of_batches, axis=1)
        batch_idxs = [list(batches[:, i]) for i in range(batches.shape[1])]
        random.shuffle(batch_idxs)
        # for i in range(len(batch_idxs)):
        #     assert len(np.unique(self.dataset[batch_idxs[i]])) == self.batch_size
        return iter(batch_idxs)

    def __len__(self):
        rec_ids, counts = np.unique(self.dataset, return_counts=True)
        data_dict = dict(zip(rec_ids, counts))
        data_dict = dict(
            sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        )
        counts_array = np.array(list(data_dict.values()))
        num_batches = 0
        while len(np.where(counts_array > 0)[0]) >= self.batch_size:
            cut_point = np.min(counts_array[: self.batch_size])
            counts_array[: self.batch_size] = (
                counts_array[: self.batch_size] - cut_point
            )
            num_batches += cut_point
            counts_array = np.sort(counts_array, kind="quicksort")[::-1]
        return num_batches


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        recording_id: np.ndarray,
        stats: t.Dict[str, t.Dict[str, float]],
        segment_length: int,
        channel_freq: t.Dict[str, int],
        batch_size: int,
    ):
        self.recording_id = recording_id
        self.batch_size = batch_size
        self.filenames = filenames
        self.dataset_path = args.dataset
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2, 3)
        self.scaling_mode = args.scaling_mode
        self.channel_freq = channel_freq
        self.segment_length = segment_length
        assert len(np.unique(self.recording_id)) >= self.batch_size

    def __len__(self):
        rec_ids, counts = np.unique(self.recording_id, return_counts=True)
        data_dict = dict(zip(rec_ids, counts))
        data_dict = dict(
            sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        )
        counts_array = np.array(list(data_dict.values()))
        num_batches = 0
        while len(np.where(counts_array > 0)[0]) >= self.batch_size:
            cut_point = np.min(counts_array[: self.batch_size])
            counts_array[: self.batch_size] = (
                counts_array[: self.batch_size] - cut_point
            )
            num_batches += cut_point
            counts_array = np.sort(counts_array, kind="quicksort")[::-1]
        return num_batches

    @staticmethod
    def segment_id(filename: str):
        return int(os.path.basename(filename).replace(".h5", ""))

    @staticmethod
    def session_id(filename: str):
        return int(os.path.basename(os.path.dirname(filename)))

    @staticmethod
    def get_collection_name(filename: str, dataset_path: str):
        return filename.replace(f"{dataset_path}/", "").split("/")[0]

    def get_another_view(self, idx):
        """
        Given a recording ID, sample a different segment from the same
        recording ID
        """
        indexes = list(
            set(np.where(self.recording_id == self.recording_id[idx])[0]).difference(
                [idx]
            )
        )
        sampled_idx = np.random.choice(indexes)
        filename = self.filenames[sampled_idx]
        another_view = {
            c: h5.get(filename, name=c).astype(np.float32) for c in self.channels
        }
        return another_view

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = process_features(
            data=data, scaling_mode=self.scaling_mode, stats=self.stats
        )
        another_view = self.get_another_view(idx=idx)
        another_view = process_features(
            data=another_view, scaling_mode=self.scaling_mode, stats=self.stats
        )

        sample = {
            "x1": data,
            "x2": another_view,
            "collection": COLLECTIONS_DICT[
                self.get_collection_name(
                    filename=filename, dataset_path=self.dataset_path
                )
            ],
        }
        return sample


class DiagnosticsDataset(Dataset):
    def __init__(
        self,
        args,
        filenames: np.ndarray,
        labels: t.Dict[str, np.ndarray],
        sleep_status: np.ndarray,
        rec_ids: np.ndarray,
        stats: t.Dict[str, t.Dict[str, float]],
        recording_id_str_to_num: t.Dict[str, int],
        target_task_ds: bool = False,
    ):
        self.filenames = filenames
        self.labels = labels
        self.sleep_status = sleep_status
        self.stats = stats
        self.channels = list(args.input_shapes.keys())
        self.task_mode = args.task_mode
        assert args.scaling_mode in (0, 1, 2, 3)
        self.scaling_mode = args.scaling_mode
        self.positive_classes = [
            v for k, v in DICT_STATE.items() if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
        ]
        self.recording_id = rec_ids
        self.recording_id_str_to_num = recording_id_str_to_num
        self.target_task_ds = target_task_ds
        self.dataset_path = args.dataset

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def get_collection_name(filename: str, dataset_path: str):
        return filename.replace(f"{dataset_path}/", "").split("/")[0]

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Dict[str, t.Any]:
        filename = self.filenames[idx]
        data = {c: h5.get(filename, name=c).astype(np.float32) for c in self.channels}
        data = process_features(
            data=data, scaling_mode=self.scaling_mode, stats=self.stats
        )
        if self.target_task_ds:
            label = {k: v[idx] for k, v in self.labels.items()}
            target = np.array(
                1 if label["status"] in self.positive_classes else 0,
                dtype=np.float32,
                ndmin=1,
            )
        else:
            label = {k: -9 for k, v in self.labels.items()}
            target = np.array([-9])
        sleep_status = self.sleep_status[idx]
        sample = {
            "data": data,
            "label": label,
            "target": target,
            "collection": COLLECTIONS_DICT[
                self.get_collection_name(
                    filename=filename, dataset_path=self.dataset_path
                )
            ],
            "sleep_status": sleep_status,
        }

        return sample
