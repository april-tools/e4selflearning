import gzip
import os
import pickle
import shutil
import typing as t
from functools import partial
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from timebase.data.static import *
from timebase.utils import h5
from timebase.utils.utils import get_sequences_boundaries_index


def shuffle(x: np.ndarray, y: np.ndarray):
    """Shuffle the 0 index of x and y, jointly."""
    indexes = np.arange(x.shape[0])
    indexes = np.random.permutation(indexes)
    return x[indexes], y[indexes]


def unzip(filename: str, unzip_dir: str):
    """Unzip filename to unzip_dir with the same basename"""
    with ZipFile(filename, mode="r") as f:
        f.extractall(
            os.path.join(unzip_dir, os.path.basename(filename).replace(".zip", ""))
        )


def unzip_session(data_dir: str, session_id: str) -> str:
    """Return the path to the unzipped recording directory for session ID"""
    unzip_dir = os.path.join(data_dir, "unzip")
    recording_dir = os.path.join(unzip_dir, session_id)
    # unzip recording to recording folder not found.
    if not os.path.isdir(recording_dir):
        zip_filename = os.path.join(data_dir, f"{session_id}.zip")
        if not os.path.exists(zip_filename):
            raise FileNotFoundError(f"session {zip_filename} not found.")
        unzip(os.path.join(data_dir, f"{session_id}.zip"), unzip_dir=unzip_dir)
    return recording_dir


def get_channel_names(channel_data: t.Dict[str, np.ndarray]) -> t.List[str]:
    """Return printable channel names"""
    channel_names = []
    for channel in channel_data.keys():
        channel = channel.upper()
        if channel == "ACC":
            channel_names.append(f"{channel}_x")
            channel_names.append(f"{channel}_y")
            channel_names.append(f"{channel}_z")
        else:
            channel_names.append(channel)

    return channel_names


def normalize(
    x: np.ndarray, x_min: t.Union[float, np.ndarray], x_max: t.Union[float, np.ndarray]
):
    """Normalize x to [0, 1]"""
    return (x - x_min) / ((x_max - x_min) + 1e-6)


def standardize(
    x: np.ndarray, x_mean: t.Union[float, np.ndarray], x_std: t.Union[float, np.ndarray]
):
    return (x - x_mean) / x_std


def generate_masks(args, channel_freq: t.Dict[str, int], segment_length: int):
    output_file = os.path.join(args.dataset, "masks.npz")
    if os.path.exists(output_file):
        os.remove(output_file)
    p_m = 1 / args.lm
    # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p_u = p_m * args.masking_ratio / (1 - args.masking_ratio)
    p = [p_m, p_u]

    masks = np.empty(shape=[NUM_MASKS, segment_length], dtype=bool)
    for num in tqdm(
        range(NUM_MASKS), desc="Creating masks...", disable=args.verbose < 2
    ):
        mask = np.ones(segment_length, dtype=bool)
        state = int(np.random.rand() > args.masking_ratio)
        # state 0 means masking, 1 means not masking
        for i in range(segment_length):
            mask[i] = state
            if np.random.rand() < p[state]:
                state = 1 - state
        # True: masked, False: unmasked. Reconstruction loss to be
        # computed on masked [True] sequences only
        masks[num] = ~mask

    np.savez_compressed(output_file, data=masks)


def stats_in_tranches(
    args,
    files2read: np.ndarray,
    channels: t.List,
    stats: t.Dict,
    buffer: int = 1000,
):
    for tranche in tqdm(
        np.split(
            files2read,
            indices_or_sections=np.linspace(
                start=0,
                stop=len(files2read),
                num=len(files2read) // buffer + 1,
                endpoint=False,
                dtype=int,
            )[1:]
            if len(files2read) > buffer
            else 1,
        ),
        desc="Tranche",
        disable=args.verbose < 2,
    ):
        tranche_collector = {
            k: np.empty(shape=[len(tranche), v[0]])
            for k, v in args.input_shapes.items()
        }
        for i, filename in enumerate(tranche):
            for channel in channels:
                tranche_collector[channel][i] = h5.get(filename, name=channel)
        for channel, v in tranche_collector.items():
            min = np.percentile(v, q=0.5)
            max = np.percentile(v, q=99.5)
            q1 = np.percentile(np.clip(a=v, a_min=min, a_max=max), 25)
            q3 = np.percentile(np.clip(a=v, a_min=min, a_max=max), 75)
            stats[channel]["min"].append(min)
            stats[channel]["max"].append(max)
            stats[channel]["mean"].append(np.mean(np.clip(a=v, a_min=min, a_max=max)))
            stats[channel]["second_moment"].append(
                np.mean(np.power(np.clip(a=v, a_min=min, a_max=max), 2))
            )
            stats[channel]["median"].append(
                np.median(np.clip(a=v, a_min=min, a_max=max))
            )
            stats[channel]["iqr"].append(q3 - q1)
    for channel in channels:
        stats[channel]["min"] = np.min(stats[channel]["min"])
        stats[channel]["max"] = np.max(stats[channel]["max"])
        stats[channel]["mean"] = np.mean(stats[channel]["mean"])
        # std = var**(1/2) var = E[X**2] - (E[X])**2
        stats[channel]["std"] = np.sqrt(
            np.mean(stats[channel]["second_moment"])
            - np.power(stats[channel]["mean"], 2)
        )
        del stats[channel]["second_moment"]
        stats[channel]["median"] = np.mean(stats[channel]["median"])
        stats[channel]["iqr"] = np.mean(stats[channel]["iqr"])


def cases_controls_difference_in_rec_status(train_ds: DataLoader, data: t.Dict):
    mask = (train_ds.dataset.labels["status"] == 5) | (
        train_ds.dataset.labels["status"] == 6
    )
    controls_idx = np.where(mask)[0]
    cases_idx = np.where(~mask)[0]

    def _extract_session_code(filename: str):
        last_slash_index = filename.rfind("/")
        second_last_slash_index = filename.rfind("/", 0, last_slash_index)
        extracted_string = filename[second_last_slash_index + 1 : last_slash_index]
        return extracted_string

    def _extract_session_codes(filenames: np.ndarray):
        session_codes = np.array([_extract_session_code(s) for s in filenames])
        return list(np.unique(session_codes))

    controls_sub_ids = list(np.unique(train_ds.dataset.labels["Sub_ID"][controls_idx]))
    cases_sub_ids = list(np.unique(train_ds.dataset.labels["Sub_ID"][cases_idx]))

    def _compute_secs_per_status(info, sub_list, ds):
        secs_off_sub, secs_wake_sub = [], []
        for sub in sub_list:
            mask = ds.labels["Sub_ID"] == sub
            sessions = _extract_session_codes(ds.filenames[mask])
            keys = ["barcelona/" + s for s in sessions]
            secs_off_session, secs_wake_session, secs_tot = 0, 0, 0
            # some subjects have more than one session_id per assessment
            for s in keys:
                secs_tot += np.sum(list(info[s]["seconds_per_status"].values()))
                secs_off_session += info[s]["seconds_per_status"][2]
                secs_wake_session += info[s]["seconds_per_status"][0]
            secs_off_sub.append(secs_off_session / secs_tot)
            secs_wake_sub.append(secs_wake_session / secs_tot)

        return secs_off_sub, secs_wake_sub

    (
        off_controls_percentage,
        wake_controls_percentage,
    ) = _compute_secs_per_status(
        info=data["sessions_info"], sub_list=controls_sub_ids, ds=train_ds.dataset
    )

    (
        off_cases_percentage,
        wake_cases_percentage,
    ) = _compute_secs_per_status(
        info=data["sessions_info"], sub_list=cases_sub_ids, ds=train_ds.dataset
    )
    t_stat, p_val = stats.ttest_ind(off_controls_percentage, off_cases_percentage)
    print(
        f"Testing difference in mean value of session fraction of "
        f"off-body time: t-statistic={t_stat}, p-val={p_val}"
    )
    t_stat, p_val = stats.ttest_ind(wake_controls_percentage, wake_cases_percentage)
    print(
        f"Testing difference in mean value of session fraction of "
        f"wake time: t-statistic={t_stat}, p-val={p_val}"
    )


def medications_info(args, train_ds: DataLoader):
    ids, ids_idx = np.unique(train_ds.dataset.labels["Sub_ID"], return_index=True)
    stati = train_ds.dataset.labels["status"][ids_idx]
    stati = np.array([{v: k for k, v in DICT_STATE.items()}.get(x, x) for x in stati])
    medications = pd.read_csv(
        os.path.join(FILE_DIRECTORY, "TIMEBASE_database_meds.csv")
    )
    idx = []
    for i, s in zip(ids, stati):
        idx.extend(
            np.where((medications["Sub_ID"] == i) & (medications["status"] == s))[0]
        )
    med_df = (
        medications.iloc[idx]
        .drop_duplicates(subset="Sub_ID", keep="first")
        .reset_index(drop=True)
    )
    med_df = med_df.loc[:, ["status"] + MEDS]
    med_df["status"] = (
        med_df["status"]
        .replace({"Eu_MDD": 0, "Eu_BD": 0})
        .apply(lambda x: 1 if x != 0 else x)
    )
    med_df.groupby("status").agg(["mean", "std"]).to_csv(
        os.path.join(args.output_dir, "meds.csv")
    )
    cont_table = med_df.groupby("status").agg(["mean"])
    cont_table.columns = cont_table.columns.droplevel(1)

    def _calculate_fractions(x):
        yes_fraction = x.mean()
        no_fraction = 1 - yes_fraction
        return yes_fraction, no_fraction

    statistics, p_values, med_names = [], [], []
    for med_name in MEDS:
        d = med_df.dropna(subset=[med_name])
        result_df = (
            d.groupby("status")[med_name].apply(_calculate_fractions).apply(pd.Series)
        )
        result_df.columns = [f"{med_name}_yes", f"{med_name}_no"]
        if not np.trace(result_df.values) == 1:
            res = stats.chi2_contingency(result_df, correction=False)
            statistics.append(res.statistic)
            p_values.append(res.pvalue)
            med_names.append(med_name)
    p_values = list(np.array(p_values) * len(p_values))
    pd.DataFrame(
        {"MED": med_names, "corrected_p": p_values, "statistic": statistics}
    ).to_csv(os.path.join(args.output_dir, "meds_chi_tests.csv"))


def flirt_features_missingness(args, datasets: t.Dict):
    pd.DataFrame(
        data=np.expand_dims(
            datasets["x_train"].isna().sum().values / len(datasets["x_train"]), axis=0
        ),
        columns=list(datasets["x_train"].columns),
    ).to_csv(os.path.join(args.output_dir, "FLIRT_features_missingness.csv"))


def compute_datasets_relative_size(
    args, pretext_train_ds: DataLoader, pretext_val_ds: DataLoader, data: t.Dict
):
    filenames = list(pretext_train_ds.dataset.filenames) + list(
        pretext_val_ds.dataset.filenames
    )
    l = len(filenames)
    filenames = list(set(filenames).difference(set(data["x_train"])))
    print(
        f"{l-len(filenames)} segments from barcelona dataset appear both in target task training set and pre-text task training set"
    )
    collection_names, counts = np.unique(
        [p.split("/")[3] for p in filenames], return_counts=True
    )
    d = pd.DataFrame(
        {
            "datasets": collection_names,
            "segments_no": counts,
            "percentage": 100 * counts / np.sum(counts),
        }
    )
    d.to_csv(os.path.join(args.output_dir, "datasets_relative_size.csv"))


def target_dataset_info(
    data: t.Dict, train_ds: DataLoader, val_ds: DataLoader, test_ds: DataLoader
):
    print(
        f"Seg # train/val/test -> {len(train_ds.dataset.filenames)}/{len(val_ds.dataset.filenames)}/{len(test_ds.dataset.filenames)}"
    )

    def _extract_session_code(filename):
        parts = filename.split("/")
        return "/".join(parts[-3:-1])

    session_codes = np.unique(
        [_extract_session_code(string) for string in train_ds.dataset.filenames]
        + [_extract_session_code(string) for string in val_ds.dataset.filenames]
        + [_extract_session_code(string) for string in test_ds.dataset.filenames]
    )
    d = {"wake": 0, "sleep": 0, "off-body": 0}
    for session in session_codes:
        for status in d.keys():
            d[status] += data["ds_info"]["wake_sleep_off"][session][status]
    d = {k: v // (60 * 60) for k, v in d.items()}
    print(f"Hours by status: {d}")
