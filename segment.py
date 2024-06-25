import argparse
import datetime
import math
import pickle
import typing as t
import warnings
from functools import partial
from shutil import rmtree

import biosppy
import flirt
import pandas as pd
from tqdm.contrib import concurrent

from timebase.data import filter_data
from timebase.data.static import *
from timebase.utils import h5
from timebase.utils.utils import set_random_seed
from timebase.utils.utils import update_dict


def extract_features(
    args,
    features: t.Dict,
    segments_unix_t0: np.array,
    ibi: np.ndarray,
    recording_unix_t0: t.Dict,
):
    if not np.isnan(ibi).any():
        timestamps_beats = pd.to_datetime(
            ibi[:, 0] + recording_unix_t0["IBI"], unit="s", origin="unix"
        )
    features_container = []
    warnings.filterwarnings(action="ignore", category=UserWarning)

    for i in range(len(segments_unix_t0)):
        # EDA
        eda = pd.DataFrame(
            data=features["EDA"][i],
            columns=["eda"],
            dtype=np.float64,
        )
        eda_timestamps = pd.to_datetime(
            segments_unix_t0[i], unit="s", origin="unix"
        ) + np.arange(len(eda)) * datetime.timedelta(seconds=CHANNELS_FREQ["EDA"] ** -1)
        eda = eda.set_index(pd.DatetimeIndex(data=eda_timestamps, name="datetime"))
        try:
            eda_features = flirt.eda.get_eda_features(
                data=eda["eda"],
                data_frequency=CHANNELS_FREQ["EDA"],
                window_length=args.segment_length,
                window_step_size=args.segment_length,
            )
            if not eda_features.shape[-1] == len(FLIRT_EDA):
                eda_features = np.empty(shape=[1, len(FLIRT_HRV)])
                eda_features.fill(np.nan)
                eda_features = pd.DataFrame(data=eda_features, columns=FLIRT_EDA)
        except:
            eda_features = np.empty(shape=[1, len(FLIRT_HRV)])
            eda_features.fill(np.nan)
            eda_features = pd.DataFrame(data=eda_features, columns=FLIRT_EDA)

        # ACC
        acc = pd.DataFrame(
            data=np.concatenate(
                [
                    np.expand_dims(axis, axis=1)
                    for axis in [
                        features["ACC_x"][i],
                        features["ACC_y"][i],
                        features["ACC_z"][i],
                    ]
                ],
                axis=1,
            ),
            columns=["acc_x", "acc_y", "acc_z"],
            dtype=np.float64,
        )
        # reverse transformation to g values
        acc = (acc * 128) / 2
        acc_timestamps = pd.to_datetime(
            segments_unix_t0[i], unit="s", origin="unix"
        ) + np.arange(len(acc)) * datetime.timedelta(
            seconds=CHANNELS_FREQ["ACC_x"] ** -1
        )
        acc = acc.set_index(pd.DatetimeIndex(data=acc_timestamps, name="datetime"))
        try:
            acc_features = flirt.acc.get_acc_features(
                data=acc,
                data_frequency=CHANNELS_FREQ["ACC_x"],
                window_length=args.segment_length,
                window_step_size=args.segment_length,
            )
            if not acc_features.shape[-1] == len(FLIRT_ACC):
                acc_features = np.empty(shape=[1, len(FLIRT_ACC)])
                acc_features.fill(np.nan)
                acc_features = pd.DataFrame(data=acc_features, columns=FLIRT_ACC)
        except:
            acc_features = np.empty(shape=[1, len(FLIRT_ACC)])
            acc_features.fill(np.nan)
            acc_features = pd.DataFrame(data=acc_features, columns=FLIRT_ACC)

        # HRV
        if (args.from_bvp2ibi_mode == 0) and (not np.isnan(ibi).any()):
            segment_start = datetime.datetime.fromtimestamp(segments_unix_t0[i])
            segment_end = segment_start + datetime.timedelta(
                seconds=args.segment_length
            )
            ibi_segment = timestamps_beats[
                (timestamps_beats >= segment_start) & (timestamps_beats <= segment_end)
            ]
            ibi = np.around(np.diff(ibi_segment).astype(np.int64) / 10**6, decimals=3)
            df_ibi = pd.DataFrame(data=ibi, columns=["ibi"]).set_index(
                pd.DatetimeIndex(data=ibi_segment[1:], tz="UTC", name="datetime")
            )
        else:
            # Signal time axis reference (seconds):
            # https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-bvp
            ts, filtered, onsets, heart_rate_ts, heart_rate = biosppy.signals.bvp.bvp(
                signal=features["BVP"][i],
                sampling_rate=CHANNELS_FREQ["BVP"],
                show=False,
            )
            # interpulse interval, pulse rate variability:
            # https://www.kubios.com/hrv-time-series/
            ipi = np.diff(ts[onsets]) * 1000
            ipi_timestamps = pd.to_datetime(
                segments_unix_t0[i], unit="s", origin="unix"
            ) + np.array([datetime.timedelta(milliseconds=ms) for ms in ipi])
            df_ibi = pd.DataFrame(data=ipi, columns=["ibi"])
            df_ibi = df_ibi.set_index(
                pd.DatetimeIndex(data=ipi_timestamps, name="datetime")
            )
        try:
            hrv_features = flirt.hrv.get_hrv_features(
                data=df_ibi["ibi"],
                window_length=args.segment_length,
                window_step_size=args.segment_length,
                domains=["td", "fd", "nl", "stat"],
            )
            if not hrv_features.shape[-1] == len(FLIRT_HRV):
                hrv_features = np.empty(shape=[1, len(FLIRT_HRV)])
                hrv_features.fill(np.nan)
                hrv_features = pd.DataFrame(data=hrv_features, columns=FLIRT_HRV)
        except:
            hrv_features = np.empty(shape=[1, len(FLIRT_HRV)])
            hrv_features.fill(np.nan)
            hrv_features = pd.DataFrame(data=hrv_features, columns=FLIRT_HRV)

        # TEMP
        # FLIRT does not extract any feature for temperature, so we extract
        # the temperature average and std across the segment
        temp = features["TEMP"][i]
        temp_features = np.concatenate(
            (np.array(np.mean(temp), ndmin=2), np.array(np.std(temp), ndmin=2)), axis=1
        )
        temp_features = pd.DataFrame(data=temp_features, columns=FLIRT_TEMP)
        features_container.append(
            np.concatenate(
                (
                    eda_features.values,
                    acc_features.values,
                    hrv_features.values,
                    temp_features.values,
                ),
                axis=1,
            )
        )

    features["FLIRT"] = np.concatenate(features_container, axis=0)


def segmentation(
    args,
    recording_path: str,
    channel_names: t.List,
    channel_freq: t.Dict[str, int],
    unix_t0: int,
    mask: np.ndarray,
) -> (t.Dict[str, np.ndarray], int):
    """
    Segment preprocessed features along the temporal dimension into
    N non-overlapping segments where each segment has size args.segment_length
    Return:
        data: t.Dict[str, np.ndarray]
                dictionary of np.ndarray, where the keys are the channels
                and each np.ndarray are in shape (num. segment, window size)
        size: int, number of extracted segments
    """
    assert (segment_length := args.segment_length) > 0
    if args.segmentation_mode == 1:
        assert segment_length % (step_size := args.step_size) == 0
    channels = [channel for channel in channel_names if channel != "IBI"]
    session_data = {k: h5.get(recording_path, k) for k in channels}
    channel_segments = {c: [] for c in channels}
    channel_segments["unix_t0"] = []
    # segment each channel using a sliding window that space out equally
    for i, channel in enumerate(channels):
        window_samples = segment_length * channel_freq[channel]
        recording = session_data[channel] * np.repeat(
            mask, repeats=channel_freq[channel]
        )
        # list of sub-arrays from recording array with no nan values
        sub_recs = [
            recording[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(recording))
        ]
        if i == 0:
            timestamps = unix_t0 + np.arange(len(recording)) * (
                datetime.timedelta(seconds=channel_freq[channel] ** -1).microseconds
                / 1e6
            )
            sub_timestamps = [
                timestamps[s]
                for s in np.ma.clump_unmasked(np.ma.masked_invalid(recording))
            ]
        # segment sub-recordings independently of each other
        for sub_i, sub_rec in enumerate(sub_recs):
            # not-overlapping segments
            if args.segmentation_mode == 0:
                num_segments = math.floor(len(sub_rec) / window_samples)
                if num_segments:
                    indexes = np.linspace(
                        start=0,
                        stop=len(sub_rec) - window_samples,
                        num=num_segments,
                        dtype=int,
                    )
                    channel_segments[channel].extend(
                        [sub_rec[idx : idx + window_samples, ...] for idx in indexes]
                    )
                    if i == 0:
                        channel_segments["unix_t0"].extend(
                            [sub_timestamps[sub_i][idx, ...] for idx in indexes]
                        )
            # sliding window
            else:
                step_samples = step_size * channel_freq[channel]
                # calculate the total number of windows in sub-recording
                num_windows = (len(sub_rec) - window_samples) // step_samples + 1
                if num_windows:
                    for idx in range(num_windows):
                        start_idx = idx * step_samples
                        end_idx = start_idx + window_samples
                        channel_segments[channel].append(sub_rec[start_idx:end_idx])
                    if i == 0:
                        for idx in range(num_windows):
                            start_idx = idx * step_samples
                            channel_segments["unix_t0"].append(
                                sub_timestamps[sub_i][start_idx, ...]
                            )

    num_channel_segments = [len(s) for s in channel_segments.values()]
    assert (
        len(set(num_channel_segments)) == 1
    ), "all channels must have equal length after segmentation"

    data = {c: np.asarray(r) for c, r in channel_segments.items()}
    return data, num_channel_segments[0]


def process_recording(args, metadata: t.Dict, session_id: str):
    recording_path = os.path.join(args.data_dir, session_id, "channels.h5")
    session_label = h5.get(recording_path, "labels")
    # wake = 0, sleep = 1, can't tell = 2
    sleep_wake_mask = h5.get(recording_path, "SLEEP")
    sleep_wake = {"wake": 0, "sleep": 1}
    features, sleep_status = {}, []
    for k, v in sleep_wake.items():
        mask = np.where(sleep_wake_mask != v, np.nan, 1)
        # resample mask so that each mask entry maps to a wall-time second
        mask = np.reshape(
            mask,
            newshape=(
                -1,
                metadata["sessions_info"][session_id]["sampling_rates"]["SLEEP"],
            ),
            order="C",
        )
        mask = np.where(np.isnan(np.sum(mask, axis=1)), np.nan, 1)
        session_data, num_segments = segmentation(
            args,
            recording_path=recording_path,
            channel_names=metadata["sessions_info"][session_id]["channel_names"],
            channel_freq=metadata["sessions_info"][session_id]["sampling_rates"],
            unix_t0=metadata["sessions_info"][session_id]["unix_t0"]["HR"],
            mask=mask,
        )
        if num_segments:
            update_dict(target=features, source=session_data)
            sleep_status.extend([v] * num_segments)

    if not len(sleep_status):
        if args.verbose == 1:
            print(f"Session {session_id} gave no segments.")
        return None

    session_output_dir = os.path.join(args.output_dir, str(session_id))
    if not os.path.isdir(session_output_dir):
        os.makedirs(session_output_dir)

    wake_sleep_off = {}
    for k, v in SLEEP_DICT.items():
        # sleep_wake_mask sampled at 32Hz (ACC sampling frequency)
        secs_in_status = (
            len(np.where(sleep_wake_mask == k)[0]) // CHANNELS_FREQ["ACC_x"]
        )
        wake_sleep_off[v] = secs_in_status

    features = {k: np.concatenate(v, axis=0) for k, v in features.items()}
    segments_unix_t0 = features["unix_t0"]
    del features["unix_t0"]
    if args.flirt:
        if not np.isnan(session_label).any():
            extract_features(
                args,
                features=features,
                segments_unix_t0=segments_unix_t0,
                ibi=h5.get(recording_path, "IBI"),
                recording_unix_t0=metadata["sessions_info"][str(session_id)]["unix_t0"],
            )
    session_paths = []
    for n in range(len(sleep_status)):
        filename = os.path.join(session_output_dir, f"{n}.h5")
        segment = {k: v[n] for k, v in features.items()}
        h5.write(filename=filename, content=segment, overwrite=True)
        session_paths.append(filename)

    session_paths = np.array(session_paths, dtype=str)
    session_labels = np.tile(session_label, reps=(len(sleep_status), 1))

    return {
        "paths": session_paths,
        "labels": session_labels,
        "segments_unix_t0": segments_unix_t0,
        "sleep_status": sleep_status,
        "wake_sleep_off": wake_sleep_off,
    }


def segmentation_wrapper(args, metadata, session_id):
    results = process_recording(args, metadata, session_id)
    return results


def main(args):
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir {args.data_dir} not found.")
    if os.path.isdir(args.output_dir):
        if args.overwrite:
            rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"output_dir {args.output_dir} already exists. Add --overwrite "
                f" flag to overwrite the existing preprocessed data."
            )
    os.makedirs(args.output_dir)

    set_random_seed(args.seed)

    filename = os.path.join(args.data_dir, "metadata.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"data_dir {filename} not found.")
    metadata = pickle.load(open(filename, "rb"))

    (
        sessions_paths,
        sessions_labels,
        sessions_sleep_status,
        sessions_segments_unix_t0,
    ) = ([], [], [], [])
    metadata["ds_info"] = {"segment_length": args.segment_length}
    metadata["ds_info"]["wake_sleep_off"] = {}
    if args.segmentation_mode:
        metadata["ds_info"]["step_size"] = args.step_size
    metadata["ds_info"]["invalid_sessions_upon_segmentation"] = []

    results = concurrent.process_map(
        partial(segmentation_wrapper, args, metadata),
        metadata["sessions_info"].keys(),
        max_workers=args.num_workers,
        chunksize=args.chunksize,
        desc="Segmenting",
    )
    for i, session_id in enumerate(metadata["sessions_info"].keys()):
        result = results[i]
        # result = process_recording(args, metadata=metadata, session_id=str(session_id))
        if result is None:
            metadata["ds_info"]["invalid_sessions_upon_segmentation"].append(session_id)
            continue
        sessions_paths.append(result["paths"])
        sessions_labels.append(result["labels"])
        sessions_sleep_status.append(result["sleep_status"])
        sessions_segments_unix_t0.append(result["segments_unix_t0"])
        metadata["ds_info"]["wake_sleep_off"][session_id] = result["wake_sleep_off"]

    # joint metadata from all sessions
    metadata["sessions_paths"] = np.concatenate(sessions_paths, axis=0)
    metadata["sessions_labels"] = np.concatenate(sessions_labels, axis=0)
    metadata["sessions_labels"] = {
        k: metadata["sessions_labels"][:, i] for i, k in enumerate(LABEL_COLS)
    }
    # 1) some recording IDs collected in Barcelona are split across multiple
    # sessions -> assign the same recording ID throughout
    # 2) some subjects from unlabelled_data have multiple recordings,
    # we assume that the underlying semantics with respect to psychiatric
    # phenotypes do not change -> assign the same recording ID throughout
    metadata["recording_id"] = filter_data.set_unique_recording_id(args, metadata)
    metadata["sessions_sleep_status"] = np.concatenate(sessions_sleep_status, axis=0)
    metadata["sessions_segments_unix_t0"] = np.concatenate(
        sessions_segments_unix_t0, axis=0
    )

    with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as file:
        pickle.dump(metadata, file)

    print(f"Saved segmented data to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/preprocessed/unsegmented",
        help="path to directory with raw data in zip files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory to store dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing preprocessed directory",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=1234)

    # segmentation configuration
    parser.add_argument(
        "--segmentation_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="control which plots are printed"
        "0) Given a segment length, non-overlapping segments are extracted"
        "1) Given a segment length and a step-size, a sliding window is used "
        "to perform segmentation",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=2**9,
        help="segmentation window length in seconds",
    )
    temp_args = parser.parse_known_args()[0]
    if temp_args.segmentation_mode == 1:
        parser.add_argument(
            "--step_size",
            type=int,
            default=2**6,
            help="segmentation window length in seconds",
        )
    del temp_args
    parser.add_argument(
        "--flirt",
        action="store_true",
        help="extract features with FLIRT on labelled sessions only",
    )
    parser.add_argument(
        "--from_bvp2ibi_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help=""
        "0) Use Empatica IBI (provided as part of the E4 output and "
        "derived through a propriety algorithm. "
        "1) Compute IBI from BVP with bioppsy open-source package",
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--chunksize", type=int, default=1)
    main(parser.parse_args())
