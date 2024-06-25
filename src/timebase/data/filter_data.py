import re
import typing as t

import pandas as pd
from biopsykit.signals.imu.activity_counts import ActivityCounts
from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection
from biopsykit.utils.array_handling import accumulate_array

from timebase.data.static import *
from timebase.utils.utils import get_sequences_boundaries_index


def set_unique_recording_id(args, metadata: t.Dict[str, t.Any]):
    """
    It is possible that a single session (i.e. a given Sub_ID at a given time T)
    has multiple recordings (i.e. the E4 starts and stops recording multiple
    times), we therefore create unique recording IDs for these sub-recordings
    which belong to a single session.

    Similarly, for unlabelled data, whenever a single individual has multiple
    sessions (e.g. taken at subsequent time points) we assign the same
    recording_id to such sessions, this is because we assume that the
    semantic does not change with respect to mood episode status

    np.unique(recording_id) < np.unique(session_id)
    """
    rec_ids = np.array(
        [
            re.search(rf"{args.output_dir}/(.*?)/\d+\.h5", path).group(1)
            for path in metadata["sessions_paths"]
        ]
    )
    rec_ids = np.array(
        [
            rec_id.rsplit("/", 1)[0]
            if ("barcelona" not in rec_id)
            and ("in-gauge_en-gage" not in rec_id)
            and (len(rec_id.rsplit("/")) > 2)
            else rec_id
            for rec_id in rec_ids
        ]
    )
    # see https://www.nature.com/articles/s41597-022-01347-w for how subject
    # id is coded in the "in-gauge_en-gage" study
    rec_ids = np.array(
        [
            os.path.join("in-gauge_en-gage", rec_id.rsplit("_", 1)[-1])
            if "in-gauge_en-gage" in rec_id
            else rec_id
            for rec_id in rec_ids
        ]
    )
    # UE4W https://zenodo.org/record/6898244 provides recording from a single
    # subject
    rec_ids = np.array(
        [
            os.path.join("ue4w", "participant_1")
            if rec_id.startswith("ue4w")
            else rec_id
            for rec_id in rec_ids
        ]
    )
    sub_id_col = metadata["sessions_labels"]["Sub_ID"]
    time_col = metadata["sessions_labels"]["time"]

    for sub_id in np.unique(np.unique(sub_id_col[~np.isnan(sub_id_col)])):
        for time in np.unique(time_col[np.where(sub_id_col == sub_id)[0]]):
            mask = np.where((sub_id_col == sub_id) & (time_col == time))[0]
            unique_recordings = np.unique(
                metadata["sessions_labels"]["Session_Code"][mask]
            )
            if len(unique_recordings) > 1:
                names = "-".join([str(int(r)) for r in unique_recordings])
                rec_ids[mask] = "barcelona/" + names
    return rec_ids


def detect_loss_of_contact(x):
    threshold, tolerance = 0.05, 0.5
    x = np.array(x)
    ratio = np.sum(x < threshold) / len(x)
    if ratio > threshold:
        return 0
    else:
        return 1


def rule2_fun(x):
    x = np.array(x)
    signal_change_ratio = np.abs((x[-1] - x[0]) / (x[0] + np.finfo(np.float32).eps))
    if (signal_change_ratio > 0.2) or (signal_change_ratio < 0.1):
        return 1
    else:
        return 0


def rule3_fun(x):
    x = np.array(x)
    min_max_diff = np.abs(np.max(x) - np.min(x))
    if min_max_diff < 0.01:
        return 1
    else:
        return 0


def minutes_qc_fun(x):
    x = np.array(x)
    eda_sampling_rate = 4
    seconds_by_samples = np.reshape(x, (-1, eda_sampling_rate), order="C")
    seconds = np.sum(seconds_by_samples, axis=1).astype(bool).astype(int)

    count = (seconds != 0).sum()
    ratio = count / len(seconds)
    if ratio > 0.5:
        return 1
    else:
        return 0


def hours_qc_fun(x):
    x = np.array(x)
    count = (x != 0).sum()
    ratio = count / len(x)
    if ratio > 0.5:
        return 1
    else:
        return 0


def scripps_clinic_algorithm(
    indexes: t.List, df_acc: pd.DataFrame, timestamps: np.ndarray, acc_freq: int
):
    # https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2869.2010.00835.x
    # initiate sleep-wake mask
    # wake = 0, sleep = 1, can't tell = 2
    sleep_wake_mask = pd.DataFrame(
        data=np.zeros(df_acc.shape[0]) + 2, columns=["sleep-wake"]
    )
    sleep_wake_mask = sleep_wake_mask.set_index(
        pd.DatetimeIndex(data=timestamps, tz="UTC", name="time")
    )
    epoch_length = 60  # seconds
    ac = ActivityCounts(acc_freq)
    sw = SleepWakeDetection(algorithm_type="scripps_clinic")
    for idx in indexes:
        df_acc_segment = df_acc.iloc[idx[0] : idx[1] + 1]
        df_ac = ac.calculate(df_acc_segment)
        df_ac = accumulate_array(df_ac, 1, 1 / epoch_length)
        df_sw = sw.predict(df_ac)
        sleep_wake_mask.iloc[
            (sleep_wake_mask.index >= df_sw.index[0])
            & (sleep_wake_mask.index <= df_sw.index[-1])
        ] = 0
        idxs = get_sequences_boundaries_index(arr=df_sw["sleep_wake"], value=1)
        for i in idxs:
            start, stop = df_sw.index[i[0]], df_sw.index[i[1]]
            sleep_wake_mask.iloc[
                (sleep_wake_mask.index >= start) & (sleep_wake_mask.index <= stop)
            ] = 1
    return np.array(sleep_wake_mask.iloc[:, 0])


def van_hees_algorithm(
    indexes: t.List, df_acc: pd.DataFrame, timestamps: np.ndarray, acc_freq: int
):
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142533
    # initiate sleep-wake mask
    # wake = 0, sleep = 1, can't tell = 2
    sleep_wake_mask = pd.DataFrame(
        data=np.zeros(df_acc.shape[0]) + 2, columns=["sleep-wake"]
    )
    sleep_wake_mask = sleep_wake_mask.set_index(
        pd.DatetimeIndex(data=timestamps, tz="UTC", name="time")
    )
    for idx in indexes:
        df_acc_segment = df_acc.iloc[idx[0] : idx[1] + 1]
        # rolling median 5 second
        df_acc_segment_median = df_acc_segment.rolling(
            int(5 * acc_freq), min_periods=0
        ).median()
        angle = np.arctan(
            df_acc_segment_median["acc_z"]
            / (
                (
                    df_acc_segment_median["acc_x"] ** 2
                    + df_acc_segment_median["acc_y"] ** 2
                )
                ** 0.5
            )
        ) * (180.0 / np.pi)
        # get 5-second average
        angle = angle.resample("5s").mean().fillna(0)
        angle_changes = np.abs(angle - angle.shift(1))
        angle_changes = angle_changes.iloc[1:] < 5
        sleep_wake_mask.iloc[
            (sleep_wake_mask.index >= angle_changes.index[0])
            & (sleep_wake_mask.index <= angle_changes.index[-1])
        ] = 0
        idxs = get_sequences_boundaries_index(arr=angle_changes, value=1)
        for i in idxs:
            start, stop = angle_changes.index[i[0]], angle_changes.index[i[1]]
            if (stop - start) > pd.Timedelta(minutes=5):
                sleep_wake_mask.iloc[
                    (sleep_wake_mask.index >= start) & (sleep_wake_mask.index <= stop)
                ] = 1

    return np.array(sleep_wake_mask.iloc[:, 0])


def artifactual_data_removal(eda: np.ndarray, session_info: t.Dict):
    session_info["sampling_rates"]["ARTIFACT"] = session_info["sampling_rates"]["EDA"]
    session_info["mask_names"].append("ARTIFACT")
    # https://pubmed.ncbi.nlm.nih.gov/35278938/
    # artifact = 1, valid = 0
    # 1) Label samples where EDA < 0.05 micro-siemens as artifact

    rule1 = np.where(eda < 0.05, 1, 0)

    # 2) If the change from last to first value sampled within a second is
    # either greater than 20% or lower than 10% of the first sampled value,
    # mark that second artifact

    rule2 = (
        pd.Series(eda)
        .rolling(
            window=session_info["sampling_rates"]["ARTIFACT"],
            center=True,
            min_periods=1,
        )
        .apply(rule2_fun)
    )
    rule2 = np.array(rule2)

    # 3) If the difference between the maximum and minium value inside 1-min
    # window is smaller than 0.01, mark that minute as artifact

    rule3 = (
        pd.Series(eda)
        .rolling(
            window=session_info["sampling_rates"]["ARTIFACT"] * 60,
            center=True,
            min_periods=1,
        )
        .apply(rule3_fun)
    )
    rule3 = np.array(rule3)

    # a) If more than 50% artifacts under any rule within a minute, set that
    # entire minute as artifact

    artifact_by_any_rule = (rule1 + rule2 + rule3).astype(bool).astype(int)
    minutes_artifact = (
        pd.Series(artifact_by_any_rule)
        .rolling(
            window=session_info["sampling_rates"]["ARTIFACT"] * 60,
            center=True,
            min_periods=session_info["sampling_rates"]["ARTIFACT"] * 60,
        )
        .apply(minutes_qc_fun)
    )
    minutes_artifact = np.where(minutes_artifact.isna(), 1, minutes_artifact)

    # b) If more than 50% artifacts under any rule within an hour, set that
    # entire hour as artifact

    artifact_by_any_rule = (
        (rule1 + rule2 + rule3 + minutes_artifact).astype(bool).astype(int)
    )

    hours_artifact = (
        pd.Series(artifact_by_any_rule)
        .rolling(
            window=session_info["sampling_rates"]["ARTIFACT"] * 60 * 60,
            center=True,
            min_periods=session_info["sampling_rates"]["ARTIFACT"] * 60 * 60,
        )
        .apply(hours_qc_fun)
    )
    hours_artifact = np.where(hours_artifact.isna(), 1, minutes_artifact)
    # artifact = 1, valid = 0 -> artifact = 0, valid = 1
    artifact_mask = np.invert(
        (rule1 + rule2 + rule3 + minutes_artifact + hours_artifact).astype(bool)
    ).astype(int)

    return artifact_mask
