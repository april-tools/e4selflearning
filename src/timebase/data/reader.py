import pickle
import typing as t
from itertools import chain

import numpy as np
import pandas as pd
import sklearn.utils
import wandb
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from timebase.data import utils
from timebase.data.dataset import ClassificationDataset
from timebase.data.dataset import ContrastiveDataset
from timebase.data.dataset import DiagnosticsDataset
from timebase.data.dataset import ImputationDataset
from timebase.data.dataset import RecIDSampler
from timebase.data.dataset import TransformationDataset
from timebase.data.static import *
from timebase.utils import h5
from timebase.utils import plots
from timebase.utils import tensorboard
from timebase.utils.utils import load_args


def compute_statistics(
    args,
    data: t.Dict[str, t.Any],
):
    """
    Compute the min, max, mean, median, std, iqr of the (pre-)training set
    """

    channels = list(args.input_shapes.keys())
    stats_filename = os.path.join(args.dataset, "stats.pkl")
    if not os.path.exists(stats_filename):
        with open(stats_filename, "wb") as file:
            pickle.dump({}, file)
    with open(stats_filename, "rb") as file:
        stats_dict = pickle.load(file)
    if args.task_mode == 0:
        # --filter_collections and --unlabelled_data_resampling_percentage
        # are used for ablation analyses
        if args.filter_collections and args.unlabelled_data_resampling_percentage < 1:
            raise ValueError(
                "--unlabelled_data_resampling_percentage and --filter_collections illegal "
                "combination. Filter collections only when "
                "--unlabelled_data_resampling_percentage 1 and downsize pre-training set only "
                "when --filter_collections None"
            )
        if args.filter_collections:
            assert (
                args.exclude_anomalies == False
            ), "--exclude_anomalies should be False when using --filter_collections"
            selected_collections = "_".join(args.filter_collections)
            cache = os.path.join(
                args.dataset,
                f"stats_{selected_collections}_split_mode_{args.split_mode}_ssl",
            )
            if args.reuse_stats and cache in stats_dict:
                pass
            else:
                stats = {
                    channel: {
                        s: []
                        for s in [
                            "min",
                            "max",
                            "mean",
                            "second_moment",
                            "median",
                            "iqr",
                        ]
                    }
                    for channel in channels
                }
                collection_files = {
                    entry: i
                    for i, entry in enumerate(data["x_pre_train"])
                    if any(substring in entry for substring in args.filter_collections)
                    # target task train is always kept in pre-training
                    or entry in data["x_train"]
                }
                if not len(
                    set(collection_files.keys()).difference(set(data["x_train"]))
                ):
                    raise ValueError("No segments for the selected collection")
                utils.stats_in_tranches(
                    args,
                    files2read=np.array(list(collection_files.keys())),
                    stats=stats,
                    channels=channels,
                )
                stats["pre_train_indeces"] = np.array(list(collection_files.values()))
                stats_dict[cache] = stats
                with open(stats_filename, "wb") as file:
                    pickle.dump(stats_dict, file)
        elif args.unlabelled_data_resampling_percentage < 1:
            assert args.exclude_anomalies == False, (
                "--exclude_anomalies should be False when using "
                "--unlabelled_data_resampling_percentage"
            )
            cache = os.path.join(
                args.dataset,
                f"stats_size_{args.unlabelled_data_resampling_percentage}_split_mode"
                f"_{args.split_mode}_ssl",
            )
            if args.reuse_stats and cache in stats_dict:
                pass
            else:
                stats = {
                    channel: {
                        s: []
                        for s in [
                            "min",
                            "max",
                            "mean",
                            "second_moment",
                            "median",
                            "iqr",
                        ]
                    }
                    for channel in channels
                }
                unlabelled_data_indeces = np.array(
                    [
                        i
                        for i, f in enumerate(data["x_pre_train"])
                        if f not in data["x_train"]
                    ]
                )
                target_task_train_indeces = list(
                    set(np.arange(len(data["x_pre_train"]))).difference(
                        set(unlabelled_data_indeces)
                    )
                )
                # only target task train set is used for pre-training
                if args.unlabelled_data_resampling_percentage == 0:
                    downsized_collections = data["x_pre_train"][
                        target_task_train_indeces
                    ]
                    indeces = target_task_train_indeces
                else:
                    # data['x_pre_train'] contains both the target task train set
                    # and unlabelled data; unlabelled data is herewith isolated
                    # and only a fraction equal to
                    # args.unlabelled_data_resampling_percentage is retained;
                    # note that the target task train set is always kept as
                    # part of the pre_training set
                    collections = np.array(
                        [
                            collection.split("/", 4)[3]
                            for collection in data["x_pre_train"]
                        ]
                    )
                    downsized_collections, _, indeces, _ = train_test_split(
                        data["x_pre_train"][unlabelled_data_indeces],
                        unlabelled_data_indeces,
                        stratify=collections[unlabelled_data_indeces],
                        test_size=1 - args.unlabelled_data_resampling_percentage,
                        random_state=args.seed,
                    )
                    downsized_collections = np.array(
                        list(downsized_collections)
                        + list(data["x_pre_train"][target_task_train_indeces])
                    )
                    indeces = np.array(list(indeces) + target_task_train_indeces)
                    assert set(downsized_collections) == set(
                        data["x_pre_train"][indeces]
                    )

                utils.stats_in_tranches(
                    args,
                    files2read=downsized_collections,
                    stats=stats,
                    channels=channels,
                )
                stats["pre_train_indeces"] = indeces
                stats_dict[cache] = stats
                with open(stats_filename, "wb") as file:
                    pickle.dump(stats_dict, file)
        else:
            cache = os.path.join(
                args.dataset,
                f"stats_split_mode_{args.split_mode}_ssl",
            )
            if args.exclude_anomalies:
                cache = cache + "_anomaly_detection"
            if args.e4selflearning:
                cache = cache + "_e4selflearning"
            if args.reuse_stats and cache in stats_dict:
                pass
            else:
                stats = {
                    channel: {
                        s: []
                        for s in [
                            "min",
                            "max",
                            "mean",
                            "second_moment",
                            "median",
                            "iqr",
                        ]
                    }
                    for channel in channels
                }
                utils.stats_in_tranches(
                    args,
                    files2read=data["x_pre_train"],
                    stats=stats,
                    channels=channels,
                )
                stats_dict[cache] = stats
                with open(stats_filename, "wb") as file:
                    pickle.dump(stats_dict, file)
    elif args.task_mode in (1, 2):
        if args.filter_collections:
            selected_collections = "_".join(args.filter_collections)
            cache = os.path.join(
                args.dataset,
                f"stats_{selected_collections}_split_mode_{args.split_mode}_ssl",
            )
        elif args.unlabelled_data_resampling_percentage < 1:
            cache = os.path.join(
                args.dataset,
                f"stats_size_{args.unlabelled_data_resampling_percentage}_split_mode"
                f"_{args.split_mode}_ssl",
            )
        else:
            cache = os.path.join(
                args.dataset,
                f"stats_split_mode_{args.split_mode}_ssl",
            )
            if args.exclude_anomalies:
                cache = cache + "_anomaly_detection"
            if args.reuse_stats and cache in stats_dict:
                pass
        if not cache in stats_dict:
            raise NotImplementedError(
                f"{cache} not found in stats.pkl, this should have been "
                f"generated during self-supervised pre-training"
            )
    else:
        cache = os.path.join(args.dataset, f"stats_split_mode_{args.split_mode}_sl")
        if args.reuse_stats and cache in stats_dict:
            pass
        else:
            if args.verbose:
                print("Compute dataset statistics...")
            channels = list(args.input_shapes.keys())
            stats = {
                channel: {
                    s: []
                    for s in [
                        "min",
                        "max",
                        "mean",
                        "second_moment",
                        "median",
                        "iqr",
                    ]
                }
                for channel in channels
            }
            utils.stats_in_tranches(
                args,
                files2read=data["x_train"],
                channels=channels,
                stats=stats,
            )
            stats_dict[cache] = stats
            with open(stats_filename, "wb") as file:
                pickle.dump(stats_dict, file)
    if getattr(args, "include_hr", False):
        if "HR" not in stats_dict[cache]:
            raise ValueError("Please re-run without --reuse_stats")
    return stats_dict[cache]


def split_into_sets(
    args, y: t.Dict, sleep_status: np.ndarray, recording_id: np.ndarray
):
    if args.e4selflearning:
        # pre-pretraining is conducted on segments marked as wake,
        # i.e. sleep_status == 1
        return {
            "pre_train": np.where(sleep_status == 0)[0],
        }
    # 0: non-cases (euthymia), 1: cases (acute mood episode)
    cases_stati = [
        v for k, v in DICT_STATE.items() if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
    ]
    cases_mask = (
        (sleep_status == 0)
        & pd.Series(y["status"]).isin(cases_stati)
        & (y["time"] == 0)
        & ((y["YMRS_SUM"] > 7) | (y["HDRS_SUM"] > 7))
    )
    controls_stati = [v for k, v in DICT_STATE.items() if k in ["Eu_BD", "Eu_MDD"]]
    controls_mask = (
        (sleep_status == 0)
        & pd.Series(y["status"]).isin(controls_stati)
        & ((y["YMRS_SUM"] <= 7) & (y["HDRS_SUM"] <= 7))
    )

    # sessions having less than n segments are dropped
    def _filter_short_sessions(mask, y):
        sub_ids, segment_counts = np.unique(y["Sub_ID"][mask], return_counts=True)
        too_short = np.where(segment_counts < 15)[0]
        if len(too_short):
            for i in too_short:
                mask[y["Sub_ID"] == sub_ids[i]] = False

    _filter_short_sessions(mask=cases_mask, y=y)
    _filter_short_sessions(mask=controls_mask, y=y)

    # Do no allow the same subject to appear across different stati,
    # we force cases and controls to form two disjoint groups in terms of Sub_ID
    cross_status_ids = set(np.unique(y["Sub_ID"][controls_mask])).intersection(
        set(np.unique(y["Sub_ID"][cases_mask]))
    )
    if len(cross_status_ids):
        if len(np.unique(y["Sub_ID"][cases_mask])) > len(
            np.unique(y["Sub_ID"][controls_mask])
        ):
            for sub_id in cross_status_ids:
                cases_mask[np.where(y["Sub_ID"] == sub_id)[0]] = False
        else:
            for sub_id in cross_status_ids:
                controls_mask[np.where(y["Sub_ID"] == sub_id)[0]] = False
    assert (
        len(
            set(np.unique(y["Sub_ID"][controls_mask])).intersection(
                set(np.unique(y["Sub_ID"][cases_mask]))
            )
        )
        == 0
    )
    # select an equal number of cases and controls, if the number is not
    # originally equal then remove as many ids as needed starting from those
    # with fewer segments first
    cases_ids, cases_counts = np.unique(y["Sub_ID"][cases_mask], return_counts=True)
    controls_ids, controls_counts = np.unique(
        y["Sub_ID"][controls_mask], return_counts=True
    )
    class_counts_cases = dict(zip(cases_ids, cases_counts))
    class_counts_controls = dict(zip(controls_ids, controls_counts))
    if len(class_counts_cases) > len(class_counts_controls):
        num_ids_to_remove = len(class_counts_cases) - len(class_counts_controls)
        sorted_classes = sorted(class_counts_cases, key=class_counts_cases.get)
        cases_mask[
            pd.Series(y["Sub_ID"]).isin(sorted_classes[:num_ids_to_remove])
        ] = False
    elif len(class_counts_controls) > len(class_counts_cases):
        num_ids_to_remove = len(class_counts_controls) - len(class_counts_cases)
        sorted_classes = sorted(class_counts_controls, key=class_counts_controls.get)
        controls_mask[
            pd.Series(y["Sub_ID"]).isin(sorted_classes[:num_ids_to_remove])
        ] = False

    # Find case-control pairs minimizing the element-wise difference
    # between cases and controls segment number arrays. For each such pair
    # retain the number of segments of the smallest pair element.
    cases_ids, cases_counts = np.unique(y["Sub_ID"][cases_mask], return_counts=True)
    controls_ids, controls_counts = np.unique(
        y["Sub_ID"][controls_mask], return_counts=True
    )
    diff_matrix = np.abs(np.subtract.outer(cases_counts, controls_counts))
    row_indices, col_indices = linear_sum_assignment(diff_matrix)
    for (k_case, v_case), (k_control, v_control) in zip(
        dict(zip(cases_ids[row_indices], cases_counts[row_indices])).items(),
        dict(zip(controls_ids[col_indices], controls_counts[col_indices])).items(),
    ):
        num_segments_to_retain = np.minimum(v_case, v_control)
        idx_case = np.where((cases_mask == True) & (y["Sub_ID"] == k_case))[0]
        idx_control = np.where((controls_mask == True) & (y["Sub_ID"] == k_control))[0]
        cases_mask[idx_case[num_segments_to_retain:]] = False
        controls_mask[idx_control[num_segments_to_retain:]] = False

    match args.split_mode:
        case 0:
            # time-split
            def _split_time(mask, y):
                sub_ids = np.unique(y["Sub_ID"][mask])
                idx_dict = {"train": [], "val": [], "test": []}
                for sub_id in sub_ids:
                    idx = np.where((mask == True) & (y["Sub_ID"] == sub_id))[0]
                    train_idx, val_idx, test_idx = np.split(
                        idx,
                        [int(len(idx) * 0.70), int(len(idx) * 0.85)],
                    )
                    # remove segments sitting at the border between splits so
                    # that there are no segments with overlapping motifs
                    # across splits
                    if "step_size" in args.ds_info:
                        # when step_size is not among keys of args.ds_info
                        # this means that segments were already created with
                        # no overlapping motifs
                        ratio = (
                            args.ds_info["segment_length"] // args.ds_info["step_size"]
                        )
                        train_idx = train_idx[: -(ratio - 1)]
                        val_idx = val_idx[: -(ratio - 1)]
                    idx_dict["train"].extend(list(train_idx))
                    idx_dict["val"].extend(list(val_idx))
                    idx_dict["test"].extend(list(test_idx))
                return idx_dict

            idx_dict_cases = _split_time(mask=cases_mask, y=y)
            idx_dict_controls = _split_time(mask=controls_mask, y=y)
        case 1:
            # subject-split
            def _split_subjects(mask, y):
                sub_ids = np.unique(y["Sub_ID"][mask])
                sub_ids = sklearn.utils.shuffle(sub_ids, random_state=args.seed)
                train_ids, val_ids, test_ids = np.split(
                    sub_ids,
                    [int(len(sub_ids) * 0.70), int(len(sub_ids) * 0.85)],
                )
                names = ["train", "val", "test"]
                idx_dict = dict(zip(names, [[], [], []]))
                ids_dict = dict(zip(names, [train_ids, val_ids, test_ids]))
                for k, v in ids_dict.items():
                    for id in v:
                        idx = np.where((mask == True) & (y["Sub_ID"] == id))[0]
                        idx_dict[k].extend(idx)

                return idx_dict

            idx_dict_cases = _split_subjects(mask=cases_mask, y=y)
            idx_dict_controls = _split_subjects(mask=controls_mask, y=y)
        case _:
            raise NotImplementedError(f"split_mode {args.split_mode} not implemented.")

    # 0: pre-training, 1: primary task train set, 2: excluded
    pretrain_mask = np.zeros(shape=len(y["Sub_ID"]))
    # sleep segments are not used in any analysis and are removed from
    # pre-training
    pretrain_mask[sleep_status == 1] = 2

    # subjects appearing in validation or test set are excluded from
    # pre-training; note this is excluding all Ts from such subjects
    pretrain_mask[
        pd.Series(y["Sub_ID"]).isin(
            np.unique(
                y["Sub_ID"][
                    idx_dict_cases["val"]
                    + idx_dict_controls["val"]
                    + idx_dict_cases["test"]
                    + idx_dict_controls["test"]
                ]
            )
        )
    ] = 2

    # train set of the main task is also used for pre-training
    pretrain_mask[idx_dict_cases["train"] + idx_dict_controls["train"]] = 1

    if (hasattr(args, "pretext_task")) and (args.pretext_task == "contrastive"):
        ids, counts = np.unique(recording_id[pretrain_mask != 2], return_counts=True)
        to_remove = ids[np.where(counts < 2)[0]]
        if len(to_remove):
            if args.verbose == 2:
                print(
                    f"Recording IDs {list(to_remove)} dropped from "
                    f"pre-training in {args.pretext_task} as only one segment "
                    f"is available"
                )
            for id in to_remove:
                pretrain_mask[np.where(recording_id == id)] = 2

    def _check_no_shared_entries(*lists):
        sets = [set(lst) for lst in lists]

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if not sets[i].isdisjoint(sets[j]):
                    return False

        return True

    assert _check_no_shared_entries(
        idx_dict_cases["train"],
        idx_dict_controls["train"],
        idx_dict_cases["val"],
        idx_dict_controls["val"],
        idx_dict_cases["test"],
        idx_dict_controls["test"],
        "Leakage in train/val/test splits",
    )

    assert _check_no_shared_entries(
        list(np.where(pretrain_mask != 2)[0]),
        idx_dict_cases["val"]
        + idx_dict_controls["val"]
        + idx_dict_cases["test"]
        + idx_dict_controls["test"],
    ), "Validation or test data leaked into pre-training data"

    match args.split_mode:
        case 0:
            assert (
                set(
                    list(np.unique(y["Sub_ID"][idx_dict_cases["train"]]))
                    + list(np.unique(y["Sub_ID"][idx_dict_controls["train"]]))
                )
                == set(
                    list(np.unique(y["Sub_ID"][idx_dict_cases["val"]]))
                    + list(np.unique(y["Sub_ID"][idx_dict_controls["val"]]))
                )
                == set(
                    list(np.unique(y["Sub_ID"][idx_dict_cases["test"]]))
                    + list(np.unique(y["Sub_ID"][idx_dict_controls["test"]]))
                )
            ), "Sub_ID are not shared across train/val/test"
        case 1:
            assert _check_no_shared_entries(
                list(np.unique(y["Sub_ID"][idx_dict_cases["train"]]))
                + list(np.unique(y["Sub_ID"][idx_dict_controls["train"]])),
                list(np.unique(y["Sub_ID"][idx_dict_cases["val"]]))
                + list(np.unique(y["Sub_ID"][idx_dict_controls["val"]])),
                list(np.unique(y["Sub_ID"][idx_dict_cases["test"]]))
                + list(np.unique(y["Sub_ID"][idx_dict_controls["test"]])),
            ), "Train/val/test do not partition Sub_IDs"

    if args.task_mode in (0, 1, 2, 3, 4):
        if args.task_mode == 0:
            if args.pretext_task == "masked_prediction" and args.exclude_anomalies:
                pretrain_mask[pd.Series(y["status"]).isin(cases_stati)] = 2
        return {
            "pre_train": np.where(pretrain_mask != 2)[0],
            "train": np.array(idx_dict_cases["train"] + idx_dict_controls["train"]),
            "val": np.array(idx_dict_cases["val"] + idx_dict_controls["val"]),
            "test": np.array(idx_dict_cases["test"] + idx_dict_controls["test"]),
        }
    else:
        idx = {
            "pre_train": np.where(pretrain_mask != 2)[0],
            "test": np.array(idx_dict_cases["test"] + idx_dict_controls["test"]),
        }
        session_codes, counts = np.unique(recording_id[idx["test"]], return_counts=True)
        # Delete this block and uncomment the one below once memory allows it
        sleep = []
        for session_code, count in zip(session_codes, counts):
            sleep_idx = np.where(
                (
                    # wake: 0, sleep: 1, off-body: 2
                    (sleep_status == 1)
                    & (recording_id == session_code)
                )
                == True
            )[0]
            if len(sleep_idx) > count:
                sleep_idx = sleep_idx[-count:]
            sleep.extend(sleep_idx)
        idx["sleep_wake"] = np.array(sleep + list(idx["test"]))

        collections = [f.split("/")[0] for f in recording_id[idx["pre_train"]]]
        selected_indexes, n = {}, 1000
        for collection in set(collections):
            if collection != "barcelona":
                indexes = [
                    i
                    for i, c in zip(idx["pre_train"], recording_id[idx["pre_train"]])
                    if collection in c
                ]
                if len(indexes) < n:
                    selected_indexes[collection] = indexes
                else:
                    selected_indexes[collection] = np.random.choice(
                        indexes, n, replace=False
                    )
        idx["test"] = np.array(
            list(idx["test"])
            + list(chain.from_iterable([list(v) for v in selected_indexes.values()]))
        )

        return idx


def construct_dataset(args, data: t.Dict):
    """Construct feature-label pairs"""
    assert args.task_mode in (0, 1, 2, 3, 4, 9)
    idx = split_into_sets(
        args,
        y=data["sessions_labels"],
        sleep_status=data["sessions_sleep_status"],
        recording_id=data["recording_id"],
    )

    if args.task_mode in (0, 1, 2, 3, 4):
        data["x_pre_train"] = data["sessions_paths"][idx["pre_train"]]
        data["y_pre_train"] = {
            k: v[idx["pre_train"]] for k, v in data["sessions_labels"].items()
        }
        data["pre_train_recording_id"] = data["recording_id"][idx["pre_train"]]

        if not args.e4selflearning:
            data["x_train"] = data["sessions_paths"][idx["train"]]
            data["y_train"] = {
                k: v[idx["train"]] for k, v in data["sessions_labels"].items()
            }
            data["train_recording_id"] = data["recording_id"][idx["train"]]

            data["x_val"] = data["sessions_paths"][idx["val"]]
            data["y_val"] = {
                k: v[idx["val"]] for k, v in data["sessions_labels"].items()
            }
            data["val_recording_id"] = data["recording_id"][idx["val"]]

            data["x_test"] = data["sessions_paths"][idx["test"]]
            data["y_test"] = {
                k: v[idx["test"]] for k, v in data["sessions_labels"].items()
            }
            data["test_recording_id"] = data["recording_id"][idx["test"]]

    else:
        data["x_pre_train"] = data["sessions_paths"][idx["pre_train"]]
        data["y_pre_train"] = {
            k: v[idx["pre_train"]] for k, v in data["sessions_labels"].items()
        }
        data["pre_train_recording_id"] = data["recording_id"][idx["pre_train"]]
        data["pre_train_sleep_status"] = data["sessions_sleep_status"][idx["pre_train"]]

        data["x_test"] = data["sessions_paths"][idx["test"]]
        data["y_test"] = {k: v[idx["test"]] for k, v in data["sessions_labels"].items()}
        data["train_recording_id"] = data["recording_id"][idx["test"]]
        data["train_sleep_status"] = data["sessions_sleep_status"][idx["test"]]

        data["x_sleep_wake"] = data["sessions_paths"][idx["sleep_wake"]]
        data["y_sleep_wake"] = {
            k: v[idx["sleep_wake"]] for k, v in data["sessions_labels"].items()
        }
        data["sleep_wake_recording_id"] = data["recording_id"][idx["sleep_wake"]]
        data["sleep_wake_sleep_status"] = data["sessions_sleep_status"][
            idx["sleep_wake"]
        ]

    del (
        data["sessions_paths"],
        data["sessions_labels"],
        data["sessions_sleep_status"],
        data["sessions_segments_unix_t0"],
        data["recording_id"],
    )


def get_subject_ids(args, sub_id: np.ndarray):
    unique_ids = np.unique(sub_id)
    ids_renaming_dict = dict(zip(unique_ids, np.arange(len(unique_ids))))
    refactored_ids = np.array(pd.Series(sub_id).replace(ids_renaming_dict), dtype=int)
    if not hasattr(args, "num_train_subjects"):
        args.num_train_subjects = len(unique_ids)
    return refactored_ids


def split_pre_train(args, data: t.Dict[str, t.Any]):
    if "pre_train_indeces" in args.ds_info["stats"]:
        # This block is used for ablation studies with masked prediction
        # pre-training, which turned out to be the best performing SSL approach
        indeces = sklearn.utils.shuffle(
            args.ds_info["stats"]["pre_train_indeces"], random_state=args.seed
        )
        split_point = int(len(indeces) * 0.85)
        pretext_train_idx = indeces[:split_point]
        pretext_val_idx = indeces[split_point:]
        del args.ds_info["stats"]["pre_train_indeces"]
    elif args.task_mode == 9:
        collections = [f.split("/")[0] for f in data["pre_train_recording_id"]]
        selected_indexes, n = {}, 50
        for collection in set(collections):
            indexes = [i for i, x in enumerate(collections) if x == collection]
            selected_indexes[collection] = np.random.choice(indexes, n, replace=False)
        return None, list(
            chain.from_iterable([list(v) for v in selected_indexes.values()])
        )
    else:
        # rec_ids are partitioned into train and val
        rec_ids, counts = np.unique(data["pre_train_recording_id"], return_counts=True)
        rec_ids, counts = sklearn.utils.shuffle(rec_ids, counts, random_state=args.seed)
        train_weight, train_rec_ids = 0, []
        counts = counts / np.sum(counts)
        for rec_id, count in sorted(
            zip(rec_ids, counts), key=lambda x: x[1], reverse=True
        ):
            if train_weight < 0.85:
                train_rec_ids.append(rec_id)
                train_weight += count
        val_rec_ids = list(set(rec_ids).difference(set(train_rec_ids)))
        pretext_train_idx = [
            i
            for i, rec_id in enumerate(data["pre_train_recording_id"])
            if rec_id in train_rec_ids
        ]
        pretext_val_idx = [
            i
            for i, rec_id in enumerate(data["pre_train_recording_id"])
            if rec_id in val_rec_ids
        ]
    return pretext_train_idx, pretext_val_idx


def get_pre_training_dataset(
    args, data: t.Dict[str, t.Any], recording_id_str_to_num: t.Dict[str, int]
):
    # settings for DataLoader
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)
    args.ds_info["stats"] = compute_statistics(args, data=data)
    pretext_train_idx, pretext_val_idx = split_pre_train(args, data=data)
    match args.pretext_task:
        case "masked_prediction":
            path2masks = os.path.join(args.dataset, "masks.npz")
            if (not os.path.exists(path2masks)) or (args.overwrite_masks):
                utils.generate_masks(
                    args,
                    channel_freq=args.ds_info["channel_freq"],
                    segment_length=args.ds_info["segment_length"],
                )
            pretext_train_ds = DataLoader(
                ImputationDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_train_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                ),
                shuffle=True,
                **dataloader_kwargs,
            )
            pretext_val_ds = DataLoader(
                ImputationDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_val_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                ),
                **dataloader_kwargs,
            )
            if args.e4selflearning:
                pretext_test_ds = None
            else:
                pretext_test_ds = DataLoader(
                    ImputationDataset(
                        args,
                        filenames=data["x_val"],
                        stats=args.ds_info["stats"],
                        segment_length=data["ds_info"]["segment_length"],
                        channel_freq=data["ds_info"]["channel_freq"],
                    ),
                    **dataloader_kwargs,
                )
        case "transformation_prediction":
            pretext_train_ds = DataLoader(
                TransformationDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_train_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                ),
                shuffle=True,
                **dataloader_kwargs,
            )
            pretext_val_ds = DataLoader(
                TransformationDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_val_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                ),
                **dataloader_kwargs,
            )
            if args.e4selflearning:
                pretext_test_ds = None
            else:
                pretext_test_ds = DataLoader(
                    TransformationDataset(
                        args,
                        filenames=data["x_val"],
                        stats=args.ds_info["stats"],
                        segment_length=data["ds_info"]["segment_length"],
                        channel_freq=data["ds_info"]["channel_freq"],
                    ),
                    **dataloader_kwargs,
                )
        case "contrastive":
            del dataloader_kwargs["batch_size"]
            pretext_train_ds = DataLoader(
                ContrastiveDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_train_idx],
                    recording_id=data["pre_train_recording_id"][pretext_train_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                    batch_size=len(
                        np.unique(data["pre_train_recording_id"][pretext_train_idx])
                    ),
                ),
                batch_sampler=RecIDSampler(
                    dataset=data["pre_train_recording_id"][pretext_train_idx],
                    batch_size=len(
                        np.unique(data["pre_train_recording_id"][pretext_train_idx])
                    ),
                ),
                **dataloader_kwargs,
            )
            pretext_val_ds = DataLoader(
                ContrastiveDataset(
                    args,
                    filenames=data["x_pre_train"][pretext_val_idx],
                    recording_id=data["pre_train_recording_id"][pretext_val_idx],
                    stats=args.ds_info["stats"],
                    segment_length=data["ds_info"]["segment_length"],
                    channel_freq=data["ds_info"]["channel_freq"],
                    batch_size=len(
                        np.unique(data["pre_train_recording_id"][pretext_val_idx])
                    )
                    // 2
                    if args.batch_size
                    > len(np.unique(data["pre_train_recording_id"][pretext_val_idx]))
                    else args.batch_size,
                ),
                batch_sampler=RecIDSampler(
                    dataset=data["pre_train_recording_id"][pretext_val_idx],
                    batch_size=len(
                        np.unique(data["pre_train_recording_id"][pretext_val_idx])
                    )
                    // 2
                    if args.batch_size
                    > len(np.unique(data["pre_train_recording_id"][pretext_val_idx]))
                    else args.batch_size,
                ),
                **dataloader_kwargs,
            )
            if args.e4selflearning:
                pretext_test_ds = None
            else:
                pretext_test_ds = DataLoader(
                    ContrastiveDataset(
                        args,
                        filenames=data["x_val"],
                        recording_id=data["val_recording_id"],
                        stats=args.ds_info["stats"],
                        segment_length=data["ds_info"]["segment_length"],
                        channel_freq=data["ds_info"]["channel_freq"],
                        batch_size=len(np.unique(data["val_recording_id"])) // 2
                        if args.batch_size > len(np.unique(data["val_recording_id"]))
                        else args.batch_size,
                    ),
                    batch_sampler=RecIDSampler(
                        dataset=data["val_recording_id"],
                        batch_size=len(np.unique(data["val_recording_id"])) // 2
                        if args.batch_size > len(np.unique(data["val_recording_id"]))
                        else args.batch_size,
                    ),
                    **dataloader_kwargs,
                )
        case _:
            raise NotImplementedError(f"{args.pretext_task} not implemented.")
    if args.use_wandb:
        wandb.config.update({"segment_length": data["ds_info"]["segment_length"]})
    return (
        pretext_train_ds,
        pretext_val_ds,
        pretext_test_ds,
    )


def get_training_datasets(
    args, data: t.Dict[str, t.Any], recording_id_str_to_num: t.Dict[str, int]
):
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)
    args.ds_info["stats"] = compute_statistics(args, data=data)
    if args.task_mode in (1, 2):
        if args.unlabelled_data_resampling_percentage < 1 or args.filter_collections:
            del args.ds_info["stats"]["pre_train_indeces"]
    train_ds = DataLoader(
        ClassificationDataset(
            args,
            filenames=data["x_train"],
            labels=data["y_train"],
            rec_ids=data["train_recording_id"],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
            subject_ids=get_subject_ids(args, sub_id=data["y_train"]["Sub_ID"]),
        ),
        shuffle=True,
        **dataloader_kwargs,
    )
    val_ds = DataLoader(
        ClassificationDataset(
            args,
            filenames=data["x_val"],
            labels=data["y_val"],
            rec_ids=data["val_recording_id"],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
            subject_ids=get_subject_ids(args, sub_id=data["y_val"]["Sub_ID"]),
        ),
        shuffle=False,
        **dataloader_kwargs,
    )
    test_ds = DataLoader(
        ClassificationDataset(
            args,
            filenames=data["x_test"],
            labels=data["y_test"],
            rec_ids=data["test_recording_id"],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
            subject_ids=get_subject_ids(args, sub_id=data["y_test"]["Sub_ID"]),
        ),
        shuffle=False,
        **dataloader_kwargs,
    )
    if args.use_wandb:
        wandb.config.update({"segment_length": data["ds_info"]["segment_length"]})
    return train_ds, val_ds, test_ds


def get_training_datasets_cml(
    args, data: t.Dict[str, t.Any], recording_id_str_to_num: t.Dict[str, int]
):
    if not args.path2featurizer:
        partitions = ["train", "val", "test"]
        features_container = dict(zip(partitions, [[], [], []]))
        for partition in partitions:
            for path in data[f"x_{partition}"]:
                features = np.expand_dims(h5.get(path, "FLIRT"), axis=0)
                features_container[partition].append(features)
        datasets = {}
        for partition in partitions:
            datasets[f"x_{partition}"] = pd.DataFrame(
                data=np.concatenate(features_container[partition], axis=0),
                columns=FLIRT_EDA + FLIRT_ACC + FLIRT_HRV + FLIRT_TEMP,
            )
            datasets[f"targets_{partition}"] = np.isin(
                data[f"y_{partition}"]["status"],
                [
                    v
                    for k, v in DICT_STATE.items()
                    if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
                ],
            ).astype(int)
            if partition in ("val", "test"):
                datasets[f"labels_{partition}"] = data[f"y_{partition}"]
                datasets[f"metadata_{partition}"] = {
                    "segment_id": np.array(
                        [
                            int(os.path.basename(str(filename)).replace(".h5", ""))
                            for filename in data[f"x_{partition}"]
                        ]
                    ),
                    "session_id": np.array(
                        [
                            int(os.path.basename(os.path.dirname(filename)))
                            for filename in data[f"x_{partition}"]
                        ]
                    ),
                    "recording_id": data[f"{partition}_recording_id"],
                    "subject_id": get_subject_ids(
                        args, sub_id=data[f"y_{partition}"]["Sub_ID"]
                    ),
                }
        return datasets
    else:
        load_args(args, dir=args.path2featurizer)
        dataloader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        if args.device.type in ("cuda", "mps"):
            gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
            dataloader_kwargs.update(gpu_kwargs)
        args.task_mode = 2
        args.ds_info["stats"] = compute_statistics(args, data=data)
        train_ds = DataLoader(
            ClassificationDataset(
                args,
                filenames=data["x_train"],
                labels=data["y_train"],
                rec_ids=data["train_recording_id"],
                stats=args.ds_info["stats"],
                recording_id_str_to_num=recording_id_str_to_num,
                subject_ids=get_subject_ids(args, sub_id=data["y_train"]["Sub_ID"]),
            ),
            shuffle=True,
            **dataloader_kwargs,
        )
        val_ds = DataLoader(
            ClassificationDataset(
                args,
                filenames=data["x_val"],
                labels=data["y_val"],
                rec_ids=data["val_recording_id"],
                stats=args.ds_info["stats"],
                recording_id_str_to_num=recording_id_str_to_num,
                subject_ids=get_subject_ids(args, sub_id=data["y_val"]["Sub_ID"]),
            ),
            shuffle=False,
            **dataloader_kwargs,
        )
        test_ds = DataLoader(
            ClassificationDataset(
                args,
                filenames=data["x_test"],
                labels=data["y_test"],
                rec_ids=data["test_recording_id"],
                stats=args.ds_info["stats"],
                recording_id_str_to_num=recording_id_str_to_num,
                subject_ids=get_subject_ids(args, sub_id=data["y_test"]["Sub_ID"]),
            ),
            shuffle=False,
            **dataloader_kwargs,
        )
        return (train_ds, val_ds, test_ds)


def get_datasets_post_hoc(
    args, data: t.Dict[str, t.Any], recording_id_str_to_num: t.Dict[str, int]
):
    dataloader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if args.device.type in ("cuda", "mps"):
        gpu_kwargs = {"prefetch_factor": 2, "pin_memory": True}
        dataloader_kwargs.update(gpu_kwargs)
    args.ds_info["stats"] = compute_statistics(args, data=data)
    _, pretext_val_idx = split_pre_train(args, data=data)
    pre_train_ds = DataLoader(
        DiagnosticsDataset(
            args,
            filenames=data["x_pre_train"][pretext_val_idx],
            labels={k: v for k, v in data["y_pre_train"].items()},
            sleep_status=data["pre_train_sleep_status"][pretext_val_idx],
            rec_ids=data["pre_train_recording_id"][pretext_val_idx],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
        ),
        shuffle=True,
        **dataloader_kwargs,
    )
    test_ds = DataLoader(
        DiagnosticsDataset(
            args,
            filenames=data["x_test"],
            labels=data["y_test"],
            sleep_status=data["train_sleep_status"],
            rec_ids=data["train_recording_id"],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
            target_task_ds=True,
        ),
        shuffle=False,
        **dataloader_kwargs,
    )
    sleep_wake_ds = DataLoader(
        DiagnosticsDataset(
            args,
            filenames=data["x_sleep_wake"],
            labels=data["y_sleep_wake"],
            sleep_status=data["sleep_wake_sleep_status"],
            rec_ids=data["sleep_wake_recording_id"],
            stats=args.ds_info["stats"],
            recording_id_str_to_num=recording_id_str_to_num,
        ),
        shuffle=False,
        **dataloader_kwargs,
    )

    return pre_train_ds, test_ds, sleep_wake_ds


def get_datasets(args, summary: tensorboard.Summary = None):
    filename = os.path.join(args.dataset, "metadata.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find metadata.pkl in {args.dataset}.")
    with open(filename, "rb") as file:
        data = pickle.load(file)
    args.ds_info = data["ds_info"]
    args.ds_info["channel_freq"] = CHANNELS_FREQ.copy()
    if not getattr(args, "include_hr", True):
        del args.ds_info["channel_freq"]["HR"]

    args.input_shapes = {
        k: h5.get(data["sessions_paths"][0], k).shape
        for k in args.ds_info["channel_freq"].keys()
    }
    numeric_recording_id = preprocessing.LabelEncoder().fit_transform(
        data["recording_id"]
    )
    recording_id_str_to_num = dict(
        zip(np.unique(data["recording_id"]), np.unique(numeric_recording_id))
    )
    construct_dataset(args, data=data)
    if (
        (summary is not None)
        and (args.plot_mode in (1, 3))
        and (not args.e4selflearning)
    ):
        plots.plot_data_summary(args, data=data, summary=summary)
    match args.task_mode:
        case 0:
            (
                pretext_train_ds,
                pretext_val_ds,
                pretext_test_ds,
            ) = get_pre_training_dataset(
                args, data=data, recording_id_str_to_num=recording_id_str_to_num
            )
            if (
                (args.plot_mode in (1, 3))
                and (not args.filter_collections)
                and (not args.e4selflearning)
            ):
                utils.compute_datasets_relative_size(
                    args,
                    pretext_train_ds=pretext_train_ds,
                    pretext_val_ds=pretext_val_ds,
                    data=data,
                )
            return (
                pretext_train_ds,
                pretext_val_ds,
                pretext_test_ds,
            )
        case 1 | 2 | 3:
            train_ds, val_ds, test_ds = get_training_datasets(
                args, data=data, recording_id_str_to_num=recording_id_str_to_num
            )
            if args.plot_mode in (1, 3):
                utils.cases_controls_difference_in_rec_status(
                    train_ds=train_ds, data=data
                )
                utils.medications_info(args, train_ds=train_ds)
                utils.target_dataset_info(
                    data=data, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds
                )
            return train_ds, val_ds, test_ds
        case 4:
            datasets = get_training_datasets_cml(
                args, data=data, recording_id_str_to_num=recording_id_str_to_num
            )
            if args.verbose == 2:
                utils.flirt_features_missingness(args, datasets=datasets)
            return datasets
        case 9:
            pre_train_ds, test_ds, sleep_wake_ds = get_datasets_post_hoc(
                args, data=data, recording_id_str_to_num=recording_id_str_to_num
            )
            return pre_train_ds, test_ds, sleep_wake_ds
