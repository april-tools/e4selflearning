import argparse
import pickle
from functools import partial
from shutil import rmtree

import pandas as pd
from tqdm.contrib import concurrent

from timebase.data import preprocessing
from timebase.data import spreadsheet
from timebase.data import utils
from timebase.data.static import *
from timebase.utils import h5
from timebase.utils.utils import set_random_seed


def get_session_label(clinical_info: pd.DataFrame, session_id: str):
    session = clinical_info[clinical_info.Session_Code == session_id]
    if session.empty:
        return None
    else:
        values = session.values[0]
        values[LABEL_COLS.index("Session_Code")] = float(
            os.path.basename(values[LABEL_COLS.index("Session_Code")])
        )
        return values.astype(np.float32)


def preprocess_session(
    args, clinical_info: pd.DataFrame, session_id: str, labelled: bool
):
    if labelled:
        recording_dir = utils.unzip_session(
            args.path2labelled_data, session_id=os.path.basename(session_id)
        )
        session_label = get_session_label(clinical_info, session_id=session_id)
        if session_label is None:
            raise ValueError(f"Cannot find session {session_id} in spreadsheet.")
        session_data, session_info, short_section = preprocessing.preprocess_dir(
            args,
            recording_dir=recording_dir,
            labelled=labelled,
        )
    else:
        session_label = np.empty(len(LABEL_COLS), dtype=np.float32)
        session_label.fill(np.nan)
        session_data, session_info, short_section = preprocessing.preprocess_dir(
            args,
            recording_dir=session_id,
            labelled=labelled,
        )
    if short_section:
        return None
    else:
        session_data["labels"] = session_label
        session_output_dir = os.path.join(args.output_dir, session_id).replace(
            os.path.join(args.path2unlabelled_data, "recast/"), ""
        )
        if not os.path.isdir(session_output_dir):
            os.makedirs(session_output_dir)
        filename = os.path.join(session_output_dir, "channels.h5")
        h5.write(filename=filename, content=session_data, overwrite=True)

        return session_info


def preprocess_wrapper(args, clinical_info, session_id, labelled):
    results = preprocess_session(
        args,
        clinical_info,
        session_id,
        labelled,
    )
    return results


def main(args):
    if not args.e4selflearning:
        if not os.path.isdir(args.path2labelled_data):
            raise FileNotFoundError(
                f"labelled data at {args.path2labelled_data} not found."
            )
    if not os.path.isdir(args.path2unlabelled_data):
        raise FileNotFoundError(
            f"path2unlabelled_data data at {args.path2unlabelled_data} " f"not found."
        )
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

    unlabelled_sessions = preprocessing.recast_unlabelled_data(args)
    all_sessions = dict(
        zip(
            unlabelled_sessions,
            len(unlabelled_sessions) * [False],
        )
    )
    recording_time = {
        "unlabelled": {0.0: 0, 1.0: 0, 2.0: 0},
    }
    if not args.e4selflearning:
        clinical_info = spreadsheet.read(args)
        args.session_codes = list(clinical_info["Session_Code"])
        clinical_info.replace({"status": DICT_STATE}, inplace=True)
        clinical_info.replace({"time": DICT_TIME}, inplace=True)
        for session in args.session_codes:
            all_sessions[session] = False
        recording_time["labelled"] = {0.0: 0, 1.0: 0, 2.0: 0}
        assert os.sep.join(args.path2unlabelled_data.split(os.sep)[:-1]) == os.sep.join(
            args.path2labelled_data.split(os.sep)[:-1]
        )
    else:
        clinical_info = None
    print(
        f"\nPreprocessing recordings from {os.sep.join(args.path2unlabelled_data.split(os.sep)[:-1])}..."
    )
    results = concurrent.process_map(
        partial(preprocess_wrapper, args, clinical_info),
        all_sessions.keys(),
        all_sessions.values(),
        max_workers=args.num_workers,
        chunksize=args.chunksize,
        desc="Preprocessing",
    )
    sessions_info, invalid_sessions = {}, []
    for i, (session_id, labelled) in enumerate(all_sessions.items()):
        # session_info = preprocess_session(
        #     args,
        #     session_id=session_id,
        #     clinical_info=clinical_info,
        #     labelled=labelled,
        # )
        session_info = results[i]
        session_id = session_id.replace(
            os.path.join(args.path2unlabelled_data, "recast/"), ""
        )
        if session_info is None:
            invalid_sessions.append(session_id)
            continue
        annotation_status = "labelled" if session_info["labelled"] else "unlabelled"
        for k, v in session_info["seconds_per_status"].items():
            recording_time[annotation_status][k] += v
        # del session_info['minutes_by_status']
        sessions_info[session_id] = session_info

    res = {
        "invalid_sessions": invalid_sessions,
        "sessions_info": sessions_info,
        "sleep_algorithm": args.sleep_algorithm,
    }
    if not args.e4selflearning:
        numeric_columns = clinical_info.select_dtypes(include=[np.number]).columns
        clinical_info[numeric_columns] = clinical_info[numeric_columns].astype(
            np.float32
        )
        res["clinical_info"] = clinical_info
    with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as file:
        pickle.dump(
            res,
            file,
        )
    if args.verbose == 1:
        for k, v in recording_time.items():
            print(f"{k} recordings:\t")
            for status, seconds in v.items():
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                remaining_seconds = seconds % 60
                print(
                    f"{hours} hours, {minutes} minutes, and {remaining_seconds} "
                    f"seconds of recording as {SLEEP_DICT[status]}"
                )

    print(f"Saved processed data to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path2unlabelled_data",
        type=str,
        default="data/raw_data/unlabelled_data",
        help="path to directory with unlabelled data",
    )
    parser.add_argument(
        "--path2labelled_data",
        type=str,
        default="data/raw_data/barcelona",
        help="path to directory with raw data in zip files collected and "
        "labelled in Barcelona, Hospital Clínic",
    )
    parser.add_argument(
        "--e4selflearning",
        action="store_true",
        help="allows users with no access to INTREPIBD/TIMEBASE data to "
        "carry out the pre-processing and the self-supervised "
        "pre-training on the datasets forming the E4SelfLearning "
        "unlabelled collection",
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
    parser.add_argument(
        "--overwrite_spreadsheet",
        action="store_true",
        help="read from timebase/data/TIMEBASE_database.xlsx",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=1234)

    # preprocessing configuration
    parser.add_argument(
        "--sleep_algorithm",
        type=str,
        default="van_hees",
        choices=["van_hees", "scripps_clinic"],
        help="algorithm used for sleep-wake detection",
    )
    parser.add_argument(
        "--wear_minimum_minutes",
        type=int,
        default=5,
        help="minimum duration (in minutes) recording periods within a session"
        "marked as on-body have to meet in order to be included in further "
        "analyses",
    )
    parser.add_argument(
        "--minimum_recorded_time",
        type=int,
        default=15,
        help="minimum duration (in minutes) a recording session has to meet in"
        " order to be considered for further analyses",
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--chunksize", type=int, default=1)
    main(parser.parse_args())
