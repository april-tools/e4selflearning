import re

import pandas as pd

from timebase.data.static import *


def read(args) -> pd.DataFrame:
    """
    Loads clinical data spreadsheet and reshapes it to long format keeping
    only variable of interest
    """
    filename = os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.csv")
    if (not os.path.exists(filename)) or (args.overwrite_spreadsheet):
        data = pd.read_excel(
            os.path.join(FILE_DIRECTORY, "TIMEBASE_database.xlsx"),
            sheet_name="TIMEBASE",
        )
        data = data[LABEL_COLS]

        # some values are erroneously above the ceiling value as per scale design
        for k, v in ITEM_MAX.items():
            data[k] = np.clip(data[k], a_min=0, a_max=v)
        data["YMRS_SUM"] = np.sum(
            data[[col for col in data.columns if bool(re.search("YMRS[0-9]", col))]],
            axis=1,
        )
        data["HDRS_SUM"] = np.sum(
            data[[col for col in data.columns if bool(re.search("HDRS[0-9]", col))]],
            axis=1,
        )

        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        data["YMRS_discretized"] = pd.cut(
            data["YMRS_SUM"],
            bins=[
                0,
                7,
                14,
                25,
                60,
            ],  # [0, 7, 14, 25, 60] <- https://clinicaltrials.gov/ct2/show/NCT00931723
            include_lowest=True,
            right=True,
            labels=False,
        )
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        data["HDRS_discretized"] = pd.cut(
            data["HDRS_SUM"],
            bins=[
                0,
                7,
                14,
                23,
                52,
            ],  # [0, 7, 14, 23, 52] <- https://en.wikipedia.org/wiki/Hamilton_Rating_Scale_for_Depression
            include_lowest=True,
            right=True,
            labels=False,
        )

        missing_session_ids = []
        for session_id in data["Session_Code"]:
            zip_file = os.path.join(args.path2labelled_data, f"{str(session_id)}.zip")
            # unzip recording to recording folder not found.
            if not os.path.exists(zip_file):
                missing_session_ids.append(session_id)

        data = data[~data["Session_Code"].isin(missing_session_ids)]
        if args.verbose and len(missing_session_ids):
            print(
                f"The following session code(s) appear in the spreadsheet but "
                f"do not appear as zip files: {missing_session_ids}"
            )
        data["Session_Code"] = [
            os.path.join(os.path.basename(args.path2labelled_data), str(session_id))
            for session_id in data["Session_Code"]
        ]

        # Check Sub_ID is consistent with NHC
        # def _find_duplicate_indices(lst):
        #     duplicate_indices = {}
        #     for idx, value in enumerate(lst):
        #         if value in duplicate_indices:
        #             duplicate_indices[value].append(idx)
        #         else:
        #             duplicate_indices[value] = [idx]
        #
        #     # Filter the dictionary to keep only the indices with more than one occurrence
        #     duplicate_indices = {
        #         value: indices
        #         for value, indices in duplicate_indices.items()
        #         if len(indices) > 1
        #     }
        #     return [v for v in duplicate_indices.values()]
        #
        # set_list1 = {
        #     frozenset(sublist)
        #     for sublist in _find_duplicate_indices(lst=data["Sub_ID"])
        # }
        # set_list2 = {
        #     frozenset(sublist) for sublist in _find_duplicate_indices(lst=data["NHC"])
        # }
        # assert set_list1 == set_list2
        # assert len(np.unique(data["Sub_ID"])) == len(np.unique(data["NHC"]))

        assert data.columns.to_list() == LABEL_COLS
        selected_columns = (
            data.filter(like="YMRS").columns.to_list()
            + data.filter(like="HDRS").columns.to_list()
            + data.filter(like="IPAQ").columns.to_list()
        )
        data[selected_columns] = data[selected_columns].fillna(-9)

        data = data.dropna(
            subset=["Sub_ID", "age", "sex", "status", "time", "Session_Code"]
        ).reset_index(drop=True)

        data.to_csv(
            os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.csv"),
            index=False,
        )
    else:
        data = pd.read_csv(filename)
    return data
