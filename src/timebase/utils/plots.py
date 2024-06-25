import json
import re
import typing as t

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import tqdm
from einops import rearrange
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from timebase.data.static import *
from timebase.utils import tensorboard

sns.set_style("ticks")
plt.style.use("seaborn-v0_8-deep")

PARAMS_PAD = 1
PARAMS_LENGTH = 2

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]
tick_fontsize, label_fontsize, title_fontsize = 9, 12, 15


def set_ticks_params(
    axis: matplotlib.axes.Axes, length: int = PARAMS_LENGTH, pad: int = PARAMS_PAD
):
    axis.tick_params(axis="both", which="both", length=length, pad=pad, colors="black")


def plot_encoders_features(
    args,
    res_sl: t.Dict,
    res_ssl: t.Dict,
    res_unsupervised: t.Dict,
    summary: tensorboard.Summary = None,
):
    labels_sl = res_sl["labels"]
    targets_sl = res_sl["targets"]
    representations_sl = res_sl["representations"]
    # B, N, D -> B, D
    representations_sl = np.mean(representations_sl, axis=-1)
    representations_sl = StandardScaler().fit_transform(representations_sl)
    umap = UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=1.0,
        n_components=3,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding_sl = umap.fit_transform(representations_sl)

    labels_ssl = res_ssl["labels"]
    targets_ssl = res_ssl["targets"]
    representations_ssl = res_ssl["representations"]
    # B, N, D -> B, D
    representations_ssl = np.mean(representations_ssl, axis=-1)
    representations_ssl = StandardScaler().fit_transform(representations_ssl)
    umap = UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=1.0,
        n_components=3,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding_ssl = umap.fit_transform(representations_ssl)
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=50,
    #     learning_rate=200,
    #     n_iter=5000,
    #     random_state=42,
    #     metric='euclidean',
    #     init='pca',
    #     early_exaggeration=4,
    #     method='exact',
    # )
    # embedding_ssl = tsne.fit_transform(representations_ssl)

    labels_unsupervised = res_unsupervised["labels"]
    targets_unsupervised = res_unsupervised["targets"]
    representations_unsupervised = res_unsupervised["representations"]
    # B, N, D -> B, D
    representations_unsupervised = np.mean(representations_unsupervised, axis=-1)
    representations_unsupervised = StandardScaler().fit_transform(
        representations_unsupervised
    )
    umap = UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=1.0,
        n_components=3,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding_unsupervised = umap.fit_transform(representations_unsupervised)

    res = {
        0: {"embedding": embedding_sl, "targets": targets_sl, "labels": labels_sl},
        1: {
            "embedding": embedding_unsupervised,
            "targets": targets_unsupervised,
            "labels": labels_unsupervised,
        },
        2: {"embedding": embedding_ssl, "targets": targets_ssl, "labels": labels_ssl},
    }

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 8),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"},
    )
    titles = ["Supervised", "Pre-training", "Self-supervised"]

    for idx in range(len(res)):
        # Target
        colors = list(np.where(np.squeeze(res[idx]["targets"]) == 1, "red", "blue"))
        mask = np.isnan(res[idx]["labels"]["Sub_ID"])
        scatter = axs[idx].scatter(
            res[idx]["embedding"][:, 0][~mask],
            res[idx]["embedding"][:, 1][~mask],
            res[idx]["embedding"][:, 2][~mask],
            c=np.array(colors)[~mask],
            s=1.5,
            marker="o",
        )
        axs[idx].scatter(
            res[idx]["embedding"][:, 0][mask],
            res[idx]["embedding"][:, 1][mask],
            res[idx]["embedding"][:, 2][mask],
            c="green",
            s=1.5,
            marker="o",
        )
        axs[idx].set_xlabel("emb 01", fontsize=label_fontsize)
        axs[idx].set_ylabel("emb 02", fontsize=label_fontsize)
        axs[idx].set_zlabel("emb 03", fontsize=label_fontsize)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        axs[idx].set_zticklabels([])
        axs[idx].grid(True)

        axs[idx].set_title(titles[idx], fontsize=title_fontsize)
        if idx == 0:
            legend_dict_color = {"case": "red", "control": "blue"}
            legend_markers = [
                plt.Line2D([0, 0], [0, 0], color=c, marker="o", linestyle="")
                for c in legend_dict_color.values()
            ]
            axs[idx].legend(
                legend_markers,
                legend_dict_color.keys(),
                loc="upper left",
                bbox_to_anchor=(-0.05, 1),
                ncol=1,
                edgecolor="white",
                facecolor="white",
            )
        scatter.set_label("_nolegend_")

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 8),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"},
    )
    colors = list(np.where(np.squeeze(res[2]["targets"]) == 1, "red", "blue"))
    mask = np.isnan(res[2]["labels"]["Sub_ID"])
    axs[0].scatter(
        res[2]["embedding"][:, 0][~mask],
        res[2]["embedding"][:, 1][~mask],
        res[2]["embedding"][:, 2][~mask],
        c=np.array(colors)[~mask],
        s=1.5,
        marker="o",
    )
    axs[0].scatter(
        res[2]["embedding"][:, 0][mask],
        res[2]["embedding"][:, 1][mask],
        res[2]["embedding"][:, 2][mask],
        c="green",
        s=1.5,
        marker="o",
    )
    axs[0].set_xlabel("emb 01", fontsize=label_fontsize)
    axs[0].set_ylabel("emb 02", fontsize=label_fontsize)
    axs[0].set_zlabel("emb 03", fontsize=label_fontsize)
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].set_zticklabels([])
    axs[0].grid(True)

    legend_dict_color = {
        "exacerbation": "red",
        "euthymia": "blue",
        "unlabelled": "green",
    }
    legend_markers = [
        plt.Line2D([0, 0], [0, 0], color=c, marker="o", linestyle="")
        for c in legend_dict_color.values()
    ]
    axs[0].legend(
        legend_markers,
        legend_dict_color.keys(),
        loc="upper left",
        bbox_to_anchor=(0.05, 1),
        ncol=3,
        edgecolor="white",
        facecolor="white",
    )
    axs[0].set_label("_nolegend_")

    ###Scales

    for r, (scale, palette) in enumerate(
        zip(["HDRS_SUM", "YMRS_SUM"], [plt.cm.Blues, plt.cm.Reds]), 1
    ):
        colors = (
            res[2]["labels"][scale][~mask] - np.min(res[2]["labels"][scale][~mask])
        ) / (
            np.max(res[2]["labels"][scale][~mask])
            - np.min(res[2]["labels"][scale][~mask])
        )
        axs[r].scatter(
            res[2]["embedding"][:, 0][~mask],
            res[2]["embedding"][:, 1][~mask],
            res[2]["embedding"][:, 2][~mask],
            c=palette(colors),
            s=1.5,
            marker="o",
        )
        axs[r].scatter(
            res[2]["embedding"][:, 0][mask],
            res[2]["embedding"][:, 1][mask],
            res[2]["embedding"][:, 2][mask],
            c="green",
            s=1.5,
            marker="o",
        )
        axs[r].set_xlabel("emb 01", fontsize=label_fontsize)
        axs[r].set_ylabel("emb 02", fontsize=label_fontsize)
        axs[r].set_zlabel("emb 03", fontsize=label_fontsize)
        axs[r].set_xticklabels([])
        axs[r].set_yticklabels([])
        axs[r].set_zticklabels([])
        axs[r].grid(True)

    cax1 = fig.add_axes([0.44, 0.73, 0.15, 0.02])
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.Blues), cax=cax1, orientation="horizontal"
    )
    label_text = "HDRS"
    label_x = 0.5  # Place the label in the middle of the colorbar
    label_y = 1.2  # Adjust this value to position the label above the colorbar
    label_horizontalalignment = "center"

    # Add the label to the custom axes
    cax1.text(
        label_x,
        label_y,
        label_text,
        transform=cax1.transAxes,  # Use the custom axes' coordinate system
        horizontalalignment=label_horizontalalignment,
        fontsize=12,
    )
    cax1.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    cb1.outline.set_visible(False)

    cax2 = fig.add_axes([0.71, 0.73, 0.15, 0.02])
    cb2 = plt.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.Reds), cax=cax2, orientation="horizontal"
    )

    label_text2 = "YMRS"
    label_x2 = 0.5  # Place the label in the middle of the colorbar
    label_y2 = 1.2  # Adjust this value to position the label above the colorbar
    label_horizontalalignment2 = "center"

    # Add the label to the second custom axes
    cax2.text(
        label_x2,
        label_y2,
        label_text2,
        transform=cax2.transAxes,
        # Use the second custom axes' coordinate system
        horizontalalignment=label_horizontalalignment2,
        fontsize=12,
    )

    cax2.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    cb2.outline.set_visible(False)

    if summary:
        summary.figure(tag="embeddings_comparison", figure=fig, step=0, mode=0)
    else:
        return fig


def clinical_demographics(args, data, s, idx_s, summary):
    cases_stati = [v for k, v in DICT_STATE.items() if k in ["MDE_BD", "MDE_MDD", "ME"]]
    controls_stati = [v for k, v in DICT_STATE.items() if k in ["Eu_BD", "Eu_MDD"]]
    cases_mask = pd.Series(data[f"y_{s}"]["status"]).isin(cases_stati)
    controls_mask = pd.Series(data[f"y_{s}"]["status"]).isin(controls_stati)

    figure, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(6 * 2, 7 * 5),
        gridspec_kw={"wspace": 0.15, "hspace": 0.15},
        dpi=args.dpi,
    )
    for idx, mask in enumerate([cases_mask, controls_mask]):
        rev_DICT_STATE = {v: k for k, v in DICT_STATE.items()}

        unique_sub_id = np.unique(data[f"y_{s}"]["Sub_ID"][mask])
        first_appear_indices = [
            np.where(data[f"y_{s}"]["Sub_ID"][mask] == val)[0][0]
            for val in unique_sub_id
        ]
        stati = data[f"y_{s}"]["status"][mask][first_appear_indices]
        stati = [rev_DICT_STATE[i] for i in stati]
        unique_stati, counts = np.unique(stati, return_counts=True)
        axs[0, idx].bar(unique_stati, counts)
        axs[0, idx].set_xlabel("status", fontsize=label_fontsize)
        axs[0, idx].set_ylabel("count", fontsize=label_fontsize)
        axs[0, idx].set_title(f"Subjects (N = {len(stati)})", fontsize=label_fontsize)

        stati = [rev_DICT_STATE[i] for i in data[f"y_{s}"]["status"][mask]]
        unique_stati, counts = np.unique(stati, return_counts=True)
        axs[1, idx].bar(unique_stati, counts)
        axs[1, idx].set_xlabel("status", fontsize=label_fontsize)
        axs[1, idx].set_ylabel("count", fontsize=label_fontsize)
        axs[1, idx].set_title(f"Segments (N = {len(stati)})", fontsize=label_fontsize)

        ax2 = sns.histplot(
            x=data[f"y_{s}"]["age"][mask],
            hue=data[f"y_{s}"]["sex"][mask].astype(int),
            palette={0: "fuchsia", 1: "aqua"},
            multiple="stack",
            stat="count",
            shrink=0.8,
            binwidth=2,
            ax=axs[2, idx],
        )
        _, counts = np.unique(
            data[f"y_{s}"]["sex"][mask].astype(int), return_counts=True
        )
        ax2.set_xlabel("Age (Years)", fontsize=label_fontsize)
        ax2.set_ylabel("count", fontsize=label_fontsize)
        ax2.set_title(
            f"F/M ratio {counts[0]/counts[1]:.02f}, Age mean (std) "
            f"{np.mean(data[f'y_{s}']['age'][mask]):.02f}"
            f"({np.std(data[f'y_{s}']['age'][mask]):.02f})",
            fontsize=label_fontsize,
        )
        legend = ax2.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax2.legend(
            handles,
            ["Female", "Male"],
            title="Sex",
            loc="best",
            edgecolor="white",
            facecolor="white",
        )

        ax3 = sns.histplot(
            x=data[f"y_{s}"]["HDRS_SUM"][mask],
            hue=pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE),
            palette={
                status: DICT_STATE_COLOR[status]
                for status in np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            },
            hue_order=list(
                np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            ),
            multiple="stack",
            stat="count",
            shrink=0.8,
            binwidth=2,
            ax=axs[3, idx],
        )
        ax3.set_xlabel("HDRS sum", fontsize=label_fontsize)
        ax3.set_ylabel("count", fontsize=label_fontsize)
        ax3.set_title(
            f"{np.mean(data[f'y_{s}']['HDRS_SUM'][mask]):.02f}"
            f"({np.std(data[f'y_{s}']['HDRS_SUM'][mask]):.02f})",
            fontsize=label_fontsize,
        )
        legend = ax3.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax3.legend(
            handles,
            list(
                np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            ),
            title="Status",
            loc="best",
            edgecolor="white",
            facecolor="white",
        )

        ax4 = sns.histplot(
            x=data[f"y_{s}"]["YMRS_SUM"][mask],
            hue=pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE),
            palette={
                status: DICT_STATE_COLOR[status]
                for status in np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            },
            hue_order=list(
                np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            ),
            multiple="stack",
            stat="count",
            shrink=0.8,
            binwidth=2,
            ax=axs[4, idx],
        )
        ax4.set_xlabel("YMRS sum", fontsize=label_fontsize)
        ax4.set_ylabel("count", fontsize=label_fontsize)
        ax4.set_title(
            f"{np.mean(data[f'y_{s}']['YMRS_SUM'][mask]):.02f}"
            f"({np.std(data[f'y_{s}']['YMRS_SUM'][mask]):.02f})",
            fontsize=label_fontsize,
        )
        legend = ax4.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax4.legend(
            handles,
            list(
                np.unique(
                    pd.Series(data[f"y_{s}"]["status"][mask]).replace(rev_DICT_STATE)
                )
            ),
            title="Status",
            loc="best",
            edgecolor="white",
            facecolor="white",
        )
    summary.figure(tag="clinical_demographics", figure=figure, step=0, mode=idx_s)


def plot_data_summary(
    args,
    summary: tensorboard.Summary,
    data: t.Dict[str, t.Any],
):
    ############################################################################

    prefix = os.path.join(args.dataset, "")
    suffix_pattern = r"/\d+\.h5"
    sessions_list = {k: [] for k in ["x_pre_train", "x_train", "x_val", "x_test"]}
    for k in sessions_list.keys():
        unique_sessions = np.unique(
            [re.sub(rf"^{re.escape(prefix)}|{suffix_pattern}$", "", s) for s in data[k]]
        )
        sessions_list[k].extend(list(unique_sessions))
    union_set = set()
    for k in ["x_train", "x_val", "x_test"]:
        union_set.update(sessions_list[k])
    pre_train_only = list(set(sessions_list["x_pre_train"]).difference(union_set))
    data_collections = np.unique([s.split("/")[0] for s in pre_train_only])
    stati_across_collections = {
        c: {"wake": 0, "sleep": 0, "off-body": 0} for c in data_collections
    }
    for session in pre_train_only:
        data_collection = session.split("/")[0]
        for k, v in data["ds_info"]["wake_sleep_off"][session].items():
            stati_across_collections[data_collection][k] += v

    if args.verbose == 2:
        print(
            f"Segmentation used a segment length of {args.ds_info['segment_length']} secs"
        )
        if hasattr(args.ds_info, "step_size"):
            print(f"Step size of {args.ds_info['step_size']} secs")

    names, wake, sleep, off_body = [], [], [], []
    for k, v in stati_across_collections.items():
        names.append(k)
        wake.append(v["wake"] // (60 * 60))
        sleep.append(v["sleep"] // (60 * 60))
        off_body.append(v["off-body"] // (60 * 60))

    union_set = set()
    for k in ["train_recording_id", "val_recording_id", "test_recording_id"]:
        union_set.update(np.unique(data[k]))
    pre_train_rec_ids = list(
        set(np.unique(data["pre_train_recording_id"])).difference(union_set)
    )
    ids, counts = np.unique(
        [rec_id.split("/")[0] for rec_id in pre_train_rec_ids], return_counts=True
    )
    num_subjects = dict(zip(ids, counts))

    # In unlabelled data, recording_id coincide with subject_id, this is
    # because all recordings from a given subject were assigned the same
    # recording_id. This is not the case with labelled data collected in
    # Barcelona where recording_id identifies different Ts. In thi case,
    # Sub_ID needs to be used to identify subjects
    union_set = set()
    for k in ["y_train", "y_val", "y_test"]:
        union_set.update(np.unique(data[k]["Sub_ID"][~np.isnan(data[k]["Sub_ID"])]))
    pre_train_sub_id = list(
        set(
            data["y_pre_train"]["Sub_ID"][~np.isnan(data["y_pre_train"]["Sub_ID"])]
        ).difference(union_set)
    )
    num_subjects["barcelona"] = len(pre_train_sub_id)
    assert names == list(num_subjects.keys())

    bcn_unlabelled_values, bcn_unlabelled_counts = np.unique(
        data["clinical_info"][
            data["clinical_info"]["Sub_ID"].isin(pre_train_sub_id)
        ].drop_duplicates(subset=["Sub_ID"])["status"],
        return_counts=True,
    )
    bcn_unlabelled_values = [
        {v: k for k, v in DICT_STATE.items()}[val] for val in bcn_unlabelled_values
    ]
    bcn_unlabelled_counts = bcn_unlabelled_counts.astype(str)
    with open(
        os.path.join(args.output_dir, "BCN_pre_train_only_stati"), "w"
    ) as json_file:
        json.dump(dict(zip(bcn_unlabelled_values, bcn_unlabelled_counts)), json_file)

    df = pd.DataFrame(
        {
            "collections": names,
            "num_subjects": list(num_subjects.values()),
            "H_wake": wake,
            "H_sleep": sleep,
            "H_off_body": off_body,
        }
    )
    df.to_csv(
        os.path.join(
            args.output_dir, "recorded_hours_per_status_across_collections.csv"
        ),
        index=False,
    )

    ############################################################################

    X = {
        "status": np.concatenate(
            (
                data["y_train"]["status"],
                data["y_val"]["status"],
                data["y_test"]["status"],
            )
        ),
        "Sub_ID": np.concatenate(
            (
                data["y_train"]["Sub_ID"],
                data["y_val"]["Sub_ID"],
                data["y_test"]["Sub_ID"],
            )
        ),
    }

    cases_mask = np.array(
        [
            True
            if s
            in [
                v
                for k, v in DICT_STATE.items()
                if k in ["MDE_BD", "MDE_MDD", "ME", "MX"]
            ]
            else False
            for s in X["status"]
        ]
    )

    cases_sub_id = np.unique(X["Sub_ID"][cases_mask])
    controls_sub_id = np.unique(X["Sub_ID"][~cases_mask])
    cases_stati = np.unique(X["status"][cases_mask])
    controls_stati = np.unique(X["status"][~cases_mask])

    res = {c: {} for c in ["cases", "controls"]}
    for name, mask, c_stati in zip(
        ["cases", "controls"],
        [cases_sub_id, controls_sub_id],
        [cases_stati, controls_stati],
    ):
        min = np.min(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["age"]
        )
        res[name]["age_min"] = str(min)
        max = np.max(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["age"]
        )
        res[name]["age_max"] = str(max)
        mean = np.mean(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["age"]
        )
        res[name]["age_mean"] = str(mean)
        std = np.std(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["age"]
        )
        res[name]["age_std"] = str(std)
        res[name]["stati"] = {}
        stati, counts = np.unique(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["status"],
            return_counts=True,
        )
        stati = [{v: k for k, v in DICT_STATE.items()}[s] for s in stati]
        for s, c in zip(stati, counts):
            res[name]["stati"][s] = str(c)
        res[name]["sex"] = {}
        sex, counts = np.unique(
            data["clinical_info"][
                data["clinical_info"]["Sub_ID"].isin(mask)
                & data["clinical_info"]["status"].isin(c_stati)
            ].drop_duplicates(subset="Sub_ID")["sex"],
            return_counts=True,
        )
        sex = [{0: "males", 1: "females"}[s] for s in sex]
        for s, c in zip(sex, counts):
            res[name]["sex"][s] = str(c)
    with open(
        os.path.join(args.output_dir, "clinical_demographics.json"), "w"
    ) as json_file:
        json.dump(res, json_file, indent=4)

    rec_ids = ["barcelona/" + f.rsplit("/", -1)[-2] for f in data["x_train"]]
    rec_ids = np.unique(rec_ids)
    d = data["clinical_info"].copy()
    d["status"].replace({v: k for k, v in DICT_STATE.items()}, inplace=True)
    d[d["Session_Code"].isin(rec_ids)].groupby("status").agg(
        {
            "HDRS_SUM": ["mean", "std"],
            "YMRS_SUM": ["mean", "std"],
        }
    ).to_csv(
        os.path.join(args.output_dir, "psychometric_scales.csv"),
        index=False,
    )

    ############################################################################

    splits = ["train", "val", "test"]
    for idx_s, s in enumerate(splits):
        clinical_demographics(args, data=data, s=s, idx_s=idx_s, summary=summary)

    ############################################################################

    rec_stati = ["wake", "sleep", "off-body"]
    barcelona = {n: [] for n in rec_stati}
    e4self_learning = {n: [] for n in rec_stati}

    for rec_name, d in data["ds_info"]["wake_sleep_off"].items():
        for rec_status, secs in d.items():
            if "barcelona" in rec_name:
                barcelona[rec_status].append(secs / (60 * 60))
            else:
                e4self_learning[rec_status].append(secs / (60 * 60))
    barcelona_dict_stats, e4self_learning_stats = {
        rec_status: {} for rec_status in rec_stati
    }, {rec_status: {} for rec_status in rec_stati}
    for rec_status in rec_stati:
        barcelona_dict_stats[rec_status]["median"] = np.median(barcelona[rec_status])
        e4self_learning_stats[rec_status]["median"] = np.median(
            e4self_learning[rec_status]
        )
        barcelona_dict_stats[rec_status]["iqr"] = np.subtract(
            *np.percentile(barcelona[rec_status], [75, 25])
        )
        e4self_learning_stats[rec_status]["iqr"] = np.subtract(
            *np.percentile(e4self_learning[rec_status], [75, 25])
        )
    res = {"barcelona": barcelona_dict_stats, "e4self_learning": e4self_learning_stats}
    with open(os.path.join(args.output_dir, "sleep_wake_stats.json"), "w") as json_file:
        json.dump(res, json_file, indent=4)


def temp_plot(
    args,
    res_ssl: t.Dict,
    res_unsupervised: t.Dict,
    summary: tensorboard.Summary = None,
):
    labels_ssl = res_ssl["labels"]
    targets_ssl = res_ssl["targets"]
    representations_ssl = res_ssl["representations"]
    # B, N, D -> B, D
    representations_ssl = np.mean(representations_ssl, axis=-1)
    representations_ssl = StandardScaler().fit_transform(representations_ssl)
    umap = UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=1.0,
        n_components=3,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding_ssl = umap.fit_transform(representations_ssl)

    labels_unsupervised = pd.Series(res_unsupervised["sleep_status"]).replace(
        SLEEP_DICT
    )
    representations_unsupervised = res_unsupervised["representations"]
    # B, N, D -> B, D
    representations_unsupervised = np.mean(representations_unsupervised, axis=-1)
    representations_unsupervised = StandardScaler().fit_transform(
        representations_unsupervised
    )
    umap = UMAP(
        n_neighbors=50,
        min_dist=0.1,
        spread=1.0,
        n_components=3,
        metric="euclidean",
        random_state=args.seed,
    )
    embedding_unsupervised = umap.fit_transform(representations_unsupervised)

    res = {
        0: {"embedding": embedding_unsupervised, "sleep": labels_unsupervised},
        1: {"embedding": embedding_ssl, "targets": targets_ssl, "labels": labels_ssl},
    }

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 8),
        dpi=args.dpi,
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0.15},
    )

    colors = res[0]["sleep"].replace(SLEEP_COLOR_DICT)
    axs[0].scatter(
        res[0]["embedding"][:, 0],
        res[0]["embedding"][:, 1],
        res[0]["embedding"][:, 2],
        c=colors,
        s=4.5,
        marker="o",
    )
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].set_zticklabels([])
    axs[0].grid(True)

    legend_markers = [
        plt.Line2D([0, 0], [0, 0], color=c, marker="o", linestyle="")
        for c in SLEEP_COLOR_DICT.values()
        if c in np.unique(colors)
    ]
    legend_labels = [s for s, c in SLEEP_COLOR_DICT.items() if c in np.unique(colors)]
    axs[0].legend(
        legend_markers,
        legend_labels,
        loc="lower left",  # Move the legend to the top center
        bbox_to_anchor=(-0.1, -0.1),
        # Adjust position to top of the plot
        ncol=3,  # Arrange entries in three columns
        edgecolor="white",
        facecolor="white",
        fontsize=30,
        markerscale=2,
    )

    # Remove the old legend
    axs[0].set_label("_nolegend_")
    axs[0].set_title("Pre-trained Encoder", fontsize=32)

    # Calculate the minimum and maximum values for x, y, and z axes
    x_min = np.min(res[0]["embedding"][:, 0])
    x_max = np.max(res[0]["embedding"][:, 0])
    y_min = np.min(res[0]["embedding"][:, 1])
    y_max = np.max(res[0]["embedding"][:, 1])
    z_min = np.min(res[0]["embedding"][:, 2])
    z_max = np.max(res[0]["embedding"][:, 2])
    # Set x, y, and z axis limits for the first scatter plot
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_zlim(z_min, z_max)

    colors = list(np.where(np.squeeze(res[1]["targets"]) == 1, "red", "blue"))
    mask = np.isnan(res[1]["labels"]["Sub_ID"])
    axs[1].scatter(
        res[1]["embedding"][:, 0][~mask],
        res[1]["embedding"][:, 1][~mask],
        res[1]["embedding"][:, 2][~mask],
        c=np.array(colors)[~mask],
        s=4.5,
        marker="o",
    )
    axs[1].scatter(
        res[1]["embedding"][:, 0][mask],
        res[1]["embedding"][:, 1][mask],
        res[1]["embedding"][:, 2][mask],
        c="green",
        s=4.5,
        marker="o",
    )
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].set_zticklabels([])
    axs[1].grid(True)
    # # Calculate the minimum and maximum values for x, y, and z axes for the second scatter plot
    # x_min = np.min(res[1]["embedding"][:, 0])
    # x_max = np.max(res[1]["embedding"][:, 0])
    # y_min = np.min(res[1]["embedding"][:, 1])
    # y_max = np.max(res[1]["embedding"][:, 1])
    # z_min = np.min(res[1]["embedding"][:, 2])
    # z_max = np.max(res[1]["embedding"][:, 2])
    #
    # # Set x, y, and z axis limits for the second scatter plot
    # axs[1].set_xlim(x_min, x_max)
    # axs[1].set_ylim(y_min, y_max)
    # axs[1].set_zlim(z_min, z_max)

    legend_dict_color = {
        "exacerbation": "red",
        "euthymia": "blue",
        "unlabelled": "green",
    }
    legend_markers = [
        plt.Line2D([0, 0], [0, 0], color=c, marker="o", linestyle="")
        for c in legend_dict_color.values()
    ]
    axs[1].legend(
        legend_markers,
        legend_dict_color.keys(),
        loc="lower left",  # Move the legend to the top center
        bbox_to_anchor=(-0.1, -0.2),
        ncol=2,
        edgecolor="white",
        facecolor="white",
        fontsize=30,
        markerscale=2,
    )
    axs[1].set_label("_nolegend_")
    axs[1].set_title("Fine-tuned Encoder", fontsize=32)

    gmm = GaussianMixture(n_components=3).fit(res[1]["embedding"][:, :3])
    labels = gmm.predict(res[1]["embedding"][:, :3])
    replacement_dict = {
        0: "exacerbation",
        1: "exacerbation",
        2: "exacerbation",
        3: "exacerbation",
        5: "euthymia",
        6: "euthymia",
        np.nan: "unlabelled",
    }
    # Use np.vectorize to apply the replacement
    replacement_func = np.vectorize(lambda x: replacement_dict.get(x, x))
    stati = replacement_func(res[1]["labels"]["status"])
    for l in np.unique(labels):
        unique, counts = np.unique(stati[labels == l], return_counts=True)
        percentages = counts / np.sum(counts)
        print(f"Cluster {l} composition:\t stati{unique} -> percentages{percentages}")

    if summary:
        summary.figure(tag="embeddings_comparison", figure=fig, step=0, mode=0)
    else:
        return fig
