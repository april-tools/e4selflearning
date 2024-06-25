import argparse
import json
import pickle
import shutil
import typing as t
from time import time

import torch
import wandb
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from timebase.data.reader import get_datasets
from timebase.data.static import *
from timebase.metrics import secondary_metrics_subjects_get_inputs
from timebase.metrics import subject_accuracy
from timebase.models.models import Classifier
from timebase.models.models import get_models
from timebase.utils import utils
from train_ann import load_pre_trained_parameters


def load(d: t.Dict[str, torch.Tensor], device: torch.device):
    """Load values in dictionary d to device"""
    return {k: v.to(device) for k, v in d.items()}


@torch.inference_mode()
def get_embeddings(
    args,
    ds: DataLoader,
    classifier: Classifier,
    verbose: int = 1,
):
    device = args.device
    targets, representations, subject_ids, labels, metadata = [], [], [], {}, {}
    classifier.to(device)
    classifier.train(False)
    for batch in tqdm(ds, disable=verbose == 0):
        inputs = load(batch["data"], device=device)
        outputs_classifier, representation = classifier(inputs)
        label = load(batch["label"], device=device)
        target = batch["target"].to(device)
        subject_id = batch["subject_id"].to(device)
        utils.update_dict(target=labels, source=label)
        utils.update_dict(target=metadata, source=batch["metadata"])
        targets.append(target)
        subject_ids.append(subject_id)
        representations.append(representation)
    res = {
        "labels": {k: torch.cat(v, dim=0).cpu().numpy() for k, v in labels.items()},
        "metadata": {k: torch.cat(v, dim=0).cpu().numpy() for k, v in metadata.items()},
        "targets": torch.concat(targets, dim=0).cpu().numpy(),
        "subject_ids": torch.concat(subject_ids, dim=0).cpu().numpy(),
        "representations": torch.concat(representations, dim=0).cpu().numpy(),
    }
    res["metadata"]["recording_id"] = np.vectorize(
        lambda x: {v: k for k, v in ds.dataset.recording_id_str_to_num.items()}.get(
            x, x
        )
    )(res["metadata"]["recording_id"])
    return res


def get_splits(args, datasets: t.Dict):
    if not args.path2featurizer:
        # Find columns that contain all np.nan values
        X_train = datasets["x_train"].values
        y_train = datasets["targets_train"]
        X_val = datasets["x_val"].values
        y_val = datasets["targets_val"]
        X_test = datasets["x_test"].values
        y_test = datasets["targets_test"]

        all_nan_columns = np.where(np.all(np.isnan(X_train), axis=0) == True)[0]
        if len(all_nan_columns):
            # Drop columns with all np.nan values
            X_train = datasets["x_train"].values[:, ~all_nan_columns]
            X_val = datasets["x_val"].values[:, ~all_nan_columns]
            X_test = datasets["x_test"].values[:, ~all_nan_columns]
        # Set non-finite values to np.nan
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_val = np.where(np.isinf(X_val), np.nan, X_val)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        # Mean value imputation
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        X_train = imp.fit_transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        train_ds, val_ds, test_ds = datasets
        classifier, _ = get_models(args, summary=None)
        load_pre_trained_parameters(
            args, classifier=classifier, path2pretraining_res=args.path2featurizer
        )
        datasets = {}
        res = get_embeddings(
            args,
            ds=train_ds,
            classifier=classifier,
        )
        X_train, y_train = (
            # np.reshape(
            #     res["representations"], newshape=(len(res["representations"]), -1)
            # ),
            np.mean(res["representations"], axis=-2),
            res["targets"],
        )

        res = get_embeddings(
            args,
            ds=val_ds,
            classifier=classifier,
        )
        (
            X_val,
            y_val,
            datasets["labels_val"],
            datasets["metadata_val"],
            datasets["subject_ids_val"],
        ) = (
            # np.reshape(
            #     res["representations"], newshape=(len(res["representations"]), -1)
            # ),
            np.mean(res["representations"], axis=-2),
            res["targets"],
            res["labels"],
            res["metadata"],
            res["subject_ids"],
        )

        res = get_embeddings(
            args,
            ds=test_ds,
            classifier=classifier,
        )
        (
            X_test,
            y_test,
            datasets["labels_test"],
            datasets["metadata_test"],
            datasets["subject_ids_test"],
        ) = (
            # np.reshape(
            #     res["representations"], newshape=(len(res["representations"]), -1)
            # ),
            np.mean(res["representations"], axis=-2),
            res["targets"],
            res["labels"],
            res["metadata"],
            res["subject_ids"],
        )

    return datasets, X_train, y_train, X_val, y_val, X_test, y_test


def main(args, wandb_sweep: bool = False):
    utils.set_random_seed(args.seed, verbose=args.verbose)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.use_wandb:
        utils.wandb_init(args, wandb_sweep=wandb_sweep)
    # 0: Self-supervised pre-training, 1: Fine-tuning, 2: Read-out, 3: NN
    # training, 4: CML (XGBoost), 9: Post-hoc analyses
    args.task_mode = 4
    utils.get_device(args)
    datasets = get_datasets(args)
    utils.save_args(args)
    datasets, X_train, y_train, X_val, y_val, X_test, y_test = get_splits(
        args, datasets=datasets
    )
    match args.learner:
        case "xgboost":
            classifier = xgb.XGBClassifier(
                random_state=args.seed,
                eval_metric="logloss",
                objective="binary:logistic",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                min_child_weight=args.min_child_weight,
                gamma=args.gamma,
            )
        case "svm":
            classifier = SVC(
                random_state=args.seed,
                C=args.C,
                kernel=args.kernel,
                degree=args.degree,
                gamma=args.gamma,
                probability=True,
            )
        case "knn":
            classifier = KNeighborsClassifier(
                n_neighbors=args.n_neighbors,
                weights=args.weights,
                algorithm=args.algorithm,
                p=args.p,
            )
        case "enet":
            classifier = SGDClassifier(
                random_state=args.seed,
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=args.l1_ratio,
                alpha=args.alpha,
            )
    start = time()
    # train
    classifier.fit(X=X_train, y=y_train)
    acc_train = accuracy_score(y_true=y_train, y_pred=classifier.predict(X_train))
    log_loss_train = log_loss(y_true=y_train, y_pred=classifier.predict_proba(X_train))

    # val
    acc_val = accuracy_score(y_true=y_val, y_pred=classifier.predict(X_val))
    log_loss_val = log_loss(y_true=y_val, y_pred=classifier.predict_proba(X_val))
    sub_acc_val = subject_accuracy(
        subject_ids=datasets["subject_ids_val"]
        if args.path2featurizer
        else datasets["labels_val"]["Sub_ID"],
        y_pred=classifier.predict_proba(X_val)[:, 1],
        y_true=y_val,
        from_logits=False,
    )
    if args.use_wandb:
        elapse = time() - start
        log = {
            "acc_train": acc_train,
            "acc_val": acc_val,
            "log_loss_train": log_loss_train,
            "log_loss_val": log_loss_val,
            "acc_subject_val": sub_acc_val,
            "elapse": elapse,
        }
        wandb.log(
            log,
            step=1,
        )
    if args.test_time:
        test_res = {
            "pred_probs": classifier.predict_proba(X_test)[:, 1],
            "targets": y_test,
            "labels": datasets["labels_test"],
            "metadata": datasets["metadata_test"],
        }
        (
            subjects_pred,
            subjects_true,
            subjects_scores,
        ) = secondary_metrics_subjects_get_inputs(
            y_pred=test_res["pred_probs"],
            y_true=y_test,
            subject_ids=datasets["metadata_test"]["subject_id"],
        )
        log = {
            "test_loss": log_loss(
                y_true=y_test, y_pred=classifier.predict_proba(X_test)
            ),
            "test_acc": accuracy_score(
                y_true=y_test, y_pred=classifier.predict(X_test)
            ),
            "test_acc_subjects": subject_accuracy(
                subject_ids=datasets["metadata_test"]["subject_id"],
                y_pred=classifier.predict_proba(X_test)[:, 1],
                y_true=y_test,
                from_logits=False,
            ),
            "test_precision": precision_score(
                y_true=test_res["targets"],
                y_pred=np.where(test_res["pred_probs"] > 0.5, 1, 0),
            ),
            "test_precision_subjects": precision_score(
                y_true=subjects_true,
                y_pred=subjects_pred,
            ),
            "test_recall": recall_score(
                y_true=test_res["targets"],
                y_pred=np.where(test_res["pred_probs"] > 0.5, 1, 0),
            ),
            "test_recall_subjects": recall_score(
                y_true=subjects_true,
                y_pred=subjects_pred,
            ),
            "test_f1_score": f1_score(
                y_true=test_res["targets"],
                y_pred=np.where(test_res["pred_probs"] > 0.5, 1, 0),
            ),
            "test_f1_subjects": f1_score(
                y_true=subjects_true,
                y_pred=subjects_pred,
            ),
            "test_auroc": roc_auc_score(
                y_true=test_res["targets"], y_score=test_res["pred_probs"]
            ),
            "test_auroc_subjects": roc_auc_score(
                y_true=subjects_true, y_score=subjects_scores
            ),
        }
        if args.use_wandb:
            wandb.log(
                log,
                step=1,
            )
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as file:
            json.dump(log, file)
        if args.save_test_model_outputs:
            with open(
                os.path.join(args.output_dir, "test_model_outputs.pkl"), "wb"
            ) as file:
                pickle.dump(test_res, file)
        print(
            f"Test accuracy: {log['test_acc']:.03f} \t"
            f"log-loss: {log['test_loss']:.03f}\t"
            f"accuracy_subject: {log['test_acc_subjects']:.03f}"
        )
    elapse = time() - start
    print(f"Elapse: {elapse:.02f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--path2featurizer",
        type=str,
        required=False,
        help="path to directory where pre-trainer featurizer is stored",
    )
    parser.add_argument(
        "--reuse_stats",
        action="store_true",
        help="reuse previously computed stats from either training or "
        "pre-training set for features scaling",
    )

    # dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to directory where preprocessed dataset is stored",
    )
    parser.add_argument(
        "--split_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="criterion for train/val/test split:"
        "0) time-split: each session is split into 70:15:15 along the temporal "
        "dimension such that segments from different splits map to "
        "different parts of the recording"
        "1) subject-split: cases and controls are split into 70:15:15 "
        "train/val/test such no subjects are not shared across splits",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="svg", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)
    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for DataLoader"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument(
        "--save_test_model_outputs", action="store_true", help="save test set outputs"
    )
    parser.add_argument(
        "--test_time", action="store_true", help="perform inference on test set"
    )
    parser.add_argument(
        "--learner", type=str, default=None, choices=["xgboost", "svm", "knn", "enet"]
    )
    temp_args = parser.parse_known_args()[0]
    match temp_args.learner:
        case "xgboost":
            # XGBoost config
            parser.add_argument("--learning_rate", type=float, default=0.001)
            parser.add_argument("--subsample", type=float, default=0.1)
            parser.add_argument("--colsample_bytree", type=float, default=0.1)
            parser.add_argument("--reg_alpha", type=float, default=0.0)
            parser.add_argument("--reg_lambda", type=float, default=0.0)
            parser.add_argument("--min_child_weight", type=float, default=0.01)
            parser.add_argument("--gamma", type=float, default=0.0)
            parser.add_argument("--n_estimators", type=int, default=5)
            parser.add_argument("--max_depth", type=int, default=3)
        case "svm":
            # SVC config
            parser.add_argument("--C", type=float, default=0.1)
            parser.add_argument("--gamma", type=float, default=0.1)
            parser.add_argument("--degree", type=int, default=3)
            parser.add_argument(
                "--kernel",
                type=str,
                choices=["linear", "poly", "rbf", "sigmoid"],
                default="linear",
            )
        case "knn":
            # KNeighborsClassifier config
            parser.add_argument("--n_neighbors", type=int, default=5)
            parser.add_argument(
                "--weights",
                type=str,
                choices=["uniform", "distance"],
                default="uniform",
            )
            parser.add_argument(
                "--algorithm",
                type=str,
                choices=["auto", "ball_tree", "kd_tree", "brute"],
                default="auto",
            )
            parser.add_argument("--p", type=int, choices=[1, 2], default=1)
        case "enet":
            # SGDClassifier config
            parser.add_argument("--l1_ratio", type=float, default=0.3)
            parser.add_argument("--alpha", type=float, default=0.001)
    del temp_args
    main(parser.parse_args())
