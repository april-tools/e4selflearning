import argparse
import json
import pickle
import shutil
import typing as t
from time import time

import torch
import wandb
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_accuracy
from tqdm import tqdm

from timebase import criterions
from timebase.criterions import Criteria
from timebase.data.reader import get_datasets
from timebase.data.static import *
from timebase.metrics import secondary_metrics_subjects_get_inputs
from timebase.metrics import subject_accuracy
from timebase.models.models import Classifier
from timebase.models.models import Critic
from timebase.models.models import get_models
from timebase.utils import tensorboard
from timebase.utils import utils
from timebase.utils import yaml
from timebase.utils.scheduler import Scheduler


def load(d: t.Dict[str, torch.Tensor], device: torch.device):
    """Load values in dictionary d to device"""
    return {k: v.to(device) for k, v in d.items()}


def load_pre_trained_parameters(
    args, classifier: Classifier, path2pretraining_res: str
):
    filename = os.path.join(path2pretraining_res, "ckpt_sslearner/model_state.pt")
    assert os.path.exists(filename), f"checkpoint was not found at {filename}"
    ckpt = torch.load(filename, map_location=args.device)
    state_dict = classifier.sslearner.state_dict()
    state_dict.update(
        {
            # the transform head is discarded
            k: v
            for k, v in ckpt["model"].items()
            if any(module in k for module in ["channel_embedding", "feature_encoder"])
        }
    )
    classifier.sslearner.load_state_dict(state_dict)
    if args.verbose:
        print(
            f"Parameters of classifier's representations extractor module "
            f"loaded from epoch {ckpt['epoch']} of SS pre-training"
        )


@torch.inference_mode()
def get_res(
    args,
    ds: DataLoader,
    classifier: Classifier,
    verbose: int = 1,
):
    device = args.device
    targets, y_pred_probs, representations, labels, metadata = [], [], [], {}, {}
    classifier.to(device)
    classifier.train(False)
    for batch in tqdm(ds, disable=verbose == 0):
        inputs = load(batch["data"], device=device)
        outputs_classifier, representation = classifier(inputs)
        label = load(batch["label"], device=device)
        target = batch["target"].to(device)
        utils.update_dict(target=labels, source=label)
        utils.update_dict(target=metadata, source=batch["metadata"])
        y_pred_probs.append(torch.sigmoid(outputs_classifier))
        targets.append(target)
        representations.append(representation)
    res = {
        "labels": {k: torch.cat(v, dim=0).cpu().numpy() for k, v in labels.items()},
        "metadata": {k: torch.cat(v, dim=0).cpu().numpy() for k, v in metadata.items()},
        "targets": torch.concat(targets, dim=0).cpu().numpy(),
        "pred_probs": torch.cat(y_pred_probs, dim=0).cpu().numpy(),
        "representations": torch.concat(representations, dim=0).cpu().numpy(),
    }
    res["metadata"]["recording_id"] = np.vectorize(
        lambda x: {v: k for k, v in ds.dataset.recording_id_str_to_num.items()}.get(
            x, x
        )
    )(res["metadata"]["recording_id"])
    return res


def train_step_time_split(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    freeze_representation_module: bool,
    critic: Critic,
    optimizer_classifier: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    classifier.to(device)
    inputs = load(batch["data"], device=device)
    targets = batch["target"].to(device)
    subject_ids = batch["subject_id"].to(device)
    classifier.train(True)
    if freeze_representation_module:
        classifier.sslearner.requires_grad_(False)
    outputs_classifier, representation = classifier(inputs)
    classifier_loss = criteria.criterion_classifier(
        y_true=targets, y_pred=outputs_classifier
    )
    classifier_loss.backward()
    optimizer_classifier.step()
    optimizer_classifier.zero_grad()
    result.update(
        {
            "loss/classifier": classifier_loss.detach(),
            "metrics/acc": binary_accuracy(
                outputs_classifier.detach(), targets.detach()
            ),
        }
    )
    outputs = {
        "y_pred": outputs_classifier.detach().cpu().numpy(),
        "y_true": targets.detach().cpu().numpy(),
        "subject_ids": subject_ids.detach().cpu().numpy(),
    }

    return result, outputs


def train_step_subject_split(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    freeze_representation_module: bool,
    critic: Critic,
    optimizer_classifier: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    classifier.to(device)
    critic.to(device)
    inputs = load(batch["data"], device=device)
    targets = batch["target"].to(device)
    subject_ids = batch["subject_id"].to(device)
    # train classifier
    classifier.train(True)
    if freeze_representation_module:
        classifier.sslearner.requires_grad_(False)
    critic.train(False)
    outputs_classifier, representation = classifier(inputs)
    classifier_loss = criteria.criterion_classifier(
        y_true=targets, y_pred=outputs_classifier
    )
    outputs_critic = critic(representation)
    representation_loss = criteria.critic_score(
        y_true=subject_ids, y_pred=outputs_critic
    )
    classifier_total_loss = classifier_loss + representation_loss
    classifier_total_loss.backward()
    optimizer_classifier.step()
    optimizer_classifier.zero_grad()
    result.update(
        {
            "loss/classifier": classifier_loss.detach(),
            "loss/representation": representation_loss.detach(),
            "loss/total": classifier_total_loss.detach(),
        }
    )

    # train critic
    representation = representation.detach()
    critic.train(True)
    outputs_critic = critic(representation)
    critic_loss = criteria.criterion_critic(y_true=subject_ids, y_pred=outputs_critic)
    critic_loss.backward()
    optimizer_critic.step()
    optimizer_critic.zero_grad()
    result.update(
        {
            "loss/critic": critic_loss.detach(),
            "metrics/acc": binary_accuracy(
                outputs_classifier.detach(), targets.detach()
            ),
        }
    )
    outputs = {
        "y_pred": outputs_classifier.detach().cpu().numpy(),
        "y_true": targets.detach().cpu().numpy(),
        "subject_ids": subject_ids.detach().cpu().numpy(),
    }

    return result, outputs


TRAIN_STEP_DICT = {
    0: train_step_time_split,
    1: train_step_subject_split,
}


def train(
    args,
    ds: DataLoader,
    classifier: Classifier,
    critic: Critic,
    optimizer_critic: torch.optim.Optimizer,
    optimizer_classifier: torch.optim.Optimizer,
    criteria: Criteria,
    summary: tensorboard.Summary,
    epoch: int,
):
    results, outputs = {}, {}
    for batch in tqdm(ds, desc="Train", disable=args.verbose <= 1):
        result, output = TRAIN_STEP_DICT[args.split_mode](
            batch=batch,
            classifier=classifier,
            freeze_representation_module=args.task_mode == 2,
            critic=critic,
            optimizer_classifier=optimizer_classifier,
            optimizer_critic=optimizer_critic,
            criteria=criteria,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
        utils.update_dict(target=outputs, source=output)
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    subjects_score = subject_accuracy(
        y_pred=np.concatenate(outputs["y_pred"], axis=0),
        y_true=np.concatenate(outputs["y_true"], axis=0),
        subject_ids=np.concatenate(outputs["subject_ids"], axis=0),
    )
    summary.scalar("metrics/subjects_accuracy", value=subjects_score, step=epoch)
    results["metrics/subjects_accuracy"] = subjects_score
    return results


@torch.inference_mode()
def validation_step(
    batch: t.Dict[str, t.Any],
    classifier: Classifier,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    classifier.to(device)
    inputs = load(batch["data"], device=device)
    targets = batch["target"].to(device)
    subject_ids = batch["subject_id"].to(device)
    classifier.train(False)
    outputs_classifier, representation = classifier(inputs)
    classifier_loss = criteria.criterion_classifier(
        y_true=targets, y_pred=outputs_classifier
    )
    result.update(
        {
            "loss/classifier": classifier_loss,
            "metrics/acc": binary_accuracy(outputs_classifier, targets),
        }
    )
    outputs = {
        "y_pred": outputs_classifier.detach().cpu().numpy(),
        "y_true": targets.detach().cpu().numpy(),
        "subject_ids": subject_ids.detach().cpu().numpy(),
    }
    return result, outputs


def validate(
    args,
    ds: DataLoader,
    classifier: Classifier,
    criteria: Criteria,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results, outputs = {}, {}
    for batch in tqdm(ds, desc="Validate", disable=args.verbose <= 1):
        result, output = validation_step(
            batch=batch,
            classifier=classifier,
            criteria=criteria,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
        utils.update_dict(target=outputs, source=output)
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    subjects_score = subject_accuracy(
        y_pred=np.concatenate(outputs["y_pred"], axis=0),
        y_true=np.concatenate(outputs["y_true"], axis=0),
        subject_ids=np.concatenate(outputs["subject_ids"], axis=0),
    )
    results["metrics/subjects_accuracy"] = subjects_score
    summary.scalar(
        "metrics/subjects_accuracy", value=subjects_score, step=epoch, mode=mode
    )
    return results


def main(args, wandb_sweep: bool = False):
    utils.set_random_seed(args.seed, verbose=args.verbose)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.use_wandb:
        utils.wandb_init(args, wandb_sweep=wandb_sweep)

    utils.get_device(args)
    if args.task_mode in (1, 2):
        utils.load_args(args, dir=args.path2pretraining_res)
    summary = tensorboard.Summary(args)

    train_ds, val_ds, test_ds = get_datasets(args, summary=summary)

    classifier, critic = get_models(args, summary=summary)
    if args.task_mode in (1, 2):
        load_pre_trained_parameters(
            args, classifier=classifier, path2pretraining_res=args.path2pretraining_res
        )

    optimizer_classifier = torch.optim.AdamW(
        params=[
            {
                "params": classifier.parameters() if args.task_mode in (1, 3)
                # only classification head is optimized in linear read-out
                else classifier.classifier.parameters(),
                "name": "classifier",
            }
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_critic = torch.optim.AdamW(
        params=[{"params": critic.parameters(), "name": "critic"}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler_classifier = Scheduler(
        args,
        model=classifier,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_classifier"),
        mode="max",
        optimizer=optimizer_classifier,
        lr_patience=args.lr_patience,
        min_epochs=args.min_epochs,
    )

    scheduler_critic = Scheduler(
        args,
        model=critic,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_critic"),
        mode="min",
        optimizer=optimizer_critic,
        lr_patience=args.lr_patience,
        min_epochs=args.min_epochs,
    )

    criteria = criterions.get_criterion(
        args,
    )

    utils.save_args(args)

    epoch = scheduler_classifier.restore(load_optimizer=True, load_scheduler=True)
    if args.split_mode == 1:
        _ = scheduler_critic.restore(load_optimizer=True, load_scheduler=True)

    results = {k: {} for k in ["train", "validation"]}
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.skip_training_loop:
            break
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")
        start = time()
        train_results = train(
            args,
            ds=train_ds,
            classifier=classifier,
            critic=critic,
            optimizer_classifier=optimizer_classifier,
            optimizer_critic=optimizer_critic,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
        )
        val_results = validate(
            args,
            ds=val_ds,
            classifier=classifier,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
        )
        elapse = time() - start

        summary.scalar("elapse", value=elapse, step=epoch, mode=0)
        summary.scalar(
            f"model/classifier/lr",
            value=optimizer_classifier.param_groups[0]["lr"],
            step=epoch,
        )
        if args.split_mode == 1:
            summary.scalar(
                f"model/critic/lr",
                value=optimizer_critic.param_groups[0]["lr"],
                step=epoch,
            )
        utils.update_dict(target=results["train"], source=train_results)
        utils.update_dict(target=results["validation"], source=val_results)
        if args.verbose:
            print(
                f'Train\t\tclassifier loss: {train_results["loss/classifier"]:.04f}\t'
                f'accuracy: {train_results["metrics/acc"]:.04f}\n'
                f'Validation\tclassifier loss: {val_results["loss/classifier"]:.04f}\t'
                f'accuracy: {val_results["metrics/acc"]:.04f}\n'
                f"Elapse: {elapse:.02f}s\n"
            )
        if args.split_mode == 1:
            scheduler_critic.step(train_results["loss/critic"], epoch=epoch)
            metric2optimize = val_results["metrics/subjects_accuracy"]
        else:
            metric2optimize = val_results["metrics/acc"]
        early_stop = scheduler_classifier.step(metric2optimize, epoch=epoch)
        if args.use_wandb:
            log = {
                "train_classifier_loss": train_results["loss/classifier"],
                "train_acc": train_results["metrics/acc"],
                "train_acc_subjects": train_results["metrics/subjects_accuracy"],
                "val_classifier_loss": val_results["loss/classifier"],
                "val_acc": val_results["metrics/acc"],
                "val_acc_subjects": val_results["metrics/subjects_accuracy"],
                "best_acc": scheduler_classifier.best_value,
                "elapse": elapse,
            }
            if args.split_mode == 1:
                log["train_critic_loss"] = train_results["loss/critic"]
            wandb.log(
                log,
                step=epoch,
            )
        if early_stop:
            break
        if np.isnan(train_results["loss/classifier"]) or np.isnan(
            val_results["loss/classifier"]
        ):
            if args.use_wandb:
                wandb.finish(exit_code=1)  # mark run as failed
            exit("\nNaN loss detected, terminate training.")
    if args.test_time:
        epoch = scheduler_classifier.restore()
        test_results = validate(
            args,
            ds=test_ds,
            classifier=classifier,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
            mode=2,
        )
        test_res = get_res(
            args,
            ds=test_ds,
            classifier=classifier,
            verbose=args.verbose,
        )
        (
            subjects_pred,
            subjects_true,
            subjects_scores,
        ) = secondary_metrics_subjects_get_inputs(
            y_pred=test_res["pred_probs"],
            y_true=test_res["targets"],
            subject_ids=test_res["labels"]["Sub_ID"],
        )
        log = {
            "test_loss": test_results["loss/classifier"],
            "test_acc": test_results["metrics/acc"],
            "test_acc_subjects": test_results["metrics/subjects_accuracy"],
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
                step=epoch,
            )
        if args.save_test_model_outputs:
            with open(
                os.path.join(args.output_dir, "test_model_outputs.pkl"), "wb"
            ) as file:
                pickle.dump(test_res, file)
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as file:
            json.dump(log, file)
        print(f"Test ACC={log['test_acc']}, ACC_subject={log['test_acc_subjects']}")
    with open(os.path.join(args.output_dir, "train_results.json"), "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument(
        "--e4selflearning",
        action="store_true",
        help="allows users with no access to INTREPIBD/TIMEBASE data to use "
        "the pre-trained E4mer for their own task",
    )
    parser.add_argument(
        "--task_mode",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="criterion for train/val/test split:"
        "0) Self-supervised learning"
        "1) Pre-trained encoder is fine-tuned"
        "2) Pre-trained encoder is frozen (features are simply "
        "read out) only the classification is trained on the target task"
        "3) Encoder and classification head are trained together end-to-end on "
        "target task directly"
        "4) Classical machine learning (XGBoost)"
        "9) Post-hoc analyses",
    )
    parser.add_argument(
        "--critic_score_lambda",
        type=float,
        default=0,
        help="when > 0, during training, the autoencoder model pays a price for "
        "encoding into h (i.e. the shared-between-tasks representation learned "
        "with feature_encoder) information that makes it easier for the critic "
        "model to tell subjects apart",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for DataLoader"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=20,
        help="number of epochs to train before enforcing in early stopping",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="number of epochs to wait before reducing lr.",
    )
    parser.add_argument(
        "--save_test_model_outputs", action="store_true", help="save test set outputs"
    )
    parser.add_argument(
        "--test_time", action="store_true", help="perform inference on test set"
    )
    parser.add_argument(
        "--skip_training_loop",
        action="store_true",
    )
    parser.add_argument(
        "--include_hr",
        action="store_true",
        help="HR is included as an input " "to the deep learning pipeline",
    )

    # optimizer configuration
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay L2 in AdamW optimizer",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="svg", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--plot_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="control which plots are printed"
        "0) no plots"
        "1) data summary plots"
        "2) training loop plots"
        "3) both data summary and training loop plots",
    )

    # misc
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--clear_output_dir", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default="")

    parser.add_argument(
        "--reuse_stats",
        action="store_true",
        help="reuse previously computed stats from either training or "
        "pre-training set for features scaling",
    )
    temp_args = parser.parse_known_args()[0]
    if temp_args.task_mode in (1, 2):
        parser.add_argument("--path2pretraining_res", type=str, required=True)
        # temp_args = utils.load_args(temp_args, dir= parser.parse_known_args()[
        #     0].path2pretraining_res)
        if temp_args.task_mode == 1:  # fine-tuning
            assert os.path.exists(parser.parse_known_args()[0].path2pretraining_res)
            parser.add_argument(
                "--a_dropout",
                type=float,
                default=0.0,
                help="dropout rate of MHA",
            )
            parser.add_argument(
                "--m_dropout",
                type=float,
                default=0.0,
                help="dropout rate of MLP",
            )
            parser.add_argument(
                "--drop_path",
                type=float,
                default=0.0,
                help="dropout rate of stochastic depth",
            )
    else:
        # dataset configuration
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="path to directory where preprocessed data are stored",
        )
        parser.add_argument(
            "--scaling_mode",
            type=int,
            default=2,
            choices=[0, 1, 2, 3],
            help="normalize features: "
            "0) no scaling "
            "1) normalize features by the overall min and max values from the "
            "training set"
            "2) standardize features by the overall mean and std from the training "
            "set"
            "3) standardize features by the overall median and iqr from the "
            "training set",
        )
        # channel embeddings configuration
        parser.add_argument(
            "--emb_num_filters",
            type=int,
            default=4,
            help="number of filters in the convolutional embedding",
        )
        # representation module configuration
        parser.add_argument(
            "--num_blocks", type=int, default=3, help="number of MHA blocks"
        )
        parser.add_argument(
            "--num_heads", type=int, default=3, help="number of attention heads"
        )
        parser.add_argument(
            "--num_units",
            type=int,
            default=64,
            help="number of hidden units, or embed_dim in MHA",
        )
        parser.add_argument(
            "--mlp_dim",
            type=int,
            default=64,
            help="hidden size in Transformer MLP",
        )
        parser.add_argument(
            "--a_dropout",
            type=float,
            default=0.0,
            help="dropout rate of MHA",
        )
        parser.add_argument(
            "--m_dropout", type=float, default=0.0, help="dropout rate of MLP"
        )
        parser.add_argument(
            "--drop_path",
            type=float,
            default=0.0,
            help="dropout rate of stochastic depth",
        )
        parser.add_argument(
            "--disable_bias",
            action="store_true",
            help="disable bias term in Transformer",
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
            "train/val/test such that subjects are not shared across splits",
        )
    del temp_args
    main(parser.parse_args())
