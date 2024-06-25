import argparse
import json
import shutil
import typing as t
from time import time

import einops
import torch
import wandb
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from timebase import criterions
from timebase.criterions import Criteria
from timebase.data.reader import get_datasets
from timebase.data.static import *
from timebase.models.models import Reconstructor
from timebase.models.models import TFClassifier
from timebase.models.models import get_models
from timebase.models.models import simCLRlike
from timebase.utils import plots
from timebase.utils import tensorboard
from timebase.utils import utils
from timebase.utils import yaml
from timebase.utils.scheduler import Scheduler


# Custom validation function for argparse
def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            "Value must be an integer greater than or equal to 1"
        )
    return ivalue


# Custom validation function
def zero_one_range(value):
    ivalue = float(value)
    if (ivalue < 0) or (ivalue > 1):
        raise argparse.ArgumentTypeError("Value must be in range [0, 1]")
    return ivalue


def positive_real(value):
    ivalue = float(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be a positive real")
    return ivalue


def load(d: t.Dict[str, torch.Tensor], device: torch.device):
    """Load values in dictionary d to device"""
    return {k: v.to(device) for k, v in d.items()}


@torch.inference_mode()
def get_res(
    args,
    ds: DataLoader,
    sslearner: t.Union[Reconstructor, TFClassifier, simCLRlike],
    verbose: int = 1,
):
    device = args.device
    match args.pretext_task:
        case "masked_prediction":
            input_name = "original"
        case "transformation_prediction":
            input_name = "transformed_data"
        case "contrastive":
            input_name = "x1"
        case _:
            raise NotImplementedError(f"{args.pretext_task} not yet implemented")
    representations, collections = [], []
    sslearner.to(device)
    sslearner.train(False)
    for batch in tqdm(ds, disable=verbose == 0):
        original = load(batch[f"{input_name}"], device=device)
        _, representation = sslearner(original)
        representations.append(einops.rearrange(representation, "b n d -> b (n d)"))
        collections.append(batch["collection"].to(device))
    return {
        "representations": torch.concat(representations, dim=0).cpu().numpy(),
        "collections": torch.cat(collections, dim=0).cpu().numpy(),
    }


def make_plots(
    args,
    ds: DataLoader,
    sslearner: t.Union[Reconstructor, TFClassifier],
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    res = get_res(
        args,
        ds=ds,
        sslearner=sslearner,
        verbose=args.verbose,
    )
    plots.training_loop_plots(
        args,
        summary=summary,
        res=res,
        step=epoch,
        mode=mode,
    )


def train_step_masked_prediction(
    batch: t.Dict[str, t.Any],
    sslearner: Reconstructor,
    optimizer_sslearner: torch.optim.Optimizer,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    original = load(batch["original"], device=device)
    corrupted = load(batch["corrupted"], device=device)
    mask = load(batch["mask"], device=device)

    sslearner.train(True)
    outputs_reconstructor, _ = sslearner(inputs=corrupted)
    reconstructor_loss = criteria.criterion_sslearner(
        x_true=original, x_hat=outputs_reconstructor, mask=mask
    )
    reconstructor_loss.backward()
    optimizer_sslearner.step()
    optimizer_sslearner.zero_grad()
    result.update(
        {
            "loss/rmse": reconstructor_loss.detach(),
        }
    )
    return result


def train_step_transformation_prediction(
    batch: t.Dict[str, t.Any],
    sslearner: TFClassifier,
    optimizer_sslearner: torch.optim.Optimizer,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    transformed_data = load(batch["transformed_data"], device=device)
    transformations = batch["transformation"].to(device)

    sslearner.train(True)
    outputs_sslearner, _ = sslearner(transformed_data)
    cross_entropy, acc = criteria.criterion_sslearner(
        y_true=transformations, y_pred=outputs_sslearner
    )
    cross_entropy.backward()
    optimizer_sslearner.step()
    optimizer_sslearner.zero_grad()
    result.update(
        {
            "loss/cross_entropy": cross_entropy.detach(),
            "metric/acc": acc.detach(),
        }
    )
    return result


def train_step_contrastive(
    batch: t.Dict[str, t.Any],
    sslearner: simCLRlike,
    optimizer_sslearner: torch.optim.Optimizer,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    x1 = load(batch["x1"], device=device)
    x2 = load(batch["x2"], device=device)

    sslearner.train(True)
    z1, _ = sslearner(x1)
    z2, _ = sslearner(x2)
    nt_xent, acc = criteria.criterion_sslearner(proj_1=z1, proj_2=z2)
    nt_xent.backward()
    optimizer_sslearner.step()
    optimizer_sslearner.zero_grad()
    result.update(
        {
            "loss/nt_xent": nt_xent.detach(),
            "metric/acc": acc.detach(),
        }
    )
    return result


TRAIN_STEP_DICT = {
    "masked_prediction": train_step_masked_prediction,
    "transformation_prediction": train_step_transformation_prediction,
    "contrastive": train_step_contrastive,
}


def train(
    args,
    ds: DataLoader,
    sslearner: t.Union[Reconstructor, TFClassifier, simCLRlike],
    optimizer_sslearner: torch.optim.Optimizer,
    criteria: Criteria,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    for batch in tqdm(ds, desc="Train", disable=args.verbose <= 1):
        result = TRAIN_STEP_DICT[args.pretext_task](
            batch=batch,
            sslearner=sslearner,
            optimizer_sslearner=optimizer_sslearner,
            criteria=criteria,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


@torch.inference_mode()
def validation_step_masked_prediction(
    batch: t.Dict[str, t.Any],
    sslearner: Reconstructor,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    original = load(batch["original"], device=device)
    corrupted = load(batch["corrupted"], device=device)
    mask = load(batch["mask"], device=device)

    sslearner.train(False)
    outputs_reconstructor, _ = sslearner(inputs=corrupted)
    reconstructor_loss = criteria.criterion_sslearner(
        x_true=original, x_hat=outputs_reconstructor, mask=mask
    )
    result.update(
        {
            "loss/rmse": reconstructor_loss,
        }
    )
    return result


@torch.inference_mode()
def validation_step_transformation_prediction(
    batch: t.Dict[str, t.Any],
    sslearner: TFClassifier,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    transformed_data = load(batch["transformed_data"], device=device)
    transformations = batch["transformation"].to(device)
    outputs_sslearner, _ = sslearner(transformed_data)
    cross_entropy, acc = criteria.criterion_sslearner(
        y_true=transformations, y_pred=outputs_sslearner
    )
    result.update(
        {
            "loss/cross_entropy": cross_entropy,
            "metric/acc": acc,
        }
    )
    return result


@torch.inference_mode()
def validation_step_contrastive(
    batch: t.Dict[str, t.Any],
    sslearner: simCLRlike,
    criteria: Criteria,
    device: torch.device,
):
    result = {}
    sslearner.to(device)
    x1 = load(batch["x1"], device=device)
    x2 = load(batch["x2"], device=device)
    z1, _ = sslearner(x1)
    z2, _ = sslearner(x2)
    nt_xent, acc = criteria.criterion_sslearner(proj_1=z1, proj_2=z2)
    result.update(
        {
            "loss/nt_xent": nt_xent,
            "metric/acc": acc,
        }
    )
    return result


VAL_STEP_DICT = {
    "masked_prediction": validation_step_masked_prediction,
    "transformation_prediction": validation_step_transformation_prediction,
    "contrastive": validation_step_contrastive,
}


def validate(
    args,
    ds: DataLoader,
    sslearner: t.Union[Reconstructor, TFClassifier, simCLRlike],
    criteria: Criteria,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results = {}
    for batch in tqdm(ds, desc="Validate", disable=args.verbose <= 1):
        result = VAL_STEP_DICT[args.pretext_task](
            batch=batch,
            sslearner=sslearner,
            criteria=criteria,
            device=args.device,
        )
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = torch.mean(torch.stack(v)).item()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
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
    summary = tensorboard.Summary(args)
    # 0: Self-supervised pre-training, 1: Fine-tuning, 2: Read-out, 3: NN
    # training, 4: CML (XGBoost), 9: Post-hoc analyses
    args.task_mode = 0

    (
        pretext_train_ds,
        pretext_val_ds,
        pretext_test_ds,
    ) = get_datasets(args, summary=summary)

    sslearner = get_models(args, summary=summary)

    optimizer_sslearner = torch.optim.AdamW(
        params=[{"params": sslearner.parameters(), "name": "sslearner"}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler_sslearner = Scheduler(
        args,
        model=sslearner,
        checkpoint_dir=os.path.join(args.output_dir, "ckpt_sslearner"),
        mode="min",
        optimizer=optimizer_sslearner,
        lr_patience=args.lr_patience,
        min_epochs=args.min_epochs,
    )

    criteria = criterions.get_criterion(args)

    utils.save_args(args)

    epoch = scheduler_sslearner.restore(load_optimizer=True, load_scheduler=True)

    match args.pretext_task:
        case "masked_prediction":
            loss_name = "rmse"
        case "transformation_prediction":
            loss_name = "cross_entropy"
        case "contrastive":
            loss_name = "nt_xent"
        case _:
            raise NotImplementedError(f"{args.pretext_task} not yet implemented")

    results = {k: {} for k in ["train", "validation"]}

    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=pretext_train_ds,
            sslearner=sslearner,
            optimizer_sslearner=optimizer_sslearner,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
        )
        val_results = validate(
            args,
            ds=pretext_val_ds,
            sslearner=sslearner,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
        )
        elapse = time() - start

        summary.scalar("elapse", value=elapse, step=epoch, mode=0)
        summary.scalar(
            f"model/sslearner/lr",
            value=optimizer_sslearner.param_groups[0]["lr"],
            step=epoch,
        )
        utils.update_dict(target=results["train"], source=train_results)
        utils.update_dict(target=results["validation"], source=val_results)
        if args.verbose:
            print(
                f"Train\t\t {loss_name}:"
                f' {train_results[f"loss/{loss_name}"]:.04f}\n'
                f"Validate\t\t {loss_name}:"
                f' {val_results[f"loss/{loss_name}"]:.04f}\n'
                f"Test\t\t {loss_name}:"
                f"Elapse: {elapse:.02f}s\n"
            )

        early_stop = scheduler_sslearner.step(
            val_results[f"loss/{loss_name}"], epoch=epoch
        )
        if args.use_wandb:
            log = {
                "train_ssl_loss": train_results[f"loss/{loss_name}"],
                "val_ssl_loss": val_results[f"loss/{loss_name}"],
                "lowest_val_loss": scheduler_sslearner.best_value,
                "elapse": elapse,
            }
            if args.pretext_task in ("contrastive", "transformation_prediction"):
                log["train_ssl_metric"] = train_results["metric/acc"]
                log["val_ssl_metric"] = val_results["metric/acc"]
            wandb.log(
                log,
                step=epoch,
            )
        if early_stop:
            break
        if np.isnan(train_results[f"loss/{loss_name}"]) or np.isnan(
            val_results[f"loss/{loss_name}"]
        ):
            if args.use_wandb:
                wandb.finish(exit_code=1)  # mark run as failed
            exit("\nNaN loss detected, terminate training.")

    if (args.test_time) and (not args.e4selflearning):
        epoch = scheduler_sslearner.restore()
        test_results = validate(
            args,
            ds=pretext_test_ds,
            sslearner=sslearner,
            criteria=criteria,
            summary=summary,
            epoch=epoch,
            mode=2,
        )
        log = {
            "test_ssl_loss": test_results[f"loss/{loss_name}"],
        }
        if args.pretext_task in ("contrastive", "transformation_prediction"):
            log["test_ssl_metric"] = test_results["metric/acc"]
        if args.use_wandb:
            wandb.log(
                log,
                step=epoch,
            )
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as file:
            json.dump(log, file)

    yaml.save(
        filename=os.path.join(args.output_dir, "train_results.yaml"), data=results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument(
        "--test_time", action="store_true", help="perform inference on test set"
    )
    parser.add_argument(
        "--filter_collections",
        nargs="*",
        choices=[
            "barcelona",
            "adarp",
            "big-ideas",
            "in-gauge_en-gage",
            "k_emocon",
            "pgg_dalia",
            "spd",
            "stress_detection_nurses_hospital",
            "toadstool",
            "ue4w",
            "weee",
            "wesad",
            "wesd",
        ],
        required=False,
        help="Select which collections should be used for SS pre-training",
    )
    parser.add_argument(
        "--unlabelled_data_resampling_percentage",
        type=zero_one_range,
        required=False,
        default=1,
        help="Select which portion [0-1] of unlabelled data should be "
        "resampled for self-supervised pre-training",
    )
    parser.add_argument(
        "--pretext_task",
        type=str,
        choices=["masked_prediction", "transformation_prediction", "contrastive"],
        help="criterion for train/val/test split:"
        "1) masked_prediction: parts of the input are selected with a mask and "
        "corrupted; the network (encoder + transform head) is trained to impute "
        "the missing (corrupted) values"
        "2) transformation_prediction: some transformations are sampled from a "
        "set of transformations and applied across channels; the network"
        " (encoder + transform head) is trained to guess what transformation (if"
        " any) was applied"
        "3) contrastive: pairs of segments from the same session are "
        "considered positive pairs where pairs from other sessions are "
        "treated as negative examples; network (encoder + transform head) is "
        "trained to map positive pairs close together in the label space and "
        "negative pairs distant apart from each other",
    )
    parser.add_argument(
        "--include_hr",
        action="store_true",
        help="HR is included as an input to the deep learning pipeline",
    )
    parser.add_argument(
        "--e4selflearning",
        action="store_true",
        help="allows users with no access to INTREPIBD/TIMEBASE data to do "
        "self-supervised pre-training on the datasets forming the "
        "E4SelfLearning unlabelled collection",
    )
    # pretext task specific settings
    temp_args = parser.parse_known_args()[0]
    match temp_args.pretext_task:
        case "masked_prediction":
            parser.add_argument(
                "--masking_ratio",
                type=zero_one_range,
                default=0.15,
                help="Proportion (in wall-time seconds) of channel values to be masked.",
            )
            parser.add_argument(
                "--lm",
                type=positive_int,
                default=3,
                help="Average length (in wall-time seconds) of masking subsequences ("
                "streaks of 0s), i.e. mean of a Geometric Distribution, counting "
                "the number of 0s before getting a one",
            )
            parser.add_argument(
                "--overwrite_masks",
                action="store_true",
                help="create new masks",
            )
            parser.add_argument(
                "--exclude_anomalies",
                action="store_true",
                help="subjects with an acute mood episode (i.e. cases in the "
                "target task) are excluded from self-supervised "
                "pre-training",
            )

        case "transformation_prediction":
            parser.add_argument(
                "--snr",
                type=positive_real,
                default=1.5,
                help="signal to noise ratio in Gaussian Noise Transformation",
            )
            parser.add_argument(
                "--num_sub_segments",
                type=positive_int,
                default=4,
                help="number of sub-segments the segment should be split when "
                "applying cropping, permutation, time warp",
            )
            parser.add_argument(
                "--stretch_factor",
                type=positive_int,
                default=4,
                help="stretching factor to apply in time-warp",
            )
        case "contrastive":
            parser.add_argument(
                "--temperature",
                type=positive_real,
                default=0.5,
                help="signal to noise ratio in Gaussian Noise Transformation",
            )

    del temp_args
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
    parser.add_argument(
        "--reuse_stats",
        action="store_true",
        help="reuse previously computed stats from training set for features "
        "scaling",
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
    main(parser.parse_args())
