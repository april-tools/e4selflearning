import argparse
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial

import wandb

import pre_train as pre_text_trainer
import train_ann as target_trainer


def forked(fn):
    """
    Does not work on Windows (except WSL2), since the fork syscall is not supported here.
    fork creates a new process which inherits all the memory without it being copied.
    Memory is copied on write instead, meaning it is very cheap to create a new process
    Reference: https://gist.github.com/schlamar/2311116?permalink_comment_id=3932763#gistcomment-3932763
    """

    def call(*args, **kwargs):
        ctx = mp.get_context("fork")
        q = ctx.Queue(1)
        is_error = ctx.Value("b", False)

        def target():
            try:
                q.put(fn(*args, **kwargs))
            except BaseException as e:
                is_error.value = True
                q.put(e)

        ctx.Process(target=target).start()
        result = q.get()
        if is_error.value:
            raise result
        return result

    return call


class Args:
    def __init__(
        self,
        id: str,
        config: wandb.Config,
        output_dir: str,
        num_workers: int = 2,
        verbose: int = 1,
        split_mode: int = 0,
        task_mode: int = None,
        pretext_task: str = None,
        path2pretraining_res: str = None,
    ):
        self.output_dir = os.path.join(
            output_dir, f"{datetime.now():%Y%m%d-%Hh%Mm}-{id}"
        )
        self.task_mode = task_mode
        self.epochs = 250
        self.device = None
        self.batch_size = 256

        if self.task_mode not in (1, 2):
            self.dataset = "data/preprocessed/sl512_ss128"
            self.split_mode = split_mode
            self.scaling_mode = 2
        if self.task_mode in (1, 2):
            self.path2pretraining_res = path2pretraining_res
        if pretext_task:
            self.pretext_task = pretext_task
            self.filter_collections = None
            self.downsize_pre_training = 1
            self.exclude_anomalies = False
            match self.pretext_task:
                case "masked_prediction":
                    self.masking_ratio = 0.15
                    self.lm = 3
                    self.overwrite_masks = False
                case "transformation_prediction":
                    self.snr = 1.5
                    self.num_sub_segments = 4
                    self.stretch_factor = 4
                case "contrastive":
                    self.temperature = 0.5
        if self.task_mode in (1, 2, 3) and split_mode == 0:
            self.critic_score_lambda = 0
        self.seed = 1234
        self.num_workers = num_workers
        self.min_epochs = 50
        self.lr_patience = 10
        self.save_test_model_outputs = False
        self.test_time = False
        self.reuse_stats = True
        self.save_plots = False
        self.format = "svg"
        self.dpi = 120
        self.plot_mode = 0
        self.verbose = verbose
        self.clear_output_dir = False
        self.use_wandb = True
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def main(
    output_dir: str,
    wandb_group: str,
    num_workers: int = 2,
    verbose: int = 1,
    split_mode: int = None,
    task_mode: int = None,
    pretext_task: str = None,
    path2pretraining_res: str = None,
):
    run = wandb.init(group=wandb_group)
    config = run.config
    run.name = run.id
    args = Args(
        id=run.id,
        config=config,
        output_dir=output_dir,
        num_workers=num_workers,
        split_mode=split_mode,
        task_mode=task_mode,
        pretext_task=pretext_task,
        path2pretraining_res=path2pretraining_res,
        verbose=verbose,
    )
    if task_mode in (1, 2, 3):
        target_trainer.main(args, wandb_sweep=True)
    else:
        pre_text_trainer.main(args, wandb_sweep=True)


@forked
def agent(params):
    wandb.agent(
        sweep_id=params.sweep_id,
        function=partial(
            main,
            output_dir=params.output_dir,
            wandb_group=params.wandb_group,
            num_workers=params.num_workers,
            verbose=params.verbose,
            split_mode=params.split_mode,
            task_mode=params.task_mode,
            pretext_task=params.pretext_task,
            path2pretraining_res=params.path2pretraining_res,
        ),
        count=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--wandb_group", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="number of trials to run with this agent",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--split_mode",
        type=int,
        default=0,
        choices=[0, 1],
        required=False,
        help="criterion for train/val/test split:"
        "0) time-split: each session is split into 70:15:15 along the temporal "
        "dimension such that segments from different splits map to "
        "different parts of the recording"
        "1) subject-split: cases and controls are split into 70:15:15 "
        "train/val/test such that subjects are not shared across splits",
    )
    parser.add_argument(
        "--task_mode",
        type=int,
        choices=[1, 2, 3],
        required=False,
        help="criterion for train/val/test split:"
        "1) Pre-trained encoder is fine-tuned"
        "2) Pre-trained encoder is frozen (features are simply "
        "read out) only the classification is trained on the target task"
        "3) Encoder and classification head are trained together end-to-end on "
        "target task directly",
    )
    parser.add_argument("--path2pretraining_res", type=str, required=False)
    if (not parser.parse_known_args()[0].path2pretraining_res) and (
        parser.parse_known_args()[0].task_mode in (1, 2)
    ):
        raise Exception(
            "--path2pretraining_res to be specified when --task_mode is in (1, 2)"
        )
    parser.add_argument(
        "--pretext_task",
        type=str,
        choices=["masked_prediction", "transformation_prediction", "contrastive"],
        required=False,
        help="criterion for train/val/test split:"
        "masked_prediction: parts of the input are selected with a mask and "
        "corrupted; the representation module is trained to impute the missing "
        "(corrupted) values"
        "transformation_prediction: some transformations are sampled from a set of "
        "transformations and applied across channels; the representation "
        "module is trained to guess what transformation (if any) was applied"
        "contrastive:",
    )
    if (not parser.parse_known_args()[0].task_mode) and (
        not parser.parse_known_args()[0].pretext_task
    ):
        raise Exception("--pretext_task to be specified in self-supervised training")
    params = parser.parse_args()

    for _ in range(params.num_trials):
        agent(params)
