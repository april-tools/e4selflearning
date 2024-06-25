import argparse
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial

import wandb

import train_cml


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
        learner: str,
        num_workers: int = 2,
        verbose: int = 1,
        split_mode: int = 0,
        path2featurizer: str = None,
    ):
        self.learner = learner
        self.output_dir = os.path.join(
            output_dir, f"{datetime.now():%Y%m%d-%Hh%Mm}-{id}"
        )
        self.path2featurizer = path2featurizer
        self.batch_size = 256
        self.num_workers = num_workers
        self.device = None
        self.split_mode = split_mode
        self.dataset = "data/preprocessed/sl512_ss128"
        self.seed = 1234
        self.save_test_model_outputs = False
        self.test_time = False
        self.reuse_stats = True
        self.format = "svg"
        self.dpi = 120
        self.verbose = verbose
        self.clear_output_dir = False
        self.use_wandb = True
        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def main(
    output_dir: str,
    wandb_group: str,
    learner: str,
    num_workers: int = 2,
    verbose: int = 1,
    split_mode: int = None,
    path2featurizer: str = None,
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
        path2featurizer=path2featurizer,
        verbose=verbose,
        learner=learner,
    )
    train_cml.main(args, wandb_sweep=True)


@forked
def agent(params):
    wandb.agent(
        sweep_id=params.sweep_id,
        function=partial(
            main,
            output_dir=params.output_dir,
            wandb_group=params.wandb_group,
            verbose=params.verbose,
            split_mode=params.split_mode,
            path2featurizer=params.path2featurizer,
            learner=params.learner,
        ),
        count=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--wandb_group", type=str, required=True)
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
    parser.add_argument("--path2featurizer", type=str, required=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--learner", type=str, default=None, choices=["xgboost", "svm", "knn", "enet"]
    )
    params = parser.parse_args()
    for _ in range(params.num_trials):
        agent(params)
