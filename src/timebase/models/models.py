import typing as t

import torch
import torchinfo
import wandb
from torch import nn

from timebase.data.static import *
from timebase.models.channel_embeddings import Embedding
from timebase.models.classification_head import ClassifierMLP
from timebase.models.critic import CriticMLP
from timebase.models.representation_module import Transformer
from timebase.models.transform_head import Decoder
from timebase.models.transform_head import ProjectionHead
from timebase.models.transform_head import TFClassificationHead
from timebase.utils import tensorboard


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any]],
    filename: str = None,
    tag: str = "model/trainable_parameters",
    summary: tensorboard.Summary = None,
    device: torch.device = torch.device("cpu"),
):
    model_info = torchinfo.summary(
        model,
        input_data=input_data,
        depth=5,
        device=device,
        verbose=0,
    )
    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    if summary is not None:
        summary.scalar(tag, model_info.trainable_params)
    return model_info


class GlbAvgPool(nn.Module):
    def __init__(self, input_shape, dim: t.Literal[-1, -2]):
        super(GlbAvgPool, self).__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.output_shape = (
            self.input_shape[-1] if self.dim == -2 else self.input_shape[-2]
        )

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=False)
        return mean


class Classifier(nn.Module):
    def __init__(self, args, input_shape: tuple = None):
        super(Classifier, self).__init__()
        self.task_mode = args.task_mode
        match self.task_mode:
            # case 1: fine-tuning, case 2: read-out
            case 1 | 2:
                self.sslearner = get_sslearner(args)
                self.classifier = ClassifierMLP(
                    input_shape=self.sslearner.feature_encoder.output_shape
                )
            # representations are learned directly on the main task in a
            # supervised way
            case 3:
                self.channel_embedding = Embedding(args)
                self.feature_encoder = Transformer(
                    args, input_shape=self.channel_embedding.output_shape
                )
                self.classifier = ClassifierMLP(
                    input_shape=self.feature_encoder.output_shape
                )

    def forward(self, inputs: t.Dict[str, torch.Tensor]):
        match self.task_mode:
            case 1 | 2:
                representation = self.sslearner(inputs)
                outputs = self.classifier(representation)
                return outputs, representation
            case 3:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                outputs = self.classifier(representation)
                return outputs, representation


class Reconstructor(nn.Module):
    def __init__(self, args):
        super(Reconstructor, self).__init__()
        self.task_mode = args.task_mode
        self.channel_embedding = Embedding(args)
        self.feature_encoder = Transformer(
            args, input_shape=self.channel_embedding.output_shape
        )
        if self.task_mode == 0:
            self.decoder = Decoder(args, input_shape=self.feature_encoder.output_shape)

    def forward(
        self,
        inputs: t.Dict[str, torch.Tensor],
    ) -> (torch.Tensor, torch.Tensor):
        match self.task_mode:
            case 0:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                outputs = self.decoder(inputs=representation)
                return outputs, representation
            case 1 | 2:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                return representation
            case _:
                raise ValueError("Task mode should be in [0, 1, 2]")


class TFClassifier(nn.Module):
    def __init__(self, args):
        super(TFClassifier, self).__init__()
        self.task_mode = args.task_mode
        self.channel_embedding = Embedding(args)
        self.feature_encoder = Transformer(
            args, input_shape=self.channel_embedding.output_shape
        )
        if self.task_mode == 0:
            self.tf_classification_head = TFClassificationHead(
                args, input_shape=self.feature_encoder.output_shape
            )

    def forward(
        self, inputs: t.Dict[str, torch.Tensor]
    ) -> (torch.Tensor, torch.Tensor):
        match self.task_mode:
            case 0:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                outputs = self.tf_classification_head(representation)
                return outputs, representation
            case 1 | 2:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                return representation
            case _:
                raise ValueError("Task mode should be in [0, 1, 2]")


class simCLRlike(nn.Module):
    def __init__(self, args):
        super(simCLRlike, self).__init__()
        self.task_mode = args.task_mode
        self.channel_embedding = Embedding(args)
        self.feature_encoder = Transformer(
            args, input_shape=self.channel_embedding.output_shape
        )
        if self.task_mode == 0:
            self.projection_head = ProjectionHead(
                args, input_shape=self.feature_encoder.output_shape
            )

    def forward(
        self, inputs: t.Dict[str, torch.Tensor]
    ) -> (torch.Tensor, torch.Tensor):
        match self.task_mode:
            case 0:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                outputs = self.projection_head(representation)
                return outputs, representation
            case 1 | 2:
                outputs = self.channel_embedding(inputs)
                representation = self.feature_encoder(outputs)
                return representation
            case _:
                raise ValueError("Task mode should be in [0, 1, 2]")


def get_sslearner(args):
    match args.pretext_task:
        case "masked_prediction":
            sslearner = Reconstructor(args)
        case "transformation_prediction":
            sslearner = TFClassifier(args)
        case "contrastive":
            sslearner = simCLRlike(args)
        case _:
            raise NotImplementedError("Function not yet implemented")
    return sslearner


class Critic(nn.Module):
    def __init__(self, args, input_shape: tuple):
        super(Critic, self).__init__()
        self.critic = CriticMLP(args, input_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.critic(inputs)
        return outputs


class NoiseInjector(nn.Module):
    def __init__(self, args):
        super(NoiseInjector, self).__init__()
        self.device = args.device
        loaded_masks = np.load(os.path.join(args.dataset, "masks.npz"))
        self.masks = (
            torch.from_numpy(loaded_masks["data"]).to(self.device).to(torch.bool)
        )
        self.channel_freq = args.ds_info["channel_freq"]
        self.num_masks = NUM_MASKS
        self.scaling_mode = args.scaling_mode
        assert self.scaling_mode in (1, 2, 3)

    def create_noise_mask(self, mask):
        if self.scaling_mode in (2, 3):
            noise = torch.normal(mean=0, std=1, size=mask.shape, device=self.device)
        else:
            noise = torch.randn(size=mask.shape, device=self.device)
        return torch.where(mask, noise, torch.zeros_like(mask, dtype=noise.dtype))

    def forward(
        self, inputs: t.Dict[str, torch.Tensor]
    ) -> (t.Dict[str, torch.Tensor], t.Dict[str, torch.Tensor]):
        with torch.no_grad():
            mask, corrupted = self._forward_impl(inputs)
        return mask, corrupted

    def _forward_impl(
        self, inputs: t.Dict[str, torch.Tensor]
    ) -> (t.Dict[str, torch.Tensor], t.Dict[str, torch.Tensor]):
        mask, corrupted = {}, {}
        for channel_name, values in inputs.items():
            num_rows = self.masks.shape[0]
            permuted_indices = torch.randperm(num_rows)
            self.masks = self.masks[permuted_indices]
            batch_size = values.shape[0]
            channel_mask = torch.repeat_interleave(
                input=self.masks[:batch_size],
                repeats=self.channel_freq[channel_name],
                dim=1,
            )
            noise_mask = self.create_noise_mask(mask=channel_mask)
            corrupted[channel_name] = values + noise_mask
            mask[channel_name] = channel_mask
        return mask, corrupted


def get_models(args, summary: tensorboard.Summary = None) -> (Classifier, Critic):
    match args.task_mode:
        case 0:
            sslearner = get_sslearner(args)
            sslearner_info = get_model_info(
                model=sslearner,
                input_data=[
                    {
                        channel: torch.randn(args.batch_size, *input_shape)
                        for channel, input_shape in args.input_shapes.items()
                    }
                ],
                filename=os.path.join(args.output_dir, "sslearner.txt"),
                summary=summary,
            )
            if args.verbose > 2:
                print(str(sslearner_info))
            if args.use_wandb:
                log = {"sslearner_size": sslearner_info.trainable_params}
                wandb.log(log, step=0)
            sslearner.to(args.device)
            return sslearner
        case 1 | 2 | 3:
            classifier = Classifier(args)
            critic = Critic(
                args,
                input_shape=classifier.sslearner.feature_encoder.output_shape
                if args.task_mode in (1, 2)
                else classifier.feature_encoder.output_shape,
            )
            classifier_info = get_model_info(
                model=classifier,
                input_data=[
                    {
                        channel: torch.randn(args.batch_size, *input_shape)
                        for channel, input_shape in args.input_shapes.items()
                    }
                ],
                filename=os.path.join(args.output_dir, "classifier.txt"),
                summary=summary,
            )
            if args.verbose > 2:
                print(str(classifier_info))
            critic_info = get_model_info(
                model=critic,
                input_data=torch.randn(args.batch_size, *critic.critic.input_shape),
                filename=os.path.join(args.output_dir, "critic.txt"),
                summary=summary,
            )
            if args.verbose > 2:
                print(str(critic_info))
            if args.use_wandb:
                log = {
                    "classifier_size": classifier_info.trainable_params,
                }
                if args.split_mode == 1:
                    log["critic_size"] = critic_info.trainable_params
                wandb.log(
                    log,
                    step=0,
                )

            classifier.to(args.device)
            critic.to(args.device)
            return classifier, critic
