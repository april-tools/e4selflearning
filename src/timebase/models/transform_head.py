import typing as t

import torch
from einops import rearrange
from torch import nn

from timebase.data.static import *

############################## MASKED PREDICTION ##############################


class DecodeUpSample(nn.Module):
    def __init__(
        self,
        args,
        channel: str,
    ):
        super(DecodeUpSample, self).__init__()
        self.segment_length = args.ds_info["segment_length"]
        self.num_channels = len(args.ds_info["channel_freq"])
        self.channel_freq = args.ds_info["channel_freq"][channel]
        self.num_units = args.num_units
        self.transpose_conv = torch.nn.ConvTranspose1d(
            in_channels=self.num_units // 4,  # self.num_channels
            out_channels=1,
            kernel_size=self.channel_freq,
            stride=self.channel_freq,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor):
        outputs = rearrange(inputs, "b n c -> b c n")
        outputs = self.transpose_conv(outputs)
        outputs = rearrange(outputs, "b 1 n -> b n")
        return outputs


class Decoder(nn.Module):
    def __init__(self, args, input_shape: tuple):
        super(Decoder, self).__init__()
        self.output_shapes = args.input_shapes
        self.channel_names = sorted(args.input_shapes.keys())
        self.channel_freq = args.ds_info["channel_freq"]
        self.num_units = args.num_units
        # self.project = nn.Linear(
        #     in_features=self.num_units,
        #     out_features=len(self.channel_freq),
        #     bias=True,
        # )
        self.project = nn.Sequential(
            nn.Linear(
                in_features=self.num_units,
                out_features=self.num_units // 2,
                bias=True,
            ),
            nn.BatchNorm1d(input_shape[-2]),
            nn.GELU(),
            nn.Linear(
                in_features=self.num_units // 2,
                out_features=self.num_units // 4,
                bias=False,
            ),
            nn.BatchNorm1d(input_shape[-2]),
        )
        decoders = {
            channel: DecodeUpSample(
                args,
                channel=channel,
            )
            for channel, output_shape in self.output_shapes.items()
        }
        self.decoders_dict = nn.ModuleDict(decoders)

    def forward(self, inputs: torch.Tensor):
        embedding = self.project(inputs)
        outputs = {}
        for channel in self.channel_names:
            outputs[channel] = self.decoders_dict[channel](embedding)
        return outputs


########################## TRANSFORMATION PREDICTION ##########################
class TFClassificationHead(nn.Module):
    def __init__(self, args, input_shape: tuple):
        super(TFClassificationHead, self).__init__()
        self.bias = True
        self.num_channels = len(args.ds_info["channel_freq"])
        self.output_shapes = len(TRANSFORMATION_NAME_DICT)
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=input_shape[-1],
                out_features=input_shape[-1] // 2,
                bias=True,
            ),
            nn.BatchNorm1d(input_shape[-2]),
            nn.GELU(),
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=input_shape[-2])
        self.to_output = nn.Linear(
            in_features=input_shape[-1] // 2,
            out_features=self.output_shapes * self.num_channels,
            bias=self.bias,
        )

        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor):
        outputs = self.mlp(inputs)
        outputs = rearrange(outputs, "b n c -> b c n")
        outputs = self.avg_pool(outputs)
        outputs = rearrange(outputs, "b c 1 -> b c")
        outputs = self.to_output(outputs)
        outputs = rearrange(
            outputs, "b (t c) -> b t c", t=self.output_shapes, c=self.num_channels
        )
        outputs = self.soft_max(outputs)
        return outputs


################################# CONTRASTIVE #################################
class ProjectionHead(nn.Module):
    def __init__(self, args, input_shape: tuple):
        super(ProjectionHead, self).__init__()
        self.bias = True
        self.output_shapes = input_shape[-1] // 2
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=input_shape[-1],
                out_features=input_shape[-1] // 2,
                bias=True,
            ),
            nn.BatchNorm1d(input_shape[-2]),
            nn.GELU(),
            nn.Linear(
                in_features=input_shape[-1] // 2,
                out_features=input_shape[-1] // 4,
                bias=True,
            ),
            nn.BatchNorm1d(input_shape[-2]),
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=input_shape[-2])

    def forward(self, inputs: torch.Tensor):
        outputs = self.mlp(inputs)
        outputs = rearrange(outputs, "b n c -> b c n")
        outputs = self.avg_pool(outputs)
        outputs = rearrange(outputs, "b c 1 -> b c")
        return outputs
