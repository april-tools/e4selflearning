import typing as t

import torch
from einops import rearrange
from torch import nn


class ConvEmbedding(nn.Module):
    """MLP embedding layer for a channel
    Expected input shape: (batch size, num. steps)
    Output shape: (batch size, embedding dim, 1)
    """

    def __init__(self, args, channel: str):
        super(ConvEmbedding, self).__init__()
        self.channel = channel
        self.channel_freq = args.ds_info["channel_freq"][self.channel]
        self.out_channels = args.emb_num_filters
        self.tc = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=self.channel_freq,
                bias=False,
            ),
            nn.GELU(),
            nn.BatchNorm1d(self.out_channels),
            nn.MaxPool1d(kernel_size=self.channel_freq, stride=self.channel_freq),
        )

    def pad_inputs(self, inputs, stride: int = 1):
        kernel_size = self.channel_freq
        output_length = (inputs.size(1) - 1) * stride + kernel_size
        padding_left = (output_length - inputs.size(1)) // 2
        padding_right = output_length - inputs.size(1) - padding_left
        padded_input = torch.nn.functional.pad(
            inputs, (padding_left, padding_right), mode="constant", value=0
        )
        return padded_input

    def forward(self, inputs: torch.Tensor):
        outputs = self.pad_inputs(inputs)
        outputs = rearrange(outputs, "b n -> b 1 n")
        outputs = self.tc(outputs)
        outputs = rearrange(outputs, "b d n -> b n d")
        return outputs


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.input_shapes = args.input_shapes
        self.channel_names = sorted(self.input_shapes.keys())
        self.channel_freq = args.ds_info["channel_freq"]
        self.segment_length = args.ds_info["segment_length"]
        self.out_channels = args.emb_num_filters

        self.output_shape = (
            self.segment_length,
            len(self.channel_names) * self.out_channels,
        )

        encoder = {
            channel: ConvEmbedding(
                args,
                channel=channel,
            )
            for channel, input_shape in self.input_shapes.items()
        }
        self.embedding = nn.ModuleDict(encoder)

    def forward(self, inputs: t.Dict[str, torch.Tensor]):
        """output shape: (batch_size, emb_dim (i.e. time-steps in new space),
        num. channels)"""
        outputs = []
        for channel in self.channel_names:
            outputs.append(self.embedding[channel](inputs[channel]))
        outputs = torch.cat(outputs, dim=-1)
        return outputs
