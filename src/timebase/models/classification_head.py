import torch
from einops import rearrange
from torch import nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_shape: tuple):
        super(ClassifierMLP, self).__init__()
        self.input_shape = input_shape
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
        self.avg_pool = nn.AvgPool1d(kernel_size=self.input_shape[-2])
        self.to_output = nn.Linear(in_features=input_shape[-1] // 4, out_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.mlp(inputs)
        outputs = rearrange(outputs, "b n c -> b c n")
        outputs = self.avg_pool(outputs)
        outputs = rearrange(outputs, "b c 1 -> b c")
        outputs = self.to_output(outputs)
        return outputs
