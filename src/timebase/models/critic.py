import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class CriticMLP(nn.Module):
    def __init__(self, args, input_shape: tuple):
        super(CriticMLP, self).__init__()
        self.input_shape = input_shape
        self.output_shapes = args.num_train_subjects
        self.avg_pool = nn.AvgPool1d(kernel_size=self.input_shape[-1])
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.input_shape[0], out_features=self.input_shape[0] // 2
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.input_shape[0] // 2, out_features=self.output_shapes
            ),
        )

    def forward(self, inputs: torch.Tensor, activate: bool = True) -> torch.Tensor:
        outputs = self.avg_pool(inputs)
        outputs = rearrange(outputs, "b n 1 -> b n")
        outputs = self.mlp(outputs)
        if activate:
            outputs = F.softmax(outputs, dim=-1)
        return outputs
