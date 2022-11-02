import torch.nn as nn
import torch.nn.functional as F
import torch
import math
# import tensorboard
from models.builder import CORES
from .attention import Attention



@CORES.registerModule
class Core(nn.Module):
    def __init__(self, cfg, in_channels: int, out_channels: int, multi_head=3, attention_times=6, down_scale=0):
        super(Core, self).__init__()
        self.cfg = cfg

        self.attention = Attention(in_channels, out_channels,
            multi_head=multi_head, attention_times=attention_times, down_scale=down_scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.clone()
        output = self.attention(output)
        
        return output