import torch
from torch import nn

from .conv import ConvBNAct


class LiteTokenMixer(nn.Module):
    """A lightweight token mixer implemented with grouped 1D convolution."""

    def __init__(self, channels: int, expand: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = channels * expand
        self.in_proj = nn.Linear(channels, hidden * 2)
        self.dwconv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(tokens).chunk(2, dim=-1)
        value = self.dwconv(value.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(self.act(value) * self.act(gate))


class LiteTokenChannelAttention(nn.Module):
    """Lite-Token Channel Attention Module.

    The module mixes spatial tokens with a compact sequence branch, preserves
    local context with depth-wise convolution, and recalibrates channels from
    global image statistics.
    """

    def __init__(self, channels: int, squeeze_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(channels // squeeze_ratio, 16)
        self.norm = nn.LayerNorm(channels)
        self.token_mixer = LiteTokenMixer(channels)
        self.local_refine = ConvBNAct(channels, channels, 3, groups=channels)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        mixed = self.token_mixer(self.norm(tokens))
        mixed = mixed.transpose(1, 2).reshape(batch, channels, height, width)
        local = self.local_refine(x)
        gate = self.channel_gate(x)
        return x + self.gamma * (mixed + local) * gate
