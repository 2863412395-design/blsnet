from torch import nn

from .modules.base import BoundaryAwareSkipEnhancement
from .modules.conv import EfficientUpBlock, ResidualBlock
from .modules.lite_token import LiteTokenChannelAttention


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_lite_token: bool = True,
    ) -> None:
        super().__init__()
        self.up = EfficientUpBlock(in_channels, out_channels)
        self.base = BoundaryAwareSkipEnhancement(
            skip_channels=skip_channels,
            decoder_channels=out_channels,
            out_channels=out_channels,
        )
        self.mix = ResidualBlock(out_channels)
        self.context = (
            LiteTokenChannelAttention(out_channels)
            if use_lite_token
            else nn.Identity()
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip, boundary_logits = self.base(skip, x)
        x = self.context(self.mix(x + skip))
        return x, boundary_logits


class BLSDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: tuple[int, int, int, int],
        use_lite_token: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4),
            LiteTokenChannelAttention(c4) if use_lite_token else nn.Identity(),
        )
        self.stage3 = DecoderStage(c4, c3, c3, use_lite_token)
        self.stage2 = DecoderStage(c3, c2, c2, use_lite_token)
        self.stage1 = DecoderStage(c2, c1, c1, use_lite_token)
        self.out_channels = c1

    def forward(self, features):
        x1, x2, x3, x4 = features
        boundaries = []

        x = self.bottleneck(x4)
        x, b3 = self.stage3(x, x3)
        boundaries.append(b3)
        x, b2 = self.stage2(x, x2)
        boundaries.append(b2)
        x, b1 = self.stage1(x, x1)
        boundaries.append(b1)

        return x, boundaries
