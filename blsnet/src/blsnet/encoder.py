from torch import nn

from .modules.conv import ConvBNAct, ResidualBlock


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stage = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.stage(x)


class PyramidEncoder(nn.Module):
    """Compact pyramid encoder with a PVT-style multi-scale interface."""

    def __init__(
        self,
        input_channels: int = 3,
        channels: tuple[int, int, int, int] = (64, 128, 320, 512),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            ConvBNAct(input_channels, channels[0], kernel_size=7, stride=2),
            ResidualBlock(channels[0]),
        )
        self.stage2 = EncoderStage(channels[0], channels[1], stride=2)
        self.stage3 = EncoderStage(channels[1], channels[2], stride=2)
        self.stage4 = EncoderStage(channels[2], channels[3], stride=2)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4


def build_encoder(name: str = "pvt_v2_b2", input_channels: int = 3):
    """Build the encoder used by BLSNet.

    The public interface keeps the manuscript's default `pvt_v2_b2` name while
    this clean release uses a compact pyramid implementation. A real pretrained
    PVT-v2 backbone can be plugged in here without changing the decoder.
    """

    if name not in {"pvt_v2_b2", "pyramid"}:
        raise ValueError(f"Unsupported encoder: {name}")
    encoder = PyramidEncoder(input_channels=input_channels)
    return encoder, encoder.channels
