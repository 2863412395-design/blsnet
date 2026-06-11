import torch
from torch import nn
import torch.nn.functional as F

from .modules.conv import ConvBNAct, ResidualBlock


class SegmentationHead(nn.Module):
    def __init__(
        self,
        decoder_channels: int,
        input_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Sequential(
            ConvBNAct(input_channels, decoder_channels, kernel_size=3, stride=2),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, stride=2),
        )
        self.out = nn.Sequential(
            ResidualBlock(decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, decoder_feature: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        image_feature = self.image_proj(image)
        if image_feature.shape[-2:] != decoder_feature.shape[-2:]:
            image_feature = F.interpolate(
                image_feature,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        logits = self.out(decoder_feature + image_feature)
        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
