import torch
from torch import nn
import torch.nn.functional as F

from .conv import ConvBNAct, ResidualBlock


class BoundaryHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BoundaryAwareSkipEnhancement(nn.Module):
    """Boundary-Aware Skip Enhancement Block (BASE).

    BASE filters shallow skip features before they enter the decoder. It uses
    multi-scale reconstruction differences as a boundary-sensitive response,
    predicts an auxiliary boundary map, and gates the enhanced skip feature.
    """

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        scale_factors: tuple[float, ...] = (0.8, 0.4),
    ) -> None:
        super().__init__()
        self.scale_factors = scale_factors
        fusion_channels = skip_channels + decoder_channels

        self.fuse = nn.Sequential(
            ConvBNAct(fusion_channels, out_channels, kernel_size=1),
            ResidualBlock(out_channels),
        )
        self.edge_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.5)
        self.refine = ResidualBlock(out_channels)
        self.boundary_head = BoundaryHead(out_channels)
        self.skip_proj = (
            nn.Identity()
            if skip_channels == out_channels
            else ConvBNAct(skip_channels, out_channels, kernel_size=1)
        )

    def _multi_scale_boundary_response(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        responses = []
        for scale in self.scale_factors:
            low = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            rec = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
            responses.append(torch.abs(x - rec))
        return torch.stack(responses, dim=0).mean(dim=0)

    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(
                decoder,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = self.fuse(torch.cat([skip, decoder], dim=1))
        boundary_response = self._multi_scale_boundary_response(fused)
        enhanced = self.refine(fused + self.edge_weight * boundary_response)
        boundary_logits = self.boundary_head(enhanced)
        boundary_prob = torch.sigmoid(boundary_logits)

        residual_skip = self.skip_proj(skip)
        out = residual_skip + enhanced * (1.0 + 0.5 * boundary_prob)
        return out, boundary_logits
