from torch import nn

from .decoder import BLSDecoder
from .encoder import build_encoder
from .heads import SegmentationHead


class BLSNet(nn.Module):
    """Boundary-Guided Lite-Token Segmentation Network."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        encoder: str = "pvt_v2_b2",
        use_lite_token: bool = True,
        use_boundary_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.use_boundary_supervision = use_boundary_supervision
        self.encoder, encoder_channels = build_encoder(
            name=encoder,
            input_channels=input_channels,
        )
        self.decoder = BLSDecoder(
            encoder_channels=encoder_channels,
            use_lite_token=use_lite_token,
        )
        self.head = SegmentationHead(
            decoder_channels=self.decoder.out_channels,
            input_channels=input_channels,
            num_classes=num_classes,
        )

    def forward(self, x, return_boundary: bool = False):
        features = self.encoder(x)
        decoder_feature, boundary_logits = self.decoder(features)
        seg_logits = self.head(decoder_feature, x)

        if return_boundary or (self.training and self.use_boundary_supervision):
            return seg_logits, boundary_logits
        return seg_logits
