import torch

from blsnet import BLSNet


def main() -> None:
    model = BLSNet(input_channels=1, num_classes=1)
    x = torch.randn(2, 1, 224, 224)
    seg_logits, boundary_logits = model(x, return_boundary=True)
    print("segmentation:", tuple(seg_logits.shape))
    print("boundary heads:", [tuple(b.shape) for b in boundary_logits])


if __name__ == "__main__":
    main()
