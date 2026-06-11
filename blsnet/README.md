# BLSNet

Boundary-Guided Lite-Token Segmentation Network for medical image segmentation.

This repository is a clean architecture-focused version of BLSNet. It keeps the
main model design from the manuscript:

- Boundary-aware skip enhancement block (BASE)
- Lite-token channel attention in the decoder
- Multi-scale boundary supervision hooks
- Boundary-guided sample reweighting utilities

The repository intentionally does not include dataset preparation scripts,
training logs, pretrained weights, generated results, or legacy project code.

## Project Layout

```text
blsnet/
  src/blsnet/
    model.py              # BLSNet entry point
    encoder.py            # compact pyramid encoder interface
    decoder.py            # boundary-guided decoder
    heads.py              # segmentation output head
    losses.py             # boundary mask and reweighting helpers
    modules/
      base.py             # BASE module
      lite_token.py       # Lite-token channel attention module
      conv.py             # common convolution blocks
  examples/
    build_model.py        # minimal construction example
```

## Quick Start

```python
import torch
from blsnet import BLSNet

model = BLSNet(input_channels=1, num_classes=1)
x = torch.randn(2, 1, 224, 224)
y = model(x)
print(y.shape)
```

During training, BLSNet can also return multi-scale boundary predictions:

```python
model.train()
seg_logits, boundary_logits = model(x, return_boundary=True)
```

## Notes

This code is organized for publication and GitHub presentation. It describes
the model structure clearly, but it is not a full reproduction package. Add
dataset loaders, training loops, metrics, and pretrained backbones separately
when an executable experiment pipeline is needed.
