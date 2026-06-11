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

Create a clean Python environment first:

```bash
conda create -n blsnet python=3.10 -y
conda activate blsnet
```

Install PyTorch according to your CUDA version. For example, with CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you only need the CPU version:

```bash
pip install torch torchvision
```

Then install the remaining dependencies and register the local package:

```bash
cd /path/to/blsnet
pip install -r requirements.txt
pip install -e .
```

Run a minimal model construction check:

```bash
python examples/build_model.py
```

You can also test the model directly in Python:

```python
import torch
from blsnet import BLSNet

model = BLSNet(input_channels=1, num_classes=1)
x = torch.randn(2, 1, 224, 224)
seg_logits, boundary_logits = model(x, return_boundary=True)

print(seg_logits.shape)
print([b.shape for b in boundary_logits])
```

## Notes

This code is organized for publication and GitHub presentation. It describes
the model structure clearly, but it is not a full reproduction package. Add
dataset loaders, training loops, metrics, and pretrained backbones separately
when an executable experiment pipeline is needed.

