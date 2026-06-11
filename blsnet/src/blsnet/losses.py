import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob + target, dim=dims)
    return 1.0 - torch.mean((2.0 * inter + eps) / (denom + eps))


def boundary_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract a soft boundary band from binary segmentation masks."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_supervision_loss(
    boundary_logits: list[torch.Tensor],
    target_mask: torch.Tensor,
    weights: tuple[float, ...] = (0.5, 0.3, 0.2),
) -> torch.Tensor:
    target_boundary = boundary_mask(target_mask)
    total = target_boundary.new_tensor(0.0)
    for idx, logits in enumerate(boundary_logits):
        resized = F.interpolate(
            target_boundary,
            size=logits.shape[-2:],
            mode="nearest",
        )
        bce = F.binary_cross_entropy_with_logits(logits, resized)
        total = total + weights[idx] * (bce + dice_loss(logits, resized))
    return total


def boundary_guided_sample_weights(
    boundary_loss_per_sample: torch.Tensor,
    target_mask: torch.Tensor,
    alpha: float = 0.5,
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sample-level dynamic reweighting from boundary loss and complexity."""

    target_boundary = boundary_mask(target_mask)
    complexity = target_boundary.flatten(1).mean(dim=1)
    difficulty = boundary_loss_per_sample.detach()
    score = alpha * difficulty + (1.0 - alpha) * complexity
    score = score / (score.mean() + eps)
    weights = score.clamp_min(eps).pow(power)
    weights = weights.clamp(min_weight, max_weight)
    return weights / (weights.mean() + eps)
