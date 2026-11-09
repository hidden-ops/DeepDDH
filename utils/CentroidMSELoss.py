from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CENTROID_LOSS_MODES = ('soft', 'legacy-hard', 'off')


class CentroidMSELoss(nn.Module):
    """Differentiable class-centroid loss based on softmax probabilities.

    Coordinates are normalized to [0, 1], making the loss independent of image
    resolution. Classes absent from the ground-truth mask are excluded, as is the
    background class by default.
    """

    def __init__(
        self,
        n_class: int,
        include_background: bool = False,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_class < 2:
            raise ValueError(f'n_class must be at least 2, got {n_class}')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}')
        self.n_class = int(n_class)
        self.include_background = bool(include_background)
        self.ignore_index = ignore_index
        self.eps = float(eps)

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        if pred_mask.ndim != 4:
            raise ValueError(
                f'pred_mask must have shape [B, C, H, W], got {tuple(pred_mask.shape)}'
            )
        batch_size, n_class, height, width = pred_mask.shape
        if n_class != self.n_class:
            raise ValueError(f'Expected {self.n_class} prediction classes, got {n_class}')
        if true_mask.shape != (batch_size, height, width):
            raise ValueError(
                'true_mask must have shape [B, H, W] matching pred_mask; '
                f'got {tuple(true_mask.shape)}'
            )

        true_mask = true_mask.to(device=pred_mask.device, dtype=torch.long)
        if self.ignore_index is None:
            valid_pixels = torch.ones_like(true_mask, dtype=torch.bool)
        else:
            valid_pixels = true_mask != self.ignore_index

        invalid_labels = valid_pixels & ((true_mask < 0) | (true_mask >= self.n_class))
        if torch.any(invalid_labels):
            values = torch.unique(true_mask[invalid_labels]).detach().cpu().tolist()
            raise ValueError(f'true_mask contains class IDs outside 0-{self.n_class - 1}: {values}')

        safe_target = torch.where(valid_pixels, true_mask, torch.zeros_like(true_mask))
        valid_weights = valid_pixels.unsqueeze(1).to(dtype=torch.float32)

        # Use float32 accumulation for numerical stability under mixed precision.
        pred_prob = F.softmax(pred_mask.float(), dim=1) * valid_weights
        true_onehot = F.one_hot(
            safe_target,
            num_classes=self.n_class,
        ).permute(0, 3, 1, 2).to(dtype=torch.float32)
        true_onehot = true_onehot * valid_weights

        y_coordinates = torch.linspace(
            0.0, 1.0, height, device=pred_mask.device, dtype=torch.float32
        ).view(1, 1, height, 1)
        x_coordinates = torch.linspace(
            0.0, 1.0, width, device=pred_mask.device, dtype=torch.float32
        ).view(1, 1, 1, width)

        pred_mass = pred_prob.sum(dim=(2, 3))
        true_mass = true_onehot.sum(dim=(2, 3))

        pred_centroid_y = (pred_prob * y_coordinates).sum(dim=(2, 3)) / pred_mass.clamp_min(self.eps)
        pred_centroid_x = (pred_prob * x_coordinates).sum(dim=(2, 3)) / pred_mass.clamp_min(self.eps)
        true_centroid_y = (true_onehot * y_coordinates).sum(dim=(2, 3)) / true_mass.clamp_min(self.eps)
        true_centroid_x = (true_onehot * x_coordinates).sum(dim=(2, 3)) / true_mass.clamp_min(self.eps)

        squared_error = (
            (pred_centroid_y - true_centroid_y).square()
            + (pred_centroid_x - true_centroid_x).square()
        ) / 2.0

        present_classes = true_mass > self.eps
        if not self.include_background:
            foreground = torch.arange(self.n_class, device=pred_mask.device) != 0
            present_classes = present_classes & foreground.unsqueeze(0)

        if not torch.any(present_classes):
            # Keep a valid zero-valued connection to logits for all-background or
            # fully ignored batches.
            return pred_mask.sum() * 0.0

        return squared_error[present_classes].mean()


class LegacyHardCentroidMSELoss(nn.Module):
    """Historical hard-argmax implementation with no logits gradient.

    This class exists only to reproduce archived experiments. New training runs
    should use :class:`CentroidMSELoss` or disable the centroid term explicitly.
    """

    def __init__(self, n_class: int):
        super().__init__()
        self.n_class = int(n_class)

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_class, height, width = pred_mask.shape
        if n_class != self.n_class:
            raise ValueError(f'Expected {self.n_class} prediction classes, got {n_class}')

        pred_class = torch.argmax(F.softmax(pred_mask, dim=1), dim=1)
        pred_onehot = F.one_hot(
            pred_class, num_classes=self.n_class
        ).permute(0, 3, 1, 2).float()
        true_onehot = F.one_hot(
            true_mask.long(), num_classes=self.n_class
        ).permute(0, 3, 1, 2).float()

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=pred_mask.device, dtype=torch.float32),
            torch.arange(width, device=pred_mask.device, dtype=torch.float32),
            indexing='ij',
        )

        pred_mass = pred_onehot.sum(dim=(2, 3)).clamp_min(1e-6)
        true_mass = true_onehot.sum(dim=(2, 3)).clamp_min(1e-6)
        pred_centroids = torch.stack(
            (
                (pred_onehot * y_grid).sum(dim=(2, 3)) / pred_mass,
                (pred_onehot * x_grid).sum(dim=(2, 3)) / pred_mass,
            ),
            dim=-1,
        )
        true_centroids = torch.stack(
            (
                (true_onehot * y_grid).sum(dim=(2, 3)) / true_mass,
                (true_onehot * x_grid).sum(dim=(2, 3)) / true_mass,
            ),
            dim=-1,
        )
        return F.mse_loss(pred_centroids, true_centroids) / 10000.0


def build_centroid_loss(mode: str, n_class: int) -> Optional[nn.Module]:
    """Build a centroid loss from an explicit reproducibility mode."""
    if mode == 'soft':
        return CentroidMSELoss(n_class=n_class)
    if mode == 'legacy-hard':
        return LegacyHardCentroidMSELoss(n_class=n_class)
    if mode == 'off':
        return None
    raise ValueError(
        f'Unsupported centroid loss mode {mode!r}; expected one of {CENTROID_LOSS_MODES}'
    )
