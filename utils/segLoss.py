import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLosses(object):
    """
    Loss functions for multi-class segmentation.

    Notes
    -----
    1. CrossEntropyLoss now uses standard PyTorch reduction logic.
    2. FocalLoss is implemented in the standard pixel-wise form:
       FL = alpha * (1 - pt)^gamma * CE
    3. `batch_average` is kept only for backward compatibility and is no longer used,
       because PyTorch's 'mean' reduction already averages properly.
    """

    def __init__(
        self,
        weight=None,
        size_average=True,
        batch_average=True,
        ignore_index=255,
        cuda=False,
        reduction=None
    ):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

        # Backward-compatible reduction handling
        if reduction is None:
            self.reduction = 'mean' if size_average else 'sum'
        else:
            self.reduction = reduction

        if batch_average:
            warnings.warn(
                "`batch_average=True` is deprecated and no longer applied explicitly. "
                "PyTorch reduction='mean' already performs averaging.",
                UserWarning
            )

    def build_loss(self, mode='ce'):
        """Choices: ['ce', 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError(f'Loss mode "{mode}" is not implemented.')

    def _move_weight_to_device(self, device):
        if self.weight is None:
            return None
        if torch.is_tensor(self.weight):
            return self.weight.to(device)
        return torch.tensor(self.weight, dtype=torch.float32, device=device)

    def CrossEntropyLoss(self, logit, target):
        """
        Standard multi-class cross-entropy loss.

        Parameters
        ----------
        logit : torch.Tensor
            Shape [N, C, H, W]
        target : torch.Tensor
            Shape [N, H, W], integer class map

        Returns
        -------
        torch.Tensor
            Scalar loss
        """
        if logit.dim() != 4:
            raise ValueError(f'Expected logit shape [N, C, H, W], got {tuple(logit.shape)}')
        if target.dim() != 3:
            raise ValueError(f'Expected target shape [N, H, W], got {tuple(target.shape)}')

        weight = self._move_weight_to_device(logit.device)

        criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        ).to(logit.device)

        loss = criterion(logit, target.long())
        return loss

    def FocalLoss(self, logit, target, gamma=2.0, alpha=0.5):
        """
        Standard multi-class focal loss.

        Parameters
        ----------
        logit : torch.Tensor
            Shape [N, C, H, W]
        target : torch.Tensor
            Shape [N, H, W], integer class map
        gamma : float
            Focusing parameter
        alpha : float or None
            Class balancing factor. If scalar, applied globally.
            If None, no alpha weighting is used.

        Returns
        -------
        torch.Tensor
            Scalar loss
        """
        if logit.dim() != 4:
            raise ValueError(f'Expected logit shape [N, C, H, W], got {tuple(logit.shape)}')
        if target.dim() != 3:
            raise ValueError(f'Expected target shape [N, H, W], got {tuple(target.shape)}')

        target = target.long()

        # Pixel-wise CE loss
        ce_loss = F.cross_entropy(
            input=logit,
            target=target,
            weight=self._move_weight_to_device(logit.device),
            ignore_index=self.ignore_index,
            reduction='none'
        )  # shape: [N, H, W]

        # Mask out ignored pixels
        valid_mask = (target != self.ignore_index).float()

        # pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** gamma) * ce_loss

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                focal_loss = float(alpha) * focal_loss
            elif torch.is_tensor(alpha):
                alpha = alpha.to(logit.device)
                # Optional: class-wise alpha map
                alpha_map = alpha[target.clamp(min=0)]
                focal_loss = alpha_map * focal_loss
            else:
                raise TypeError("alpha must be float, int, torch.Tensor, or None")

        focal_loss = focal_loss * valid_mask

        valid_count = valid_mask.sum()

        if self.reduction == 'mean':
            loss = focal_loss.sum() / valid_count.clamp_min(1.0)
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        elif self.reduction == 'none':
            loss = focal_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')

        return loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = SegmentationLosses(cuda=(device.type == 'cuda'))

    # logits: [N, C, H, W]
    a = torch.rand(2, 6, 128, 128).to(device=device)

    # target: integer class labels [N, H, W], values in [0, C-1]
    b = torch.randint(0, 6, (2, 128, 128), device=device)

    print('logit shape:', a.size())
    print('target shape:', b.size())
    print('CE:', loss_fn.CrossEntropyLoss(a, b).item())
    print('Focal(gamma=0, alpha=None):', loss_fn.FocalLoss(a, b, gamma=0, alpha=None).item())
    print('Focal(gamma=2, alpha=0.5):', loss_fn.FocalLoss(a, b, gamma=2, alpha=0.5).item())