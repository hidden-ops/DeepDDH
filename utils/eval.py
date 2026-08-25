import torch
import torch.nn as nn
from tqdm import tqdm


def _get_main_logits(net, imgs, model_name):
    """
    Extract the main segmentation logits for different model variants.
    Returns:
        logits: Tensor of shape [B, C, H, W] for multi-class
                or [B, 1, H, W] for binary segmentation
    """
    outputs = net(imgs)

    if model_name == 'bisenet':
        # aux_pred0, aux_pred1, main_pred, smax_pred
        _, _, _, smax_pred = outputs
        return smax_pred

    elif model_name == 'danet':
        # main_pred is usually a tuple/list, use the main output
        main_pred = outputs
        return main_pred[1]

    elif model_name == 'ddhnet_v2_b':
        # aux_1, aux_2, main_pred
        _, _, main_pred = outputs
        return main_pred

    else:
        return outputs


def _compute_multiclass_dice_stats(pred_labels, true_masks, num_classes):
    """
    Accumulate per-class intersection and cardinality for dataset-level Dice.
    pred_labels: [B, H, W]
    true_masks:  [B, H, W]
    """
    intersections = torch.zeros(num_classes, device=pred_labels.device, dtype=torch.float64)
    cardinalities = torch.zeros(num_classes, device=pred_labels.device, dtype=torch.float64)

    for cls in range(num_classes):
        pred_c = (pred_labels == cls)
        true_c = (true_masks == cls)

        intersections[cls] += torch.logical_and(pred_c, true_c).sum().double()
        cardinalities[cls] += pred_c.sum().double() + true_c.sum().double()

    return intersections, cardinalities


@torch.no_grad()
def eval_net(net, loader, model_name, device, return_dict=False, ignore_background=True, eps=1e-6):
    """
    Validation function.

    Backward-compatible behaviour:
    - if return_dict=False:
        * multi-class segmentation (n_classes > 1): returns average validation CE loss
        * binary segmentation (n_classes == 1): returns dataset-level Dice score

    Recommended modern usage:
    - set return_dict=True, then read:
        results['val_loss']
        results['mean_dice']
        results['per_class_dice']
    """
    was_training = net.training
    net.eval()

    n_val = len(loader)
    if n_val == 0:
        raise ValueError("Validation loader is empty.")

    ce_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    bce_criterion = nn.BCEWithLogitsLoss().to(device)

    total_val_loss = 0.0

    # For multi-class Dice accumulation
    total_intersections = None
    total_cardinalities = None

    # For binary Dice accumulation
    bin_intersection = torch.tensor(0.0, device=device)
    bin_cardinality = torch.tensor(0.0, device=device)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)

            # Multi-class segmentation
            if net.n_classes > 1:
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                logits = _get_main_logits(net, imgs, model_name)

                # Validation loss: CE
                val_loss = ce_criterion(logits, true_masks)
                total_val_loss += val_loss.item()

                # Validation Dice: dataset-level per-class Dice
                pred_labels = torch.argmax(logits, dim=1)  # [B, H, W]
                intersections, cardinalities = _compute_multiclass_dice_stats(
                    pred_labels, true_masks, net.n_classes
                )

                if total_intersections is None:
                    total_intersections = intersections
                    total_cardinalities = cardinalities
                else:
                    total_intersections += intersections
                    total_cardinalities += cardinalities

            # Binary segmentation
            else:
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                logits = _get_main_logits(net, imgs, model_name)

                # Ensure target shape matches logits: [B, 1, H, W]
                if true_masks.dim() == 3:
                    true_masks = true_masks.unsqueeze(1)

                val_loss = bce_criterion(logits, true_masks)
                total_val_loss += val_loss.item()

                pred_masks = (torch.sigmoid(logits) > 0.5).float()
                true_masks_bin = (true_masks > 0.5).float()

                bin_intersection += (pred_masks * true_masks_bin).sum()
                bin_cardinality += pred_masks.sum() + true_masks_bin.sum()

            pbar.update(1)

    avg_val_loss = total_val_loss / n_val

    # Prepare final metrics
    if net.n_classes > 1:
        dice_per_class = (2.0 * total_intersections + eps) / (total_cardinalities + eps)

        start_cls = 1 if ignore_background and net.n_classes > 1 else 0
        dice_slice = dice_per_class[start_cls:]
        card_slice = total_cardinalities[start_cls:]

        valid_mask = card_slice > 0
        if valid_mask.any():
            mean_dice = dice_slice[valid_mask].mean().item()
        else:
            mean_dice = float('nan')

        results = {
            'val_loss': avg_val_loss,  # CE loss, lower is better
            'mean_dice': mean_dice,    # higher is better
            'per_class_dice': dice_per_class.detach().cpu().tolist(),
            'n_val_batches': n_val
        }

    else:
        dice_score = ((2.0 * bin_intersection + eps) / (bin_cardinality + eps)).item()

        results = {
            'val_loss': avg_val_loss,   # BCE loss, lower is better
            'mean_dice': dice_score,    # higher is better
            'per_class_dice': [dice_score],
            'n_val_batches': n_val
        }

    if was_training:
        net.train()

    # Backward compatibility with your old code
    if not return_dict:
        if net.n_classes > 1:
            return results['val_loss']
        else:
            return results['mean_dice']

    return results