import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddhnet_model.dilated_ddhnet_bony import DDHNet_Bony
from utils.dataset_bony import BasicDataset
from utils.segLoss import SegmentationLosses
from utils.CentroidMSELoss import CentroidMSELoss


tr_dir_img = str(PROJECT_ROOT / 'data' / 'Training' / 'imgs') + os.sep
tr_dir_mask = str(PROJECT_ROOT / 'data' / 'Training' / 'segs') + os.sep
tr_dir_bony = str(PROJECT_ROOT / 'data' / 'Training' / 'bony_masks') + os.sep

val_dir_img = str(PROJECT_ROOT / 'data' / 'Testing' / 'imgs') + os.sep
val_dir_mask = str(PROJECT_ROOT / 'data' / 'Testing' / 'segs') + os.sep
val_dir_bony = str(PROJECT_ROOT / 'data' / 'Testing' / 'bony_masks') + os.sep

base_path = str(PROJECT_ROOT / 'checkpoint' / 'Stage-2')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_project_path(path: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved.resolve())


def validate_args(args: argparse.Namespace) -> None:
    if args.n_classes < 2 or args.bony_class < 2:
        raise ValueError('--n-classes and --bony-class must both be at least 2.')
    if args.epochs < 1:
        raise ValueError('--epochs must be at least 1.')
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError('--batch-sizes values must all be at least 1.')
    if args.lr <= 0:
        raise ValueError('--lr must be positive.')
    if not 0 < args.scale <= 1:
        raise ValueError('--scale must be in the interval (0, 1].')
    if args.patience < 1:
        raise ValueError('--patience must be at least 1.')
    if args.min_delta < 0:
        raise ValueError('--min-delta cannot be negative.')
    if args.num_workers_train < 0 or args.num_workers_val < 0:
        raise ValueError('DataLoader worker counts cannot be negative.')
    if args.seg_centroid_weight < 0 or args.bony_centroid_weight < 0:
        raise ValueError('Centroid-loss weights cannot be negative.')


def to_python_scalar(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return x


def save_json(obj: Dict[str, Any], path: str) -> None:
    serialisable = {}
    for k, v in obj.items():
        if isinstance(v, list):
            serialisable[k] = [to_python_scalar(i) for i in v]
        else:
            serialisable[k] = to_python_scalar(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


def append_csv_row(csv_path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_curve(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_learning_curves(history: List[Dict[str, Any]], out_dir: str) -> None:
    epochs = [row['epoch'] for row in history]

    curve_specs = [
        ('train_total_loss', 'Train total loss', 'curve_train_total_loss.png'),
        ('train_seg_loss', 'Train seg loss', 'curve_train_seg_loss.png'),
        ('train_bony_loss', 'Train bony loss', 'curve_train_bony_loss.png'),
        ('val_total_loss', 'Validation total loss', 'curve_val_total_loss.png'),
        ('val_seg_loss', 'Validation seg loss', 'curve_val_seg_loss.png'),
        ('val_bony_loss', 'Validation bony loss', 'curve_val_bony_loss.png'),
        ('val_seg_mean_dice', 'Validation seg mean Dice', 'curve_val_seg_mean_dice.png'),
        ('val_bony_mean_dice', 'Validation bony mean Dice', 'curve_val_bony_mean_dice.png'),
        ('val_joint_mean_dice', 'Validation joint mean Dice', 'curve_val_joint_mean_dice.png'),
        ('monitor_value', 'Monitor value', 'curve_monitor_metric.png'),
    ]

    for key, ylabel, filename in curve_specs:
        y = [row[key] for row in history]
        plot_curve(
            epochs, y,
            xlabel='Epoch',
            ylabel=ylabel,
            title=f'Stage-2 {ylabel} by Epoch',
            save_path=os.path.join(out_dir, filename)
        )


def get_milestone_epochs(total_epochs: int) -> List[int]:
    raw = [
        max(1, int(round(total_epochs * 0.25))),
        max(1, int(round(total_epochs * 0.50))),
        max(1, int(round(total_epochs * 0.75))),
        max(1, int(round(total_epochs * 1.00))),
    ]
    return sorted(set(raw))


def build_model(n_classes: int, bony_class: int):
    return DDHNet_Bony(
        n_classes=n_classes,
        bony_class=bony_class,
        n_channels=3,
        pretrained_model=True
    )


def build_criteria(device: torch.device, n_classes: int, bony_class: int):
    seg_criterion = SegmentationLosses(cuda=(device.type == 'cuda'), batch_average=False)
    bony_criterion = SegmentationLosses(cuda=(device.type == 'cuda'), batch_average=False)
    seg_centroid_loss = CentroidMSELoss(n_class=n_classes)
    bony_centroid_loss = CentroidMSELoss(n_class=bony_class)
    return seg_criterion, bony_criterion, seg_centroid_loss, bony_centroid_loss


def resolve_load_path(load_path: str, load_template: str, seed: int, batch_size: int) -> Optional[str]:
    if load_template:
        return resolve_project_path(load_template.format(seed=seed, batch_size=batch_size))
    if load_path:
        return resolve_project_path(load_path)
    return None


def load_partial_checkpoint(net, checkpoint_path: str, device: torch.device) -> None:
    checkpoint_path = resolve_project_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Stage-1 checkpoint does not exist: {checkpoint_path}')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(f'No model state_dict found in checkpoint: {checkpoint_path}')

    checkpoint = {
        key.removeprefix('module.'): value
        for key, value in checkpoint.items()
    }

    model_state_dict = net.state_dict()
    filtered_state_dict = {
        k: v for k, v in checkpoint.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    skipped_keys = sorted(set(checkpoint) - set(filtered_state_dict))
    if not filtered_state_dict:
        raise ValueError(
            f'Checkpoint has no key-and-shape-compatible parameters for Stage-2: {checkpoint_path}'
        )

    missing_keys, unexpected_keys = net.load_state_dict(filtered_state_dict, strict=False)

    logging.info(f'Loaded partial checkpoint from: {checkpoint_path}')
    if skipped_keys:
        logging.info('Skipped %d incompatible or Stage-1-only checkpoint keys.', len(skipped_keys))
    if missing_keys:
        logging.info(f'Missing keys not loaded from checkpoint: {missing_keys}')
    if unexpected_keys:
        logging.info(f'Unexpected keys in checkpoint: {unexpected_keys}')


def _compute_multiclass_dice_stats(pred_labels, true_masks, num_classes):
    intersections = torch.zeros(num_classes, device=pred_labels.device, dtype=torch.float64)
    cardinalities = torch.zeros(num_classes, device=pred_labels.device, dtype=torch.float64)

    for cls in range(num_classes):
        pred_c = (pred_labels == cls)
        true_c = (true_masks == cls)
        intersections[cls] += torch.logical_and(pred_c, true_c).sum().double()
        cardinalities[cls] += pred_c.sum().double() + true_c.sum().double()

    return intersections, cardinalities


def _finalise_mean_dice(total_intersections, total_cardinalities, ignore_background=True, eps=1e-6):
    dice_per_class = (2.0 * total_intersections + eps) / (total_cardinalities + eps)
    start_cls = 1 if ignore_background and len(dice_per_class) > 1 else 0
    dice_slice = dice_per_class[start_cls:]
    card_slice = total_cardinalities[start_cls:]
    valid_mask = card_slice > 0

    if valid_mask.any():
        mean_dice = dice_slice[valid_mask].mean().item()
    else:
        mean_dice = float('nan')

    return mean_dice, dice_per_class.detach().cpu().tolist()


def compute_stage2_losses(
    net,
    imgs,
    true_masks,
    true_bony,
    seg_criterion,
    bony_criterion,
    seg_centroid_loss,
    bony_centroid_loss=None,
    use_bony_centroid_loss=False,
    seg_centroid_weight=0.1,
    bony_centroid_weight=0.1
):
    seg_logits, bony_logits = net(imgs)

    # seg branch
    seg_ce_loss = seg_criterion.CrossEntropyLoss(seg_logits, true_masks)
    seg_fl_loss = seg_criterion.FocalLoss(seg_logits, true_masks, gamma=2, alpha=0.5)
    seg_ct_loss = seg_centroid_loss(seg_logits, true_masks)
    seg_loss = seg_ce_loss + seg_fl_loss + seg_centroid_weight * seg_ct_loss

    # bony branch
    bony_ce_loss = bony_criterion.CrossEntropyLoss(bony_logits, true_bony)
    bony_fl_loss = bony_criterion.FocalLoss(bony_logits, true_bony, gamma=2, alpha=0.5)

    if use_bony_centroid_loss and bony_centroid_loss is not None:
        bony_ct_loss = bony_centroid_loss(bony_logits, true_bony)
        bony_loss = bony_ce_loss + bony_fl_loss + bony_centroid_weight * bony_ct_loss
    else:
        bony_ct_loss = torch.tensor(0.0, device=imgs.device)
        bony_loss = bony_ce_loss + bony_fl_loss

    total_loss = seg_loss + bony_loss

    return {
        'total_loss': total_loss,
        'seg_loss': seg_loss,
        'bony_loss': bony_loss,
        'seg_ce_loss': seg_ce_loss,
        'seg_fl_loss': seg_fl_loss,
        'seg_ct_loss': seg_ct_loss,
        'bony_ce_loss': bony_ce_loss,
        'bony_fl_loss': bony_fl_loss,
        'bony_ct_loss': bony_ct_loss,
        'seg_logits': seg_logits,
        'bony_logits': bony_logits,
    }


@torch.no_grad()
def evaluate_stage2(
    net,
    val_loader,
    device,
    seg_criterion,
    bony_criterion,
    seg_centroid_loss,
    bony_centroid_loss,
    n_classes,
    bony_class,
    ignore_background=True,
    use_bony_centroid_loss=False,
    seg_centroid_weight=0.1,
    bony_centroid_weight=0.1
):
    was_training = net.training
    net.eval()

    total_total_loss = 0.0
    total_seg_loss = 0.0
    total_bony_loss = 0.0
    n_batches = 0

    seg_intersections = torch.zeros(n_classes, device=device, dtype=torch.float64)
    seg_cardinalities = torch.zeros(n_classes, device=device, dtype=torch.float64)

    bony_intersections = torch.zeros(bony_class, device=device, dtype=torch.float64)
    bony_cardinalities = torch.zeros(bony_class, device=device, dtype=torch.float64)

    for batch in val_loader:
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)
        true_bony = batch['bony'].to(device=device, dtype=torch.long)

        loss_dict = compute_stage2_losses(
            net=net,
            imgs=imgs,
            true_masks=true_masks,
            true_bony=true_bony,
            seg_criterion=seg_criterion,
            bony_criterion=bony_criterion,
            seg_centroid_loss=seg_centroid_loss,
            bony_centroid_loss=bony_centroid_loss,
            use_bony_centroid_loss=use_bony_centroid_loss,
            seg_centroid_weight=seg_centroid_weight,
            bony_centroid_weight=bony_centroid_weight
        )

        total_total_loss += loss_dict['total_loss'].item()
        total_seg_loss += loss_dict['seg_loss'].item()
        total_bony_loss += loss_dict['bony_loss'].item()
        n_batches += 1

        seg_pred = torch.argmax(loss_dict['seg_logits'], dim=1)
        bony_pred = torch.argmax(loss_dict['bony_logits'], dim=1)

        seg_i, seg_c = _compute_multiclass_dice_stats(seg_pred, true_masks, n_classes)
        bony_i, bony_c = _compute_multiclass_dice_stats(bony_pred, true_bony, bony_class)

        seg_intersections += seg_i
        seg_cardinalities += seg_c
        bony_intersections += bony_i
        bony_cardinalities += bony_c

    avg_total_loss = total_total_loss / max(n_batches, 1)
    avg_seg_loss = total_seg_loss / max(n_batches, 1)
    avg_bony_loss = total_bony_loss / max(n_batches, 1)

    seg_mean_dice, seg_per_class_dice = _finalise_mean_dice(
        seg_intersections, seg_cardinalities, ignore_background=ignore_background
    )
    bony_mean_dice, bony_per_class_dice = _finalise_mean_dice(
        bony_intersections, bony_cardinalities, ignore_background=ignore_background
    )
    joint_mean_dice = float(np.nanmean([seg_mean_dice, bony_mean_dice]))

    if was_training:
        net.train()

    return {
        'val_total_loss': avg_total_loss,
        'val_seg_loss': avg_seg_loss,
        'val_bony_loss': avg_bony_loss,
        'val_seg_mean_dice': seg_mean_dice,
        'val_bony_mean_dice': bony_mean_dice,
        'val_joint_mean_dice': joint_mean_dice,
        'val_seg_per_class_dice': seg_per_class_dice,
        'val_bony_per_class_dice': bony_per_class_dice,
        'n_val_batches': n_batches
    }


def infer_monitor_mode(monitor_metric: str) -> str:
    if monitor_metric in ['val_total_loss', 'val_seg_loss', 'val_bony_loss']:
        return 'min'
    elif monitor_metric in ['val_seg_mean_dice', 'val_bony_mean_dice', 'val_joint_mean_dice']:
        return 'max'
    else:
        raise ValueError(f'Unsupported monitor_metric: {monitor_metric}')


def get_monitor_value(monitor_metric: str, val_results: Dict[str, Any]) -> float:
    if monitor_metric not in val_results:
        raise ValueError(f'Monitor metric "{monitor_metric}" not found in validation results.')
    return float(val_results[monitor_metric])


def is_better(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == 'max':
        return current > (best + min_delta)
    elif mode == 'min':
        return current < (best - min_delta)
    else:
        raise ValueError(f'Unsupported monitor mode: {mode}')


def train_one_run(
    seed: int,
    batch_size: int,
    lr: float,
    epochs: int,
    img_scale: float,
    device: torch.device,
    n_classes: int,
    bony_class: int,
    run_dir: str,
    load_path: Optional[str] = None,
    num_workers_train: int = 8,
    num_workers_val: int = 4,
    save_all_epoch_ckpt: bool = False,
    save_milestone_ckpt: bool = True,
    monitor_metric: str = 'val_total_loss',
    patience: int = 10,
    min_delta: float = 1e-5,
    use_bony_centroid_loss: bool = False,
    seg_centroid_weight: float = 0.1,
    bony_centroid_weight: float = 0.1,
    ignore_background: bool = True
) -> Dict[str, Any]:

    monitor_mode = infer_monitor_mode(monitor_metric)
    milestone_epochs = get_milestone_epochs(epochs)

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, 'checkpoints'))
    ensure_dir(os.path.join(run_dir, 'tb'))

    set_seed(seed)

    train_set = BasicDataset(tr_dir_img, tr_dir_mask, tr_dir_bony, batch_size, img_scale)
    val_set = BasicDataset(val_dir_img, val_dir_mask, val_dir_bony, batch_size, img_scale)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    net = build_model(n_classes=n_classes, bony_class=bony_class)
    net.to(device=device)

    if load_path:
        load_partial_checkpoint(net, load_path, device=device)

    seg_criterion, bony_criterion, seg_centroid_loss, bony_centroid_loss = build_criteria(
        device=device, n_classes=n_classes, bony_class=bony_class
    )

    optimizer = optim.RMSprop(
        net.parameters(),
        lr=lr,
        weight_decay=1e-12,
        momentum=0.95
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=monitor_mode,
        patience=2
    )

    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))

    config = {
        'seed': seed,
        'batch_size': batch_size,
        'lr': lr,
        'epochs': epochs,
        'img_scale': img_scale,
        'n_classes': n_classes,
        'bony_class': bony_class,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'patience': patience,
        'min_delta': min_delta,
        'load_path': load_path,
        'milestone_epochs': milestone_epochs,
        'use_bony_centroid_loss': use_bony_centroid_loss,
        'seg_centroid_weight': seg_centroid_weight,
        'bony_centroid_weight': bony_centroid_weight,
        'ignore_background': ignore_background
    }
    save_json(config, os.path.join(run_dir, 'config.json'))

    logging.info(
        f'Starting Stage-2 run | seed={seed} | batch_size={batch_size} | lr={lr} | '
        f'train={len(train_set)} | val={len(val_set)} | monitor_metric={monitor_metric} | '
        f'monitor_mode={monitor_mode} | load_path={load_path}'
    )

    history = []
    history_csv = os.path.join(run_dir, 'history.csv')

    best_monitor_value = -float('inf') if monitor_mode == 'max' else float('inf')
    best_epoch = -1
    best_metrics = None

    best_ckpt_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    last_ckpt_path = os.path.join(run_dir, 'checkpoints', 'last_model.pth')

    epochs_without_improve = 0
    stopped_early = False
    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        net.train()

        running_total_loss = 0.0
        running_seg_loss = 0.0
        running_bony_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_set), desc=f'Stage-2 Seed {seed} | Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                true_bony = batch['bony']

                assert imgs.shape[1] == net.n_channels, (
                    f'Network expects {net.n_channels} channels, but got {imgs.shape[1]}.'
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_bony = true_bony.to(device=device, dtype=torch.long)

                optimizer.zero_grad()

                loss_dict = compute_stage2_losses(
                    net=net,
                    imgs=imgs,
                    true_masks=true_masks,
                    true_bony=true_bony,
                    seg_criterion=seg_criterion,
                    bony_criterion=bony_criterion,
                    seg_centroid_loss=seg_centroid_loss,
                    bony_centroid_loss=bony_centroid_loss,
                    use_bony_centroid_loss=use_bony_centroid_loss,
                    seg_centroid_weight=seg_centroid_weight,
                    bony_centroid_weight=bony_centroid_weight
                )

                total_loss = loss_dict['total_loss']
                total_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                batch_total_loss = total_loss.item()
                batch_seg_loss = loss_dict['seg_loss'].item()
                batch_bony_loss = loss_dict['bony_loss'].item()

                running_total_loss += batch_total_loss
                running_seg_loss += batch_seg_loss
                running_bony_loss += batch_bony_loss
                num_batches += 1
                global_step += 1

                writer.add_scalar('batch/train_total_loss', batch_total_loss, global_step)
                writer.add_scalar('batch/train_seg_loss', batch_seg_loss, global_step)
                writer.add_scalar('batch/train_bony_loss', batch_bony_loss, global_step)

                pbar.set_postfix(total=f'{batch_total_loss:.6f}', seg=f'{batch_seg_loss:.6f}', bony=f'{batch_bony_loss:.6f}')
                pbar.update(imgs.shape[0])

        train_total_loss = running_total_loss / max(num_batches, 1)
        train_seg_loss = running_seg_loss / max(num_batches, 1)
        train_bony_loss = running_bony_loss / max(num_batches, 1)

        val_results = evaluate_stage2(
            net=net,
            val_loader=val_loader,
            device=device,
            seg_criterion=seg_criterion,
            bony_criterion=bony_criterion,
            seg_centroid_loss=seg_centroid_loss,
            bony_centroid_loss=bony_centroid_loss,
            n_classes=n_classes,
            bony_class=bony_class,
            ignore_background=ignore_background,
            use_bony_centroid_loss=use_bony_centroid_loss,
            seg_centroid_weight=seg_centroid_weight,
            bony_centroid_weight=bony_centroid_weight
        )

        monitor_value = get_monitor_value(monitor_metric, val_results)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('epoch/train_total_loss', train_total_loss, epoch)
        writer.add_scalar('epoch/train_seg_loss', train_seg_loss, epoch)
        writer.add_scalar('epoch/train_bony_loss', train_bony_loss, epoch)
        writer.add_scalar('epoch/val_total_loss', val_results['val_total_loss'], epoch)
        writer.add_scalar('epoch/val_seg_loss', val_results['val_seg_loss'], epoch)
        writer.add_scalar('epoch/val_bony_loss', val_results['val_bony_loss'], epoch)
        writer.add_scalar('epoch/val_seg_mean_dice', val_results['val_seg_mean_dice'], epoch)
        writer.add_scalar('epoch/val_bony_mean_dice', val_results['val_bony_mean_dice'], epoch)
        writer.add_scalar('epoch/val_joint_mean_dice', val_results['val_joint_mean_dice'], epoch)
        writer.add_scalar('epoch/monitor_value', monitor_value, epoch)
        writer.add_scalar('epoch/lr', current_lr, epoch)

        scheduler.step(monitor_value)

        improved = is_better(monitor_value, best_monitor_value, monitor_mode, min_delta=min_delta)

        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_metrics = dict(val_results)
            best_metrics['best_monitor_value'] = monitor_value
            best_metrics['best_epoch'] = best_epoch
            epochs_without_improve = 0
            torch.save(net.state_dict(), best_ckpt_path)
        else:
            epochs_without_improve += 1

        torch.save(net.state_dict(), last_ckpt_path)

        if save_all_epoch_ckpt:
            torch.save(net.state_dict(), os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:03d}.pth'))

        if save_milestone_ckpt and epoch in milestone_epochs:
            torch.save(net.state_dict(), os.path.join(run_dir, 'checkpoints', f'milestone_epoch_{epoch:03d}.pth'))

        row = {
            'epoch': epoch,
            'seed': seed,
            'batch_size': batch_size,
            'lr': lr,
            'train_total_loss': train_total_loss,
            'train_seg_loss': train_seg_loss,
            'train_bony_loss': train_bony_loss,
            'val_total_loss': val_results['val_total_loss'],
            'val_seg_loss': val_results['val_seg_loss'],
            'val_bony_loss': val_results['val_bony_loss'],
            'val_seg_mean_dice': val_results['val_seg_mean_dice'],
            'val_bony_mean_dice': val_results['val_bony_mean_dice'],
            'val_joint_mean_dice': val_results['val_joint_mean_dice'],
            'monitor_metric': monitor_metric,
            'monitor_value': monitor_value,
            'best_so_far': int(improved),
            'best_epoch_so_far': best_epoch,
            'lr_current': current_lr
        }
        history.append(row)

        append_csv_row(
            history_csv,
            row=row,
            fieldnames=[
                'epoch', 'seed', 'batch_size', 'lr',
                'train_total_loss', 'train_seg_loss', 'train_bony_loss',
                'val_total_loss', 'val_seg_loss', 'val_bony_loss',
                'val_seg_mean_dice', 'val_bony_mean_dice', 'val_joint_mean_dice',
                'monitor_metric', 'monitor_value',
                'best_so_far', 'best_epoch_so_far', 'lr_current'
            ]
        )

        logging.info(
            f'[Stage-2 Seed {seed}] Epoch {epoch}/{epochs} | '
            f'train_total={train_total_loss:.6f} | '
            f'val_total={val_results["val_total_loss"]:.6f} | '
            f'val_seg_loss={val_results["val_seg_loss"]:.6f} | '
            f'val_bony_loss={val_results["val_bony_loss"]:.6f} | '
            f'val_seg_dice={val_results["val_seg_mean_dice"]:.6f} | '
            f'val_bony_dice={val_results["val_bony_mean_dice"]:.6f} | '
            f'val_joint_dice={val_results["val_joint_mean_dice"]:.6f} | '
            f'monitor({monitor_metric})={monitor_value:.6f} | best_epoch={best_epoch}'
        )

        if epochs_without_improve >= patience:
            logging.info(
                f'[Stage-2 Seed {seed}] Early stopping at epoch {epoch} '
                f'(no improvement for {patience} consecutive epochs).'
            )
            stopped_early = True
            break

    runtime_sec = time.time() - start_time
    writer.close()

    save_learning_curves(history, run_dir)

    if best_metrics is None:
        best_metrics = {
            'best_epoch': -1,
            'best_monitor_value': None
        }

    save_json(best_metrics, os.path.join(run_dir, 'best_metrics.json'))

    run_summary = {
        'seed': seed,
        'batch_size': batch_size,
        'lr': lr,
        'epochs_requested': epochs,
        'epochs_completed': history[-1]['epoch'],
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'best_epoch': best_epoch,
        'best_monitor_value': best_monitor_value,
        'best_val_total_loss': best_metrics.get('val_total_loss'),
        'best_val_seg_loss': best_metrics.get('val_seg_loss'),
        'best_val_bony_loss': best_metrics.get('val_bony_loss'),
        'best_val_seg_mean_dice': best_metrics.get('val_seg_mean_dice'),
        'best_val_bony_mean_dice': best_metrics.get('val_bony_mean_dice'),
        'best_val_joint_mean_dice': best_metrics.get('val_joint_mean_dice'),
        'final_train_total_loss': history[-1]['train_total_loss'],
        'final_val_total_loss': history[-1]['val_total_loss'],
        'final_val_seg_loss': history[-1]['val_seg_loss'],
        'final_val_bony_loss': history[-1]['val_bony_loss'],
        'final_val_seg_mean_dice': history[-1]['val_seg_mean_dice'],
        'final_val_bony_mean_dice': history[-1]['val_bony_mean_dice'],
        'final_val_joint_mean_dice': history[-1]['val_joint_mean_dice'],
        'stopped_early': int(stopped_early),
        'runtime_sec': runtime_sec,
        'load_path': load_path,
        'best_checkpoint': best_ckpt_path,
        'last_checkpoint': last_ckpt_path
    }
    save_json(run_summary, os.path.join(run_dir, 'run_summary.json'))

    return run_summary


def collect_run_summaries(root_dir: str) -> List[Dict[str, Any]]:
    summaries = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'run_summary.json' in filenames:
            json_path = os.path.join(dirpath, 'run_summary.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                summaries.append(json.load(f))
    return summaries


def save_experiment_summary(root_dir: str) -> None:
    summaries = collect_run_summaries(root_dir)
    if len(summaries) == 0:
        return

    csv_path = os.path.join(root_dir, 'experiment_summary.csv')
    fieldnames = [
        'seed', 'batch_size', 'lr',
        'epochs_requested', 'epochs_completed',
        'monitor_metric', 'monitor_mode',
        'best_epoch', 'best_monitor_value',
        'best_val_total_loss', 'best_val_seg_loss', 'best_val_bony_loss',
        'best_val_seg_mean_dice', 'best_val_bony_mean_dice', 'best_val_joint_mean_dice',
        'final_train_total_loss', 'final_val_total_loss',
        'final_val_seg_loss', 'final_val_bony_loss',
        'final_val_seg_mean_dice', 'final_val_bony_mean_dice', 'final_val_joint_mean_dice',
        'stopped_early', 'runtime_sec',
        'load_path', 'best_checkpoint', 'last_checkpoint'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    grouped = {}
    for row in summaries:
        key = (row['batch_size'], row['monitor_metric'])
        grouped.setdefault(key, []).append(row)

    stat_path = os.path.join(root_dir, 'experiment_summary_stats.csv')
    with open(stat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'batch_size', 'monitor_metric', 'n_runs',
                'mean_best_val_total_loss', 'sd_best_val_total_loss',
                'mean_best_val_seg_mean_dice', 'sd_best_val_seg_mean_dice',
                'mean_best_val_bony_mean_dice', 'sd_best_val_bony_mean_dice',
                'mean_best_val_joint_mean_dice', 'sd_best_val_joint_mean_dice',
                'mean_best_monitor_value', 'sd_best_monitor_value'
            ]
        )
        writer.writeheader()

        for (batch_size, monitor_metric), rows in grouped.items():
            best_total = np.array([float(r['best_val_total_loss']) for r in rows], dtype=np.float32)
            best_seg_dice = np.array([float(r['best_val_seg_mean_dice']) for r in rows], dtype=np.float32)
            best_bony_dice = np.array([float(r['best_val_bony_mean_dice']) for r in rows], dtype=np.float32)
            best_joint_dice = np.array([float(r['best_val_joint_mean_dice']) for r in rows], dtype=np.float32)
            best_monitor = np.array([float(r['best_monitor_value']) for r in rows], dtype=np.float32)

            writer.writerow({
                'batch_size': batch_size,
                'monitor_metric': monitor_metric,
                'n_runs': len(rows),
                'mean_best_val_total_loss': float(best_total.mean()),
                'sd_best_val_total_loss': float(best_total.std(ddof=1)) if len(best_total) > 1 else 0.0,
                'mean_best_val_seg_mean_dice': float(best_seg_dice.mean()),
                'sd_best_val_seg_mean_dice': float(best_seg_dice.std(ddof=1)) if len(best_seg_dice) > 1 else 0.0,
                'mean_best_val_bony_mean_dice': float(best_bony_dice.mean()),
                'sd_best_val_bony_mean_dice': float(best_bony_dice.std(ddof=1)) if len(best_bony_dice) > 1 else 0.0,
                'mean_best_val_joint_mean_dice': float(best_joint_dice.mean()),
                'sd_best_val_joint_mean_dice': float(best_joint_dice.std(ddof=1)) if len(best_joint_dice) > 1 else 0.0,
                'mean_best_monitor_value': float(best_monitor.mean()),
                'sd_best_monitor_value': float(best_monitor.std(ddof=1)) if len(best_monitor) > 1 else 0.0,
            })


def get_args():
    parser = argparse.ArgumentParser(
        description='Stage-2 refinement training for DeepDDH with seed-wise reproducible outputs'
    )
    parser.add_argument('--n-classes', type=int, default=8)
    parser.add_argument('--bony-class', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=1e-5)

    parser.add_argument(
        '--monitor-metric',
        type=str,
        choices=[
            'val_total_loss',
            'val_seg_loss',
            'val_bony_loss',
            'val_seg_mean_dice',
            'val_bony_mean_dice',
            'val_joint_mean_dice'
        ],
        default='val_total_loss'
    )

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--load', type=str, default='')
    checkpoint_group.add_argument(
        '--load-template',
        type=str,
        default='',
        help='Optional template, e.g. ./checkpoint/Stage-1/dilated_ddhnet/bs_{batch_size}/seed_{seed}/checkpoints/best_model.pth'
    )

    parser.add_argument('--save-all-epoch-ckpt', action='store_true')
    parser.add_argument('--no-milestone-ckpt', action='store_true')
    parser.add_argument('--num-workers-train', type=int, default=8)
    parser.add_argument('--num-workers-val', type=int, default=4)
    parser.add_argument('--base-path', type=str, default=base_path)

    parser.add_argument('--use-bony-centroid-loss', action='store_true')
    parser.add_argument('--seg-centroid-weight', type=float, default=0.1)
    parser.add_argument('--bony-centroid-weight', type=float, default=0.1)
    parser.add_argument('--include-background-in-dice', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    validate_args(args)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    root_dir = os.path.join(resolve_project_path(args.base_path), 'ddhnet_bony')
    ensure_dir(root_dir)

    try:
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                load_path = resolve_load_path(
                    load_path=args.load,
                    load_template=args.load_template,
                    seed=seed,
                    batch_size=batch_size
                )

                run_dir = os.path.join(
                    root_dir,
                    f'bs_{batch_size}',
                    f'seed_{seed}'
                )

                summary = train_one_run(
                    seed=seed,
                    batch_size=batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    img_scale=args.scale,
                    device=device,
                    n_classes=args.n_classes,
                    bony_class=args.bony_class,
                    run_dir=run_dir,
                    load_path=load_path,
                    num_workers_train=args.num_workers_train,
                    num_workers_val=args.num_workers_val,
                    save_all_epoch_ckpt=args.save_all_epoch_ckpt,
                    save_milestone_ckpt=(not args.no_milestone_ckpt),
                    monitor_metric=args.monitor_metric,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    use_bony_centroid_loss=args.use_bony_centroid_loss,
                    seg_centroid_weight=args.seg_centroid_weight,
                    bony_centroid_weight=args.bony_centroid_weight,
                    ignore_background=(not args.include_background_in_dice)
                )

                logging.info(
                    f'Finished Stage-2 run | seed={seed} | batch_size={batch_size} | '
                    f'best_epoch={summary["best_epoch"]} | '
                    f'best_monitor_value={summary["best_monitor_value"]:.6f}'
                )

        save_experiment_summary(root_dir)
        logging.info(f'Experiment summaries saved under: {root_dir}')

    except KeyboardInterrupt:
        logging.info('Interrupted by user. Exiting safely.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
