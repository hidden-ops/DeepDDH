import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any


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

from model.unet import UNet
from model.segnet import SegNet
from model.fcn import FCN
from model.bisenet import BiSeNet
from attention.danet.danet import DANet
from model.aunet import AUNet_Res
from model.scse_unet import SCSERes
from ddhnet_model.ddhnet import DDHNet
from ddhnet_model.dilated_ddhnet import dilated_DDHNet
from attention.danet.danet import danet_DDNet

from utils.eval import eval_net
from utils.dataset import BasicDataset
from utils.segLoss import SegmentationLosses


tr_dir_img = str(PROJECT_ROOT / 'data' / 'Training' / 'imgs') + os.sep
tr_dir_mask = str(PROJECT_ROOT / 'data' / 'Training' / 'segs') + os.sep

val_dir_img = str(PROJECT_ROOT / 'data' / 'Testing' / 'imgs') + os.sep
val_dir_mask = str(PROJECT_ROOT / 'data' / 'Testing' / 'segs') + os.sep

base_path = str(PROJECT_ROOT / 'checkpoint' / 'Stage-1')
MODEL_CHOICES = (
    'dilated_ddhnet', 'ddhnet', 'danet_ddhnet', 'unet', 'segnet',
    'fcn', 'aunet', 'danet', 'bisenet', 'scse',
)


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
    if args.n_classes < 2:
        raise ValueError('--n-classes must be at least 2 for multi-class training.')
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
    if not 0 <= args.dropout1 <= 1 or not 0 <= args.dropout2 <= 1:
        raise ValueError('--dropout1 and --dropout2 must be in the interval [0, 1].')


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
    train_total_losses = [row['train_total_loss'] for row in history]
    val_losses = [row['val_loss'] for row in history]
    val_mean_dices = [row['val_mean_dice'] for row in history]
    monitor_values = [row['monitor_value'] for row in history]

    plot_curve(
        epochs, train_total_losses,
        xlabel='Epoch',
        ylabel='Train total loss',
        title='Stage-1 Train Total Loss by Epoch',
        save_path=os.path.join(out_dir, 'curve_train_total_loss.png')
    )
    plot_curve(
        epochs, val_losses,
        xlabel='Epoch',
        ylabel='Validation loss',
        title='Stage-1 Validation Loss by Epoch',
        save_path=os.path.join(out_dir, 'curve_val_loss.png')
    )
    plot_curve(
        epochs, val_mean_dices,
        xlabel='Epoch',
        ylabel='Validation mean Dice',
        title='Stage-1 Validation Mean Dice by Epoch',
        save_path=os.path.join(out_dir, 'curve_val_mean_dice.png')
    )
    plot_curve(
        epochs, monitor_values,
        xlabel='Epoch',
        ylabel='Monitor value',
        title='Stage-1 Monitor Metric by Epoch',
        save_path=os.path.join(out_dir, 'curve_monitor_metric.png')
    )


def build_model(model_name: str, n_classes: int, dropout1: float = 1.0, dropout2: float = 1.0):
    if model_name == 'unet':
        net = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
    elif model_name == 'segnet':
        net = SegNet(n_channels=3, n_classes=n_classes)
    elif model_name == 'fcn':
        net = FCN(n_channels=3, n_classes=n_classes, pretrained_model=True)
    elif model_name == 'aunet':
        net = AUNet_Res(n_channels=3, n_classes=n_classes, pretrained_model=True, learned_bilinear=True)
    elif model_name == 'danet':
        net = DANet(n_classes=n_classes, n_channels=3, backbone='resnet50')
    elif model_name == 'bisenet':
        net = BiSeNet(n_classes=n_classes, n_channels=3, pretrained_model=True)
    elif model_name == 'scse':
        net = SCSERes(n_classes=n_classes, n_channels=3, pretrained_model=True)
    elif model_name == 'ddhnet':
        net = DDHNet(
            n_classes=n_classes,
            n_channels=3,
            dropout1=dropout1,
            dropout2=dropout2,
            pretrained_model=True
        )
    elif model_name == 'dilated_ddhnet':
        net = dilated_DDHNet(n_classes=n_classes, n_channels=3, pretrained_model=True)
    elif model_name == 'danet_ddhnet':
        net = danet_DDNet(n_classes=n_classes, n_channels=3, backbone='resnet50')
    else:
        raise ValueError(f'Unsupported model_name: {model_name}')
    return net


def build_criterion(device: torch.device):
    return SegmentationLosses(cuda=(device.type == 'cuda'), batch_average=False)


def compute_segmentation_loss(model_name: str, net, imgs, true_masks, criterion) -> torch.Tensor:
    if model_name == 'bisenet':
        aux_pred0, aux_pred1, main_pred, _ = net(imgs)
        aux_loss0 = criterion.CrossEntropyLoss(aux_pred0, true_masks) + \
                    criterion.FocalLoss(aux_pred0, true_masks, gamma=2, alpha=0.5)
        aux_loss1 = criterion.CrossEntropyLoss(aux_pred1, true_masks) + \
                    criterion.FocalLoss(aux_pred1, true_masks, gamma=2, alpha=0.5)
        main_loss = criterion.CrossEntropyLoss(main_pred, true_masks) + \
                    criterion.FocalLoss(main_pred, true_masks, gamma=2, alpha=0.5)
        loss = (aux_loss0 + aux_loss1 + main_loss) / 3.0

    elif model_name == 'danet':
        main_pred = net(imgs)
        aux_loss = criterion.CrossEntropyLoss(main_pred[0], true_masks) + \
                   criterion.FocalLoss(main_pred[0], true_masks, gamma=2, alpha=0.5)
        main_loss = criterion.CrossEntropyLoss(main_pred[1], true_masks) + \
                    criterion.FocalLoss(main_pred[1], true_masks, gamma=2, alpha=0.5)
        loss = (aux_loss + main_loss) / 2.0

    else:
        masks_pred = net(imgs)
        loss = criterion.CrossEntropyLoss(masks_pred, true_masks) + \
               criterion.FocalLoss(masks_pred, true_masks, gamma=2, alpha=0.5)

    return loss


@torch.no_grad()
def compute_validation_total_loss(net, val_loader, model_name, device, criterion) -> float:
    """
    Validation loss using the same training objective:
    total_loss = CrossEntropyLoss + FocalLoss
    This is useful for train-vs-validation loss curves.
    """
    was_training = net.training
    net.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)

        loss = compute_segmentation_loss(model_name, net, imgs, true_masks, criterion)
        total_loss += loss.item()
        n_batches += 1

    if was_training:
        net.train()

    return total_loss / max(n_batches, 1)


def infer_monitor_mode(monitor_metric: str) -> str:
    if monitor_metric in ['val_loss', 'val_total_loss']:
        return 'min'
    elif monitor_metric == 'val_mean_dice':
        return 'max'
    else:
        raise ValueError(f'Unsupported monitor_metric: {monitor_metric}')


def is_better(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == 'max':
        return current > (best + min_delta)
    elif mode == 'min':
        return current < (best - min_delta)
    else:
        raise ValueError(f'Unsupported monitor mode: {mode}')


def get_monitor_value(
    monitor_metric: str,
    val_total_loss: float,
    val_loss: float,
    val_mean_dice: float
) -> float:
    if monitor_metric == 'val_total_loss':
        return val_total_loss
    elif monitor_metric == 'val_loss':
        return val_loss
    elif monitor_metric == 'val_mean_dice':
        return val_mean_dice
    else:
        raise ValueError(f'Unsupported monitor_metric: {monitor_metric}')


def train_one_run(
    model_name: str,
    seed: int,
    batch_size: int,
    lr: float,
    epochs: int,
    img_scale: float,
    device: torch.device,
    n_classes: int,
    run_dir: str,
    num_workers_train: int = 8,
    num_workers_val: int = 4,
    save_all_epoch_ckpt: bool = False,
    monitor_metric: str = 'val_loss',
    patience: int = 5,
    min_delta: float = 0.001,
    dropout1: float = 1.0,
    dropout2: float = 1.0
) -> Dict[str, Any]:

    monitor_mode = infer_monitor_mode(monitor_metric)

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, 'checkpoints'))
    ensure_dir(os.path.join(run_dir, 'tb'))

    set_seed(seed)

    train_set = BasicDataset(tr_dir_img, tr_dir_mask, batch_size, img_scale)
    val_set = BasicDataset(val_dir_img, val_dir_mask, batch_size, img_scale)

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

    net = build_model(
        model_name=model_name,
        n_classes=n_classes,
        dropout1=dropout1,
        dropout2=dropout2
    )
    net.to(device=device)

    criterion = build_criterion(device)
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
        'model_name': model_name,
        'seed': seed,
        'batch_size': batch_size,
        'lr': lr,
        'epochs': epochs,
        'img_scale': img_scale,
        'n_classes': n_classes,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'patience': patience,
        'min_delta': min_delta,
        'dropout1': dropout1,
        'dropout2': dropout2
    }
    save_json(config, os.path.join(run_dir, 'config.json'))

    logging.info(
        f'Starting run | model={model_name} | seed={seed} | batch_size={batch_size} | '
        f'lr={lr} | train={len(train_set)} | val={len(val_set)} | '
        f'monitor_metric={monitor_metric} | monitor_mode={monitor_mode}'
    )

    history = []
    history_csv = os.path.join(run_dir, 'history.csv')

    best_monitor_value = -float('inf') if monitor_mode == 'max' else float('inf')
    best_epoch = -1
    best_val_loss = None
    best_val_total_loss = None
    best_val_mean_dice = None
    best_per_class_dice = None

    best_ckpt_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    last_ckpt_path = os.path.join(run_dir, 'checkpoints', 'last_model.pth')

    epochs_without_improve = 0
    stopped_early = False
    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_set), desc=f'Seed {seed} | Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == net.n_channels, (
                    f'Network expects {net.n_channels} channels, '
                    f'but got {imgs.shape[1]}.'
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                loss = compute_segmentation_loss(model_name, net, imgs, true_masks, criterion)
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1
                global_step += 1

                writer.add_scalar('batch/train_total_loss', batch_loss, global_step)
                pbar.set_postfix(loss=f'{batch_loss:.6f}')
                pbar.update(imgs.shape[0])

        train_total_loss = epoch_loss / max(num_batches, 1)

        # 1) validation loss with same training objective (CE + Focal)
        val_total_loss = compute_validation_total_loss(net, val_loader, model_name, device, criterion)

        # 2) validation metrics from eval.py
        val_results = eval_net(
            net,
            val_loader,
            model_name,
            device,
            return_dict=True,
            ignore_background=True
        )
        val_loss = float(val_results['val_loss'])
        val_mean_dice = float(val_results['mean_dice'])
        per_class_dice = val_results['per_class_dice']

        monitor_value = get_monitor_value(
            monitor_metric=monitor_metric,
            val_total_loss=val_total_loss,
            val_loss=val_loss,
            val_mean_dice=val_mean_dice
        )

        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('epoch/train_total_loss', train_total_loss, epoch)
        writer.add_scalar('epoch/val_total_loss', val_total_loss, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/val_mean_dice', val_mean_dice, epoch)
        writer.add_scalar('epoch/monitor_value', monitor_value, epoch)
        writer.add_scalar('epoch/lr', current_lr, epoch)

        scheduler.step(monitor_value)

        improved = is_better(monitor_value, best_monitor_value, monitor_mode, min_delta=min_delta)

        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_val_total_loss = val_total_loss
            best_val_loss = val_loss
            best_val_mean_dice = val_mean_dice
            best_per_class_dice = per_class_dice
            epochs_without_improve = 0
            torch.save(net.state_dict(), best_ckpt_path)
        else:
            epochs_without_improve += 1

        torch.save(net.state_dict(), last_ckpt_path)

        if save_all_epoch_ckpt:
            epoch_ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:03d}.pth')
            torch.save(net.state_dict(), epoch_ckpt_path)

        row = {
            'epoch': epoch,
            'seed': seed,
            'model_name': model_name,
            'batch_size': batch_size,
            'lr': lr,
            'train_total_loss': train_total_loss,
            'val_total_loss': val_total_loss,
            'val_loss': val_loss,
            'val_mean_dice': val_mean_dice,
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
                'epoch', 'seed', 'model_name', 'batch_size', 'lr',
                'train_total_loss', 'val_total_loss', 'val_loss', 'val_mean_dice',
                'monitor_metric', 'monitor_value',
                'best_so_far', 'best_epoch_so_far', 'lr_current'
            ]
        )

        logging.info(
            f'[Seed {seed}] Epoch {epoch}/{epochs} | '
            f'train_total_loss={train_total_loss:.6f} | '
            f'val_total_loss={val_total_loss:.6f} | '
            f'val_loss={val_loss:.6f} | '
            f'val_mean_dice={val_mean_dice:.6f} | '
            f'monitor({monitor_metric})={monitor_value:.6f} | '
            f'best_epoch={best_epoch} | lr={current_lr:.8f}'
        )

        if epochs_without_improve >= patience:
            logging.info(
                f'[Seed {seed}] Early stopping at epoch {epoch} '
                f'(no improvement for {patience} consecutive epochs).'
            )
            stopped_early = True
            break

    runtime_sec = time.time() - start_time
    writer.close()

    save_learning_curves(history, run_dir)

    best_metrics = {
        'best_epoch': best_epoch,
        'best_monitor_value': best_monitor_value,
        'best_val_total_loss': best_val_total_loss,
        'best_val_loss': best_val_loss,
        'best_val_mean_dice': best_val_mean_dice,
        'best_per_class_dice': best_per_class_dice
    }
    save_json(best_metrics, os.path.join(run_dir, 'best_metrics.json'))

    run_summary = {
        'model_name': model_name,
        'seed': seed,
        'batch_size': batch_size,
        'lr': lr,
        'epochs_requested': epochs,
        'epochs_completed': history[-1]['epoch'],
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'best_epoch': best_epoch,
        'best_monitor_value': best_monitor_value,
        'best_val_total_loss': best_val_total_loss,
        'best_val_loss': best_val_loss,
        'best_val_mean_dice': best_val_mean_dice,
        'final_train_total_loss': history[-1]['train_total_loss'],
        'final_val_total_loss': history[-1]['val_total_loss'],
        'final_val_loss': history[-1]['val_loss'],
        'final_val_mean_dice': history[-1]['val_mean_dice'],
        'stopped_early': int(stopped_early),
        'runtime_sec': runtime_sec,
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
        'model_name', 'seed', 'batch_size', 'lr',
        'epochs_requested', 'epochs_completed',
        'monitor_metric', 'monitor_mode',
        'best_epoch', 'best_monitor_value',
        'best_val_total_loss', 'best_val_loss', 'best_val_mean_dice',
        'final_train_total_loss', 'final_val_total_loss', 'final_val_loss', 'final_val_mean_dice',
        'stopped_early', 'runtime_sec',
        'best_checkpoint', 'last_checkpoint'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    grouped = {}
    for row in summaries:
        key = (row['model_name'], row['batch_size'], row['monitor_metric'])
        grouped.setdefault(key, []).append(row)

    stat_path = os.path.join(root_dir, 'experiment_summary_stats.csv')
    with open(stat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model_name', 'batch_size', 'monitor_metric', 'n_runs',
                'mean_best_val_loss', 'sd_best_val_loss',
                'mean_best_val_mean_dice', 'sd_best_val_mean_dice',
                'mean_best_monitor_value', 'sd_best_monitor_value'
            ]
        )
        writer.writeheader()

        for (model_name, batch_size, monitor_metric), rows in grouped.items():
            best_val_losses = np.array([float(r['best_val_loss']) for r in rows], dtype=np.float32)
            best_val_mean_dices = np.array([float(r['best_val_mean_dice']) for r in rows], dtype=np.float32)
            best_monitor_values = np.array([float(r['best_monitor_value']) for r in rows], dtype=np.float32)

            writer.writerow({
                'model_name': model_name,
                'batch_size': batch_size,
                'monitor_metric': monitor_metric,
                'n_runs': len(rows),
                'mean_best_val_loss': float(best_val_losses.mean()),
                'sd_best_val_loss': float(best_val_losses.std(ddof=1)) if len(best_val_losses) > 1 else 0.0,
                'mean_best_val_mean_dice': float(best_val_mean_dices.mean()),
                'sd_best_val_mean_dice': float(best_val_mean_dices.std(ddof=1)) if len(best_val_mean_dices) > 1 else 0.0,
                'mean_best_monitor_value': float(best_monitor_values.mean()),
                'sd_best_monitor_value': float(best_monitor_values.std(ddof=1)) if len(best_monitor_values) > 1 else 0.0,
            })


def get_args():
    parser = argparse.ArgumentParser(
        description='Stage-1 pretraining for DeepDDH with seed-wise reproducible outputs'
    )
    parser.add_argument('--model-name', choices=MODEL_CHOICES, default='dilated_ddhnet')
    parser.add_argument('--n-classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument(
        '--monitor-metric',
        type=str,
        choices=['val_total_loss', 'val_loss', 'val_mean_dice'],
        default='val_loss',
        help='Metric used for scheduler, early stopping, and best checkpoint selection'
    )
    parser.add_argument('--dropout1', type=float, default=1.0)
    parser.add_argument('--dropout2', type=float, default=1.0)
    parser.add_argument('--save-all-epoch-ckpt', action='store_true')
    parser.add_argument('--num-workers-train', type=int, default=8)
    parser.add_argument('--num-workers-val', type=int, default=4)
    parser.add_argument('--base-path', type=str, default=base_path)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    validate_args(args)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    root_dir = os.path.join(resolve_project_path(args.base_path), args.model_name)
    ensure_dir(root_dir)

    try:
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                run_dir = os.path.join(
                    root_dir,
                    f'bs_{batch_size}',
                    f'seed_{seed}'
                )

                summary = train_one_run(
                    model_name=args.model_name,
                    seed=seed,
                    batch_size=batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    img_scale=args.scale,
                    device=device,
                    n_classes=args.n_classes,
                    run_dir=run_dir,
                    num_workers_train=args.num_workers_train,
                    num_workers_val=args.num_workers_val,
                    save_all_epoch_ckpt=args.save_all_epoch_ckpt,
                    monitor_metric=args.monitor_metric,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    dropout1=args.dropout1,
                    dropout2=args.dropout2
                )
                logging.info(
                    f'Finished run | seed={seed} | batch_size={batch_size} | '
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
