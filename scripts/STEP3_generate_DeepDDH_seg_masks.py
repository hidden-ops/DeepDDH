"""Run a trained Stage-2 DeepDDH model and save its two segmentation maps."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddhnet_model.dilated_ddhnet_bony import DDHNet_Bony
from utils.inference_dataset import InferenceDataset


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested, but CUDA is not available to PyTorch.')
    return torch.device(requested)


def load_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Stage-2 checkpoint does not exist: {checkpoint_path}')

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f'Expected a state-dict checkpoint, got {type(checkpoint).__name__}.')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f'No model state_dict found in checkpoint: {checkpoint_path}')

    return {
        key.removeprefix('module.'): value
        for key, value in checkpoint.items()
    }


def encode_mask(mask: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == 'class_id':
        encoded = mask
    elif encoding == 'times_30':
        encoded = mask * 30
    else:
        raise ValueError(f'Unsupported mask encoding: {encoding}')
    return encoded.astype(np.uint8, copy=False)


def save_mask(mask: np.ndarray, path: Path, original_size, encoding: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists; pass --overwrite to replace it: {path}')

    image = Image.fromarray(encode_mask(mask, encoding), mode='L')
    if image.size != original_size:
        image = image.resize(original_size, resample=Image.Resampling.NEAREST)
    image.save(path)


def run_inference(args: argparse.Namespace) -> None:
    if args.n_classes < 2 or args.bony_classes < 2:
        raise ValueError('--n-classes and --bony-classes must both be at least 2.')
    if args.batch_size < 1:
        raise ValueError(f'--batch-size must be at least 1, got {args.batch_size}')
    if args.num_workers < 0:
        raise ValueError(f'--num-workers cannot be negative, got {args.num_workers}')
    if args.mask_encoding == 'times_30' and max(args.n_classes - 1, args.bony_classes - 1) > 8:
        raise ValueError('times_30 encoding supports class IDs only up to 8 in an 8-bit PNG.')

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_seg_dir = Path(args.output_seg_dir).expanduser().resolve()
    output_bony_dir = Path(args.output_bony_dir).expanduser().resolve()
    if output_seg_dir == output_bony_dir:
        raise ValueError('Anatomical and bony outputs must use different directories.')
    output_seg_dir.mkdir(parents=True, exist_ok=True)
    output_bony_dir.mkdir(parents=True, exist_ok=True)

    dataset = InferenceDataset(
        image_dir=args.input_dir,
        scale=args.scale,
        normalize=args.normalize,
    )
    if not args.overwrite:
        existing_outputs = [
            path
            for source_path in dataset.files
            for path in (
                output_seg_dir / f'{source_path.stem}.png',
                output_bony_dir / f'{source_path.stem}.png',
            )
            if path.exists()
        ]
        if existing_outputs:
            preview = ', '.join(str(path) for path in existing_outputs[:5])
            raise FileExistsError(
                f'{len(existing_outputs)} output file(s) already exist. '
                f'Pass --overwrite to replace them. First paths: {preview}'
            )
    scaled_sizes = set(dataset.scaled_sizes())
    if args.batch_size > 1 and len(scaled_sizes) > 1:
        raise ValueError(
            'Images have different spatial sizes after scaling. '
            'Use --batch-size 1 or resize inputs to a common size.'
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = DDHNet_Bony(
        n_classes=args.n_classes,
        bony_class=args.bony_classes,
        n_channels=3,
        pretrained_model=False,
    )
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()

    use_amp = args.amp and device.type == 'cuda'
    if args.amp and device.type != 'cuda':
        logging.warning('--amp is ignored because the selected device is not CUDA.')

    logging.info('Device: %s', device)
    logging.info('Checkpoint: %s', checkpoint_path)
    logging.info('Images: %d', len(dataset))
    logging.info('Mask encoding: %s', args.mask_encoding)

    start_time = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc='DeepDDH inference', unit='batch'):
            images = batch['image'].to(
                device=device,
                dtype=torch.float32,
                non_blocking=(device.type == 'cuda'),
            )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                seg_logits, bony_logits = model(images)

            seg_predictions = torch.argmax(seg_logits, dim=1).cpu().numpy()
            bony_predictions = torch.argmax(bony_logits, dim=1).cpu().numpy()

            for index, case_id in enumerate(batch['id']):
                original_size = (
                    int(batch['original_width'][index]),
                    int(batch['original_height'][index]),
                )
                save_mask(
                    seg_predictions[index],
                    output_seg_dir / f'{case_id}.png',
                    original_size,
                    args.mask_encoding,
                    args.overwrite,
                )
                save_mask(
                    bony_predictions[index],
                    output_bony_dir / f'{case_id}.png',
                    original_size,
                    args.mask_encoding,
                    args.overwrite,
                )

    logging.info('Saved anatomical maps to: %s', output_seg_dir)
    logging.info('Saved bony maps to: %s', output_bony_dir)
    logging.info('Elapsed time: %.2f seconds', time.time() - start_time)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='STEP3: generate DeepDDH anatomical and bony-substructure segmentation maps.'
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing input image files.')
    parser.add_argument('--checkpoint', required=True, help='Trained Stage-2 model state_dict (.pth).')
    parser.add_argument(
        '--output-seg-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP3' / 'seg_masks'),
        help='Output directory for 8-class anatomical maps.',
    )
    parser.add_argument(
        '--output-bony-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP3' / 'bony_masks'),
        help='Output directory for 7-class bony-substructure maps.',
    )
    parser.add_argument('--n-classes', type=int, default=8)
    parser.add_argument('--bony-classes', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--amp', action='store_true', help='Use CUDA automatic mixed precision.')
    parser.add_argument(
        '--normalize',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Use [0, 1] network inputs. Already-normalised inputs are preserved; '
            'conventional 8-bit inputs are divided by 255. Use --no-normalize only '
            'for a checkpoint intentionally trained on an unnormalised intensity scale.'
        ),
    )
    parser.add_argument(
        '--mask-encoding',
        choices=['class_id', 'times_30'],
        default='class_id',
        help='Save raw class IDs (recommended) or the legacy class_id*30 grayscale encoding.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run_inference(get_args())
