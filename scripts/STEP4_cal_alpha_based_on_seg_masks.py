"""Extract DeepDDH landmarks and calculate parallel, alpha, and beta angles."""

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVALID = -9999.0
SUPPORTED_MASK_EXTENSIONS = {'.png'}

# Bony class -> index in the five-keypoint array. This preserves the mapping
# used by the original class_id*30 implementation: 60, 90, 120, 180, 150.
BONY_CLASS_TO_KEYPOINT = {2: 0, 3: 1, 4: 2, 6: 3, 5: 4}


def read_class_map(path: Path, max_class_id: int, encoding: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Mask file does not exist: {path}')
    with Image.open(path) as image:
        array = np.asarray(image.convert('L'), dtype=np.int32)

    unique_values = np.unique(array)
    if encoding == 'auto':
        if unique_values.max(initial=0) <= max_class_id:
            encoding = 'class_id'
        elif np.all(unique_values % 30 == 0):
            encoding = 'times_30'
        else:
            raise ValueError(
                f'Cannot infer encoding for {path}; values are {unique_values.tolist()}. '
                'Expected raw class IDs or multiples of 30.'
            )

    if encoding == 'times_30':
        if not np.all(unique_values % 30 == 0):
            raise ValueError(f'{path} contains values that are not divisible by 30.')
        array = array // 30
    elif encoding != 'class_id':
        raise ValueError(f'Unsupported mask encoding: {encoding}')

    decoded_values = np.unique(array)
    if decoded_values.min(initial=0) < 0 or decoded_values.max(initial=0) > max_class_id:
        raise ValueError(
            f'{path} contains decoded class IDs {decoded_values.tolist()}, '
            f'outside the expected range 0-{max_class_id}.'
        )
    return array


def find_upper_edge(class_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the first foreground pixel and its class for every non-empty column."""
    foreground = class_map != 0
    valid_columns = np.flatnonzero(foreground.any(axis=0))
    if valid_columns.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)

    rows = np.argmax(foreground[:, valid_columns], axis=0)
    positions = np.column_stack((rows, valid_columns)).astype(np.int32)
    labels = class_map[rows, valid_columns].astype(np.int32)
    return positions, labels


def extract_five_keypoints(edge_positions: np.ndarray, edge_labels: np.ndarray) -> np.ndarray:
    """Use the first upper-edge occurrence of bony classes 2-6 as landmarks."""
    keypoints = np.full((5, 2), INVALID, dtype=np.float64)
    for class_id, keypoint_index in BONY_CLASS_TO_KEYPOINT.items():
        matches = np.flatnonzero(edge_labels == class_id)
        if matches.size:
            keypoints[keypoint_index] = edge_positions[matches[0]]
    return keypoints


def calculate_centroid(binary_map: np.ndarray) -> Optional[np.ndarray]:
    rows, columns = np.nonzero(binary_map)
    if columns.size == 0:
        return None
    return np.array([rows.mean(), columns.mean()], dtype=np.float64)


def calculate_line_angle(point1: Sequence[float], point2: Sequence[float]) -> float:
    """Return an unoriented line angle in degrees in the interval (-90, 90]."""
    row1, column1 = float(point1[0]), float(point1[1])
    row2, column2 = float(point2[0]), float(point2[1])
    if row1 == row2 and column1 == column2:
        raise ValueError('Cannot calculate an angle from two identical points.')

    angle = math.degrees(math.atan2(row2 - row1, column2 - column1))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def is_valid_point(point: Sequence[float]) -> bool:
    return bool(np.all(np.asarray(point) != INVALID))


def calculate_case(
    bony_map: np.ndarray,
    seg_map: np.ndarray,
    case_name: str,
    labrum_class: int = 5,
    min_labrum_pixels: int = 20,
) -> Dict[str, object]:
    if bony_map.shape != seg_map.shape:
        raise ValueError(
            f'Segmentation maps for {case_name} have different shapes: '
            f'bony={bony_map.shape}, anatomical={seg_map.shape}'
        )

    edge_positions, edge_labels = find_upper_edge(bony_map)
    five_keypoints = extract_five_keypoints(edge_positions, edge_labels)

    labrum_binary = seg_map == labrum_class
    labrum_centroid = None
    if int(labrum_binary.sum()) >= min_labrum_pixels:
        labrum_centroid = calculate_centroid(labrum_binary)

    all_keypoints = np.full((6, 2), INVALID, dtype=np.float64)
    all_keypoints[:5] = five_keypoints
    if labrum_centroid is not None:
        all_keypoints[5] = labrum_centroid

    parallel_angle = None
    alpha_angle = None
    beta_angle = None

    if is_valid_point(five_keypoints[0]) and is_valid_point(five_keypoints[1]):
        parallel_angle = -calculate_line_angle(five_keypoints[0], five_keypoints[1])

    if (
        parallel_angle is not None
        and is_valid_point(five_keypoints[2])
        and is_valid_point(five_keypoints[3])
    ):
        cut_angle = calculate_line_angle(five_keypoints[2], five_keypoints[3])
        alpha_angle = cut_angle - parallel_angle

    if (
        parallel_angle is not None
        and is_valid_point(five_keypoints[4])
        and labrum_centroid is not None
    ):
        labrum_angle = calculate_line_angle(five_keypoints[4], labrum_centroid)
        beta_angle = -labrum_angle + parallel_angle

    all_angles = [
        parallel_angle if parallel_angle is not None else INVALID,
        alpha_angle if alpha_angle is not None else INVALID,
        beta_angle if beta_angle is not None else INVALID,
    ]
    if alpha_angle is not None and beta_angle is not None:
        status = 'complete'
    elif parallel_angle is not None:
        status = 'parallel_only'
    else:
        status = 'invalid'

    return {
        'name': case_name,
        'status': status,
        'all_key_points': [
            [float(point[0]), float(point[1])]
            for point in all_keypoints
        ],
        'all_angles': [float(value) for value in all_angles],
        'parallel_angle': None if parallel_angle is None else float(parallel_angle),
        'alpha_angle': None if alpha_angle is None else float(alpha_angle),
        'beta_angle': None if beta_angle is None else float(beta_angle),
        'labrum_pixels': int(labrum_binary.sum()),
    }


def judge_pose_result(results: Sequence[Dict[str, object]]) -> str:
    parallel_angles = [
        float(result['parallel_angle'])
        for result in results
        if result.get('parallel_angle') is not None
    ]
    if not parallel_angles:
        return 'Wrong pose'
    if all(angle > 5 for angle in parallel_angles):
        return 'Positive'
    if all(angle < -5 for angle in parallel_angles):
        return 'Negative'
    if all(abs(angle) > 5 for angle in parallel_angles):
        return 'Too fast'
    return 'Proposal'


def list_pngs(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Mask directory does not exist: {directory}')
    files = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MASK_EXTENSIONS
    }
    if not files:
        raise RuntimeError(f'No PNG masks found in: {directory}')
    return files


def generate_results(
    seg_dir: Path,
    bony_dir: Path,
    encoding: str,
    labrum_class: int,
    min_labrum_pixels: int,
) -> Tuple[List[Dict[str, object]], str]:
    seg_files = list_pngs(seg_dir)
    bony_files = list_pngs(bony_dir)

    missing_seg = sorted(set(bony_files) - set(seg_files))
    missing_bony = sorted(set(seg_files) - set(bony_files))
    if missing_seg or missing_bony:
        raise ValueError(
            'Anatomical and bony mask stems must match exactly. '
            f'Missing anatomical masks: {missing_seg}; missing bony masks: {missing_bony}'
        )

    results = []
    for stem in tqdm(sorted(seg_files), desc='Angle calculation', unit='case'):
        seg_map = read_class_map(seg_files[stem], max_class_id=7, encoding=encoding)
        bony_map = read_class_map(bony_files[stem], max_class_id=6, encoding=encoding)
        results.append(
            calculate_case(
                bony_map=bony_map,
                seg_map=seg_map,
                case_name=seg_files[stem].name,
                labrum_class=labrum_class,
                min_labrum_pixels=min_labrum_pixels,
            )
        )
    return results, judge_pose_result(results)


def save_json(results, quality_assessment, args, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'quality_assessment': quality_assessment,
        'mask_encoding': args.encoding,
        'labrum_class': args.labrum_class,
        'min_labrum_pixels': args.min_labrum_pixels,
        'results': results,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(results: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'name', 'status', 'parallel_angle', 'alpha_angle', 'beta_angle',
        'labrum_pixels', 'all_key_points',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                **{key: result[key] for key in fieldnames if key != 'all_key_points'},
                'all_key_points': json.dumps(result['all_key_points']),
            })


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='STEP4: extract keypoints and calculate parallel, alpha, and beta angles.'
    )
    parser.add_argument(
        '--seg-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP3' / 'seg_masks'),
        help='Directory containing anatomical segmentation PNGs.',
    )
    parser.add_argument(
        '--bony-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP3' / 'bony_masks'),
        help='Directory containing bony-substructure segmentation PNGs.',
    )
    parser.add_argument(
        '--output-json',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP4' / 'measurements.json'),
    )
    parser.add_argument(
        '--output-csv',
        default=str(PROJECT_ROOT / 'outputs' / 'STEP4' / 'measurements.csv'),
    )
    parser.add_argument('--encoding', choices=['auto', 'class_id', 'times_30'], default='auto')
    parser.add_argument(
        '--labrum-class',
        type=int,
        default=5,
        help='Anatomical segmentation class used to calculate the labrum centroid.',
    )
    parser.add_argument('--min-labrum-pixels', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if not 1 <= args.labrum_class <= 7:
        raise ValueError(f'--labrum-class must be in 1-7, got {args.labrum_class}')
    if args.min_labrum_pixels < 1:
        raise ValueError('--min-labrum-pixels must be at least 1.')

    seg_dir = Path(args.seg_dir).expanduser().resolve()
    bony_dir = Path(args.bony_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if seg_dir == bony_dir:
        raise ValueError('--seg-dir and --bony-dir must be different directories.')
    if output_json == output_csv:
        raise ValueError('--output-json and --output-csv must be different files.')

    start_time = time.time()
    results, quality_assessment = generate_results(
        seg_dir=seg_dir,
        bony_dir=bony_dir,
        encoding=args.encoding,
        labrum_class=args.labrum_class,
        min_labrum_pixels=args.min_labrum_pixels,
    )
    save_json(results, quality_assessment, args, output_json)
    save_csv(results, output_csv)

    complete = sum(result['status'] == 'complete' for result in results)
    logging.info('Cases: %d; complete measurements: %d', len(results), complete)
    logging.info('Quality assessment: %s', quality_assessment)
    logging.info('JSON: %s', output_json)
    logging.info('CSV: %s', output_csv)
    logging.info('Elapsed time: %.2f seconds', time.time() - start_time)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
