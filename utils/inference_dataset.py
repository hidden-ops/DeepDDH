from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class InferenceDataset(Dataset):
    """Load an unlabeled image directory for DeepDDH inference."""

    def __init__(self, image_dir: str, scale: float = 1.0, normalize: bool = True):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.scale = scale
        self.normalize = normalize

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f'Input image directory does not exist: {self.image_dir}')
        if not 0 < scale <= 1:
            raise ValueError(f'scale must be in (0, 1], got {scale}')

        self.files: List[Path] = sorted(
            (
                path for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower()
        )
        if not self.files:
            supported = ', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise RuntimeError(
                f'No supported images found in {self.image_dir}. '
                f'Supported extensions: {supported}'
            )

        stems = [path.stem for path in self.files]
        duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
        if duplicates:
            raise ValueError(
                'Input filenames must have unique stems because outputs are PNG files. '
                f'Duplicate stems: {duplicates}'
            )

    def __len__(self) -> int:
        return len(self.files)

    def scaled_sizes(self) -> List[Tuple[int, int]]:
        sizes = []
        for path in self.files:
            with Image.open(path) as image:
                width, height = image.size
            sizes.append((int(width * self.scale), int(height * self.scale)))
        return sizes

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            original_width, original_height = image.size
            scaled_width = int(original_width * self.scale)
            scaled_height = int(original_height * self.scale)
            if scaled_width <= 0 or scaled_height <= 0:
                raise ValueError(f'scale={self.scale} is too small for {path.name}')
            if (scaled_width, scaled_height) != image.size:
                image = image.resize((scaled_width, scaled_height), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)

        if self.normalize and array.size and float(array.max()) > 1.0:
            array = array / 255.0
        array = np.ascontiguousarray(array.transpose(2, 0, 1))

        return {
            'image': torch.from_numpy(array),
            'id': path.stem,
            'original_width': original_width,
            'original_height': original_height,
        }
