import random
from os.path import splitext
from os import listdir
from pathlib import Path
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, bony_dir, batch_size, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.bony_dir = bony_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, got {batch_size}')

        for name, directory in (
            ('image', imgs_dir),
            ('anatomical mask', masks_dir),
            ('bony mask', bony_dir),
        ):
            if not Path(directory).is_dir():
                raise FileNotFoundError(f'{name.capitalize()} directory does not exist: {directory}')

        self.ids = sorted(
            splitext(file)[0] for file in listdir(imgs_dir)
            if not file.startswith('.') and file.lower().endswith('.png')
        )
        if not self.ids:
            raise RuntimeError(f'No PNG training images found in: {imgs_dir}')

        missing_masks = [idx for idx in self.ids if not (Path(masks_dir) / f'{idx}.png').is_file()]
        missing_bony = [idx for idx in self.ids if not (Path(bony_dir) / f'{idx}.png').is_file()]
        if missing_masks or missing_bony:
            raise FileNotFoundError(
                'Dataset triplets are incomplete. '
                f'Missing anatomical masks (first 10): {missing_masks[:10]}; '
                f'missing bony masks (first 10): {missing_bony[:10]}'
            )

        random_add_num = len(self.ids) % batch_size
        for _ in range(random_add_num):
            random_integer = random.randint(0, len(self.ids) - 1)
            self.ids.append(self.ids[random_integer])

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_image(cls, pil_img, scale):
        """Preserve upstream [0, 1] inputs; safely scale conventional 8-bit inputs."""
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small'
        if (new_w, new_h) != pil_img.size:
            pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        img_nd = np.asarray(pil_img)
        if img_nd.ndim == 2:
            img_nd = np.repeat(img_nd[..., None], 3, axis=2)
        elif img_nd.ndim == 3 and img_nd.shape[2] > 3:
            img_nd = img_nd[:, :, :3]

        img_nd = img_nd.astype(np.float32, copy=False)
        if img_nd.size and float(img_nd.max()) > 1.0:
            img_nd = img_nd / 255.0

        return np.ascontiguousarray(img_nd.transpose((2, 0, 1)), dtype=np.float32)

    @classmethod
    def preprocess_mask(cls, pil_img, scale):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small'
        if (new_w, new_h) != pil_img.size:
            pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

        mask_nd = np.asarray(pil_img)
        if mask_nd.ndim == 3:
            mask_nd = mask_nd[:, :, 0]
        return np.ascontiguousarray(mask_nd, dtype=np.int64)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.png')
        img_file = glob(self.imgs_dir + idx + '.png')
        bony_file = glob(self.bony_dir + idx + '.png')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(bony_file) == 1, \
            f'Either no heatmap or multiple heatmaps found for the ID {idx}: {bony_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        bony = Image.open(bony_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        assert img.size == bony.size, \
            f'Image and bony mask {idx} should be the same size, but are {img.size} and {bony.size}'

        img = self.preprocess_image(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale)
        bony = self.preprocess_mask(bony, self.scale)

        return {
            'image': torch.from_numpy(img).to(dtype=torch.float32),
            'mask': torch.from_numpy(mask).to(dtype=torch.long),
            'bony': torch.from_numpy(bony).to(dtype=torch.long),
        }
