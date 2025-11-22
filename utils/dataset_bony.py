import random
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import matplotlib.pyplot as plt

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, bony_dir, batch_size, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.bony_dir = bony_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        random_add_num = len(self.ids) % batch_size
        for i in range(random_add_num):
            random_integer = random.randint(0, len(self.ids) - 1)
            self.ids.append(self.ids[random_integer])

        #print('id:', self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mark = 'mask'):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        #print(pil_img.size)
        img_nd = np.array(pil_img)
        #print(img_nd.shape)

        if len(img_nd.shape) == 2 :
            img_nd = np.expand_dims(img_nd, axis=2)
            img_nd = np.repeat(img_nd, 3, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if mark == 'img':
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        #print('index:', idx)
        mask_file = glob(self.masks_dir + idx + '.png')
        img_file = glob(self.imgs_dir + idx + '.png')
        bony_file = glob(self.bony_dir + idx + '.png')

        #print('mask_file:', mask_file)
        #print('img_file:', img_file)
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

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        bony = self.preprocess(bony, self.scale)

        return {'image': torch.from_numpy(img).type(torch.ByteTensor), \
                'mask': torch.from_numpy(mask[0]).type(torch.ByteTensor), \
                'bony': torch.from_numpy(bony[0]).type(torch.ByteTensor)}



