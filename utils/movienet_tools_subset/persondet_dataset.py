import os.path as osp
import numpy as np

import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .persondet_formatting import Collect, ImageToTensor, OneSampleCollate
from .persondet_transforms import Normalize, Pad, Resize


class DataProcessor(object):
    """image preprocess pipeline."""

    def __init__(self, gpu, img_scale=(1333, 800)):
        self.pipeline = Compose([
            Resize(img_scale, True),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            Pad(size_divisor=32),
            ImageToTensor(['img']),
            Collect(
                keys=['img'],
                meta_keys=('ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg')),
            OneSampleCollate(device=gpu),
        ])

    def __call__(self, img):
        """process an image.

        Args:
            img (np.array<uint8>): the input image, in BGR
        """
        img_info = {}
        img_info['img'] = img
        img_info['img_shape'] = img.shape
        img_info['ori_shape'] = img.shape
        img_info['flip'] = False
        img_info['bbox_fields'] = []
        return self.pipeline(img_info)


class CustomDataset(Dataset):
    """Custom dataset for detection."""

    def __init__(self, img_list, img_scale=(1333, 800), img_prefix=''):
        if isinstance(img_list, list):
            self.img_list = img_list
        elif isinstance(img_list, str):
            assert osp.isfile(img_list)
            self.img_list = [x.strip() for x in open(img_list)]
        else:
            raise ValueError(
                'param "img_list" must be list or str, now it is {}'.format(
                    type(img_list)))
        self.img_prefix = img_prefix
        self.pipeline = Compose([
            Resize(img_scale, True),
            Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            Pad(size_divisor=32),
            ImageToTensor(['img']),
            Collect(keys=['img'])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.prepare_test_img(idx)

    def prepare_test_img(self, idx):
        this_img = self.img_list[idx]
        if isinstance(this_img, str):
            filename = osp.join(self.img_prefix, this_img)
            img = mmcv.imread(filename)
        elif isinstance(this_img, np.ndarray):
            filename = ''
            img = this_img

        img_info = {}
        img_info['filename'] = filename
        img_info['img_prefix'] = self.img_prefix
        img_info['img'] = img
        img_info['img_shape'] = img.shape
        img_info['ori_shape'] = img.shape
        img_info['flip'] = False
        img_info['bbox_fields'] = []
        return self.pipeline(img_info)
