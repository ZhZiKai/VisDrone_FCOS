# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

import numpy as np
# import cv2
# from PIL import Image, ImageDraw

import pdb

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        # pdb.set_trace()

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# -------------------------------- T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train and cfg.INPUT.RANDOM_CROP_FIXED else 0) ----------------------------

class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, target):
        # print('i am doing random crop')
        if self.crop_size <= 0:
            return image, target
        w, h = image.size
        assert target.mode == "xyxy"
        num_original_targets = len(target)
        # print(image)
        # print(target)


        if num_original_targets > 0:
            must_included_box = target.bbox[np.random.randint(len(target))]
            x0, y0 = int(must_included_box[0]), int(must_included_box[1])
            # print(target.bbox.shape)
            # print(must_included_box)
            # print(x0)
            # print(y0)

        else:
            x0 = 0
            y0 = 0

        crop_x0 = np.random.randint(
            max(x0 - self.crop_size // 2, 0),
            x0 + 1
        )
        crop_y0 = np.random.randint(
            max(y0 - self.crop_size // 2, 0),
            y0 + 1
        )

        crop_h = min(h - crop_y0, self.crop_size)
        crop_w = min(w - crop_x0, self.crop_size)
        # print(crop_x0)
        # print(crop_y0)
        # print(crop_h)
        # print(crop_w)
        image = F.crop(image, crop_y0, crop_x0, crop_h, crop_w)
        padding = (0, 0, self.crop_size - crop_w, self.crop_size - crop_h)
        image = F.pad(image, padding=padding)

        target = target.crop([
            crop_x0, crop_y0,
            crop_x0 + self.crop_size - 1,
            crop_y0 + self.crop_size - 1
        ])

        # pdb.set_trace()
        # (Pdb) target.mode
        # 'xyxy'
        target = self.filter_empty_bboxes(target)
        assert num_original_targets == 0 or len(target) > 0

        # print(image)
        # print(target)
        # print(target.bbox.shape)
        # print(target.get_field("labels").shape)
        # pdb.set_trace()
        return image, target

    def filter_empty_bboxes(self, target):
        assert target.mode == "xyxy"
        bbox = target.bbox
        is_empty = (bbox[:, 0] >= bbox[:, 2]) | (bbox[:, 1] >= bbox[:, 3])
        # print(is_empty)
        # pdb.set_trace()
        return target[~is_empty.byte()]

