# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

import pdb

def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    # pdb.set_trace()
    # (Pdb) min_size
    # 1500
    # (Pdb) max_size
    # 2500
    # (Pdb) cfg.INPUT.CROP_SIZE
    # 800
    # (Pdb) is_train
    # False
    # (Pdb) flip_prob
    # 0
 
    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train and cfg.INPUT.RANDOM_CROP_FIXED else 0),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def build_transforms_mstest(cfg, is_train=False):
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.TEST.MIN_SIZE_RANGE_TEST
        min_size_small = min_size[0]
        min_size_medium = min_size[1]
        min_size_large = min_size[2]
        # pdb.set_trace()
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    # pdb.set_trace()
    # (Pdb) min_size_large
    # 1500
    # (Pdb) max_size
    # 2500
    # (Pdb) is_train
    # False
    # (Pdb) cfg.INPUT.RANDOM_CROP_FIXED
    # True
    # (Pdb) cfg.INPUT.CROP_SIZE
    # 800
    # (Pdb) flip_prob
    # 0

    #################################### small ####################################
    transform_small = T.Compose(
        [
            T.Resize(min_size_small, max_size),
            T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train and cfg.INPUT.RANDOM_CROP_FIXED else 0),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    #################################### medium ####################################
    transform_medium = T.Compose(
        [
            T.Resize(min_size_medium, max_size),
            T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train and cfg.INPUT.RANDOM_CROP_FIXED else 0),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    #################################### large ####################################
    transform_large = T.Compose(
        [
            T.Resize(min_size_large, max_size),
            T.RandomCrop(cfg.INPUT.CROP_SIZE if is_train and cfg.INPUT.RANDOM_CROP_FIXED else 0),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    # pdb.set_trace()
    return [transform_small, transform_medium, transform_large]