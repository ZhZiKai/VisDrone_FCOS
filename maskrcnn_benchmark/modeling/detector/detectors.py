# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

import pdb

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    # pdb.set_trace()
    # (Pdb) meta_arch
    # <class 'maskrcnn_benchmark.modeling.detector.generalized_rcnn.GeneralizedRCNN'>
    return meta_arch(cfg)
