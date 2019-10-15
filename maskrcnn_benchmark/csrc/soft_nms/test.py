import numpy as np

from .soft_nms import soft_nms_cpu


def _boxlist_soft_nms(boxlist, nms_thresh, method=1, sigma=0.5, min_score=0.001):
    return soft_nms_cpu(boxlist, nms_thresh, method, sigma, min_score)

