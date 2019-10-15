# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

import pdb

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        # pdb.set_trace()
        # (Pdb) type(self.rpn)
        # <class 'maskrcnn_benchmark.modeling.rpn.fcos.fcos.FCOSModule'>
        # (Pdb) self.roi_heads
        # []


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        pdb.set_trace()
        # (Pdb) images.tensors.shape
        # torch.Size([2, 3, 1024, 1792])
        # (Pdb) feature.shape
        # *** NameError: name 'feature' is not defined
        # (Pdb) len(features)
        # 6


        proposals, proposal_losses = self.rpn(images, features, targets)

        #################### train
        # pdb.set_trace()
        # (Pdb) proposals
        # (Pdb) proposal_losses
        # {'loss_cls': tensor(1.1167, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_reg': tensor(5.8484, device='cuda:0', grad_fn=<DivBackward0>), 
        # 'loss_centerness': tensor(0.6951, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)}

        #################### test
        # (Pdb) len(proposals)
        # 2
        # (Pdb) proposals[0]
        # BoxList(num_boxes=1143, image_width=1777, image_height=1000, mode=xyxy)
        # (Pdb) proposals[1]
        # BoxList(num_boxes=1308, image_width=1777, image_height=1000, mode=xyxy)
        # (Pdb) proposal_losses
        # {}
        # (Pdb) proposals[0].fields()
        # ['labels', 'scores']
        # (Pdb) proposals[0].bbox
        # tensor([[484.7408, 630.1026, 543.2703, 663.3972],
        #         [554.7380, 141.0157, 584.8567, 157.2129],
        #         [513.7943, 539.8823, 568.1558, 575.9161],
        #         ...,
        #         [571.4872, 259.4807, 684.8684, 323.5432],
        #         [651.8997, 319.1873, 740.9105, 377.8519],
        #         [572.8201, 258.3826, 678.3829, 323.1006]])

        # (Pdb) self.roi_heads
        # []

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
