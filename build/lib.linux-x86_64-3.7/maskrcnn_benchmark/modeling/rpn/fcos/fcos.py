import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale

import pdb


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        # pdb.set_trace()
        # (Pdb) self.cls_tower
        # Sequential(
        # (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (2): ReLU()
        # (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (5): ReLU()
        # (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (8): ReLU()
        # (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (11): ReLU()
        # )
        # (Pdb) self.bbox_tower
        # Sequential(
        # (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (2): ReLU()
        # (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (5): ReLU()
        # (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (8): ReLU()
        # (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        # (11): ReLU()
        # )
        # (Pdb) self.cls_logits
        # Conv2d(256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (Pdb) self.bbox_pred
        # Conv2d(256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (Pdb) self.centerness
        # Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        if cfg.MODEL.BACKBONE.USE_P2:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(6)])
        else:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # pdb.set_trace()
        for l, feature in enumerate(x):
            # pdb.set_trace()
            
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))

            # pdb.set_trace()
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))

            # pdb.set_trace()
            # (Pdb) feature.shape
            # torch.Size([2, 256, 128, 128])
            # (Pdb) logits[0].shape
            # torch.Size([2, 80, 128, 128])
            # (Pdb) centerness[0].shape
            # torch.Size([2, 1, 128, 128])
            # (Pdb) bbox_reg[0].shape
            # torch.Size([2, 4, 128, 128])
        # pdb.set_trace()
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        if cfg.MODEL.BACKBONE.USE_P2:
            self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES_ADDP2
        else:
            self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # pdb.set_trace()
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
        # pdb.set_trace()
        # (Pdb) len(locations)
        # 5
        # (Pdb) locations[0].shape
        # torch.Size([16384, 2])
        # (Pdb) features[0].shape
        # torch.Size([2, 256, 128, 128])
        # (Pdb) 128*128
        # 16384
        # (Pdb) locations[0]
        # tensor([[   4.,    4.],
        #         [  12.,    4.],
        #         [  20.,    4.],
        #         ...,
        #         [1004., 1020.],
        #         [1012., 1020.],
        #         [1020., 1020.]], device='cuda:0')
        
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        # pdb.set_trace()
        # (Pdb) type(self.loss_evaluator)
        # <class 'maskrcnn_benchmark.modeling.rpn.fcos.loss.FCOSLossComputation'>
        # (Pdb) locations[0]
        # tensor([[   2.,    2.],
        #         [   6.,    2.],
        #         [  10.,    2.],
        #         ...,
        #         [1014., 1022.],
        #         [1018., 1022.],
        #         [1022., 1022.]], device='cuda:0')
        # (Pdb) locations[0].shape
        # torch.Size([65536, 2])
        # (Pdb) box_cls[0].shape
        # torch.Size([1, 80, 256, 256])
        # (Pdb) box_regression[0].shape
        # torch.Size([1, 4, 256, 256])
        # (Pdb) centerness[0].shape
        # torch.Size([1, 1, 256, 256])
        # (Pdb) targets[0].bbox.shape
        # torch.Size([19, 4])

        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            # pdb.set_trace()
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
