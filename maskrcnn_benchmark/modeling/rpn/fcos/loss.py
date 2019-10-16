"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

import pdb

INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.use_p2 = cfg.MODEL.BACKBONE.USE_P2
        # pdb.set_trace()
        # (Pdb) cfg.MODEL.FCOS.LOSS_GAMMA
        # 2.0
        # (Pdb) cfg.MODEL.FCOS.LOSS_ALPHA
        # 0.25
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        if self.use_p2:
            self.strides = cfg.MODEL.FCOS.FPN_STRIDES_ADDP2
        else:
            self.strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius = cfg.MODEL.FCOS.POS_RADIUS

        self.loc_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask      

    def prepare_targets(self, points, targets):
        if self.use_p2:
            object_sizes_of_interest = [
                [-1, 32], 
                [32, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, INF],
            ]
        else:
            object_sizes_of_interest = [
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, INF],
            ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        # pdb.set_trace()
        # (Pdb) expanded_object_sizes_of_interest[0].shape
        # torch.Size([65536, 2])
        # (Pdb) expanded_object_sizes_of_interest[0]
        # tensor([[-1., 32.],
        #         [-1., 32.],
        #         [-1., 32.],
        #         ...,
        #         [-1., 32.],
        #         [-1., 32.],
        #         [-1., 32.]], device='cuda:0')
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        
        # pdb.set_trace()
        # (Pdb) points_all_level.shape
        # torch.Size([87360, 2])
        # (Pdb) targets[0].bbox.shape
        # torch.Size([5, 4])
        # (Pdb) expanded_object_sizes_of_interest.shape          
        # torch.Size([87360, 2])
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        # pdb.set_trace()
        # (Pdb) labels[0].shape
        # torch.Size([87360])
        # (Pdb) reg_targets[0].shape
        # torch.Size([87360, 4])

        for i in range(len(labels)):
            # (Pdb) num_points_per_level
            # [65536, 16384, 4096, 1024, 256, 64]
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )
        # pdb.set_trace()
        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            # pdb.set_trace()
            # (Pdb) ys[:, None].shape
            # torch.Size([87360, 1])
            # (Pdb) bboxes[:, 1][None].shape
            # torch.Size([1, 53])
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            # pdb.set_trace()
            # (Pdb) t.shape
            # torch.Size([87360, 53])
            # (Pdb) reg_targets_per_im.shape
            # torch.Size([87360, 53, 4])
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # pdb.set_trace()
            # (Pdb) is_in_boxes.shape
            # torch.Size([87360, 15])
            # (Pdb) is_in_boxes[:,0].nonzero()
            # tensor([[   94],
            #         [   95],
            #         [   96],
            #         [  350],
            #         [  351],
            #         [  352],
            #         [  606],
            #         [  607],
            #         [  608],
            #         [  862],
            #         [  863],
            #         [  864],
            #         [ 1118],
            #         [ 1119],
            #         [ 1120],
            #         [65583],
            #         [65711]], device='cuda:0')
            # (Pdb) reg_targets_per_im.shape
            # torch.Size([87360, 15, 4])
            # (Pdb) reg_targets_per_im[94,0,:]
            # tensor([ 0.5350,  2.0000,  8.3900, 17.9111], device='cuda:0')
            # (Pdb) reg_targets_per_im[95,0,:]
            # tensor([ 4.5350,  2.0000,  4.3900, 17.9111], device='cuda:0')
            # (Pdb) locations.shape
            # torch.Size([87360, 2])
            # (Pdb) locations[94]
            # tensor([378.,   2.], device='cuda:0')
            # (Pdb) targets[0].bbox.shape
            # torch.Size([15, 4])
            # (Pdb) targets[0].bbox[0]
            # tensor([377.4650,   0.0000, 386.3900,  19.9111], device='cuda:0')
            
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # pdb.set_trace()
            # (Pdb) reg_targets_per_im.shape
            # torch.Size([87360, 20, 4])
            # (Pdb) max_reg_targets_per_im.shape
            # torch.Size([87360, 20])
            # (Pdb) reg_targets_per_im[0,0,:].max()
            # tensor(664.1428, device='cuda:0')
            # (Pdb) max_reg_targets_per_im[0,0]
            # tensor(664.1428, device='cuda:0')
            
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            # pdb.set_trace()
            # (Pdb) is_cared_in_the_level.shape
            # torch.Size([87360, 20])
            # (Pdb) is_cared_in_the_level[:,0].max()
            # tensor(1, device='cuda:0', dtype=torch.uint8)
            # (Pdb) is_cared_in_the_level[:,0].min()
            # tensor(0, device='cuda:0', dtype=torch.uint8)
            
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # pdb.set_trace()
            # (Pdb) area[None].shape
            # torch.Size([1, 37])
            # (Pdb) locations_to_gt_area.shape
            # torch.Size([87360, 37])

            # pdb.set_trace()
            # (Pdb) is_in_boxes.shape
            # torch.Size([87360, 40])
            # (Pdb) is_cared_in_the_level.shape
            # torch.Size([87360, 40])
            # (Pdb) is_in_boxes.max()
            # tensor(1, device='cuda:0', dtype=torch.uint8)
            # (Pdb) is_cared_in_the_level.max()
            # tensor(1, device='cuda:0', dtype=torch.uint8)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            
            # pdb.set_trace()
            # (Pdb) labels_per_im.shape
            # torch.Size([6])
            # (Pdb) labels_per_im
            # tensor([ 5,  5, 10,  2,  2,  2], device='cuda:0')
            # (Pdb) locations_to_gt_inds.nonzero().shape
            # torch.Size([143, 1])

            # (Pdb) locations_to_gt_inds.shape
            # torch.Size([87360])
            # (Pdb) locations_to_gt_inds
            # tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0')
            # (Pdb) locations_to_gt_inds.max()
            # tensor(47, device='cuda:0')
            # (Pdb) locations_to_gt_inds.min()
            # tensor(0, device='cuda:0')
            
            labels_per_im = labels_per_im[locations_to_gt_inds]  #  locations_to_gt_inds
            # pdb.set_trace()
            # (Pdb) labels_per_im.shape
            # torch.Size([87360])
            # (Pdb) tmp = (labels_per_im[:]==5)
            # (Pdb) tmp.nonzero().shape
            # torch.Size([87265, 1])
            # (Pdb) tmp = (labels_per_im[:]==10)
            # (Pdb) tmp.nonzero().shape
            # torch.Size([21, 1])
            # (Pdb) tmp = (labels_per_im[:]==2)
            # (Pdb) tmp.nonzero().shape
            # torch.Size([74, 1])
            # (Pdb) 87265+21+74
            # 87360

            # pdb.set_trace()
            # (Pdb) labels_per_im.shape
            # torch.Size([87360])
            # (Pdb) locations_to_min_area.shape
            # torch.Size([87360])
            # (Pdb) tmp =locations_to_min_area == INF
            # (Pdb) tmp.nonzero().shape
            # torch.Size([86374, 1])
            labels_per_im[locations_to_min_area == INF] = 0    #  locations_to_min_area
            # pdb.set_trace()
            # (Pdb) labels_per_im.nonzero().shape
            # torch.Size([986, 1])
            # (Pdb) 86374+986
            # 87360
            # (Pdb) labels_per_im.shape
            # torch.Size([87360])
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        
        # pdb.set_trace()
        # (Pdb) targets[0].get_field('labels')
        # tensor([2, 3, 3, 6], device='cuda:0')
        # (Pdb) targets[0].get_field('labels').shape
        # torch.Size([4])
        # (Pdb) targets[0].bbox.shape
        # torch.Size([4, 4])
        # TODO
        labels, reg_targets = self.prepare_targets(locations, targets)
        # pdb.set_trace()
        # (Pdb) len(labels)
        # 6
        # (Pdb) len(reg_targets)
        # 6
        # (Pdb) labels[0].shape
        # torch.Size([65536])
        # (Pdb) labels[0].nonzero().shape      # ?????
        # torch.Size([0, 1])
        # (Pdb) labels[1].nonzero().shape
        # torch.Size([52, 1])
        # (Pdb) labels[2].nonzero().shape
        # torch.Size([46, 1])
        # (Pdb) labels[3].nonzero().shape
        # torch.Size([32, 1])
        # (Pdb) labels[4].nonzero().shape      # ?????
        # torch.Size([0, 1])
        # (Pdb) labels[5].nonzero().shape      # ?????
        # torch.Size([0, 1])

        # (Pdb) reg_targets[0].shape
        # torch.Size([65536, 4])

        # exit()

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            # pdb.set_trace()
            # (Pdb) len(box_cls)
            # 6
            # (Pdb) box_cls[l].shape
            # torch.Size([1, 80, 256, 256])
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            # pdb.set_trace()
            # (Pdb) box_cls_flatten[0].shape
            # torch.Size([65536, 80])
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))

            # pdb.set_trace()
            # (Pdb) labels[l].shape
            # torch.Size([65536])
            labels_flatten.append(labels[l].reshape(-1))
            # pdb.set_trace()
            # (Pdb) labels_flatten[0].shape
            # torch.Size([65536])


            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            # pdb.set_trace()
        # pdb.set_trace()
        # (Pdb) len(box_cls_flatten)
        # 6
        # (Pdb) box_cls_flatten[0].shape
        # torch.Size([65536, 80])
        # (Pdb) box_cls_flatten[1].shape
        # torch.Size([16384, 80])
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        # pdb.set_trace()
        # (Pdb) box_cls_flatten.shape
        # torch.Size([87360, 80])

        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        
        # remove the loss of ignore class. 
        # pos_inds = torch.nonzero(labels_flatten > 1).squeeze(1)

        # pdb.set_trace()
        # (Pdb) box_cls_flatten.shape
        # torch.Size([87360, 80])
        # (Pdb) labels_flatten.int().shape
        # torch.Size([87360])
        # (Pdb) labels_flatten.int().max()
        # tensor(11, device='cuda:0', dtype=torch.int32)
        # (Pdb) labels_flatten.int().min()
        # tensor(0, device='cuda:0', dtype=torch.int32)
        # (Pdb) labels_flatten.int().nonzero().shape
        # torch.Size([64, 1])
        # (Pdb) pos_inds.numel()
        # 64
        # (Pdb) N
        # 1
        #################################### 
        #          TODO (cls_loss)         # 
        ####################################
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero
        # pdb.set_trace()
        # (Pdb) cls_loss
        # tensor(1.4201, device='cuda:0', grad_fn=<DivBackward0>)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        
        ####################################
        # TODO (reg_loss; centerness_loss) #
        ####################################
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    # pdb.set_trace()
    return loss_evaluator
