import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

import pdb

class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        nms_method
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.nms_method = nms_method

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            
            # pdb.set_trace()
            # (Pdb) detections.shape
            # torch.Size([1000, 4])
            # (Pdb) per_class.shape
            # torch.Size([1000])
            # (Pdb) per_box_cls.shape
            # torch.Size([1000])
            # (Pdb) self.min_size
            # 0

            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)

            # pdb.set_trace()
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        # pdb.set_trace()
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # pdb.set_trace()
            # (Pdb) self.num_classes
            # 81
            # (Pdb) labels.dtype
            # torch.int64
            # (Pdb) scores.dtype
            # torch.float32
            # (Pdb) boxes.dtype
            # torch.float32

            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)

                ############################## softNMS ##############################
                if self.nms_method == "nms":
                    # pdb.set_trace()
                    # (Pdb) boxlist_for_class.bbox.shape
                    # torch.Size([291, 4])
                    # (Pdb) boxlist_for_class.bbox[0]
                    # tensor([1422.0798,  192.1235, 1482.6444,  257.5991], device='cuda:0')
                    # (Pdb) boxlist_for_class.bbox[0].dtype
                    # torch.float32
                    # (Pdb) boxlist_for_class.get_field('scores').shape
                    # torch.Size([291])
                    # (Pdb) boxlist_for_class.get_field('scores')[0]
                    # tensor(0.0988, device='cuda:0')
                    # (Pdb) boxlist_for_class.get_field('scores')[0].dtype
                    # torch.float32
                    # (Pdb) self.nms_thresh
                    # 0.6
                    boxlist_for_class = boxlist_nms(
                        boxlist_for_class, self.nms_thresh,
                        score_field="scores"
                    )
                elif self.nms_method == "soft_nms":
                    boxlist_for_class = boxlist_soft_nms(
                        boxlist_for_class, self.nms_thresh,
                        score_field="scores"
                    )
                else:
                    print('the nms method is wrong')
                ############################## softNMS ##############################

                num_labels = len(boxlist_for_class)

                # pdb.set_trace()
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            # pdb.set_trace()
            # (Pdb) len(result)
            # 80
            # (Pdb) result[0]
            # BoxList(num_boxes=185, image_width=1777, image_height=1000, mode=xyxy)

            result = cat_boxlist(result)
            
            # pdb.set_trace()
            # (Pdb) result
            # BoxList(num_boxes=529, image_width=1777, image_height=1000, mode=xyxy)

            number_of_detections = len(result)

            # pdb.set_trace()

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    # pdb.set_trace()
    # (Pdb) config.MODEL.FCOS.PRE_NMS_TOP_N
    # 1000
    # (Pdb) config.MODEL.FCOS.INFERENCE_TH
    # 0.05
    # (Pdb) config.MODEL.FCOS.NMS_TH
    # 0.6
    # (Pdb) config.TEST.DETECTIONS_PER_IMG
    # 5000

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        nms_method=config.TEST.NMS   
    )

    return box_selector
