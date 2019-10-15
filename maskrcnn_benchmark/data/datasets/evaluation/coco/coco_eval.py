import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.config import cfg

import pdb
import numpy as np
import cv2
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms

import json
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    # pdb.set_trace()
    # (Pdb) box_only
    # False
    if box_only:
        pdb.set_trace()
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    
    # pdb.set_trace()
    # (Pdb) iou_types
    # ('bbox',)
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")

        # pdb.set_trace()
        # (Pdb) predictions
        # [BoxList(num_boxes=1143, image_width=2133, image_height=1200, mode=xyxy)]
        # (Pdb) iou_types
        # ('bbox',)

        ############################## mstest ##############################
        if not cfg.TEST.MS_TEST:
            coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
            # pdb.set_trace()
        else:
            coco_results["bbox"] = prepare_for_coco_detection_mstest(predictions, dataset)
            # pdb.set_trace()
        ############################## mstest ##############################

        # pdb.set_trace()
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if 'keypoints' in iou_types:
        logger.info('Preparing keypoints results')
        coco_results['keypoints'] = prepare_for_coco_keypoint(predictions, dataset)

    # pdb.set_trace()
    results = COCOResults(*iou_types)

    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            # pdb.set_trace()
            ##########################  print  ......                                    ##########################
            ##########################  Average Precision  (AP) @[ IoU=0.50:0.95  ...... ##########################
            
            ############################## mstest ##############################
            if not cfg.TEST.MS_TEST:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
                )
            else:
                dataset_small = dataset[0]
                dataset_medium = dataset[1]
                dataset_large = dataset[2]
                # pdb.set_trace()
                res = evaluate_predictions_on_coco(
                    dataset_small.coco, coco_results[iou_type], file_path, iou_type
                )
            ############################## mstest ##############################

            ##########################  Average Precision  (AP) @[ IoU=0.50:0.95  ...... ##########################
            # pdb.set_trace()
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def prepare_for_coco_detection(predictions, dataset):
    # pdb.set_trace()
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        prediction = prediction.resize((image_width, image_height))
        
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
        # pdb.set_trace()
    return coco_results

def prepare_for_coco_detection_mstest(predictions, dataset):

    # pdb.set_trace()

    predictions_s = predictions[0]
    predictions_m = predictions[1]
    predictions_l = predictions[2]

    dataset_s = dataset[0]
    dataset_m = dataset[1]
    dataset_l = dataset[2]

    coco_results = []
    # one image.
    for image_id, predictions in enumerate(zip(predictions_s, predictions_m, predictions_l)):

        prediction_s = predictions[0]
        prediction_m = predictions[1]
        prediction_l = predictions[2]

        original_id = dataset_l.id_to_img_map[image_id]

        if len(predictions_l) == 0:
            continue

        img_info = dataset_l.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        img_id_json = img_info['id']

        # rescale predict bbox to original images size.
        prediction_s = prediction_s.resize((image_width, image_height))
        prediction_m = prediction_m.resize((image_width, image_height))
        prediction_l = prediction_l.resize((image_width, image_height))

        # get single-scale results from type BoxList.
        bbox_s = prediction_s.bbox
        score_s = prediction_s.get_field('scores').unsqueeze(1)
        label_s = prediction_s.get_field('labels').unsqueeze(1)

        bbox_m = prediction_m.bbox
        score_m = prediction_m.get_field('scores').unsqueeze(1)
        label_m = prediction_m.get_field('labels').unsqueeze(1)

        bbox_l = prediction_l.bbox
        score_l = prediction_l.get_field('scores').unsqueeze(1)
        label_l = prediction_l.get_field('labels').unsqueeze(1)

        # concat single-scale result and convert to type BoxList. (small, medium, large)
        min_size = 0 
        w = prediction_l.size[0]
        h = prediction_l.size[1]

        detections = torch.from_numpy(np.row_stack((bbox_s, bbox_m, bbox_l))).cuda()
        per_class = torch.from_numpy(np.row_stack((label_s, label_m, label_l))).cuda()
        per_class = torch.squeeze(per_class, dim=1)
        per_box_cls = torch.from_numpy(np.row_stack((score_s, score_m, score_l))).cuda()
        per_box_cls = torch.squeeze(per_box_cls, dim=1)

        boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
        boxlist.add_field("labels", per_class)
        boxlist.add_field("scores", per_box_cls)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        boxlist = remove_small_boxes(boxlist, min_size)

        # multi-scale results apply NMS. (small, medium, large)
        nms_method = cfg.TEST.MS_TEST_NMS
        nms_thresh = cfg.TEST.MS_TEST_NMS_THR

        num_classes = 81
        scores = boxlist.get_field("scores")
        labels = boxlist.get_field("labels")
        boxes = boxlist.bbox 
        result = []      

        # multi-scale test + NMS
        for j in range(1, num_classes):
            inds = (labels == j).nonzero().view(-1)
            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)

            if nms_method == "nms":
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, nms_thresh,
                    score_field="scores"
                )  
            elif nms_method == "soft_nms":
                boxlist_for_class = boxlist_soft_nms(
                    boxlist_for_class, nms_thresh,
                    score_field="scores"
                )
            else:
                print('the nms method is wrong')

            num_labels = len(boxlist_for_class)

            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j,
                                        dtype=torch.int64,
                                        device=scores.device)
            )

            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        boxlist = result

        boxlist = boxlist.convert("xywh")
        boxes = boxlist.bbox.tolist()
        scores = boxlist.get_field("scores").tolist()
        labels = boxlist.get_field("labels").tolist()

        mapped_labels = [dataset_l.contiguous_category_id_to_json_id[int(i)] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
            
    return coco_results


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'keypoints': keypoint,
            'score': scores[k]} for k, keypoint in enumerate(keypoints)])
    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    # pdb.set_trace()
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    pr_curve = True
    # Plot Percision-Recall Curve.(Use COCOAPI Results)
    # https://zhuanlan.zhihu.com/p/60707912
    if pr_curve:
        import matplotlib.pyplot as plt
        classes = coco_eval.eval['precision'].shape[2]
        AP_thes = coco_eval.eval['precision'].shape[0]
        pr_array_all_class = np.empty(101)
        mAP = np.empty(101)
        mAP_all_class = np.empty(101)

        mAP_small = np.empty(101)
        mAP_small_all_class = np.empty(101)

        mAP_medium = np.empty(101)
        mAP_medium_all_class = np.empty(101)

        mAP_large = np.empty(101)
        mAP_large_all_class = np.empty(101)

        mAP_car = np.empty(101)
        

        # mAP_all_class_small
        pos_classes = 0   # [0,12]
        for i in range(classes):
            # pdb.set_trace()
            # print("****************************************************")
            # print(coco_eval.eval['precision'][0, :, i, 0, 2])
            # AP50
            if coco_eval.eval['precision'][0, :, i, 0, 2][0] != -1:
                pos_classes = pos_classes + 1
                pr_array_all_class = pr_array_all_class + coco_eval.eval['precision'][0, :, i, 0, 2]
                # pdb.set_trace()
            
            # mAP
                for j in range(AP_thes):
                    mAP = mAP + coco_eval.eval['precision'][j, :, i, 0, 2]
                    mAP_small = mAP_small + coco_eval.eval['precision'][j, :, i, 1, 2]
                    mAP_medium = mAP_medium + coco_eval.eval['precision'][j, :, i, 2, 2]
                    mAP_large = mAP_large + coco_eval.eval['precision'][j, :, i, 3, 2]
                    mAP_car = mAP_car + coco_eval.eval['precision'][j, :, 4, 0, 2]

                mAP = mAP / AP_thes
                mAP_small = mAP_small / AP_thes
                mAP_medium = mAP_medium / AP_thes
                mAP_large = mAP_large / AP_thes
                mAP_car = mAP_car / AP_thes

            mAP_all_class = mAP_all_class + mAP 
            # pdb.set_trace()
            mAP_small_all_class = mAP_small_all_class + mAP_small
            mAP_medium_all_class = mAP_medium_all_class + mAP_medium
            mAP_large_all_class = mAP_large_all_class + mAP_large

        AP50_all_class = pr_array_all_class / pos_classes
        mAP_all_class = mAP_all_class / pos_classes
        mAP_small_all_class = mAP_small_all_class / pos_classes
        mAP_medium_all_class = mAP_medium_all_class / pos_classes
        mAP_large_all_class = mAP_large_all_class / pos_classes

        # pdb.set_trace()
        # (Pdb) mAP_all_class[40]
        # 0.5222252067476771
        # (Pdb) AP50_all_class[40]
        # 0.7389909069320636

        # (Pdb) mAP_all_class[50]
        # 0.4360258764442643
        # (Pdb) AP50_all_class[50]
        # 0.6768210069117998

        # pdb.set_trace()
        # pr_array2 = coco_eval.eval['precision'][2, :, 2, 0, 2]
        # pr_array3 = coco_eval.eval['precision'][4, :, 2, 0, 2]
        x = np.arange(0.0, 1.01, 0.01)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)

        # plt.plot(x, AP50_all_class, 'b-', label='IoU=0.5')

        # plt.plot(x, mAP_all_class, 'c-', label='IoU=0.5:0.95')
        # plt.plot(x, mAP_small_all_class, 'b-', label='area = small')
        # plt.plot(x, mAP_medium_all_class, 'c-', label='area = medium')
        # plt.plot(x, mAP_large_all_class, 'y-', label='area = large')

        plt.plot(x, mAP_car, 'b-', label='mAP_car')

        # plt.plot(x, pr_array2, 'c-', label = 'IoU=0.6')
        # plt.plot(x, pr_array3, 'y-', label = 'IoU=0.7')

        plt.legend(loc="lower left")
        # pdb.set_trace()
        plt.savefig('./class_2.jpg')
        plt.show

    compute_thresholds_for_classes(coco_eval)

    return coco_eval





def compute_thresholds_for_classes(coco_eval):
    '''
    The function is used to compute the thresholds corresponding to best f-measure.
    The resulting thresholds are used in fcos_demo.py.
    :param coco_eval:
    :return:
    '''
    import numpy as np
    # dimension of precision: [TxRxKxAxM]
    precision = coco_eval.eval['precision']
    # we compute thresholds with IOU being 0.5
    precision = precision[0, :, :, 0, -1]
    scores = coco_eval.eval['scores']
    scores = scores[0, :, :, 0, -1]

    recall = np.linspace(0, 1, num=precision.shape[0])
    recall = recall[:, None]

    f_measure = (2 * precision * recall) / (np.maximum(precision + recall, 1e-6))
    max_f_measure = f_measure.max(axis=0)
    max_f_measure_inds = f_measure.argmax(axis=0)
    scores = scores[max_f_measure_inds, range(len(max_f_measure_inds))]

    print("Maximum f-measures for classes:")
    print(list(max_f_measure))
    print("Score thresholds for classes (used in demos for visualization purposes):")
    print(list(scores))


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
