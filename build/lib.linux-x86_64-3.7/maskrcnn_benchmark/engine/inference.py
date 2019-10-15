# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

import pdb
import numpy as np

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        
        images, targets, image_ids = batch
        # pdb.set_trace()
        # (Pdb) images.image_sizes
        # [torch.Size([1200, 2133]), torch.Size([1200, 2133]), torch.Size([1200, 2133]), torch.Size([1200, 2133])]
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

        # pdb.set_trace()
        # (Pdb) len(image_ids)
        # 4
        # (Pdb) len(output)
        # 4
        # (Pdb)  output[0]
        # BoxList(num_boxes=1143, image_width=2133, image_height=1200, mode=xyxy)
        # (Pdb)  output[0].fields()
        # ['scores', 'labels']
        # (Pdb) output[0].get_field('scores').shape
        # torch.Size([1143])
        # (Pdb) output[0].get_field('labels').shape
        # torch.Size([1143])

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        # pdb.set_trace()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # pdb.set_trace()


    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    
    # pdb.set_trace()
    # (Pdb) type(predictions)
    # <class 'list'>
    # (Pdb) len(predictions)
    # 1
    # (Pdb) predictions[0]
    # BoxList(num_boxes=1143, image_width=2133, image_height=1200, mode=xyxy)
    # (Pdb) predictions[0].get_field('labels').shape
    # torch.Size([1143])
    # (Pdb) predictions[0].get_field('scores').shape
    # torch.Size([1143])

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)




def ms_inference(
        model,
        data_loader_val_mstest,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    # pdb.set_trace()

    data_loader_small = data_loader_val_mstest[0][0]
    data_loader_medium = data_loader_val_mstest[1][0]
    data_loader_large = data_loader_val_mstest[2][0]

    #################################### small ####################################
    dataset_small = data_loader_small.dataset

    logger.info("Start evaluation on {} dataset_small({} images).".format(dataset_name, len(dataset_small)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    predictions_small = compute_on_dataset(model, data_loader_small, device, inference_timer)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset_small), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset_small),
            num_devices,
        )
    )

    predictions_small = _accumulate_predictions_from_multiple_gpus(predictions_small)

    #################################### medium ####################################
    dataset_medium = data_loader_medium.dataset

    logger.info("Start evaluation on {} dataset_medium({} images).".format(dataset_name, len(dataset_medium)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    predictions_medium = compute_on_dataset(model, data_loader_medium, device, inference_timer)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset_medium), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset_medium),
            num_devices,
        )
    )

    predictions_medium = _accumulate_predictions_from_multiple_gpus(predictions_medium)

    #################################### large ####################################
    dataset_large = data_loader_large.dataset

    logger.info("Start evaluation on {} dataset_large({} images).".format(dataset_name, len(dataset_large)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    predictions_large = compute_on_dataset(model, data_loader_large, device, inference_timer)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset_large), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset_large),
            num_devices,
        )
    )

    predictions_large = _accumulate_predictions_from_multiple_gpus(predictions_large)   

    if not is_main_process():
        return

    dataset = [dataset_small, dataset_medium, dataset_large]
    predictions = [predictions_small, predictions_medium, predictions_large]

    # dataset = dataset_large
    # predictions = predictions_large

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)

