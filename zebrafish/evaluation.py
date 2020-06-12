import copy
import os
from collections import defaultdict
from typing import Any, Callable, List, NamedTuple

import numpy as np
import pandas as pd
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.structures.boxes import matched_boxlist_iou
from matplotlib.path import Path

from zebrafish.configs import load_config
from zebrafish.dataset import get_dataset, get_name_with_prefix, load_all_image_in_dataset
from zebrafish.utils import get_all_stored_model_paths


def image_data_to_mask(image_data):
    masks = []
    width = image_data["width"]
    height = image_data["height"]

    for annotation in image_data["annotations"]:
        masks.append(polygons_to_mask(annotation["segmentation"], width, height))

    return masks


def polygons_to_mask(polygons, width, height):
    mask = np.zeros([height, width], dtype=bool)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    for polygon in polygons:

        path = Path(np.array(polygon).reshape(-1, 2))
        mask_polymask = path.contains_points(points)
        mask_polymask = mask_polymask.reshape((height, width))

        mask = mask + mask_polymask

    return mask


def IoU_score(predicted_mask, true_mask):
    overlap = predicted_mask * true_mask
    union = predicted_mask + true_mask

    return overlap.sum() / float(union.sum())


def coco_evaluation(cfg, dataset_name):
    predictor = DefaultPredictor(cfg)

    dataset_name_with_prefix = get_name_with_prefix(dataset_name, cfg.DATASETS.USE_DIRECTION_CLASSES)
    evaluator = COCOEvaluator(dataset_name_with_prefix, cfg, distributed=False, output_dir=cfg.OUTPUT_DIR)

    predictions = [predictor(img) for img in load_all_image_in_dataset(dataset_name, cfg)]

    dataset = get_dataset(dataset_name, cfg)

    evaluator.reset()
    evaluator.process(dataset, predictions)

    return evaluator.evaluate()


def coco_evaluation_all_model(dataset_name, models_path="output"):
    results = []
    paths = get_all_stored_model_paths(models_path=models_path)
    models = []
    used_directions = []
    max_iterations = []
    lrs = []

    for path in paths:
        if os.path.isfile(os.path.join(path, "model_final.pth")):
            models.append(path)

    for path in models:
        cfg = load_config(path)
        results.append(coco_evaluation(cfg, dataset_name))
        used_directions.append(cfg.DATASETS.USE_DIRECTION_CLASSES)
        max_iterations.append(cfg.SOLVER.MAX_ITER)
        lrs.append(cfg.SOLVER.BASE_LR)

    bbox = defaultdict(list)
    bbox["model"] = models
    bbox["used_direction"] = used_directions
    bbox["max_iteration"] = max_iterations
    bbox["lrs"] = lrs

    segm = defaultdict(list)
    segm["model"] = models
    segm["used_direction"] = used_directions
    segm["max_iteration"] = max_iterations
    segm["lrs"] = lrs

    for result, model in zip(results, models):
        for k, v in result["bbox"].items():
            bbox[k].append(v)

        for k, v in result["segm"].items():
            segm[k].append(v)

    bbox = {k: v for k, v in bbox.items() if len(v) == len(models) or k == "model"}
    segm = {k: v for k, v in segm.items() if len(v) == len(models) or k == "model"}

    bbox_df = pd.DataFrame.from_dict(bbox)
    segm_df = pd.DataFrame.from_dict(segm)

    return bbox_df, segm_df


Match = NamedTuple('Match', [('gt_index', int), ('pred_index', int), ("IoU", float), ("gt_class", int), ("pred_class", int)])


def evaluate_confusion_on_dataset(cfg, dataset_name, threshold_score=0.5, IoU_threshold=0.5):
    predictor = DefaultPredictor(cfg)
    dataset = get_dataset(dataset_name, cfg)

    all_y_pred = []
    all_y_true = []
    for image_info in dataset:
        gt_item = _mapper(image_info)
        gt_instances = gt_item["instances"]
        prediction = predictor(gt_item["image"])
        pred_instances = prediction["instances"]

        y_pred, y_true = evaluate_confusion(gt_instances, pred_instances, threshold_score, IoU_threshold)

        all_y_pred += y_pred
        all_y_true += y_true

    return all_y_pred, all_y_true


def evaluate_confusion(gt_instances, pred_instances, threshold_score, IoU_threshold):
    n_gt_instances = len(gt_instances)
    n_pred_instances = len(pred_instances)

    matches = []

    for i in range(n_gt_instances):
        gt_instance = gt_instances[i]
        gt_box = gt_instance.gt_boxes
        gt_box.tensor = gt_box.tensor.cuda()

        gt_class = gt_instance.gt_classes.cpu().numpy()[0]
        for j in range(n_pred_instances):
            pred_instance = pred_instances[j]
            pred_box = pred_instance.pred_boxes
            pred_score = pred_instance.scores[0].cpu().numpy()
            IoU = float(matched_boxlist_iou(gt_box, pred_box)[0].cpu().numpy())
            pred_class = pred_instance.pred_classes.cpu().numpy()[0]

            if pred_score >= threshold_score and IoU >= IoU_threshold:
                matches.append(Match(gt_index=i, pred_index=j, IoU=IoU, gt_class=gt_class, pred_class=pred_class))

    matches = _filter_matches_by_key(matches, lambda m: m.gt_index)
    matches = _filter_matches_by_key(matches, lambda m: m.pred_index)

    matched_gts = list(map(lambda x: x.gt_index, matches))
    matched_preds = list(map(lambda x: x.pred_index, matches))

    unmatched_gts = list(filter(lambda x: x not in matched_gts, range(n_gt_instances)))
    unmatched_pred = list(filter(lambda x: x not in matched_preds, range(n_pred_instances)))

    y_pred = []
    y_true = []

    for match in matches:
        y_pred.append(match.pred_class)
        y_true.append(match.gt_class)

    for gt_idx in unmatched_gts:
        gt_instance = gt_instances[gt_idx]
        gt_class = gt_instance.gt_classes.cpu().numpy()[0]
        y_pred.append(-1)
        y_true.append(gt_class)

    for gt_idx in unmatched_pred:
        pred_instance = pred_instances[gt_idx]
        pred_class = pred_instance.pred_classes.cpu().numpy()[0]
        y_pred.append(pred_class)
        y_true.append(-1)

    return y_pred, y_true


def _filter_matches_by_key(matches: List[Match], key: Callable[[Match], Any]):
    matches_per_key = dict()
    for match in matches:
        k = key(match)
        if k not in matches_per_key or match.IoU > matches_per_key[k].IoU:
            matches_per_key[k] = match

    return list(matches_per_key.values())


def _mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    dataset_dict["image"] = image

    instances = utils.annotations_to_instances(dataset_dict["annotations"], image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
