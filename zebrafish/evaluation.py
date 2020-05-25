import os
from collections import defaultdict

import numpy as np
import pandas as pd
from detectron2.evaluation import COCOEvaluator
from matplotlib.path import Path
from detectron2.engine import DefaultPredictor

from zebrafish.configs import load_config
from zebrafish.dataset import get_name_with_prefix, get_dataset, load_all_image_in_dataset
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

def coco_evaluation_all_model(dataset_name, models_path = "output"):
    results = []
    paths = get_all_stored_model_paths(models_path=models_path)
    models = []

    for path in paths:
        if os.path.isfile(os.path.join(path, "model_final.pth")):
            models.append(path)

    for path in models:
        cfg = load_config(path)
        results.append(coco_evaluation(cfg, dataset_name))

    bbox = defaultdict(list)
    segm = defaultdict(list)

    for result, model in zip(results, models):
        bbox["model"] = model
        for k, v in result["bbox"].items():
            bbox[k].append(v)

        segm["model"] = model
        for k, v in result["segm"].items():
            segm[k].append(v)

    bbox = {k: v for k,v in bbox.items() if len(v) == len(models) or k == "model"}
    segm = {k: v for k, v in segm.items() if len(v) == len(models) or k == "model"}

    bbox_df = pd.DataFrame.from_dict(bbox)
    segm_df = pd.DataFrame.from_dict(segm)

    return bbox_df, segm_df