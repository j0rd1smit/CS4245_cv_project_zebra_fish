import json
import copy
import os

import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


def plot_train_vs_validation_loss(cfg, show=True):
    experiment_metrics = _load_json_arr(cfg.OUTPUT_DIR + '/metrics.json')

    x_train = [x['iteration'] for x in experiment_metrics]
    y_train = [x['total_loss'] for x in experiment_metrics]

    x_val = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    y_val = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]

    plt.plot(x_train, y_train, label="train loss")
    plt.plot(x_val, y_val, label="validation loss")
    plt.grid()
    plt.legend()
    if show:
        plt.show()


def _load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_dataset(dataset_name, n_images = 1):
    metadata = MetadataCatalog.get(dataset_name)

    for image_data in DatasetCatalog.get(dataset_name)[:n_images]:
        img = cv2.imread(image_data["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(image_data)

        plt.figure(figsize=(14, 10))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def plot_prediction(cfg, dataset_name, n_images = None, threshold = 0.7):
    cfg = copy.deepcopy(cfg)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
    cfg.DATASETS.TEST = (dataset_name,)
    predictor = DefaultPredictor(cfg)

    data = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    if n_images is not None and n_images >= 0:
        data = data[:n_images]

    for image_data in data[:n_images]:
        im = cv2.imread(image_data["file_name"])
        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()