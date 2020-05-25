import json
import os

import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import colors as mcolors



from zebrafish.dataset import get_meta_dataset, get_name_with_prefix
from zebrafish.model import prediction_to_an_image_wide_mask, segement_image
from zebrafish.utils import get_all_stored_model_paths


def plot_all_model_validation_loss(models_path = "output", show_training_loss= False, show=True):
    models = get_all_stored_model_paths(models_path)


    plot_train_vs_validation_loss(models, show_training_loss=show_training_loss, show=show)


def plot_train_vs_validation_loss(output_dirs, labels = None, show_training_loss= False, show=True):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    assert len(output_dirs) <= len(colors)

    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in output_dirs]

    assert len(labels) == len(output_dirs)

    for c, output_dir, label in zip(colors, output_dirs, labels):
        experiment_metrics = _load_json_arr(output_dir + '/metrics.json')

        x_train = [x['iteration'] for x in experiment_metrics]
        y_train = [x['total_loss'] for x in experiment_metrics]

        x_val = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x ]
        y_val = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]

        if show_training_loss:
            plt.plot(x_train, y_train, "--", c=c)
        plt.plot(x_val, y_val, "-", label=label, c=c)

    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.legend()
    if show:
        plt.show()


def _load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_dataset(dataset_name, cfg, n_images = 1):
    dataset_name_with_prefix = get_name_with_prefix(dataset_name, cfg.DATASETS.USE_DIRECTION_CLASSES)
    metadata = MetadataCatalog.get(dataset_name_with_prefix)

    for image_data in DatasetCatalog.get(dataset_name)[:n_images]:
        img = cv2.imread(image_data["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(image_data)

        plt.figure(figsize=(14, 10))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.title(image_data["file_name"])
        plt.show()


def plot_prediction(image, prediction, cfg):
    meta_data = get_meta_dataset("train", cfg)

    v = Visualizer(image[:, :, ::-1],
                   metadata=meta_data,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(prediction["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()


def plot_segementation(image, prediction):

    mask = prediction_to_an_image_wide_mask(prediction)
    masked_image = segement_image(image, mask)

    plt.figure(figsize=(14, 10))
    plt.imshow(masked_image)
    plt.show()


def plot_segementation_vs_real(image, prediction):

    mask = prediction_to_an_image_wide_mask(prediction)
    masked_image = segement_image(image, mask)

    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 10))

    ax0.imshow(image)
    ax1.imshow(masked_image)
    plt.show()



