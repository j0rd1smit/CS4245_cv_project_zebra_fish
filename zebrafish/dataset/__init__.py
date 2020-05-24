import itertools
import json
import os

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split


def register_datasets(path_to_dataset_dir, validation_size =  0.25, test_size = 0.25, seed=42, clear=True):
    if clear:
        DatasetCatalog.clear()

    dataset_names = ["train", "test"] if validation_size == 0.0 else ["train", "val", "test"]
    dataset_configs = dict(zip(dataset_names, get_dataset_configs(path_to_dataset_dir, validation_size, test_size, seed)))

    for name in dataset_names:
        DatasetCatalog.register(name, lambda: config_to_dataset(path_to_dataset_dir, dataset_configs[name]))
        MetadataCatalog.get(name).set(thing_classes=["zebra_fish"]) #TODO


def get_dataset_configs(path_to_dataset_dir, validation_size =  0.25, test_size = 0.25, seed=42):
    assert test_size > 0
    assert validation_size >= 0
    assert validation_size + test_size < 1

    with open(os.path.join(path_to_dataset_dir, "via_region_data.json")) as f:
        dataset = list(json.load(f).items())

    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=seed)
    if validation_size > 0:
        train_set, validation_set = train_test_split(train_set, test_size=validation_size / (1 - test_size), random_state=seed)

        return dict(train_set), dict(validation_set), dict(test_set)

    return dict(train_set), dict(test_set)


def config_to_dataset(img_dir, config):
    i = 0

    dataset_dicts = []
    for k, image_data in config.items():
        record = dict()

        record["file_name"] = os.path.join(img_dir, image_data["filename"])
        record["image_id"] = i
        i += 1

        h, w = cv2.imread(record["file_name"]).shape[:2]
        record["height"] = h
        record["width"] = w

        annotations = []

        for region in image_data["regions"]:
            shape_attributes = region["shape_attributes"]
            px = shape_attributes["all_points_x"]
            py = shape_attributes["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            annotation = {
                "bbox": [min(px), min(py), max(px), max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,  # There is only 1 catagory
                "iscrowd": 0
            }
            annotations.append(annotation)

        record["annotations"] = annotations

        dataset_dicts.append(record)
    return dataset_dicts



if __name__ == '__main__':
    print(os.getcwd())
    train, val, test = get_dataset_configs("../../dataset")
    print(len(train))
    print(len(val))
    print(len(test))