import itertools
import json
import os

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split



DIRECTION_CLASSES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIRECTION_PREFIX = "_direction"
WITHOUT_DIRECTION_PREFIX = "_without_direction"


def load_all_image_in_dataset(name, cfg):
    images = []

    for image_data in get_dataset(name, cfg):
        images.append(cv2.imread(image_data["file_name"]))

    return images


def get_dataset(name, cfg):
    return DatasetCatalog.get(get_name_with_prefix(name, cfg.DATASETS.USE_DIRECTION_CLASSES))


def get_meta_dataset(name, cfg):
    return MetadataCatalog.get(get_name_with_prefix(name, cfg.DATASETS.USE_DIRECTION_CLASSES))


def get_name_with_prefix(name, use_direction_classes):
    return name + DIRECTION_PREFIX if use_direction_classes else name + WITHOUT_DIRECTION_PREFIX


def register_datasets(path_to_dataset_dir, validation_size =  0.1, test_size = 0.1, seed=42, clear=True):
    if clear:
        DatasetCatalog.clear()

    register_datasets_type(path_to_dataset_dir, True, validation_size=validation_size, test_size=test_size, seed=seed, clear=False)
    register_datasets_type(path_to_dataset_dir, False, validation_size=validation_size, test_size=test_size, seed=seed, clear=False)


def register_datasets_type(path_to_dataset_dir, use_direction_classes, validation_size =  0.1, test_size = 0.1, seed=42, clear=True):
    if clear:
        DatasetCatalog.clear()

    dataset_names = ["train", "test"] if validation_size == 0.0 else ["train", "val", "test"]
    configs = get_dataset_configs(path_to_dataset_dir, validation_size, test_size, seed)


    dataset_configs = dict(zip(dataset_names, configs))

    if use_direction_classes:
        thing_classes = list(DIRECTION_CLASSES)
    else:
        thing_classes = ["zebra_fish"]

    def __fetch_data_set(name):
        return lambda: config_to_dataset(path_to_dataset_dir, dataset_configs[name], use_direction_classes)

    for name in dataset_names:
        DatasetCatalog.register(get_name_with_prefix(name, use_direction_classes), __fetch_data_set(name))

        MetadataCatalog.get(get_name_with_prefix(name, use_direction_classes)).set(thing_classes=thing_classes)


def get_dataset_configs(path_to_dataset_dir, validation_size =  0.1, test_size = 0.1, seed=42):
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


def config_to_dataset(img_dir, config, use_direction_classes):
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

            if use_direction_classes:
                assert "class" in region["region_attributes"], "No class found for " + record["file_name"]
                category_id = DIRECTION_CLASSES.index(region["region_attributes"]["class"])
                assert category_id != -1, "Unknown class label: " + region["region_attributes"]["class"]
            else:
                category_id = 0

            annotation = {
                "bbox": [min(px), min(py), max(px), max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,  # There is only 1 catagory
                "iscrowd": 0
            }
            annotations.append(annotation)

        record["annotations"] = annotations

        dataset_dicts.append(record)
    return dataset_dicts






if __name__ == '__main__':
    from zebrafish.configs import get_default_instance_segmentation_config
    base_path = "/home/jordi/Documents/Github/CS4245_cv_project_zebra_fish"
    os.chdir(base_path)
    register_datasets("dataset")

    cfg = get_default_instance_segmentation_config(
        False
    )

    print(len(get_dataset("train", cfg)))
    print(len(get_dataset("val", cfg)))
    print(len(get_dataset("test", cfg)))
    print()

    print(len(DatasetCatalog.get("train_without_direction")))
    print(len(DatasetCatalog.get("val_without_direction")))
    print(len(DatasetCatalog.get("test_without_direction")))

    #help(DatasetCatalog)
    print(DatasetCatalog.list())