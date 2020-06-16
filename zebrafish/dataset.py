import itertools
import json
import os

import cv2
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

FISH_TYPE = "FISH_TYPE"
DIRECTION_TYPE = "DIRECTION_TYPE"
UNLABELED_TYPE = "UNLABELED_TYPE"

TYPE_TO_CLASS_NAME = {
    FISH_TYPE: ["Deformed", "Healthy"],
    DIRECTION_TYPE: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    UNLABELED_TYPE: ["Zebra fish"]
}
TYPE_TO_ANNOTIATION_FILE = {
    FISH_TYPE: "via_region_data_fish_type.json",
    DIRECTION_TYPE: "via_region_data.json",
    UNLABELED_TYPE: "via_region_data.json",
}


def load_all_image_in_dataset(name, cfg):
    images = []

    for image_data in get_dataset(name, cfg):
        images.append(cv2.imread(image_data["file_name"]))

    return images


def get_dataset(name, cfg):
    return DatasetCatalog.get(get_name_with_prefix(name, cfg.DATASETS.TYPE, flip=cfg.DATA_TRANSFORMATIONS.FLIP, rotate=cfg.DATA_TRANSFORMATIONS.ROTATION))


def get_meta_dataset(name, cfg):
    return MetadataCatalog.get(get_name_with_prefix(name, cfg.DATASETS.TYPE, flip=cfg.DATA_TRANSFORMATIONS.FLIP, rotate=cfg.DATA_TRANSFORMATIONS.ROTATION))


def get_name_with_prefix(name, type, flip, rotate):
    assert type in TYPE_TO_CLASS_NAME, f'Unknown dataset type: {type}'
    f = "Flip" if flip else "NoFlip"
    r = "Rotate" if rotate else "NoRotate"
    return f'{name}_{type}_{f}_{r}'


def register_datasets(path_to_dataset_dir, validation_size =  0.1, test_size = 0.1, seed=42, clear=True):
    if clear:
        DatasetCatalog.clear()

    for t in TYPE_TO_CLASS_NAME:
        register_datasets_type(path_to_dataset_dir, t, validation_size=validation_size, test_size=test_size, seed=seed, clear=False)


def register_datasets_type(path_to_dataset_dir, date_set_type, validation_size =  0.1, test_size = 0.1, seed=42, clear=True):
    if clear:
        DatasetCatalog.clear()

    dataset_names = ["train", "test"] if validation_size == 0.0 else ["train", "val", "test"]
    configs = get_dataset_configs(path_to_dataset_dir, date_set_type, validation_size, test_size, seed)

    dataset_configs = dict(zip(dataset_names, configs))

    thing_classes = TYPE_TO_CLASS_NAME[date_set_type]
    class_names = None if date_set_type == UNLABELED_TYPE else TYPE_TO_CLASS_NAME[date_set_type]

    def __fetch_data_set(name, flips: bool, rotate: bool):
        return lambda: config_to_dataset(path_to_dataset_dir, dataset_configs[name], flips, rotate, class_names)

    for name in dataset_names:
        for flip, rotate in [(False, False), (False, True), (True, False), (True, True)]:
            DatasetCatalog.register(get_name_with_prefix(name, date_set_type, flip, rotate), __fetch_data_set(name, flip, rotate))
            MetadataCatalog.get(get_name_with_prefix(name, date_set_type, flip, rotate)).set(thing_classes=thing_classes)


def get_dataset_configs(path_to_dataset_dir, date_set_type, validation_size =  0.1, test_size = 0.1, seed=42):
    assert test_size > 0
    assert validation_size >= 0
    assert validation_size + test_size < 1

    with open(os.path.join(path_to_dataset_dir, TYPE_TO_ANNOTIATION_FILE[date_set_type])) as f:
        dataset = list(json.load(f).items())

    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=seed)

    if validation_size > 0:
        train_set, validation_set = train_test_split(train_set, test_size=validation_size / (1 - test_size), random_state=seed)

        return dict(train_set), dict(validation_set), dict(test_set)

    return dict(train_set), dict(test_set)


def config_to_dataset(img_dir, config, flips: bool, rotate: bool, class_names = None):
    i = 0
    n_options = 1
    if flips:
        n_options *= 2
    if rotate:
        n_options *= 4

    dataset_dicts = []
    for k, image_data in config.items():
        for j in range(n_options):
            record = dict()

            # Set flags for augmentation
            if flips and j % 2 == 0:
                record["flip"] = True
            else:
                record["flip"] = False

            record["rotate"] = round(j / 2) * 90 if rotate else 0

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

                if class_names is not None:
                    assert "class" in region["region_attributes"], "No class found for " + record["file_name"]
                    assert region["region_attributes"][
                               "class"] in class_names, f'Unknown class label: {region["region_attributes"]["class"]} it is not in: {class_names}'
                    category_id = class_names.index(region["region_attributes"]["class"])
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
    random.seed(42)
    random.shuffle(dataset_dicts)
    return dataset_dicts




if __name__ == '__main__':
    from zebrafish.configs import get_default_instance_segmentation_config
    base_path = "/home/jordi/Documents/Github/CS4245_cv_project_zebra_fish"
    os.chdir(base_path)
    register_datasets("dataset")

    cfg = get_default_instance_segmentation_config(FISH_TYPE)
    cfg.DATA_TRANSFORMATIONS.ROTATION = True
    cfg.DATA_TRANSFORMATIONS.FLIP = True

    print(len(get_dataset("train", cfg)))
    print(len(get_dataset("val", cfg)))
    print(len(get_dataset("test", cfg)))
    print()

    cfg = get_default_instance_segmentation_config(UNLABELED_TYPE)

    print(len(get_dataset("train", cfg)))
    print(len(get_dataset("val", cfg)))
    print(len(get_dataset("test", cfg)))
    print()


    #help(DatasetCatalog)
    print(DatasetCatalog.list())