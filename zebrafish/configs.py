import os
import pickle
import re

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.model_zoo import model_zoo

from zebrafish.dataset import TYPE_TO_CLASS_NAME, UNLABELED_TYPE, get_name_with_prefix


def load_config(path_to_model):
    path_to_model = os.path.abspath(path_to_model)
    pickle_path = os.path.join(path_to_model, "pickle")

    with open(pickle_path, "rb") as f:
        cfg = pickle.load(f)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        return cfg


def update_output_dir_path(cfg):
    data_transformations = f"DA_RF{cfg.DATA_TRANSFORMATIONS.RESIZE_FACTOR}"
    if cfg.DATA_TRANSFORMATIONS.FLIP:
        data_transformations += "_FL"
    if cfg.DATA_TRANSFORMATIONS.ROTATION:
        data_transformations += "_RO"

    model = f"BS{cfg.SOLVER.IMS_PER_BATCH}_FA{cfg.MODEL.BACKBONE.FREEZE_AT}_WUI{cfg.SOLVER.WARMUP_ITERS}_RIOBS_{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}"

    test_aug = "TAUG" if cfg.TEST.AUG.ENABLED else "NTAUG"

    path = f"output/{cfg.TYPE}_{model}_{data_transformations}_{test_aug}_{cfg.DATASETS.TRAIN[0]}"
    print(path)

    cfg.OUTPUT_DIR = os.path.abspath(path)
    cfg.PICKLE = os.path.join(cfg.OUTPUT_DIR, "pickle")


def get_default_instance_segmentation_config(dataset_type, training_set_name="train", test_set_name="val", max_iter = 1000, threshold=0.7):
    cfg = get_cfg()


    cfg.DATA_TRANSFORMATIONS = CN()
    cfg.DATA_TRANSFORMATIONS.ROTATION = True
    cfg.DATA_TRANSFORMATIONS.FLIP = True
    cfg.DATA_TRANSFORMATIONS.RESIZE_FACTOR = 1

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # only segmentation and bounding boxes
    cfg.TYPE = "mask_rcnn_R_50_FPN_3x"
    assert cfg.TYPE in cfg.MODEL.WEIGHTS, "miss match between type and weights"

    # General
    cfg.SEED = 42



    # Data set
    cfg.DATASETS.TRAIN = (get_name_with_prefix(training_set_name, dataset_type),)
    cfg.DATASETS.TEST = (get_name_with_prefix(test_set_name, dataset_type),)
    cfg.DATASETS.TYPE = dataset_type

    cfg.DATALOADER.NUM_WORKERS = 0

    # validation
    cfg.TEST.EVAL_PERIOD = 60

    # Solver
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = int(0.5 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / (cfg.SOLVER.WARMUP_ITERS + 1)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(TYPE_TO_CLASS_NAME[dataset_type])
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    update_output_dir_path(cfg)

    return cfg

def get_rs_101_instance_segmentation_config(dataset_type, training_set_name="train", test_set_name="val", max_iter = 1000, threshold=0.7):
    cfg = get_cfg()
    cfg.DATA_TRANSFORMATIONS = CN()
    cfg.DATA_TRANSFORMATIONS.ROTATION = True
    cfg.DATA_TRANSFORMATIONS.FLIP = True
    cfg.DATA_TRANSFORMATIONS.RESIZE_FACTOR = 1

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # only segmentation and bounding boxes
    cfg.TYPE = "mask_rcnn_R_101_FPN_3x"
    assert cfg.TYPE in cfg.MODEL.WEIGHTS, "miss match between type and weights"

    # General
    cfg.SEED = 42



    # Data set
    cfg.DATASETS.TRAIN = (get_name_with_prefix(training_set_name, dataset_type),)
    cfg.DATASETS.TEST = (get_name_with_prefix(test_set_name, dataset_type),)
    cfg.DATASETS.TYPE = dataset_type

    cfg.DATALOADER.NUM_WORKERS = 2

    # validation
    cfg.TEST.EVAL_PERIOD = 60

    # Solver
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / (cfg.SOLVER.WARMUP_ITERS + 1)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(TYPE_TO_CLASS_NAME[dataset_type])
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    update_output_dir_path(cfg)

    return cfg


def update_cfg_to_load_model(path_to_model_folder, cfg):
    cfg.OUTPUT_DIR = os.path.abspath(path_to_model_folder)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    return cfg
