import os
import pickle

from detectron2.config import get_cfg
import datetime

from detectron2.model_zoo import model_zoo

from zebrafish.dataset import DIRECTION_CLASSES, get_name_with_prefix


def load_config(path_to_model):
    path_to_model = os.path.abspath(path_to_model)
    pickle_path = os.path.join(path_to_model, "pickle")

    with open(pickle_path, "rb") as f:
        cfg = pickle.load(f)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        return cfg

def get_new_output_dir_path():
    return f"output/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


def get_default_instance_segmentation_config(use_direction_classes, training_set_name="train", test_set_name="val", max_iter = 1000, threshold=0.7):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # only segmentation and bounding boxes

    # General
    cfg.SEED = 42
    cfg.OUTPUT_DIR = os.path.abspath(get_new_output_dir_path())
    cfg.PICKLE = os.path.join(cfg.OUTPUT_DIR, "pickle")



    # Data set
    cfg.DATASETS.TRAIN = (get_name_with_prefix(training_set_name, use_direction_classes),)
    cfg.DATASETS.TEST = (get_name_with_prefix(test_set_name, use_direction_classes),)  # TODO can I add test set also here?
    cfg.DATASETS.USE_DIRECTION_CLASSES = use_direction_classes

    cfg.DATALOADER.NUM_WORKERS = 2

    # validation
    cfg.TEST.EVAL_PERIOD = 60

    # Solver
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = int(0.15 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / (cfg.SOLVER.WARMUP_ITERS + 1)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 if not use_direction_classes else len(DIRECTION_CLASSES)
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold


    return cfg


def update_cfg_to_load_model(path_to_model_folder, cfg):
    cfg.OUTPUT_DIR = os.path.abspath(path_to_model_folder)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    return cfg
