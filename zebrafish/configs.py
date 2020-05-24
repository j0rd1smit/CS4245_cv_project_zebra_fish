from detectron2.config import get_cfg
import datetime

from detectron2.model_zoo import model_zoo

from zebrafish.dataset import DIRECTION_CLASSES


def get_default_config(cfg, training_set_name, validation_set_name, use_direction_classes, max_iter = 1000):
    # General
    cfg.SEED = 42
    cfg.OUTPUT_DIR = get_new_output_dir_path()

    # Data set
    cfg.DATASETS.TRAIN = (training_set_name,)
    cfg.DATASETS.TEST = (validation_set_name,) #TODO can I add test set also here?

    # validation
    cfg.TEST.EVAL_PERIOD = 20


    # Solver
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.IMS_PER_BATCH = 5
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = int(0.5 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    #TODO cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 if not use_direction_classes else len(DIRECTION_CLASSES)
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    return cfg


def get_new_output_dir_path():
    return f"output/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"


def get_instance_segmentation_config(training_set_name, validation_set_name, max_iter = 1000):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # only segmentation and bounding boxes

    cfg = get_default_config(cfg, training_set_name, validation_set_name, max_iter,)


    return cfg

