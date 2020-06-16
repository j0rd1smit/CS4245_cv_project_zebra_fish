import os
import pickle

from zebrafish.dataset import get_name_with_prefix


def train(cfg, trainer_builder, resume=False):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATASETS.TRAIN = (get_name_with_prefix("train", cfg.DATASETS.TYPE, flip=cfg.DATA_TRANSFORMATIONS.FLIP, rotate=cfg.DATA_TRANSFORMATIONS.ROTATION),)
    cfg.DATASETS.TEST = (get_name_with_prefix("test", cfg.DATASETS.TYPE, flip=cfg.DATA_TRANSFORMATIONS.FLIP, rotate=cfg.DATA_TRANSFORMATIONS.ROTATION),)

    with open(cfg.PICKLE, "wb") as f:
        pickle.dump(cfg, f)

    trainer = trainer_builder(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()