import os

from zebrafish.configs import get_new_output_dir_path
from zebrafish.trainers.CocoTrainer import CocoTrainer


def train(cfg, resume=False):
    if not resume:
        cfg.OUTPUT_DIR = get_new_output_dir_path()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()