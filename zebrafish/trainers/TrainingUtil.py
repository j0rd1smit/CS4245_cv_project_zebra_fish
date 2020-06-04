import os
import pickle

from zebrafish.configs import get_new_output_dir_path


def train(cfg, trainer_builder, resume=False):
    if not resume:
        cfg.OUTPUT_DIR = get_new_output_dir_path()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(cfg.PICKLE, "wb") as f:
        pickle.dump(cfg, f)

    trainer = trainer_builder(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()