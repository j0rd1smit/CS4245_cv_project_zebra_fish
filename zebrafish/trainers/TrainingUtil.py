import os
import pickle



def train(cfg, trainer_builder, resume=False):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(cfg.PICKLE, "wb") as f:
        pickle.dump(cfg, f)

    trainer = trainer_builder(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()