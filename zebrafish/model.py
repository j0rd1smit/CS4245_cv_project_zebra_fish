import os
import numpy as np

from detectron2.engine import DefaultPredictor


def predict(cfg, images, threshold = 0.7):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)

    outputs = []
    for image in images:
        prediction = predictor(image)
        outputs.append(prediction)

    return outputs


def segement_image(image, mask):
    mask = np.dstack([mask, mask, mask])

    return np.where(mask, image, np.zeros_like(image))


def prediction_to_an_image_wide_mask(prediction):
    masks = predictions_to_masks(prediction)

    mask = np.sum(masks, axis=0) >= 1


    return mask


def predictions_to_masks(prediction):
    return prediction["instances"].pred_masks.cpu().numpy()