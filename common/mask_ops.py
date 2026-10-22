"""
Compact instance masks for evaluation (no pycocotools dependency).

A mask is stored as CropMask(x0, y0, crop) : a bool array covering only its bounding box in original-image
pixels. Mask IoU between two CropMasks only touches the overlap of their boxes, so evaluating 1,024 test
images x 100 detections stays in memory (full 608 x 608 masks would need tens of GB).

common.metrics.evaluate_detections is reused unchanged: per image, `boxes` is a numpy object array of
CropMask and the IoU function is mask_iou. Matching, `ignore` handling and AP are identical to detection.
"""

import numpy as np


def semantic_confusion(pred_cls, gt_cls, num_classes, conf=None, void=255):
    """Accumulate a (C + 1) x (C + 1) confusion matrix (index 0 = background); void pixels are skipped."""
    n = num_classes + 1
    if conf is None:
        conf = np.zeros((n, n), np.int64)
    valid = gt_cls != void
    conf += np.bincount(n * gt_cls[valid].astype(np.int64) + pred_cls[valid], minlength=n * n).reshape(n, n)
    return conf


def miou_from_confusion(conf):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - np.diag(conf)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou, float(np.nanmean(iou))
