"""
PASCAL VOC2007 dataset for the image track.

Output contract (same as the legacy ImageYoloDataset):
    image   : FloatTensor [3, S, S], RGB, letterboxed, 0..1
    targets : FloatTensor [N, 6] = [batch_index, class, x, y, w, h] normalized

Training excludes "difficult" objects (VOC convention). Evaluation keeps them
as `ignore` ground truths.

[Unified] Training augmentation (profile v1.1, `aug`): the basic augmentation of the original
YOLO papers, shared by every version -
    YOLOv1 paper : random scaling and translation of up to 20 % of the image size,
                   random exposure and saturation of up to a factor 1.5 in HSV
    YOLOv2 / v3  : Darknet cfg hue = .1, saturation = 1.5, exposure = 1.5
plus the common horizontal flip. Version-specific strong augmentation (Mosaic, MixUp, ...) is
not used. Boxes follow the same affine transform; boxes left with less than 25 % of their area
(or narrower than 2 px) after clipping are dropped.
"""


import cv2
import numpy as np
import torch
from PIL import Image


def letterbox_params(w, h, size):
    scale = min(size / w, size / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    return scale, (size - rw) // 2, (size - rh) // 2, rw, rh


def letterbox(image, size):
    """PIL RGB -> (letterboxed PIL, scale, pad_x, pad_y). Gray (128) padding, legacy."""
    w, h = image.size
    scale, px, py, rw, rh = letterbox_params(w, h, size)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(image.resize((rw, rh), Image.BILINEAR), (px, py))
    return canvas, scale, px, py


def _rand_scale(s):
    """Darknet rand_scale: a factor in [1, s], inverted with probability 0.5."""
    f = np.random.uniform(1.0, s)
    return f if np.random.random() < 0.5 else 1.0 / f


def hsv_jitter(arr, hue, saturation, exposure):
    """uint8 RGB -> uint8 RGB with random hue shift and saturation / exposure scaling."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-hue, hue) * 180.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * _rand_scale(saturation), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * _rand_scale(exposure), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def to_tensor(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
