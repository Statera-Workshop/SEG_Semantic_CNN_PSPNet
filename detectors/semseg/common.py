"""Shared pieces of the semantic segmentation models (tasks/semseg_task.py contract)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255


class SemStats:
    """Stands in for a YOLO layer in model.yolo_layers so common.trainer logs the losses and pixel accuracy."""

    def __init__(self):
        self.metrics = {}

    @torch.no_grad()
    def update(self, losses, logits, target):
        valid = target != IGNORE
        prob = F.softmax(logits.float(), 1)
        m = {k: float(v.detach()) for k, v in losses.items()}
        if valid.any():
            m["conf_obj"] = float(prob.max(1).values[valid].mean())        # mean confidence on labelled pixels
            m["cls_acc"] = float((logits.argmax(1)[valid] == target[valid]).float().mean() * 100)   # pixel acc
        self.metrics = m


class SemSegBase(nn.Module):
    """Subclasses implement `logits(x) -> (main [B, K, h, w], aux [B, K, h', w'] or None)` and set `aux_weight`."""

    aux_weight = 0.0

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes            # object classes; K = num_classes + 1
        self.K = num_classes + 1
        self.stats = SemStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        main, aux = self.logits(x)
        if targets is None:
            return main
        t = targets["sem"]
        up = F.interpolate(main, t.shape[-2:], mode="bilinear", align_corners=False)
        loss_main = F.cross_entropy(up, t, ignore_index=IGNORE)
        losses = {"loss_ce": loss_main}
        total = loss_main
        if aux is not None and self.aux_weight:
            loss_aux = F.cross_entropy(F.interpolate(aux, t.shape[-2:], mode="bilinear", align_corners=False), t,
                                       ignore_index=IGNORE)
            losses["loss_aux"] = loss_aux
            total = total + self.aux_weight * loss_aux
        losses["losses"] = total
        self.stats.update(losses, up.detach(), t)
        return total, main

    def load_darknet_weights(self, path):
        raise NotImplementedError("no Darknet weights; train from scratch.")
