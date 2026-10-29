"""Standard and class-weighted cross-entropy for semantic segmentation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional weights and safe handling of unlabeled batches."""

    def __init__(self, weight=None, ignore_index=4):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        labeled = target != self.ignore_index
        if not labeled.any():
            return logits.sum() * 0
        if self.weight is not None and not (self.weight[target[labeled]] > 0).any():
            return logits.sum() * 0
        return F.cross_entropy(
            logits, target, weight=self.weight, ignore_index=self.ignore_index
        )


def count_training_classes(label_paths, num_classes, ignore_index=4):
    """Count labeled pixels only in training rasters, streaming GeoTIFF blocks."""
    import rasterio

    counts = np.zeros(num_classes, dtype=np.int64)
    for path in label_paths:
        with rasterio.open(path) as raster:
            for _, window in raster.block_windows(1):
                block = raster.read(1, window=window, masked=True)
                pixels = np.asarray(block.data)
                valid = ~np.ma.getmaskarray(block)
                valid &= np.isfinite(pixels)
                valid &= (pixels >= 0) & (pixels < num_classes)
                valid &= pixels != ignore_index
                if valid.any():
                    counts += np.bincount(
                        pixels[valid].astype(np.int64), minlength=num_classes
                    )
    return counts


def inverse_sqrt_class_weights(counts, max_weight=10.0):
    """Inverse-sqrt relative frequency, mean 1 over observed classes, capped at 10."""
    counts = np.asarray(counts, dtype=np.float64)
    active = counts > 0
    if not active.any():
        raise ValueError("No labeled training pixels found for class weighting")
    frequencies = counts[active] / counts[active].sum()
    weights = np.zeros_like(counts)
    weights[active] = 1 / np.sqrt(frequencies)
    weights[active] /= weights[active].mean()
    weights[active] = np.minimum(weights[active], max_weight)
    return torch.as_tensor(weights, dtype=torch.float32)
