import numpy as np


def preprocess_volume(volume, slice_axis=2):
    """
    Match the U-Net training preprocessing:
    clip HU to [-1000, 400], scale to [0, 1],
    then move the slice axis to the front.
    """
    if volume.ndim != 3:
        raise ValueError("Expected a 3D CT volume.")

    normalized = np.asarray(volume, dtype=np.float32).copy()

    if not np.isfinite(normalized).all():
        raise ValueError("CT volume contains invalid values.")

    np.clip(normalized, -1000.0, 400.0, out=normalized)
    normalized -= -1000.0
    normalized /= 1400.0

    return np.moveaxis(normalized, slice_axis, 0)
