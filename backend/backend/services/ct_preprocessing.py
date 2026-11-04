import numpy as np


def preprocess_volume(volume, slice_axis=2):
    """
    Convert CT volume into normalized 2D slices.

    Returns:
        slices: Array of shape (number_of_slices, height, width)
    """

    if volume.ndim != 3:
        raise ValueError("Expected a 3D CT volume.")

    volume = np.asarray(volume, dtype=np.float32)

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Clip CT intensity values.
    volume = np.clip(volume, -1000, 400)

    # Normalize intensity values.
    low = np.percentile(volume, 1)
    high = np.percentile(volume, 99)

    if high <= low:
        raise ValueError("Invalid CT intensity range.")

    volume = np.clip(volume, low, high)

    volume = (volume - low) / (high - low)
    volume = (volume * 255).astype(np.uint8)

    # Put the slice dimension first.
    slices = np.moveaxis(volume, slice_axis, 0)

    return slices