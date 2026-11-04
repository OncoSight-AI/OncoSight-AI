import numpy as np


def reconstruct_volume(
    predicted_slices,
    original_shape,
    slice_axis
):

    if not predicted_slices:
        raise ValueError("No predicted slices received.")

    stacked = np.stack(
        predicted_slices,
        axis=0
    )

    reconstructed = np.moveaxis(
        stacked,
        0,
        slice_axis
    )

    if reconstructed.shape != tuple(original_shape):
        raise ValueError(
            f"Reconstructed shape {reconstructed.shape} "
            f"does not match original shape {original_shape}."
        )

    return reconstructed.astype(np.uint8)