from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(file_path):
    """
    Load a 3D NIfTI CT volume.

    Returns:
        volume: 3D numpy array
        affine: NIfTI affine matrix
        header: NIfTI header
        metadata: Volume metadata
    """

    file_path = Path(file_path)

    image = nib.load(str(file_path))
    volume = image.get_fdata(dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D CT volume, received {volume.ndim}D data."
        )

    if not np.isfinite(volume).all():
        raise ValueError("CT volume contains invalid values.")

    spacing = image.header.get_zooms()[:3]

    metadata = {
        "dimensions": list(volume.shape),
        "spacing": list(spacing),
        "orientation": nib.aff2axcodes(image.affine),
        "format": "NIfTI",
        "slice_axis": int(
            np.argmax(np.abs(image.affine[2, :3]))
        )
    }

    return volume, image.affine, image.header, metadata