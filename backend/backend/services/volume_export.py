from pathlib import Path

import nibabel as nib
import numpy as np


def save_mask(mask, affine, header, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mask = np.asarray(mask, dtype=np.uint8)

    output_image = nib.Nifti1Image(
        mask,
        affine,
        header.copy()
    )

    output_image.set_data_dtype(np.uint8)

    nib.save(
        output_image,
        str(output_path)
    )

    return str(output_path)