from pathlib import Path
import uuid

import numpy as np

from services.volume_loader import load_nifti
from services.ct_preprocessing import preprocess_volume
from services.unet_service import UNetSegmentationService
from services.volume_reconstruction import reconstruct_volume
from services.volume_export import save_mask


class PredictionPipeline:
    def __init__(self, model_path, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = UNetSegmentationService(
            model_path=model_path,
            threshold=0.50,
            image_size=256
        )

    def predict(self, input_path):
        scan_id = uuid.uuid4().hex

        volume, affine, header, metadata = load_nifti(input_path)
        slice_axis = metadata["slice_axis"]

        slices = preprocess_volume(volume, slice_axis)
        predicted_slices = []

        batch_size = 2

        for start in range(0, len(slices), batch_size):
            batch = slices[start:start + batch_size]
            predicted_slices.extend(self.model.predict_batch(batch))

        mask = reconstruct_volume(
            predicted_slices,
            volume.shape,
            slice_axis
        )

        mask_path = self.output_dir / f"{scan_id}_mask.nii.gz"
        save_mask(mask, affine, header, mask_path)

        slices_with_segmentation = sum(
            int(np.any(item)) for item in predicted_slices
        )

        return {
            "scan_id": scan_id,
            "model": "U-Net (best_lung_unet_weighted.pth)",
            "total_slices": len(slices),
            "slices_with_segmentation": slices_with_segmentation,
            "segmented_voxels": int(mask.sum()),
            "mask_shape": list(mask.shape),
            "mask_path": str(mask_path),
            "threshold": 0.50
        }
