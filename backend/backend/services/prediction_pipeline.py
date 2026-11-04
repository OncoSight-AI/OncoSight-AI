from pathlib import Path
import uuid

from services.volume_loader import load_nifti
from services.ct_preprocessing import preprocess_volume
from services.yolo_service import YOLOSegmentationService
from services.volume_reconstruction import reconstruct_volume
from services.volume_export import save_mask


class PredictionPipeline:

    def __init__(self, model_path, output_dir):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.yolo = YOLOSegmentationService(
            model_path=model_path
        )

    def predict(self, input_path):

        scan_id = uuid.uuid4().hex

        volume, affine, header, metadata = load_nifti(
            input_path
        )

        slice_axis = metadata["slice_axis"]

        slices = preprocess_volume(
            volume,
            slice_axis
        )

        predicted_slices = []
        total_detections = 0
        slices_with_detections = 0

        # Small batches help limit GPU memory use.
        batch_size = 2

        for start in range(0, len(slices), batch_size):

            batch = slices[
                start:start + batch_size
            ]

            results = self.yolo.predict_batch(batch)

            for result in results:

                predicted_slices.append(result["mask"])

                count = result["detection_count"]

                total_detections += count

                if count > 0:
                    slices_with_detections += 1

        mask = reconstruct_volume(
            predicted_slices,
            volume.shape,
            slice_axis
        )

        mask_path = self.output_dir / (
            f"{scan_id}_mask.nii.gz"
        )

        save_mask(
            mask,
            affine,
            header,
            mask_path
        )

        return {
            "scan_id": scan_id,
            "total_slices": len(slices),
            "slices_with_detections": slices_with_detections,
            "total_detections": total_detections,
            "segmented_voxels": int(mask.sum()),
            "mask_shape": list(mask.shape),
            "mask_path": str(mask_path)
        }