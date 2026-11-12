
from pathlib import Path
import uuid

import numpy as np
import torch

from services.volume_loader import load_nifti
from services.ct_preprocessing import preprocess_volume
from services.yolo_service import YOLOSegmentationService
from services.volume_reconstruction import reconstruct_volume
from services.volume_export import save_mask


class PredictionPipeline:

    def __init__(self, model_path, output_dir):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = 0 if torch.cuda.is_available() else "cpu"

        self.model = YOLOSegmentationService(
            model_path=model_path,
            confidence=0.25,
            image_size=512,
            device=device
        )

        self.device = device
        self.confidence_threshold = 0.25

    def predict(self, input_path):

        scan_id = uuid.uuid4().hex

        volume, affine, header, metadata = load_nifti(input_path)

        slice_axis = int(metadata["slice_axis"])

        if slice_axis not in (0, 1, 2):
            raise ValueError("Invalid CT slice axis.")

        slices = preprocess_volume(volume, slice_axis)

        predicted_slices = []
        per_slice_detections = []
        per_slice_details = []
        all_detections = []

        batch_size = 2

        for start in range(0, len(slices), batch_size):

            batch = slices[start:start + batch_size]

            yolo_batch = []

            for ct_slice in batch:

                gray = np.clip(
                    ct_slice * 255.0,
                    0,
                    255
                ).astype(np.uint8)

                rgb = np.repeat(
                    gray[..., np.newaxis],
                    3,
                    axis=2
                )

                yolo_batch.append(rgb)

            batch_predictions = self.model.predict_batch(yolo_batch)

            for batch_index, prediction in enumerate(batch_predictions):

                slice_index = start + batch_index

                predicted_slices.append(prediction["mask"])

                detection_count = prediction["detection_count"]

                per_slice_detections.append(detection_count)

                slice_detections = []

                for detection in prediction["detections"]:

                    confidence = detection.get("confidence")

                    detection_info = {
                        "slice_index": int(slice_index),
                        "slice_axis": slice_axis,
                        "detection_number": int(
                            detection["detection_number"]
                        ),
                        "segmented_pixels": int(
                            detection["segmented_pixels"]
                        ),
                        "confidence": (
                            float(confidence)
                            if confidence is not None
                            else None
                        ),
                        "confidence_percent": (
                            round(float(confidence) * 100, 2)
                            if confidence is not None
                            else None
                        )
                    }

                    slice_detections.append(detection_info)
                    all_detections.append(detection_info)

                per_slice_details.append({
                    "slice_index": int(slice_index),
                    "slice_axis": slice_axis,
                    "detection_count": int(detection_count),
                    "average_confidence": prediction.get(
                        "average_confidence"
                    ),
                    "highest_confidence": prediction.get(
                        "highest_confidence"
                    ),
                    "detections": slice_detections
                })

        mask = reconstruct_volume(
            predicted_slices,
            volume.shape,
            slice_axis
        )

        mask_path = self.output_dir / f"{scan_id}_mask.nii.gz"

        save_mask(
            mask,
            affine,
            header,
            mask_path
        )

        slices_with_segmentation = sum(
            int(np.any(item)) for item in predicted_slices
        )

        total_detections = sum(per_slice_detections)

        valid_confidences = [
            item["confidence"]
            for item in all_detections
            if item["confidence"] is not None
        ]

        average_detection_confidence = (
            float(np.mean(valid_confidences))
            if valid_confidences
            else None
        )

        highest_detection_confidence = (
            float(np.max(valid_confidences))
            if valid_confidences
            else None
        )

        best_detection = (
            max(
                (
                    item for item in all_detections
                    if item["confidence"] is not None
                ),
                key=lambda item: item["confidence"],
                default=None
            )
        )

        if best_detection is None and all_detections:
            best_detection = all_detections[0]

        best_detection_slice = (
            best_detection["slice_index"]
            if best_detection is not None
            else None
        )

        return {
            "scan_id": scan_id,
            "model": "YOLOv8n-seg (lung_nodule)",
            "detection_class": "lung_nodule",

            "slice_axis": slice_axis,
            "best_detection_slice": best_detection_slice,

            "total_slices": len(slices),
            "slices_with_segmentation": slices_with_segmentation,

            "total_detections": total_detections,
            "per_slice_detections": per_slice_detections,
            "per_slice_details": per_slice_details,
            "detections": all_detections,

            "average_detection_confidence": (
                average_detection_confidence
            ),
            "average_detection_confidence_percent": (
                round(average_detection_confidence * 100, 2)
                if average_detection_confidence is not None
                else None
            ),

            "highest_detection_confidence": (
                highest_detection_confidence
            ),
            "highest_detection_confidence_percent": (
                round(highest_detection_confidence * 100, 2)
                if highest_detection_confidence is not None
                else None
            ),

            "confidence_available": bool(valid_confidences),

            "segmented_voxels": int(mask.sum()),
            "mask_shape": list(mask.shape),
            "mask_path": str(mask_path),

            "confidence_threshold": self.confidence_threshold,
            "device": str(self.device)
        }
