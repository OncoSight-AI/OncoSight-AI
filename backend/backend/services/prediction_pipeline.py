
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

        # Use NVIDIA GPU when available.
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

        slice_axis = metadata["slice_axis"]

        # Preprocess CT volume.
        slices = preprocess_volume(volume, slice_axis)

        predicted_slices = []
        per_slice_detections = []
        per_slice_details = []
        all_detections = []

        batch_size = 2

        for start in range(0, len(slices), batch_size):

            batch = slices[start:start + batch_size]

            # Convert normalized grayscale slices to uint8 RGB.
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
                        "slice_index": slice_index,
                        "detection_number": detection["detection_number"],
                        "segmented_pixels": detection["segmented_pixels"],
                        "confidence": confidence,
                        "confidence_percent": (
                            round(confidence * 100, 2)
                            if confidence is not None
                            else None
                        )
                    }

                    slice_detections.append(detection_info)
                    all_detections.append(detection_info)

                per_slice_details.append({
                    "slice_index": slice_index,
                    "detection_count": detection_count,
                    "average_confidence": prediction.get(
                        "average_confidence"
                    ),
                    "highest_confidence": prediction.get(
                        "highest_confidence"
                    ),
                    "detections": slice_detections
                })

        # Reconstruct 2D predictions into the original 3D volume.
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

        # Calculate statistics using actual YOLO confidence scores.
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

        return {
            "scan_id": scan_id,
            "model": "YOLOv8n-seg (lung_nodule)",
            "detection_class": "lung_nodule",

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
