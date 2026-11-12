
import numpy as np
import cv2
from ultralytics import YOLO


class YOLOSegmentationService:

    def __init__(
        self,
        model_path,
        confidence=0.25,
        image_size=512,
        device='cpu'
    ):

        self.model = YOLO(str(model_path))

        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def predict_batch(self, slices):

        results = self.model.predict(
            source=list(slices),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            retina_masks=True,
            verbose=False
        )

        predictions = []

        for result in results:

            height, width = result.orig_shape

            combined_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            detections = []

            # Extract actual confidence scores from YOLO
            if result.boxes is not None and len(result.boxes) > 0:
                confidence_scores = (
                    result.boxes.conf.cpu().numpy()
                )
            else:
                confidence_scores = np.array([])

            # Process segmentation masks
            if result.masks is not None:

                masks = result.masks.data.cpu().numpy()

                for index, mask in enumerate(masks):

                    mask = (
                        mask > 0.5
                    ).astype(np.uint8)

                    if mask.shape != (height, width):

                        mask = cv2.resize(
                            mask,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST
                        )

                    combined_mask = np.maximum(
                        combined_mask,
                        mask
                    )

                    # Match the mask with its YOLO detection score
                    confidence_score = (
                        float(confidence_scores[index])
                        if index < len(confidence_scores)
                        else None
                    )

                    detections.append({
                        "detection_number": index + 1,
                        "segmented_pixels": int(mask.sum()),
                        "confidence": confidence_score,
                        "confidence_percent": (
                            round(confidence_score * 100, 2)
                            if confidence_score is not None
                            else None
                        )
                    })

            # Calculate confidence statistics for this slice
            valid_scores = [
                item["confidence"]
                for item in detections
                if item["confidence"] is not None
            ]

            average_confidence = (
                float(np.mean(valid_scores))
                if valid_scores
                else None
            )

            highest_confidence = (
                float(np.max(valid_scores))
                if valid_scores
                else None
            )

            predictions.append({
                "mask": combined_mask,
                "detections": detections,
                "detection_count": len(detections),
                "average_confidence": average_confidence,
                "highest_confidence": highest_confidence
            })

        return predictions
