import numpy as np
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

            if result.masks is not None:

                masks = result.masks.data.cpu().numpy()

                for index, mask in enumerate(masks):

                    mask = (
                        mask > 0.5
                    ).astype(np.uint8)

                    if mask.shape != (height, width):
                        import cv2

                        mask = cv2.resize(
                            mask,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST
                        )

                    combined_mask = np.maximum(
                        combined_mask,
                        mask
                    )

                    detections.append({
                        "detection_number": index + 1,
                        "segmented_pixels": int(mask.sum())
                    })

            predictions.append({
                "mask": combined_mask,
                "detections": detections,
                "detection_count": len(detections)
            })

        return predictions