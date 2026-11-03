# OncoSight AI — YOLOv8 Lung Nodule Segmentation

## Test Evaluation Visual Report

This folder contains visual outputs from the YOLOv8n-seg model trained for experimental lung nodule segmentation using LIDC-IDRI CT data.

### Model Information
- Model: YOLOv8n-seg
- Task: 2D lung nodule detection and segmentation
- Input image size: 512 × 512
- Test images: 1,515
- Ground-truth instances: 106
- Held-out test patients: 7

### Test Metrics

| Metric | Bounding Box | Mask Segmentation |
|---|---:|---:|
| Precision | 0.460 | 0.533 |
| Recall | 0.329 | 0.292 |
| mAP@50 | 0.342 | 0.307 |
| mAP@50–95 | 0.146 | 0.131 |

### Folder Structure

- `graphs/` — Precision, recall, F1, PR curves and confusion matrices.
- `predictions/` — YOLO-annotated CT test images.
- `val_batch*_labels.jpg` — Ground-truth validation batch visualizations.
- `val_batch*_pred.jpg` — Validation batch model predictions.

### Important Limitations

This is an experimental research prototype, not a clinically validated diagnostic system. The model performs 2D nodule detection and segmentation; it does not classify cancer or determine malignancy.

The test set contains only seven held-out patients, so results should be interpreted cautiously.