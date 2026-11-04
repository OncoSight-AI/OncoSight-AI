import os
import shutil
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.prediction_pipeline import PredictionPipeline


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            PROJECT_DIR
            / "models"
            / "best.pt"
        )
    )
)

OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="OncoSight AI Backend",
    description="Research prototype for lung nodule candidate segmentation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

pipeline = None
inference_lock = Lock()


@app.on_event("startup")
def load_model():

    global pipeline

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    pipeline = PredictionPipeline(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR
    )


@app.get("/")
def root():

    return {
        "project": "OncoSight AI",
        "status": "running",
        "purpose": "Lung nodule candidate segmentation research prototype"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy" if pipeline else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "model_loaded": pipeline is not None
    }


@app.post("/predict-volume")
async def predict_volume(
    file: UploadFile = File(...)
):

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    filename = file.filename or ""

    if not (
        filename.endswith(".nii")
        or filename.endswith(".nii.gz")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .nii and .nii.gz files are supported."
        )

    # Save upload to a temporary file.
    temporary_path = UPLOAD_DIR / (
        f"upload_{os.urandom(8).hex()}.nii.gz"
    )

    try:

        size = 0
        max_size = 512 * 1024 * 1024

        with temporary_path.open("wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds 512 MB."
                    )

                buffer.write(chunk)

        with inference_lock:

            result = pipeline.predict(
                temporary_path
            )

        return {
            "success": True,
            "message": "CT segmentation completed.",
            "prediction": result
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

        await file.close()


@app.get("/download-mask/{scan_id}")
def download_mask(scan_id: str):

    # Prevent path traversal.
    if not scan_id.isalnum():
        raise HTTPException(
            status_code=400,
            detail="Invalid scan ID."
        )

    mask_path = OUTPUT_DIR / (
        f"{scan_id}_mask.nii.gz"
    )

    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Segmentation mask not found."
        )

    return FileResponse(
        path=mask_path,
        filename=f"{scan_id}_mask.nii.gz",
        media_type="application/gzip"
    )