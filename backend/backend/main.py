import os
import hashlib
import urllib.request
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.prediction_pipeline import PredictionPipeline


BASE_DIR = Path(__file__).resolve().parent

MODEL_URL = os.getenv("MODEL_URL", "").strip()

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "best_lung_unet_weighted.pth")
    )
)

EXPECTED_SHA256 = (
    "e5ac0f41fbabe9eca37ad8134df59a1a1f264de749abb032022c68b31f63a8ce"
)

OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_model():
    if MODEL_PATH.exists():
        return

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL environment variable is missing.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_model = MODEL_PATH.with_suffix(".download")

    print("Downloading trained U-Net model...")

    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_model)

        sha256 = hashlib.sha256()

        with temporary_model.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                sha256.update(chunk)

        if sha256.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("Model SHA256 checksum verification failed.")

        temporary_model.replace(MODEL_PATH)

        print("U-Net model downloaded and verified successfully.")

    finally:
        if temporary_model.exists():
            temporary_model.unlink()


app = FastAPI(
    title="OncoSight AI Backend",
    description="Research prototype for lung CT segmentation using U-Net.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://oncosight-ai-px9c.onrender.com"
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

    download_model()

    pipeline = PredictionPipeline(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR
    )

    print("OncoSight AI U-Net backend is ready.")


@app.get("/")
def root():
    return {
        "project": "OncoSight AI",
        "status": "running",
        "model": "U-Net",
        "purpose": "Lung CT segmentation research prototype"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if pipeline else "model_not_loaded",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.model.device) if pipeline else None
    }


@app.post("/predict-volume")
async def predict_volume(file: UploadFile = File(...)):

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    filename = file.filename or ""

    if not (
        filename.lower().endswith(".nii")
        or filename.lower().endswith(".nii.gz")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .nii and .nii.gz files are supported."
        )

    suffix = (
        ".nii.gz"
        if filename.lower().endswith(".nii.gz")
        else ".nii"
    )

    temporary_path = UPLOAD_DIR / (
        f"upload_{os.urandom(8).hex()}{suffix}"
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
            result = pipeline.predict(temporary_path)

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

    if not scan_id.isalnum():
        raise HTTPException(
            status_code=400,
            detail="Invalid scan ID."
        )

    mask_path = OUTPUT_DIR / f"{scan_id}_mask.nii.gz"

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