import os
import io
import sys
from typing import List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import config
from src.predict import ImagePredictor

app = FastAPI(
    title="Binary Image Classification API",
    description="Urban Sanitation & Environmental Intelligence API for Clean vs Garbage Classification",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ImagePredictor()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "backend": predictor.backend,
        "classes": config.CLASS_LABELS,
        "target_size": config.IMG_SIZE
    }

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not a valid image.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predictor.predict(image)
        # Ensure all fields are standard python types for JSON serialization
        response_data = {
            "label": str(result["label"]),
            "class_id": int(result["class_id"]),
            "raw_probability": float(result["raw_probability"]),
            "confidence": float(result["confidence"]),
            "status": str(result["status"]),
            "filename": file.filename
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if file.content_type.startswith("image/"):
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            res = predictor.predict(image)
            results.append({
                "label": str(res["label"]),
                "class_id": int(res["class_id"]),
                "raw_probability": float(res["raw_probability"]),
                "confidence": float(res["confidence"]),
                "status": str(res["status"]),
                "filename": file.filename
            })
    return {"total_processed": len(results), "results": results}

@app.get("/samples")
def list_samples():
    samples_dir = config.SAMPLES_DIR
    if not os.path.exists(samples_dir):
        return {"samples": []}
    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    return {"samples": files}

# Mount static web UI directory at root /
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
