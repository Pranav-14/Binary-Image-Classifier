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
        result["filename"] = file.filename
        return JSONResponse(content=result)
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
            res["filename"] = file.filename
            results.append(res)
    return {"total_processed": len(results), "results": results}

@app.get("/samples")
def list_samples():
    samples_dir = config.SAMPLES_DIR
    if not os.path.exists(samples_dir):
        return {"samples": []}
    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    return {"samples": files}

# Mount static web UI files if present
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Binary Image Classifier API is Running</h1><p>Visit <a href='/docs'>/docs</a> for Swagger API UI.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
