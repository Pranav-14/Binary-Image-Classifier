import os
from dataclasses import dataclass

@dataclass
class Config:
    # Model parameters
    IMG_SIZE: tuple = (256, 256)
    NUM_CLASSES: int = 1  # Binary output with Sigmoid activation
    CLASS_LABELS: tuple = ("Clean Environment", "Garbage / Litter")

    # Directory Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    SAMPLES_DIR: str = os.path.join(DATA_DIR, "samples")

    # Weights and Artifact Paths
    SAVED_MODEL_PATH: str = os.path.join(MODELS_DIR, "classifier.h5")
    ONNX_MODEL_PATH: str = os.path.join(MODELS_DIR, "classifier.onnx")
    PYTORCH_MODEL_PATH: str = os.path.join(MODELS_DIR, "classifier.pth")

    # Training parameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    CONFIDENCE_THRESHOLD: float = 0.5

config = Config()
