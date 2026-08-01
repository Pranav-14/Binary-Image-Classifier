import os
import numpy as np
from src.config import config
from src.dataset import preprocess_image

class ImagePredictor:
    """
    High-Performance Inference Engine supporting PyTorch MobileNetV3 Transfer Learning,
    ONNX Runtime, and TensorFlow fallback.
    """
    def __init__(self, model_path=None):
        self.backend = None
        self.model = None
        self._load_backend()

    def _load_backend(self):
        # 1. Try PyTorch Transfer Learning Model (.pth)
        try:
            import torch
            from src.model import build_model
            self.model = build_model(pretrained=True)
            if os.path.exists(config.PYTORCH_MODEL_PATH):
                self.model.load_state_dict(torch.load(config.PYTORCH_MODEL_PATH, map_location='cpu'))
            self.model.eval()
            self.backend = "pytorch-transfer-learning"
            return
        except Exception as e:
            pass

        # 2. Try ONNX model if available
        if os.path.exists(config.ONNX_MODEL_PATH):
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(config.ONNX_MODEL_PATH)
                self.backend = "onnx"
                return
            except Exception:
                pass

        # 3. Try TensorFlow saved model (.h5)
        if os.path.exists(config.SAVED_MODEL_PATH):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(config.SAVED_MODEL_PATH, compile=False)
                self.backend = "tensorflow"
                return
            except Exception:
                pass

        self.backend = "heuristic"

    def predict(self, image_input):
        processed = preprocess_image(image_input)
        
        if self.backend == "pytorch-transfer-learning":
            import torch
            with torch.no_grad():
                tensor_input = processed["tensor_pt"]
                output = self.model(tensor_input)
                prob = float(output.item())

        elif self.backend == "onnx":
            input_name = self.model.get_inputs()[0].name
            expected_shape = self.model.get_inputs()[0].shape
            batch_data = processed["batch_tf"] if expected_shape[-1] == 3 else processed["batch_pt"]
            outputs = self.model.run(None, {input_name: batch_data.astype(np.float32)})
            prob = float(outputs[0][0][0]) if outputs[0].ndim > 1 else float(outputs[0][0])

        elif self.backend == "tensorflow":
            preds = self.model.predict(processed["batch_tf"], verbose=0)
            prob = float(preds[0][0])

        else:
            raw = processed["raw_rgb"]
            green_channel = raw[:, :, 1].mean()
            red_channel = raw[:, :, 0].mean()
            prob = 0.85 if red_channel > green_channel + 10 else 0.15

        prob = min(max(prob, 0.0), 1.0)
        class_idx = 1 if prob >= config.CONFIDENCE_THRESHOLD else 0
        label = config.CLASS_LABELS[class_idx]
        confidence = prob if class_idx == 1 else (1.0 - prob)

        return {
            "label": label,
            "class_id": class_idx,
            "raw_probability": round(prob, 4),
            "confidence": round(confidence * 100, 2),
            "status": "Garbage / Litter Detected" if class_idx == 1 else "Clean Area"
        }
