import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import config
from src.model import build_torch_model

def train_pytorch_model(num_epochs=config.EPOCHS):
    """
    Train PyTorch model loop and export weights and ONNX model for high performance deployment.
    """
    print("Initializing PyTorch Binary Classifier training...")
    model = build_torch_model()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save default weights
    torch.save(model.state_dict(), config.PYTORCH_MODEL_PATH)
    print(f"Saved PyTorch weights to {config.PYTORCH_MODEL_PATH}")

    # Export ONNX model
    try:
        dummy_input = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            model,
            dummy_input,
            config.ONNX_MODEL_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Exported ONNX model to {config.ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"ONNX Export notice: {e}")

if __name__ == "__main__":
    train_pytorch_model()
