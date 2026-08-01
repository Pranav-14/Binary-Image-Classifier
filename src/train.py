import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.config import config
from src.model import build_model
from src.dataset import WasteDataset, train_transforms, val_transforms, VALID_EXTENSIONS

def train_transfer_learning_model(epochs=config.EPOCHS, lr=config.LEARNING_RATE):
    """
    Train PyTorch MobileNetV3 Transfer Learning model and export model weights and ONNX.
    """
    print("=" * 60)
    print("  PyTorch MobileNetV3 Transfer Learning Training Engine")
    print("=" * 60)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference & Training Device: {device}")

    # Gather local dataset images
    samples = []
    samples_dir = config.SAMPLES_DIR
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.lower().endswith(VALID_EXTENSIONS):
                # 0 for clean, 1 for garbage/litter
                label = 1 if any(k in f.lower() for k in ["garb", "d1", "d3", "litter", "test3", "test5"]) else 0
                samples.append((os.path.join(samples_dir, f), label))

    print(f"Total training/validation samples loaded: {len(samples)}")
    if len(samples) < 2:
        print("Notice: Insufficient local dataset images. Exporting pre-trained weights.")
        model = build_model(pretrained=True).to(device)
        save_and_export_model(model, device)
        return

    # Train / Val Split
    dataset = WasteDataset(samples, transform=train_transforms)
    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()

        val_acc = (correct / val_size) * 100.0 if val_size > 0 else 100.0
        val_epoch_loss = val_loss / val_size if val_size > 0 else epoch_loss

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            save_and_export_model(model, device)

def save_and_export_model(model, device):
    model.eval()
    torch.save(model.state_dict(), config.PYTORCH_MODEL_PATH)
    print(f"--> Saved PyTorch weights to {config.PYTORCH_MODEL_PATH}")

    # Export ONNX Model
    try:
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
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
        print(f"--> Exported high-performance ONNX model to {config.ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"ONNX export notice: {e}")

if __name__ == "__main__":
    train_transfer_learning_model(epochs=5)
