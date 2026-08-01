import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.config import config

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# PyTorch Data Augmentation and Normalization Transforms
train_transforms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset for Clean vs Garbage Image Classification
    """
    def __init__(self, samples_list, transform=val_transforms):
        self.samples = samples_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def clean_corrupted_images(data_dir=config.DATA_DIR):
    corrupted = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(VALID_EXTENSIONS):
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Corrupted image detected {file_path}: {e}")
                corrupted.append(file_path)
    return corrupted

def preprocess_image(image_input, target_size=config.IMG_SIZE):
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img_pil = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input).convert('RGB')
    else:
        raise TypeError("Unsupported image input type")

    img_resized = img_pil.resize(target_size, Image.Resampling.BILINEAR)
    raw_rgb = np.array(img_resized)
    
    # ImageNet Normalized Tensor
    tensor_input = val_transforms(img_pil).unsqueeze(0) # (1, 3, 256, 256)
    
    normalized = raw_rgb.astype(np.float32) / 255.0
    batch_tf = np.expand_dims(normalized, axis=0) # (1, H, W, C)
    batch_pt = np.transpose(batch_tf, (0, 3, 1, 2))

    return {
        "raw_rgb": raw_rgb,
        "batch_tf": batch_tf,
        "batch_pt": batch_pt,
        "tensor_pt": tensor_input
    }

if __name__ == "__main__":
    print("=== Dataset Module Test Run ===")
    sample_dir = config.SAMPLES_DIR
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        if samples:
            test_path = os.path.join(sample_dir, samples[0])
            print(f"Testing preprocess_image on sample: {samples[0]}")
            output = preprocess_image(test_path)
            print(f"Raw RGB array shape: {output['raw_rgb'].shape}")
            print(f"PyTorch ImageNet Tensor shape: {output['tensor_pt'].shape}")
