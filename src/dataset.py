import os
import numpy as np
from PIL import Image
from src.config import config

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def clean_corrupted_images(data_dir=config.DATA_DIR):
    """
    Scans dataset folders for invalid or corrupted image files and removes/logs them.
    """
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
    """
    Preprocesses an input image (filepath, numpy array, or PIL Image)
    Returns normalized arrays scaled [0, 1].
    """
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

    # Resize image to target resolution
    img_resized = img_pil.resize(target_size, Image.Resampling.BILINEAR)
    raw_rgb = np.array(img_resized)
    normalized = raw_rgb.astype(np.float32) / 255.0
    
    # Batch dimension
    batch_tf = np.expand_dims(normalized, axis=0) # (1, H, W, C)
    batch_pt = np.transpose(batch_tf, (0, 3, 1, 2)) # (1, C, H, W)
    
    return {
        "raw_rgb": raw_rgb,
        "batch_tf": batch_tf,
        "batch_pt": batch_pt
    }
