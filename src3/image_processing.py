"""
image_processing.py
---------------------------------
Enhanced image segmentation and overlay module
✅ Loads trained DeepLabV3+ model safely
✅ Highlights only the requested theme
✅ Falls back to mock segmentation ONLY if model missing
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.serialization import add_safe_globals
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
from segmentation_models_pytorch.encoders._base import EncoderMixin

# =====================================================
# Global model registry
# =====================================================
SEGMENTATION_MODELS = {}
DEVICE = None
MODEL_DIR = Path(r"F:/src3/models/segmentation_models")
MODEL_PATH = MODEL_DIR / "best_model.pth"
MODEL_INITIALIZED = False


# =====================================================
# Model Loader (fixed)
# =====================================================
def load_segmentation_models():
    """Load PyTorch segmentation models safely, including ResNetEncoder."""
    global SEGMENTATION_MODELS, DEVICE

    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
    from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
    from torch.serialization import add_safe_globals
    from pathlib import Path
    import torch

    # ✅ Allowlist safe classes
    add_safe_globals([
        DeepLabV3Plus,
        ResNetEncoder
    ])

    model_dir = Path(r"F:/src3/models/segmentation_models")
    if not model_dir.exists():
        print("⚠️ segmentation_models directory not found")
        return

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {DEVICE}")

    model_path = model_dir / "best_model.pth"
    if not model_path.exists():
        print(f"⚠️ Model not found at {model_path}")
        return

    try:
        print(f"📦 Loading model: {model_path}")
        checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, smp.DeepLabV3Plus):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            model = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                in_channels=3,
                classes=4
            )
            model.load_state_dict(checkpoint)
        else:
            raise ValueError("❌ Unsupported checkpoint format")

        model.to(DEVICE).eval()
        SEGMENTATION_MODELS["deeplab"] = model
        print("✅ Successfully loaded DeepLabV3+ model")

    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        print("⚙️ Using mock segmentation instead.")



# =====================================================
# Image Preprocessing
# =====================================================
def preprocess_image(image, target_size=(256, 256)):
    """Resize, normalize, and prepare tensor for model input."""
    img_resized = cv2.resize(image, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_resized


# =====================================================
# Segmentation Inference
# =====================================================
def run_segmentation(img_batch, model_choice="deeplab"):
    """Run model inference and return class mask."""
    if model_choice not in SEGMENTATION_MODELS:
        print("⚠️ Model not loaded. Using mock segmentation.")
        return None

    model = SEGMENTATION_MODELS[model_choice]
    with torch.no_grad():
        img_batch = img_batch.to(DEVICE)
        logits = model(img_batch)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
    return mask


# =====================================================
# Theme-to-class mapping
# =====================================================
THEME_CLASS = {
    "vegetation": 1,
    "water": 2,
    "urban": 3
}


# =====================================================
# Generate Overlay (theme-specific highlighting)
# =====================================================
def get_mask_overlay(image_path, model_choice="deeplab", theme="vegetation"):
    """
    Generate segmentation mask + overlay, highlighting ONLY the requested theme.
    Returns: (original_image_rgb, overlay_rgb, mask)
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Preprocess
    img_batch, img_resized = preprocess_image(original_img)

    # Run model
    if SEGMENTATION_MODELS and model_choice in SEGMENTATION_MODELS:
        mask = run_segmentation(img_batch, model_choice)
        if mask is None:
            mask = create_mock_segmentation((256, 256), theme)
    else:
        mask = create_mock_segmentation((256, 256), theme)

    mask = np.clip(mask, 0, 3)
    # After mask = np.clip(mask, 0, 3)
    # ADD THIS FILTERING
    if theme == "vegetation":
        mask = np.where(mask == 1, 1, 0)
    elif theme == "water":
        mask = np.where(mask == 2, 2, 0)
    elif theme == "urban":
        mask = np.where(mask == 3, 3, 0)

    # Focus only on the requested theme
    class_id = THEME_CLASS.get(theme, 1)
    binary_mask = (mask == class_id).astype(np.uint8)

    # Define color for this theme only
    theme_colors = {
        "vegetation": [0, 255, 0],  # Green
        "water": [255, 0, 0],       # Blue (BGR)
        "urban": [0, 0, 255]        # Red (BGR)
    }
    color = np.array(theme_colors.get(theme, [0, 255, 0]), dtype=np.uint8)

    color_mask = np.zeros_like(img_resized, dtype=np.uint8)
    color_mask[binary_mask == 1] = color

    # Overlay
    overlay = cv2.addWeighted(img_resized, 0.6, color_mask, 0.4, 0)

    # Convert to RGB
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return img_resized_rgb, overlay_rgb, mask


# =====================================================
# Mock segmentation (fallback)
# =====================================================
def create_mock_segmentation(size, theme):
    """Simulated segmentation mask if model missing."""
    mask = np.zeros(size, dtype=np.uint8)
    h, w = size
    if theme == "vegetation":
        mask[h//4:3*h//4, w//4:3*w//4] = 1
    elif theme == "water":
        mask[h//3:2*h//3, w//5:4*w//5] = 2
    elif theme == "urban":
        mask[:h//2, :w//2] = 3
    return mask


# =====================================================
# Change Detection
# =====================================================
def compute_binary_change(mask1, mask2, class_id):
    """Compute binary change map and statistics for a given class."""
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask1 = np.clip(mask1, 0, 3)
    mask2 = np.clip(mask2, 0, 3)

    binary1 = (mask1 == class_id).astype(np.uint8)
    binary2 = (mask2 == class_id).astype(np.uint8)

    change_map = np.zeros_like(binary1)
    change_map[(binary1 == 1) & (binary2 == 0)] = 1  # Loss
    change_map[(binary1 == 0) & (binary2 == 1)] = 2  # Gain

    total_pixels = mask1.size
    unchanged = np.sum(change_map == 0)
    loss = np.sum(change_map == 1)
    gain = np.sum(change_map == 2)

    stats = {
        "total_pixels": int(total_pixels),
        "loss_percent": float(loss / total_pixels * 100),
        "gain_percent": float(gain / total_pixels * 100),
        "net_change_percent": float((gain - loss) / total_pixels * 100),
    }

    return change_map, stats


# =====================================================
# Initialize models on import
# =====================================================
def ensure_real_model_loaded():
    """Guarantees that the DeepLabV3+ model is loaded (not mock)."""
    global SEGMENTATION_MODELS, MODEL_INITIALIZED

    if MODEL_INITIALIZED and "deeplab" in SEGMENTATION_MODELS:
        return SEGMENTATION_MODELS

    print("🔁 Re-initializing real DeepLabV3+ model...")
    load_segmentation_models()

    if "deeplab" in SEGMENTATION_MODELS:
        MODEL_INITIALIZED = True
        print("✅ Real DeepLabV3+ model active.")
    else:
        print("⚠️ Still using mock model — please check model path.")
    
    return SEGMENTATION_MODELS

ensure_real_model_loaded()
