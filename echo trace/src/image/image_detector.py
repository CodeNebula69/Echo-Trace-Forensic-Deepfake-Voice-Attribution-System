import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -------------------------
# 1. Load Better Diffusion-Aware Model
# -------------------------
_MODEL_NAME = "umm-maybe/AI-image-detector"  
_IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(_MODEL_NAME)
_IMAGE_MODEL = AutoModelForImageClassification.from_pretrained(_MODEL_NAME)
_IMAGE_MODEL.eval()

def _deepfake_model_fake_score(pil_image: Image.Image) -> float:
    """
    Returns fake probability from the classifier, dynamically checking labels.
    """
    inputs = _IMAGE_PROCESSOR(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = _IMAGE_MODEL(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    # Dynamically find the 'fake' / 'ai' label index
    id2label = _IMAGE_MODEL.config.id2label
    fake_idx = 1  # Default fallback
    
    for idx, label in id2label.items():
        if any(keyword in label.lower() for keyword in ["fake", "ai", "artificial", "generated"]):
            fake_idx = idx
            break

    return float(probs[fake_idx])

# -------------------------
# 2. Main Detector Logic (Fourier Focused)
# -------------------------
def detect_image_fake(image_path: str):
    """
    Returns:
      is_fake (bool): True if classified as fake / AI-generated.
      report (str): Human-readable explanation with metrics.
    """
    # ---- Load image ----
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image from path: {image_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Safe PIL conversion for HF Processor
    pil_image = Image.fromarray(img_rgb)
    img_gray_f = img_gray.astype(np.float32)

    # ---- Calculate RMS Contrast ----
    rms_contrast = np.std(img_gray_f) + 1e-5  

    # ---- 1) Fourier transform artifacts (The Game Changer) ----
    f = np.fft.fft2(img_gray_f)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    fourier_std = float(np.std(mag))
    
    # Relative Fourier: Normalizing against the image's overall contrast
    rel_fourier = fourier_std / rms_contrast
    
    # Threshold setup based on testing (Fake was ~0.35, Real was ~0.44)
    checker_pattern = rel_fourier <= 0.38  
    checker_flag = int(checker_pattern)

    # ---- 2) Deepfake classifier score ----
    model_fake_score = _deepfake_model_fake_score(pil_image)

    # ---- 3) Fuse scores (New Rule-Based Logic) ----
    # Heavily relying on Fourier frequency, AI model as secondary
    final_score = (
        0.70 * checker_flag + 
        0.30 * model_fake_score
    )

    # Decision Boundary
    is_fake = final_score >= 0.50

    if is_fake:
        decision_text = "FAKE / AI-generated (Detected via Frequency Spectrum)"
    else:
        decision_text = "Likely REAL (Normal Frequency Spread)"

    # ---- 4) Human-readable report ----
    report = (
        f"[Image Data] RMS Contrast: {rms_contrast:.2f} | "
        f"Rel. Fourier: {rel_fourier:.2f} (flag={checker_pattern}) | "
        f"AI Model Prob: {model_fake_score:.2f} | "
        f"Final Fusion Score: {final_score:.2f} | "
        f"Decision: {decision_text}"
    )

    return is_fake, report

# Run the function:
# is_fake, report = detect_image_fake("test_image.jpg")
# print(report)
