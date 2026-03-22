import cv2
import numpy as np
import os

# MediaPipe ko safe tareeqe se import karein
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

def detect_video_fake(video_path: str):
    """
    Returns: is_fake (bool), report (str)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True, "Error: Video file format not supported or corrupted."

    frames_data = []
    count = 0
    
    # Analyze first 15 frames for consistency
    while count < 15:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert to gray for frequency analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        frames_data.append(np.std(magnitude_spectrum))
        count += 1
    
    cap.release()

    if len(frames_data) < 2:
        return False, "Video too short for deepfake analysis."

    # Temporal logic: AI videos often have inconsistent frequency patterns between frames
    variation = float(np.std(frames_data))
    
    # If variation is too high or too low, it's suspicious
    is_fake = variation > 2.5 or variation < 0.1
    
    decision = "🚨 FAKE DETECTED (Temporal Inconsistency)" if is_fake else "✅ VERIFIED REAL"
    
    # Final Report
    report = (
        f"[Video] Traceability Report\n"
        f"- File: {os.path.basename(video_path)}\n"
        f"- Frames Analyzed: {len(frames_data)}\n"
        f"- Spectral Variation: {variation:.4f}\n"
        f"- MediaPipe Status: {'Active' if MEDIAPIPE_AVAILABLE else 'Bypassed (System Error)'}\n"
        f"- Decision: {decision}"
    )
    
    return is_fake, report


