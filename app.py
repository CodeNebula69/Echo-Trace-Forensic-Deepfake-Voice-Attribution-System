import sys
import os
from pathlib import Path
import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from video.video_detector import detect_video_fake
from audio.audio_detector import detect_audio_fake
from image.image_detector import detect_image_fake

def analyze_file(file):
    if file is None:
        return "No file", "Please upload a file."

    file_path = file.name
    ext = file_path.split(".")[-1].lower()

    # Route based on file type
    if ext in ["mp4", "avi", "mov", "mkv"]:
        if os.path.getsize(file_path) < 8 * 1024 * 1024:
            is_fake, report = detect_audio_fake(file_path)
            modality = "Audio (Extracted from Video)"
        else:
            is_fake, report = detect_video_fake(file_path)
            modality = "Video"
    elif ext in ["mp3", "wav", "m4a"]:
        is_fake, report = detect_audio_fake(file_path)
        modality = "Audio"
    else:
        # 📸 Photo Detection
        is_fake, report = detect_image_fake(file_path)
        modality = "Image"

    # 🚨 FINAL VERDICT LOGIC
    verdict = "🚨 FAKE DETECTED" if is_fake else "✅ VERIFIED REAL"
    
    full_report = (
        f"--- Echo-Trace Analysis ---\n"
        f"Modality: {modality}\n"
        f"Verdict: {verdict}\n"
        f"Details: {report}"
    )
    
    return verdict, full_report

# app.py mein niche wala section update karo
# app.py ki last lines ko replace karo
iface = gr.Interface(
    fn=analyze_file,
    inputs=gr.File(label="Upload Media"),
    outputs=[
        gr.Textbox(label="Result"), 
        gr.Textbox(label="Full Report", lines=8)
    ],
    title="Echo-Trace: Deepfake Detector",
    description="Running on NVIDIA RTX 4060 GPU"
)

if __name__ == "__main__":
    # 🚀 504 ERROR FIX: Queue enable karne se timeout nahi aayega
    iface.queue().launch(
        server_name="0.0.0.0", 
        server_port=7870, # Naya port conflict se bachne ke liye
        share=True,
        show_error=True
    )