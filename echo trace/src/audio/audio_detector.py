import os
import tempfile
import warnings
import torch
import librosa
from pathlib import Path
from pydub import AudioSegment
from transformers import pipeline

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    try:
        from moviepy import AudioFileClip
    except ImportError:
        AudioFileClip = None

warnings.filterwarnings("ignore")

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU found. Using CPU.")

_AUDIO_MODEL = None

def get_model():
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        try:
            _AUDIO_MODEL = pipeline(
                "audio-classification",
                model="Hemgg/Deepfake-audio-detection",
                device=device
            )
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
    return _AUDIO_MODEL

def _to_wav(input_path: str, max_duration_sec=20):
    ext = Path(input_path).suffix.lower()
    temp_wav = tempfile.mktemp(suffix=".wav")

    try:
        if ext in ['.mp4', '.mov', '.avi'] and AudioFileClip:
            clip = AudioFileClip(input_path)
            sub_clip = clip.subclip(0, min(clip.duration, max_duration_sec))
            sub_clip.write_audiofile(temp_wav, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
            clip.close()
            return temp_wav
        
        audio = AudioSegment.from_file(input_path)
        if len(audio) > max_duration_sec * 1000:
            audio = audio[:max_duration_sec * 1000]
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav, format="wav")
        return temp_wav

    except Exception as e:
        y, sr = librosa.load(input_path, sr=16000, duration=max_duration_sec)
        import soundfile as sf
        sf.write(temp_wav, y, sr)
        return temp_wav

def detect_audio_fake(audio_path: str):
    model = get_model()
    if model is None:
        return False, "❌ Audio Model not loaded."

    wav_path = None
    original_ext = Path(audio_path).suffix.lower()

    try:
        wav_path = _to_wav(audio_path)

        result = model(wav_path)
        if isinstance(result, list): result = result[0]
        
        label = result.get("label", "").upper()
        score = float(result.get("score", 0.0))

        is_fake_label = any(word in label for word in ["FAKE", "SPOOF", "AIVOICE", "AI"])
        model_fake_prob = score if is_fake_label else (1.0 - score)
        
        # 🚨 ENGINEERING HEURISTIC FOR WHATSAPP FILES
        # Agar file .mp4 ya .ogg hai (WhatsApp format), toh 15% penalty kaat do compression ki wajah se
        if original_ext in ['.mp4', '.ogg', '.m4a']:
            model_fake_prob = max(0.0, model_fake_prob - 0.15)
        
        # Threshold wapas normal kar diya hai
        is_fake = model_fake_prob > 0.85

        decision = "🚨 FAKE DETECTED" if is_fake else "✅ VERIFIED REAL"
        
        report = (
            f"[Audio Traceability]\n"
            f"- Model Label: {label} (Raw Score: {score:.2f})\n"
            f"- Adjusted AI Probability: {model_fake_prob:.2f} (Compression Adjusted)\n"
            f"- Final Verdict: {decision}"
        )
        return is_fake, report

    except Exception as e:
        return False, f"❌ Processing Error: {str(e)}"

    finally:
        if wav_path and os.path.exists(wav_path):
            try: os.remove(wav_path)
            except: pass
        if device == 0:
            torch.cuda.empty_cache()