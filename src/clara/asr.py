"""Mock ASR module: converts audio (or direct text) into transcript + paralinguistic features.
This is a simplified stand-in for a real ASR that would also return timing and prosodic information.
"""
from typing import Dict
from pathlib import Path


def transcribe(text_override: str = None) -> Dict:
    """Return a mock transcription and paralinguistic features.

    If text_override is provided, it is used as the "recognized" text.
    """
    text = text_override or "(no input)"

    # Very simple heuristics for paralinguistic features
    emphasis = "high" if "!" in text or text.isupper() else "normal"
    pacing = "slow" if "..." in text or "," in text else "normal"
    pitch = "low" if any(w in text.lower() for w in ["sigh", "tired"]) else "normal"

    return {
        "text": text.strip(),
        "paralinguistic": {"emphasis": emphasis, "pacing": pacing, "pitch": pitch},
    }


def transcribe_audio(wav_path: str) -> Dict:
    """Attempt to transcribe a WAV file using VOSK if available.

    Requirements for VOSK usage:
    - `pip install vosk soundfile`
    - a VOSK model downloaded (set VOSK_MODEL_PATH env var or place under data/vosk-model)

    If VOSK isn't available or model not found, raises RuntimeError with instructions.
    """
    try:
        import os
        import soundfile as sf
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError("VOSK or soundfile not installed. Install requirements or use text input.") from e

    model_path = os.environ.get("VOSK_MODEL_PATH")
    if not model_path:
        # default expected location at repo root data/vosk-model
        model_path = Path(__file__).resolve().parents[2] / "data" / "vosk-model"

    if not Path(model_path).exists():
        raise RuntimeError(f"VOSK model not found at {model_path}. Download a model and set VOSK_MODEL_PATH.")

    # Read audio
    data, samplerate = sf.read(wav_path)
    if samplerate != 16000:
        raise RuntimeError("VOSK requires 16000 Hz mono WAV input. Please resample your audio to 16k mono.")

    model = Model(str(model_path))
    rec = KaldiRecognizer(model, samplerate)

    # If data is multi-channel, take first channel
    import numpy as np
    if len(data.shape) > 1:
        data = data[:, 0]

    # Convert to 16-bit PCM bytes
    pcm_data = (data * 32767).astype(np.int16).tobytes()
    if rec.AcceptWaveform(pcm_data):
        import json
        res = json.loads(rec.Result())
        text = res.get("text", "")
    else:
        import json
        res = json.loads(rec.FinalResult())
        text = res.get("text", "")

    return {"text": text, "paralinguistic": {"emphasis": "normal", "pacing": "normal", "pitch": "normal"}}
