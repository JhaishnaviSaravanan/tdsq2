from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import io
import soundfile as sf
from scipy import stats

app = FastAPI()

class AudioInput(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/")
def process_audio(data: AudioInput):
    audio_bytes = base64.b64decode(data.audio_base64)
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(float)

    return {
        "rows": int(len(audio)),
        "columns": ["amplitude"],
        "mean": {"amplitude": float(np.mean(audio))},
        "std": {"amplitude": float(np.std(audio))},
        "variance": {"amplitude": float(np.var(audio))},
        "min": {"amplitude": float(np.min(audio))},
        "max": {"amplitude": float(np.max(audio))},
        "median": {"amplitude": float(np.median(audio))},
        "mode": {"amplitude": float(stats.mode(audio, keepdims=True)[0][0])},
        "range": {"amplitude": float(np.max(audio) - np.min(audio))},
        "allowed_values": {"amplitude": []},
        "value_range": {"amplitude": [float(np.min(audio)), float(np.max(audio))]},
        "correlation": []
    }
