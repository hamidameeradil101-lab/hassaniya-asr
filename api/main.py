"""Hassaniya ASR API: transcribes an uploaded audio clip with the fine-tuned Whisper-medium model."""

import io
import os
from math import gcd

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "checkpoint/whisper-medium-hassaniya-checkpoint-1580"
)
TARGET_SR = 16000
MAX_SECONDS = 30  # Whisper sees at most 30 s per clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Hassaniya ASR")

# Loaded once, when the server starts, not on every request.
processor = WhisperProcessor.from_pretrained(CHECKPOINT_PATH)
model = WhisperForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH, torch_dtype=DTYPE
).to(DEVICE)
model.eval()


def load_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a 16 kHz mono float32 waveform."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Could not read audio (use wav/flac/ogg).")

    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SR:
        divisor = gcd(sample_rate, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // divisor, sample_rate // divisor)

    return audio.astype(np.float32)


def transcribe_array(audio: np.ndarray) -> str:
    """Run the model on a 16 kHz mono waveform and return the text."""
    inputs = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE, dtype=DTYPE)

    # Same decoding settings as evaluation, including the guards against looping output.
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="arabic",
            task="transcribe",
            max_length=256,
            no_repeat_ngram_size=5,
            num_beams=1,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# heavy model call does not block the server's event loop.
@app.post("/transcribe")
def transcribe(file: UploadFile):
    audio = load_audio(file.file.read())
    duration = len(audio) / TARGET_SR

    if duration > MAX_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is {duration:.1f}s; the limit is {MAX_SECONDS}s.",
        )

    return {"transcription": transcribe_array(audio), "duration_seconds": round(duration, 2)}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}
