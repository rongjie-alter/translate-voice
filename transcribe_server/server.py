#!/usr/bin/env python3
"""Transcription server: kotoba-tech/kotoba-whisper-v2.1 → Japanese text."""

import os
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import pipeline

MODEL_ID = os.environ.get("TRANSCRIBE_MODEL", "kotoba-tech/kotoba-whisper-v2.1")
HOST = os.environ.get("TRANSCRIBE_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRANSCRIBE_PORT", "8001"))
CHUNK_LENGTH_S = int(os.environ.get("CHUNK_LENGTH_S", "30"))


def _get_device() -> str:
    env = os.environ.get("GPU_DEVICE", "auto")
    if env != "auto":
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = _get_device()
dtype = torch.float32 if device == "cpu" else torch.float16
model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" else {}

print(f"Loading {MODEL_ID} on {device} ({dtype}) …")
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    torch_dtype=dtype,
    device=device,
    model_kwargs=model_kwargs,
    punctuator=True,
)
print("Transcription model ready.")

app = FastAPI(title="Transcribe Server")


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    content = await audio.read()
    suffix = Path(audio.filename or "audio.ogg").suffix or ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipe(
            tmp_path,
            chunk_length_s=CHUNK_LENGTH_S,
            generate_kwargs={"language": "ja", "task": "transcribe"},
            return_timestamps=False,
        )
        return {"result": {"text": result["text"].strip()}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
