"""Isolated Pocket TTS worker using this machine's existing CUDA runtime."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
from pathlib import Path
import queue
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import numpy as np
from pocket_tts_turkish import TurkishTTS
from pydantic import BaseModel, Field
import soxr
import torch
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "wite-tech/pocket-tts-turkish"
VOICES = ("female_1", "female_2", "female_3", "male_1", "male_2", "male_3")
logger = logging.getLogger("pocket-worker")


class Speech(BaseModel):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "female_1"
    sample_rate: int = 24000
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class AudioResponse(StreamingResponse):
    def __init__(self, content, finish, **kwargs):
        super().__init__(content, **kwargs)
        self.finish = finish

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.finish()


def create_app(device="cpu", *, model_id=MODEL_ID, voices=VOICES, loader=None, worker_name="Pocket TTS"):
    key = next((line[8:] for line in (ROOT / ".env").read_text().splitlines() if line.startswith("API_KEY=")), "")
    if not key:
        raise RuntimeError("A nonempty API_KEY is required")
    slots = threading.BoundedSemaphore(1)

    def authorize(request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            raise HTTPException(403, "Invalid API key")

    @asynccontextmanager
    async def lifespan(app):
        def load():
            torch.set_num_threads(1)
            model = (
                loader(device)
                if loader
                else TurkishTTS.from_pretrained(ROOT / "downloads/pocket-tts-turkish", device=device)
            )
            for voice in voices:
                for _ in model.stream("Merhaba.", voice=voice, seed=0):
                    pass
            return model

        app.state.model = await asyncio.to_thread(load)
        logger.info("%s ready on %s; all voices warmed", worker_name, device)
        yield

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(authorize)])

    @app.get("/health")
    def health():
        return {"status": "ready", "device": device, "voices": voices, "model": model_id}

    @app.post("/v1/audio/speech")
    async def speech(body: Speech, request: Request):
        if body.model != model_id or body.voice not in voices or not body.input.strip():
            raise HTTPException(422, "Invalid model, voice or text")
        if body.sample_rate not in (8000, 16000, 24000, 48000):
            raise HTTPException(422, "Unsupported sample rate")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Pocket TTS capacity reached", headers={"Retry-After": "1"})
        stopped = threading.Event()
        chunks = queue.Queue(maxsize=2)
        completion_lock = threading.Lock()
        completed = set()
        deadline = time.monotonic() + 120

        def finish(part):
            with completion_lock:
                if part in completed:
                    return
                completed.add(part)
                if len(completed) == 2:
                    slots.release()

        def response_done():
            stopped.set()
            finish("response")

        def send(item):
            while not stopped.is_set() and time.monotonic() < deadline:
                try:
                    chunks.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            iterator = None
            try:
                source_rate = app.state.model.sample_rate
                resampler = (
                    soxr.ResampleStream(source_rate, body.sample_rate, 1, dtype="float32")
                    if body.sample_rate != source_rate
                    else None
                )
                iterator = app.state.model.stream(body.input, voice=body.voice, seed=body.seed)
                for audio in iterator:
                    if stopped.is_set() or time.monotonic() >= deadline:
                        break
                    if resampler:
                        audio = resampler.resample_chunk(audio)
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    if pcm and not send(pcm):
                        break
                if not stopped.is_set():
                    if resampler:
                        tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                        send((np.clip(tail, -1, 1) * 32767).astype("<i2").tobytes())
                    send(None)
            except Exception:
                logger.error("Pocket TTS synthesis failed", exc_info=False)
                send(RuntimeError("Synthesis failed"))
            finally:
                try:
                    if iterator is not None:
                        iterator.close()
                finally:
                    finish("producer")

        async def receive():
            while time.monotonic() < deadline:
                if await request.is_disconnected():
                    raise asyncio.CancelledError
                try:
                    item = await asyncio.to_thread(chunks.get, True, 0.1)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise item
                return item
            raise TimeoutError("Synthesis timed out")

        threading.Thread(target=produce, name="pocket-synthesis", daemon=True).start()
        try:
            first = await receive()
            if not first:
                raise RuntimeError("No audio generated")
        except BaseException as exc:
            response_done()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(503, "Pocket TTS could not generate audio") from None

        async def audio():
            item = first
            while item is not None:
                yield item
                item = await receive()

        return AudioResponse(
            audio(),
            response_done,
            media_type="audio/pcm",
            headers={"Cache-Control": "no-store", "X-Audio-Sample-Rate": str(body.sample_rate)},
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(args.device), host="0.0.0.0", port=args.port)
