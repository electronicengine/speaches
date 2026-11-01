from __future__ import annotations

import asyncio
import concurrent.futures
from functools import partial
import logging
import threading
import time
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from ema_lightning import EMA
    from sherpa_onnx import OnlineRecognizer
    from starlette.types import Receive, Scope, Send

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from speaches.executors.turkish import stt

MODEL = "canberkkkkkk/ema-lightning"
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TURKISH__")
    device: str = "cpu"
    max_streams: int = Field(default=16, ge=1)
    stt_max_streams: int = Field(default=4, ge=1)
    timeout: float = Field(default=120, gt=0)


class SpeechRequest(BaseModel):
    model: Literal[MODEL] = MODEL
    voice: Literal["default"] = "default"
    input: str = Field(min_length=1, max_length=4096)
    response_format: Literal["pcm", "wav", "flac", "mp3"] = "pcm"
    speed: float = Field(default=1, ge=0.25, le=4)
    sample_rate: Literal[8000, 16000, 24000, 48000] = 24000
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class CancellableStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        on_complete: Callable[[], None],
        timeout: float,
        media_type: str,
        headers: dict[str, str],
    ) -> None:
        super().__init__(content, media_type=media_type, headers=headers)
        self.on_complete = on_complete
        self.timeout = timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await asyncio.wait_for(super().__call__(scope, receive, send), self.timeout)
        finally:
            self.on_complete()


class TurkishRuntime:
    def __init__(
        self,
        settings: Settings,
        model: EMA,
        recognizer: OnlineRecognizer,
        duxx_recognizer: OnlineRecognizer | None = None,
        ayaz_recognizer: OnlineRecognizer | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.recognizer = recognizer
        self.duxx_recognizer = duxx_recognizer
        self.ayaz_recognizer = ayaz_recognizer
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_streams)
        self.stt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.stt_max_streams)
        self.active = 0
        self.stt_active = 0
        self.stops: set[threading.Event] = set()

    @classmethod
    def load(cls) -> TurkishRuntime:
        from ema_lightning import EMA

        settings = Settings()
        model = EMA(device=settings.device)
        logger.info("Warming up EMA before accepting requests")
        started = time.monotonic()
        # The first stream benchmarks batch sizes. Complete it before readiness
        # so the first caller does not pay for calibration and lazy setup.
        for _ in model.stream("Merhaba.", seed=0, sample_rate=24000):
            pass
        logger.info("EMA warm-up completed in %.2f seconds", time.monotonic() - started)
        from speaches.dependencies import get_config

        ayaz = stt.load_recognizer(stt.AYAZ_MODEL) if get_config().enable_ayaz_stt else None
        return cls(settings, model, stt.load_recognizer(), stt.load_recognizer(stt.DUXX_MODEL), ayaz)

    async def close(self) -> None:
        for stop in self.stops:
            stop.set()
        await asyncio.to_thread(self.stt_executor.shutdown, wait=True, cancel_futures=True)
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)

    async def transcription(  # noqa: C901, PLR0912
        self,
        request: Request,
        file: UploadFile,
        language: str | None,
        response_format: str,
        stream: bool = False,
        model: str = stt.MODEL,
    ) -> Response:
        allowed_languages = (None, *stt.AYAZ_LANGUAGES) if model == stt.AYAZ_MODEL else (None, "tr")
        if language not in allowed_languages or response_format not in {"json", "verbose_json", "text"} or stream:
            raise HTTPException(422, "Unsupported STT language or format; use non-streaming json, verbose_json or text")
        recognizer = self.recognizer
        attribution = "Powered by mihu"
        transcribe = stt.transcribe
        if model == stt.DUXX_MODEL:
            recognizer, attribution = self.duxx_recognizer, "duxx"
        elif model == stt.AYAZ_MODEL:
            recognizer, attribution = self.ayaz_recognizer, "Ayaz ASR by Veli Ayaz"
            transcribe = partial(stt.transcribe, language=language)
            if recognizer is None:
                raise HTTPException(503, "Ayaz STT is disabled")
        if file.size is not None and file.size > 16 * 1024 * 1024:
            raise HTTPException(413, "Audio upload exceeds 16 MiB")
        if self.stt_active >= self.settings.stt_max_streams:
            raise HTTPException(429, "STT capacity reached", headers={"Retry-After": "1"})
        self.stt_active += 1
        stopped = threading.Event()
        self.stops.add(stopped)
        future = None

        def release(_: asyncio.Future | None = None) -> None:
            self.stt_active -= 1
            self.stops.discard(stopped)
            if future is not None and not future.cancelled():
                future.exception()

        try:
            async with asyncio.timeout(self.settings.timeout):
                data = await file.read(16 * 1024 * 1024 + 1)
                if len(data) > 16 * 1024 * 1024:
                    raise HTTPException(413, "Audio upload exceeds 16 MiB")
                future = asyncio.get_running_loop().run_in_executor(
                    self.stt_executor,
                    transcribe,
                    recognizer,
                    data,
                    stopped,
                )
                future.add_done_callback(release)
                while not future.done():
                    if await request.is_disconnected():
                        raise HTTPException(499, "Caller disconnected")
                    await asyncio.sleep(0.02)
                text, duration = future.result()
                headers = {
                    "Cache-Control": "no-store",
                    "X-Attribution": attribution,
                }
                if response_format == "text":
                    return PlainTextResponse(text, headers=headers)
                result = {"text": text}
                if response_format == "verbose_json":
                    result.update(
                        language=language or ("unknown" if model == stt.AYAZ_MODEL else "tr"),
                        duration=duration,
                        task="transcribe",
                        segments=[],
                    )
                return JSONResponse(result, headers=headers)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except TimeoutError:
            raise HTTPException(504, "Transcription timed out") from None
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 - sanitize errors at the inference boundary
            logger.error("STT request failed")  # noqa: TRY400 - upstream errors can contain caller audio/text
            raise HTTPException(500, "Transcription failed") from None
        finally:
            stopped.set()
            if future is None:
                release()

    async def speech(self, body: SpeechRequest, request: Request) -> StreamingResponse:  # noqa: C901, PLR0915
        if not body.input.strip():
            raise HTTPException(422, "Text must not be blank")
        cfg = self.settings
        # No await between check and reservation: all admission is on this event loop.
        if self.active >= cfg.max_streams:
            raise HTTPException(429, "TTS capacity reached", headers={"Retry-After": "1"})
        self.active += 1
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=2)
        stopped = threading.Event()
        self.stops.add(stopped)
        deadline = loop.time() + cfg.timeout

        def send(item: bytes | Exception | None) -> bool:
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stopped.is_set():
                try:
                    future.result(timeout=0.05)
                    return True
                except concurrent.futures.TimeoutError:
                    pass
            future.cancel()
            return False

        producer_done = False
        response_done = False
        released = False

        def release_if_done() -> None:
            nonlocal released
            if producer_done and response_done and not released:
                released = True
                self.active -= 1
                self.stops.discard(stopped)

        def producer_finished() -> None:
            nonlocal producer_done
            producer_done = True
            release_if_done()

        def response_finished() -> None:
            nonlocal response_done
            stopped.set()
            response_done = True
            release_if_done()

        def produce() -> None:  # noqa: C901, PLR0912
            iterator = None
            buffered = bytearray()
            try:
                iterator = self.model.stream(
                    body.input,
                    speed=body.speed,
                    seed=body.seed,
                    sample_rate=body.sample_rate,
                )
                for chunk in iterator:
                    if stopped.is_set():
                        break
                    # PCM signed int16 little-endian mono; no shared buffers or files.
                    pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
                    if body.response_format in {"wav", "flac"}:
                        buffered.extend(pcm)
                    else:
                        if body.response_format == "mp3":
                            from speaches.audio import convert_audio_format

                            pcm = convert_audio_format(pcm, body.sample_rate, "mp3")
                        if pcm and not send(pcm):
                            break
                if not stopped.is_set():
                    if body.response_format in {"wav", "flac"}:
                        from speaches.audio import convert_audio_format

                        send(convert_audio_format(bytes(buffered), body.sample_rate, body.response_format))
                    send(None)
            except Exception:  # noqa: BLE001 - sanitize errors at the inference boundary
                # Do not log caller text or upstream exceptions containing caller data.
                logger.error("EMA synthesis failed")  # noqa: TRY400 - do not log caller text
                if not stopped.is_set():
                    send(RuntimeError("Synthesis failed"))
            finally:
                try:
                    if iterator is not None:
                        iterator.close()
                finally:
                    loop.call_soon_threadsafe(producer_finished)

        async def receive(*, check_disconnect: bool = False) -> bytes | None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            while True:
                if check_disconnect and await request.is_disconnected():
                    raise asyncio.CancelledError
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    item = await asyncio.wait_for(queue.get(), min(remaining, 0.1))
                    break
                except TimeoutError:
                    continue
            if isinstance(item, Exception):
                raise item
            return item

        self.executor.submit(produce)
        try:
            first = await receive(check_disconnect=True)
        except BaseException as error:
            response_finished()
            if isinstance(error, TimeoutError):
                raise HTTPException(504, "Synthesis timed out") from None
            if isinstance(error, Exception):
                raise HTTPException(500, "Synthesis failed") from None
            raise

        async def audio() -> AsyncIterator[bytes]:
            try:
                item = first
                while item is not None:
                    yield item
                    item = await receive()
            finally:
                # Closing the iterator in its owning thread cancels EMA's scheduler work.
                stopped.set()

        return CancellableStreamingResponse(
            audio(),
            on_complete=response_finished,
            timeout=max(0.001, deadline - loop.time()),
            media_type=f"audio/{body.response_format}",
            headers={
                "Cache-Control": "no-store",
                "X-Audio-Sample-Rate": str(body.sample_rate),
            },
        )
