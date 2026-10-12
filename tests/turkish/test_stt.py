import asyncio
import threading
import time

import httpx
import pytest
from . import support as server

HEADERS = {"Authorization": "Bearer test-key"}


@pytest.mark.parametrize("model", [server.turkish_stt.MODEL, server.turkish_stt.DUXX_MODEL])
@pytest.mark.asyncio
async def test_isolated_concurrent_transcripts_and_validation(monkeypatch, model):
    def transcribe(recognizer, audio, stopped):
        time.sleep(0.03)
        return audio.decode(), 1.0

    monkeypatch.setattr(server.turkish_stt, "transcribe", transcribe)
    app = server.create_app(server.Settings(api_key="test-key"), lambda _: object(), lambda: object())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

            async def call(i):
                return await client.post(
                    "/v1/audio/transcriptions",
                    headers=HEADERS,
                    files={"file": ("a.wav", str(i).encode())},
                    data={"model": model},
                )

            responses = await asyncio.gather(*(call(i) for i in range(4)))
            assert [r.json()["text"] for r in responses] == ["0", "1", "2", "3"]
            assert all(r.headers["cache-control"] == "no-store" for r in responses)
            assert (await client.post("/v1/audio/transcriptions")).status_code == 403
            assert (
                await client.post(
                    "/v1/audio/transcriptions",
                    headers=HEADERS,
                    files={"file": ("a.wav", b"x")},
                    data={"model": model, "language": "en"},
                )
            ).status_code == 422
            assert app.state.turkish.stt_active == 0


@pytest.mark.asyncio
async def test_timed_out_worker_holds_capacity(monkeypatch):
    gate = threading.Event()

    def transcribe(recognizer, audio, stopped):
        gate.wait(2)
        return "done", 1.0

    monkeypatch.setattr(server.turkish_stt, "transcribe", transcribe)
    app = server.create_app(
        server.Settings(api_key="test-key", timeout=0.08),
        lambda _: object(),
        lambda: object(),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

            async def call():
                return await client.post(
                    "/v1/audio/transcriptions",
                    headers=HEADERS,
                    files={"file": ("a.wav", b"x")},
                    data={"model": server.turkish_stt.MODEL},
                )

            try:
                responses = await asyncio.gather(*(call() for _ in range(4)))
                assert all(r.status_code == 504 for r in responses)
                assert app.state.turkish.stt_active == 4
                assert (await call()).status_code == 429
            finally:
                gate.set()
            await asyncio.sleep(0.05)
            assert app.state.turkish.stt_active == 0
