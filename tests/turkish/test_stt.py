import asyncio
import threading
import time

import httpx
import pytest

from . import support as server

HEADERS = {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_ayaz_routes_to_its_recognizer_and_keeps_language_per_request(monkeypatch):
    ayaz_recognizer = object()
    calls = []

    def transcribe(recognizer, audio, stopped, *, language=None):
        assert recognizer is ayaz_recognizer
        calls.append((audio.decode(), language))
        return audio.decode(), 1.0

    monkeypatch.setattr(server.turkish_stt, "transcribe", transcribe)
    app = server.create_app(server.Settings(), lambda _: object())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

            async def call(text, language=None):
                data = {"model": server.turkish_stt.AYAZ_MODEL, "response_format": "verbose_json"}
                if language is not None:
                    data["language"] = language
                return await client.post(
                    "/v1/audio/transcriptions", headers=HEADERS, files={"file": ("a.wav", text.encode())}, data=data
                )

            assert (await call("disabled")).status_code == 503
            app.state.turkish.ayaz_recognizer = ayaz_recognizer
            responses = await asyncio.gather(call("turkish", "tr"), call("english", "en"), call("auto"))
            assert [r.json()["text"] for r in responses] == ["turkish", "english", "auto"]
            assert [r.json()["language"] for r in responses] == ["tr", "en", "unknown"]
            assert all(r.headers["x-attribution"] == "Ayaz ASR by Veli Ayaz" for r in responses)
            assert set(calls) == {("turkish", "tr"), ("english", "en"), ("auto", None)}
            assert (await call("invalid", "not-a-language")).status_code == 422
            assert app.state.turkish.stt_active == 0


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
