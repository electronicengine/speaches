import asyncio
import threading
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pytest

from .support import MODEL, Settings, create_app

HEADERS = {"Authorization": "Bearer test-key"}


class Model:
    def __init__(self):
        self.calls = []
        self.closed = []

    def stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        try:
            for _ in range(3):
                yield np.array([int(text) / 100], dtype=np.float32)
        finally:
            self.closed.append(text)


@asynccontextmanager
async def client_for(model, **kwargs):
    app = create_app(Settings(api_key="test-key", **kwargs), lambda _: model)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ema") as client:
            yield app, client


@pytest.mark.asyncio
async def test_concurrent_users_receive_only_their_own_pcm():
    model = Model()
    async with client_for(model, max_streams=32) as (_, client):
        responses = await asyncio.gather(
            *[
                client.post(
                    "/v1/audio/speech",
                    headers=HEADERS,
                    json={
                        "model": MODEL,
                        "voice": "default",
                        "response_format": "pcm",
                        "input": str(i),
                        "seed": i,
                        "speed": 1.25,
                        "sample_rate": 8000,
                    },
                )
                for i in range(1, 33)
            ]
        )
    for i, response in enumerate(responses, 1):
        assert response.status_code == 200
        expected = (np.array([i / 100], dtype=np.float32) * 32767).astype("<i2").tobytes()
        assert response.content == expected * 3
        assert response.headers["x-audio-sample-rate"] == "8000"
        assert response.headers["cache-control"] == "no-store"
    assert len(model.closed) == 32
    assert all(options["sample_rate"] == 8000 and options["speed"] == 1.25 for _, options in model.calls)


@pytest.mark.asyncio
async def test_auth_model_discovery_and_validation_do_not_run_inference(monkeypatch):
    for kind in ("kokoro", "piper", "whisper"):
        monkeypatch.setattr(f"speaches.routers.models.{kind}_model_registry.list_local_models", lambda: iter(()))
    model = Model()
    async with client_for(model) as (_, client):
        assert (
            await client.post(
                "/v1/audio/speech", json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": "1"}
            )
        ).status_code == 403
        assert (await client.get("/v1/models")).status_code == 403
        data = next(item for item in (await client.get("/v1/models", headers=HEADERS)).json()["data"] if item["id"] == MODEL)
        assert data["id"] == MODEL
        assert data["voices"] == [{"id": "default", "name": "default", "language": "tr"}]
        for body in [
            {"input": " "},
            {"input": "x" * 4097},
            {"input": "1", "voice": "other"},
            {"input": "1", "sample_rate": 22050},
            {"input": "1", "speed": 5},
        ]:
            assert (
                await client.post(
                    "/v1/audio/speech",
                    json={"model": MODEL, "voice": "default", "response_format": "pcm", **body},
                    headers=HEADERS,
                )
            ).status_code == 422
    assert not model.calls


@pytest.mark.asyncio
async def test_capacity_reserved_until_timed_out_worker_exits():
    started, finish = threading.Event(), threading.Event()

    class Slow(Model):
        def stream(self, text, **kwargs):
            started.set()
            finish.wait(3)
            yield np.zeros(4)

    async with client_for(Slow(), max_streams=1, timeout=0.2) as (app, client):
        task = asyncio.create_task(
            client.post(
                "/v1/audio/speech",
                headers=HEADERS,
                json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": "1"},
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        rejected = await client.post(
            "/v1/audio/speech",
            headers=HEADERS,
            json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": "2"},
        )
        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "1"
        assert (await task).status_code == 504
        assert app.state.turkish.active == 1
        finish.set()
        for _ in range(100):
            if app.state.turkish.active == 0:
                break
            await asyncio.sleep(0.01)
        assert app.state.turkish.active == 0
        assert (
            await client.post(
                "/v1/audio/speech",
                headers=HEADERS,
                json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": "3"},
            )
        ).status_code == 200


@pytest.mark.asyncio
async def test_failure_is_sanitized_and_other_caller_survives():
    class Faulty(Model):
        def stream(self, text, **kwargs):
            if text == "secret-text":
                raise ValueError("secret-text and private model path")
            yield from super().stream(text, **kwargs)

    async with client_for(Faulty()) as (_, client):
        bad, good = await asyncio.gather(
            *[
                client.post(
                    "/v1/audio/speech",
                    headers=HEADERS,
                    json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": text},
                )
                for text in ["secret-text", "2"]
            ]
        )
        assert bad.status_code == 500
        assert "secret-text" not in bad.text
        assert good.status_code == 200


@pytest.mark.asyncio
async def test_disconnect_closes_producer_and_reclaims_capacity():
    # Raw ASGI permits a disconnect while StreamingResponse is still sending.
    disconnected = asyncio.Event()

    class LongModel(Model):
        def stream(self, text, **kwargs):
            try:
                for _ in range(10000):
                    self.calls.append(text)
                    yield np.zeros(4)
            finally:
                self.closed.append(text)

    model = LongModel()
    app = create_app(Settings(api_key="test-key", max_streams=1), lambda _: model)
    body = b'{"model":"canberkkkkkk/ema-lightning","voice":"default","response_format":"pcm","input":"1"}'
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            disconnected.set()
            await asyncio.sleep(0.1)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer test-key"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("ema", 8000),
    }
    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
        for _ in range(100):
            if app.state.turkish.active == 0:
                break
            await asyncio.sleep(0.01)
        assert app.state.turkish.active == 0
        assert model.closed == ["1"]
        assert len(model.calls) < 10000


@pytest.mark.asyncio
async def test_slow_client_holds_capacity_and_response_deadline_releases_it():
    class Single(Model):
        def stream(self, text, **kwargs):
            try:
                yield np.zeros(4)
            finally:
                self.closed.append(text)

    model = Single()
    blocked = asyncio.Event()
    never = asyncio.Event()
    delivered = False
    app = create_app(Settings(api_key="test-key", max_streams=1, timeout=0.3), lambda _: model)

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {
                "type": "http.request",
                "body": b'{"model":"canberkkkkkk/ema-lightning","voice":"default","response_format":"pcm","input":"1"}',
                "more_body": False,
            }
        await never.wait()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            blocked.set()
            await never.wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer test-key"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("ema", 8000),
    }
    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(blocked.wait(), 1)
        for _ in range(100):
            if model.closed:
                break
            await asyncio.sleep(0.001)
        assert model.closed == ["1"]
        assert app.state.turkish.active == 1
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ema") as client:
            assert (
                await client.post(
                    "/v1/audio/speech",
                    headers=HEADERS,
                    json={"model": MODEL, "voice": "default", "response_format": "pcm", "input": "2"},
                )
            ).status_code == 429
        with pytest.raises(TimeoutError):
            await task
        assert app.state.turkish.active == 0
