import json
from types import SimpleNamespace

from fastapi import HTTPException
import httpx
from pydantic import SecretStr
import pytest

from speaches.config import Config
from speaches.executors import pocket


@pytest.mark.asyncio
async def test_pocket_stream_preserves_pcm_and_forwards_private_credentials(monkeypatch):
    closed = []

    class Audio(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\x01\x00" * 240
            yield b"\x02\x00" * 240

        async def aclose(self):
            closed.append(True)

    def handle(request):
        assert str(request.url) == "http://worker/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer private-key"
        body = json.loads(request.content)
        assert body["voice"] == "male_2" and body["sample_rate"] == 16000
        assert body["response_format"] == "pcm"
        return httpx.Response(200, stream=Audio())

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        pocket,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
            Timeout=httpx.Timeout,
            HTTPError=httpx.HTTPError,
        ),
    )
    config = Config(pocket_tts_url="http://worker", api_key=SecretStr("private-key"))
    response = await pocket.speech(
        config,
        {
            "model": pocket.MODEL,
            "voice": "male_2",
            "input": "Merhaba",
            "sample_rate": 16000,
            "response_format": "pcm",
            "speed": 1,
        },
    )
    parts = [part async for part in response.body_iterator]
    assert parts == [b"\x01\x00" * 240, b"\x02\x00" * 240]
    assert closed and response.headers["x-audio-sample-rate"] == "16000"


@pytest.mark.asyncio
async def test_pocket_rejects_unsupported_voice_before_contacting_worker():
    config = Config(pocket_tts_url="http://worker")
    with pytest.raises(HTTPException) as error:
        await pocket.speech(config, {"voice": "unknown"})
    assert error.value.status_code == 422


def test_pocket_discovery_lists_all_six_voices_only_when_enabled():
    assert pocket.models(Config(pocket_tts_url=None)) == []
    config = Config(pocket_tts_url="http://worker")
    assert pocket.models(config, "automatic-speech-recognition") == []
    (model,) = pocket.models(config, "text-to-speech")
    assert model.id == pocket.MODEL and model.sample_rate == 24000
    assert [voice["id"] for voice in model.voices] == pocket.VOICES
