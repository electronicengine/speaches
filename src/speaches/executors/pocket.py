"""Stream Pocket TTS from the isolated local inference worker."""

from collections.abc import AsyncIterator

from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx

from speaches.api_types import Model, ModelTask
from speaches.config import Config

MODEL = "wite-tech/pocket-tts-turkish"
VOICES = ["female_1", "female_2", "female_3", "male_1", "male_2", "male_3"]


def models(config: Config, task: ModelTask | None = None) -> list[Model]:
    if not config.pocket_tts_url or task not in (None, "text-to-speech"):
        return []
    return [
        Model(
            id=MODEL,
            owned_by="wite-tech",
            task="text-to-speech",
            language=["tr"],
            sample_rate=24000,
            voices=[{"id": voice, "name": voice, "language": "tr"} for voice in VOICES],
            attribution="Pocket TTS Turkish by Wite Tech, derived from Kyutai Pocket TTS (CC BY 4.0)",
            license_url=f"https://huggingface.co/{MODEL}#license-and-attribution",
        )
    ]


async def speech(config: Config, parameters: dict) -> Response:
    return await worker_speech(
        config, parameters, worker_url=config.pocket_tts_url, voices=VOICES, model_name="Pocket TTS"
    )


async def worker_speech(
    config: Config, parameters: dict, *, worker_url: str | None, voices: list[str], model_name: str
) -> Response:
    if not worker_url:
        raise HTTPException(503, f"{model_name} is disabled")
    if parameters.get("voice") not in voices:
        raise HTTPException(422, f"Unknown {model_name} voice")
    if parameters.get("speed", 1.0) != 1.0:
        raise HTTPException(422, f"{model_name} currently supports speed=1 only")
    response_format = parameters.get("response_format", "pcm")
    sample_rate = parameters.get("sample_rate") or 24000
    parameters = {**parameters, "response_format": "pcm", "sample_rate": sample_rate}
    headers = {"Authorization": f"Bearer {config.api_key.get_secret_value()}"} if config.api_key else {}
    client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))
    try:
        request = client.build_request(
            "POST", worker_url.rstrip("/") + "/v1/audio/speech", json=parameters, headers=headers
        )
        upstream = await client.send(request, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(503, f"{model_name} worker is unavailable") from None
    if upstream.status_code != 200:
        status = upstream.status_code if upstream.status_code in (422, 429) else 503
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(status, f"{model_name} worker could not accept the request")

    async def audio() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    result_headers = {"Cache-Control": "no-store", "X-Audio-Sample-Rate": str(sample_rate)}
    if response_format == "pcm":
        return StreamingResponse(audio(), media_type="audio/pcm", headers=result_headers)
    from speaches.audio import convert_audio_format

    pcm = b"".join([chunk async for chunk in audio()])
    return Response(
        convert_audio_format(pcm, sample_rate, response_format),
        media_type=f"audio/{response_format}",
        headers=result_headers,
    )
