"""Optional real-model regression against the source-built Speaches service."""

import asyncio
import os

import httpx
import numpy as np
import pytest

URL = os.getenv("SPEACHES_TEST_BASE_URL")
KEY = os.getenv("SPEACHES_TEST_API_KEY")
pytestmark = pytest.mark.skipif(not URL or not KEY, reason="No live Speaches test service configured")


@pytest.mark.asyncio
async def test_real_model_concurrent_audio_matches_each_isolated_caller():
    texts = [
        "Merhaba, size nasıl yardımcı olabilirim?",
        "Siparişiniz yola çıktı.",
        "Randevunuz onaylandı.",
        "Bugün hava çok güzel.",
        "Ödemeniz hesabınıza yatırıldı.",
        "İyi günler dileriz.",
        "Verasist sesli asistan hizmetine hoş geldiniz.",
        "Teşekkür ederim, görüşmek üzere.",
    ]
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {KEY}"}) as client:

        async def speech(index, text):
            response = await client.post(
                f"{URL.rstrip('/')}/audio/speech",
                json={
                    "model": "canberkkkkkk/ema-lightning",
                    "voice": "default",
                    "response_format": "pcm",
                    "input": text,
                    "seed": index,
                    "sample_rate": 24000,
                },
            )
            response.raise_for_status()
            audio = np.frombuffer(response.content, dtype="<i2").astype(np.float64)
            assert audio.size > 4800
            return audio

        isolated = [await speech(i, text) for i, text in enumerate(texts)]
        concurrent = await asyncio.gather(*[speech(i, text) for i, text in enumerate(texts)])
    for alone, shared in zip(isolated, concurrent):
        assert alone.shape == shared.shape
        # Batched PyTorch arithmetic can differ slightly; detect mixing or drift.
        assert np.corrcoef(alone, shared)[0, 1] > 0.9999
        assert np.max(np.abs(alone - shared)) <= 16


@pytest.mark.asyncio
async def test_real_stt_distinct_callers_and_speech_formats():
    import io
    import wave

    texts = [
        "Bugün hava çok güzel.",
        "Siparişiniz yarın gelecek.",
        "Saat dokuzda toplantımız var.",
        "İstanbul çok kalabalık bir şehir.",
    ]
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {KEY}"}) as client:
        samples = []
        for text in texts:
            r = await client.post(
                f"{URL.rstrip('/')}/audio/speech",
                json={
                    "model": "canberkkkkkk/ema-lightning",
                    "voice": "default",
                    "input": text,
                    "seed": 0,
                    "response_format": "wav",
                    "sample_rate": 16000,
                },
            )
            r.raise_for_status()
            with wave.open(io.BytesIO(r.content)) as wav:
                assert wav.getframerate() == 16000
            samples.append(r.content)

        async def transcribe(data):
            r = await client.post(
                f"{URL.rstrip('/')}/audio/transcriptions",
                files={"file": ("audio.wav", data, "audio/wav")},
                data={"model": "mihuai/turkish-stt", "language": "tr", "response_format": "verbose_json"},
            )
            r.raise_for_status()
            assert r.headers["x-attribution"] == "Powered by mihu"
            return r.json()["text"]

        expected = [await transcribe(sample) for sample in samples]
        assert all(expected) and len(set(expected)) == 4
        assert await asyncio.gather(*(transcribe(sample) for sample in samples)) == expected
        invalid = await client.post(
            f"{URL.rstrip('/')}/audio/transcriptions",
            files={"file": ("bad.wav", b"not audio")},
            data={"model": "mihuai/turkish-stt"},
        )
        assert invalid.status_code == 422
