# Verasist speech deployment

This checkout extends upstream revision `7dc79ccfa35e5eeaf5920f8e88119ac3f5e34846`,
matching the deployed Speaches API/dependencies. Existing deployment fixes for
Piper catalog entries and RTC are retained. Dependencies are locked in `uv.lock`;
PyAV 14.2.0 is pinned because 14.4.0 no longer provides Linux Python 3.12 wheels.
The image builds from Ubuntu and this source checkout, without a prebuilt
Speaches application image.

Run from the Verasist repository root:

```sh
export SPEACHES_API_KEY='your-private-service-key'
docker compose -f docker-compose-models.yaml build speaches
docker compose -f docker-compose-models.yaml up -d speaches piper-importer whisper-importer kokoro-importer
```

The existing Speaches model volume retains Piper, Kokoro and Whisper files.
`ENABLE_TURKISH_MODELS=true` loads EMA and mihu into the same Speaches process.
They use the same API key, port, model catalog and audio endpoints as the other
Speaches models. No separate EMA server/container is needed.

- TTS: `canberkkkkkk/ema-lightning`, voice `default`, Turkish. PCM, WAV, FLAC or
  MP3; sample rates 8000, 16000, 24000, 48000; speed 0.25–4; optional seed.
- STT: `mihuai/turkish-stt`, language `tr`. Multipart `file`; JSON, verbose JSON
  or text responses. Maximum file size 16 MiB and decoded duration 120 seconds.
  Verasist submits completed utterances using its existing Speaches STT adapter.
- Admission: `TURKISH__MAX_STREAMS=16`, `TURKISH__STT_MAX_STREAMS=4`,
  `TURKISH__TIMEOUT=120`. Excess requests return 429 with Retry-After. Inference
  streams/buffers are private to each request. Cancellation retains the capacity
  slot until the worker has stopped; slow TTS consumers apply backpressure.
- The models are resident while enabled. Catalog deletion is refused while
  active. Stop the service before removing their cached weights.

In Verasist choose **Local TTS — Speaches** and **Local STT — Speaches** in the
model package builder. STT defaults to automatic language detection (`unknown`)
with the multilingual Whisper base model. Select a language explicitly when
needed; mihu supports Turkish only, so automatic selection resolves to `tr`.
Earlier `ema_lightning` and `mihu_stt` provider IDs remain
compatible and resolve to the same Speaches endpoint through platform credentials.

## Attribution and model terms

**Powered by mihu** applies to `mihuai/turkish-stt`. Its full Community License is
preserved in `NOTICE.mihu`. Combined annual revenue of USD 500,000 or more requires
a separate commercial agreement. The catalog and Verasist provider description
also expose the attribution and license link.

## Focused verification

```sh
uv sync --frozen --extra dev --extra ui
uv run --frozen --extra dev --extra ui pytest tests/turkish -q
```

Production migration order: build; copy the two Turkish model caches into the
existing Speaches volume; verify a temporary candidate; replace Speaches; update
platform credentials for Speaches and legacy provider IDs; verify Verasist;
remove the separate EMA container and deployment files. Keep the previous image
and private credential snapshot until the verification passes.

Turkish STT also includes `duxx/turkish-stt-zipformer`, pinned to
`cebec199e0a9a1bdcd195b9124922b4a33e98bc8`, INT8, modified beam search
with 12 paths. Its CC BY-NC 4.0 license permits non-commercial use only.
It shares the bounded STT pool but uses its own recognizer and private streams.
