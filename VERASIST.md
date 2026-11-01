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

## Ayaz ASR

Set `ENABLE_TURKISH_MODELS=true` and `ENABLE_AYAZ_STT=true` to expose
`VeliAyaz/ayaz-asr-tr-streaming-0.6b` in the same STT catalog and endpoint.
The INT8 sherpa-onnx 560 ms export is pinned to revision
`a6dbed373347312475dfc3888a163dae9ac67ffb` and runs on CPU. LICENSE and NOTICE
are downloaded with the weights. The existing Whisper GPU backend is unchanged.

Requests use the existing multipart `/v1/audio/transcriptions` interface with
JSON, verbose JSON or text responses. Each request creates a separate stream;
`language=tr` pins Turkish, another supported language pins that language, and
omitting the language enables automatic detection. The current Verasist adapter
submits completed utterances; partial transcripts during live capture are not
exposed by this integration. The shared STT upload, duration, capacity and
cancellation limits also apply to Ayaz.

## Pocket TTS Turkish on this Windows workstation

`wite-tech/pocket-tts-turkish` is served by `scripts/pocket_tts_server.py` in an
isolated Windows environment, reusing the CUDA PyTorch libraries already in
`H:/workspace/tts-test/.venv`. `scripts/setup-pocket.ps1` prepares this environment
without changing that shared runtime. The model is downloaded to
`downloads/pocket-tts-turkish`, pinned to revision
`fcbd37c74dc127be846face0ff36062e9a939f7b`; code packages are pinned in
`scripts/requirements-pocket.txt`. Model attribution is CC BY 4.0, Wite Tech and
Kyutai, as recorded in the original downloaded model card and the API catalog.

`POCKET_TTS_URL=http://host.docker.internal:8010` enables the Speaches proxy and
catalog entry. The private worker uses the same API key as Speaches. Run
`scripts/pocket-service.ps1 -Action start -Device cuda` to start it, or use
`scripts/local-service.ps1 start`, which starts it before Docker. Stop/status
actions also include the worker. All six bundled voices are warmed before ready.

Select Local TTS — Speaches and `wite-tech/pocket-tts-turkish` in Verasist,
then `female_1`, `female_2`, `female_3`, `male_1`, `male_2` or `male_3`.
PCM is streamed immediately; WAV/FLAC/MP3 responses buffer for encoding. The
worker resamples continuously to 8/16/24/48 kHz and supports `speed=1`.
One synthesis runs at a time; concurrent requests return 429 rather than mixing
audio. Disconnects stop synthesis and release admission after the producer exits.

## Trendyol TTS

`TRENDYOL_TTS_URL=http://host.docker.internal:8011` exposes `Trendyol/Trendyol-TTS`
with voice `default`. The native worker reuses the existing standalone merged
weights in `H:/workspace/tts-test/models/Trendyol-TTS` and VoxCPM 2.0.3; no LoRA
merging or second download is necessary. Start/status/stop through
`scripts/pocket-service.ps1 -Engine trendyol -Action start` or the main
`scripts/local-service.ps1` commands. The same private service key is required.

Inference runs in BF16 on CUDA with cfg=2.0, 16 inference steps, denoising off,
and the upstream text normalizer. Native 48 kHz audio is resampled continuously
to the requested output rate. Streaming PCM and buffered WAV/FLAC/MP3 use the
same Speaches endpoint. Speed currently supports 1.0 only.

This model needs about 5.5 GB of GPU memory on this workstation. Its parameters
are parked in RAM and its GPU KV caches released after every generation. The
Speaches Trendyol route unloads idle Whisper models before starting synthesis;
active requests are preserved. This permits sequential GPU STT/TTS in one
pipeline but does not guarantee enough VRAM for concurrent GPU calls. The
worker's one-request admission limit also applies. The tested eager Windows
backend is substantially slower than real time; Pocket remains the lower
latency TTS option. Triton compilation is unavailable in the current runtime.
