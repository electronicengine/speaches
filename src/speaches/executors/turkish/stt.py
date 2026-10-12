# ruff: noqa: EM101
from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from sherpa_onnx import OnlineRecognizer

import numpy as np

MODEL = "mihuai/turkish-stt"
REVISION = "4f868b7a4a7cfaccb97fdaf12cdc8fdfabc8caad"


DUXX_MODEL = "duxx/turkish-stt-zipformer"
DUXX_REVISION = "cebec199e0a9a1bdcd195b9124922b4a33e98bc8"


def load_recognizer(model: str = MODEL) -> OnlineRecognizer:
    from pathlib import Path

    from huggingface_hub import snapshot_download
    import sherpa_onnx

    if model == DUXX_MODEL:
        names = {part: f"{part}-epoch-1-avg-1-chunk-32-left-128.int8.onnx" for part in ("encoder", "decoder", "joiner")}
        revision = DUXX_REVISION
    else:
        names = {"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.int8.onnx"}
        revision = REVISION
    root = Path(
        snapshot_download(
            model, revision=revision, allow_patterns=[*names.values(), "tokens.txt", "LICENSE", "README.md"]
        )
    )
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(root / "tokens.txt"),
        encoder=str(root / names["encoder"]),
        decoder=str(root / names["decoder"]),
        joiner=str(root / names["joiner"]),
        num_threads=1,
        decoding_method="modified_beam_search",
        max_active_paths=12 if model == DUXX_MODEL else 4,
        provider="cpu",
        enable_endpoint_detection=False,
    )


def transcribe(recognizer: OnlineRecognizer, audio: bytes, stopped: threading.Event) -> tuple[str, float]:
    import soundfile as sf
    import soxr

    try:
        with sf.SoundFile(io.BytesIO(audio)) as file:
            if not 8000 <= file.samplerate <= 192000 or file.channels > 2:
                raise ValueError("Unsupported audio rate or channels")
            duration = len(file) / file.samplerate
            if not 0 < duration <= 120:
                raise ValueError("Audio must contain at most 120 seconds")
            samples = file.read(dtype="float32", always_2d=True).mean(axis=1)
            rate = file.samplerate
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Invalid audio file") from exc
    if not np.isfinite(samples).all():
        raise ValueError("Invalid audio samples")
    if stopped.is_set():
        return "", duration
    if rate != 16000:
        samples = soxr.resample(samples, rate, 16000)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        if stopped.is_set():
            return "", duration
        recognizer.decode_stream(stream)
    result = recognizer.get_result(stream)
    return result if isinstance(result, str) else result.text, duration
