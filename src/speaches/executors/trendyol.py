from speaches.api_types import Model, ModelTask
from speaches.config import Config

MODEL = "Trendyol/Trendyol-TTS"


def models(config: Config, task: ModelTask | None = None) -> list[Model]:
    if not config.trendyol_tts_url or task not in (None, "text-to-speech"):
        return []
    return [
        Model(
            id=MODEL,
            owned_by="Trendyol",
            task="text-to-speech",
            language=["tr"],
            sample_rate=48000,
            voices=[{"id": "default", "name": "default", "language": "tr"}],
            attribution="Trendyol TTS, based on OpenBMB VoxCPM2",
            license_url="https://huggingface.co/Trendyol/Trendyol-TTS#license",
        )
    ]
