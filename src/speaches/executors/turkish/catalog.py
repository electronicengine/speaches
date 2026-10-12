from speaches.api_types import Model, ModelTask
from speaches.dependencies import get_config

EMA_ID = "canberkkkkkk/ema-lightning"
MIHU_ID = "mihuai/turkish-stt"


def models(task: ModelTask | None = None) -> list[Model]:
    if not get_config().enable_turkish_models:
        return []
    items = [
        Model(
            id="duxx/turkish-stt-zipformer",
            owned_by="duxx",
            task="automatic-speech-recognition",
            language=["tr"],
            attribution="duxx — CC BY-NC 4.0 (non-commercial only)",
            license_url="https://huggingface.co/duxx/turkish-stt-zipformer",
        ),
        Model(
            id=EMA_ID,
            owned_by="canberkkkkkk",
            task="text-to-speech",
            language=["tr"],
            sample_rate=24000,
            voices=[{"id": "default", "name": "default", "language": "tr"}],
        ),
        Model(
            id=MIHU_ID,
            owned_by="mihuai",
            task="automatic-speech-recognition",
            language=["tr"],
            attribution="Powered by mihu",
            license_url="https://huggingface.co/mihuai/turkish-stt/blob/main/LICENSE",
        ),
    ]
    return [item for item in items if task is None or item.task == task]
