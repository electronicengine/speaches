from speaches.api_types import Model, ModelTask
from speaches.dependencies import get_config
from speaches.executors import pocket, trendyol
from speaches.executors.turkish.stt import AYAZ_LANGUAGES, AYAZ_MODEL

EMA_ID = "canberkkkkkk/ema-lightning"
MIHU_ID = "mihuai/turkish-stt"


def models(task: ModelTask | None = None) -> list[Model]:
    if not get_config().enable_turkish_models:
        return pocket.models(get_config(), task) + trendyol.models(get_config(), task)
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
    if get_config().enable_ayaz_stt:
        items.append(
            Model(
                id=AYAZ_MODEL,
                owned_by="VeliAyaz",
                task="automatic-speech-recognition",
                language=AYAZ_LANGUAGES,
                attribution="Ayaz ASR by Veli Ayaz; NVIDIA Nemotron base model",
                license_url=f"https://huggingface.co/{AYAZ_MODEL}/blob/main/LICENSE",
            )
        )
    return (
        [item for item in items if task is None or item.task == task]
        + pocket.models(get_config(), task)
        + trendyol.models(get_config(), task)
    )
