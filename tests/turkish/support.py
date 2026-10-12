from contextlib import asynccontextmanager
from unittest.mock import patch
from pydantic import SecretStr
from speaches.config import Config
from speaches.dependencies import get_config
from speaches.main import create_app as create_speaches_app
from speaches.executors.turkish import stt as turkish_stt
from speaches.executors.turkish.runtime import MODEL, Settings as RuntimeSettings, TurkishRuntime


class Settings(RuntimeSettings):
    api_key: str = "test-key"


def create_app(settings, model_factory, stt_factory=lambda: object()):
    cfg = Config(enable_ui=False, enable_turkish_models=True, api_key=SecretStr(settings.api_key))
    with patch("speaches.main.get_config", return_value=cfg):
        app = create_speaches_app()
    app.dependency_overrides[get_config] = lambda: cfg

    @asynccontextmanager
    async def lifespan(app):
        runtime = TurkishRuntime(settings, model_factory(settings), stt_factory())
        app.state.turkish = runtime
        with patch("speaches.executors.turkish.catalog.get_config", return_value=cfg):
            try:
                yield
            finally:
                await runtime.close()

    app.router.lifespan_context = lifespan
    return app
