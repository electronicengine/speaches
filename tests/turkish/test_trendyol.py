import threading

from speaches.config import Config, WhisperConfig
from speaches.executors import trendyol
from speaches.executors.whisper.model_manager import WhisperModelManager
from speaches.model_manager import SelfDisposingModel


def test_reclaiming_idle_whisper_models_preserves_active_requests_without_deadlock():
    manager = WhisperModelManager(WhisperConfig(ttl=-1))
    for name in ("idle", "active"):
        manager.loaded_models[name] = SelfDisposingModel(
            name, load_fn=object, ttl=-1, model_unloaded_callback=manager._handle_model_unloaded
        )
    with manager.loaded_models["idle"]:
        pass
    with manager.loaded_models["active"]:
        worker = threading.Thread(target=manager.unload_idle_models, daemon=True)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert set(manager.loaded_models) == {"active"}
        assert manager.loaded_models["active"].ref_count == 1


def test_trendyol_discovery_is_separate_from_pocket():
    assert trendyol.models(Config(trendyol_tts_url=None)) == []
    config = Config(trendyol_tts_url="http://trendyol-worker")
    assert trendyol.models(config, "automatic-speech-recognition") == []
    (model,) = trendyol.models(config, "text-to-speech")
    assert model.id == "Trendyol/Trendyol-TTS"
    assert model.sample_rate == 48000
    assert model.voices == [{"id": "default", "name": "default", "language": "tr"}]
