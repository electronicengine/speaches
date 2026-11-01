"""Trendyol VoxCPM2 worker; park weights in RAM between GPU requests."""

import gc
from pathlib import Path

from pocket_tts_server import create_app
import torch
import uvicorn
from voxcpm import VoxCPM

MODEL_ID = "Trendyol/Trendyol-TTS"
MODEL_PATH = Path("H:/workspace/tts-test/models/Trendyol-TTS")


class TrendyolEngine:
    def __init__(self, device):
        if device != "cuda":
            raise ValueError("This Trendyol worker requires CUDA")
        self.model = VoxCPM.from_pretrained(
            str(MODEL_PATH), load_denoiser=False, optimize=False, device=device, local_files_only=True
        )
        self.sample_rate = self.model.tts_model.sample_rate

    def stream(self, text, voice="default", seed=None):
        model = self.model.tts_model
        try:
            model.to("cuda")
            for lm in (model.base_lm, model.residual_lm):
                if lm.kv_cache is None:
                    lm.setup_cache(1, model.config.max_length, "cuda", next(lm.parameters()).dtype)
            torch.manual_seed(42 if seed is None else seed)
            yield from self.model.generate_streaming(
                text=text,
                cfg_value=2.0,
                inference_timesteps=16,
                max_len=1024,
                normalize=True,
                denoise=False,
                retry_badcase=False,
            )
        finally:
            model.to("cpu")
            model.base_lm.kv_cache = None
            model.residual_lm.kv_cache = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            args.device, model_id=MODEL_ID, voices=("default",), loader=TrendyolEngine, worker_name="Trendyol TTS"
        ),
        host="0.0.0.0",
        port=8011,
    )
