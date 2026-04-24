from __future__ import annotations

from fastapi import FastAPI

from module.stt import NemoAsrAdapter, STTService, SttConfig, create_stt_router


def _service_factory() -> STTService:
    config = SttConfig.from_env()
    adapter = NemoAsrAdapter(config)
    return STTService(config=config, adapter=adapter)

app = FastAPI(title="STT Service")
app.include_router(create_stt_router(service_factory=_service_factory))
