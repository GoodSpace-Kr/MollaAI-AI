from __future__ import annotations

from fastapi import FastAPI

from .config import SttConfig
from .engine import NemoAsrAdapter
from .service import STTService
from .transport import create_stt_router


def _service_factory() -> STTService:
    config = SttConfig.from_env()
    adapter = NemoAsrAdapter(config)
    return STTService(config=config, adapter=adapter)


app = FastAPI(title="STT Service")
app.include_router(create_stt_router(service_factory=_service_factory))
