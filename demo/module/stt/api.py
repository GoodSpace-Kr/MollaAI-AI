from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .config import SttConfig
from .service import STTService
from .types import TranscriptSegment


AudioEncoding = Literal["pcm16", "float32"]


@dataclass(frozen=True, slots=True)
class WebSocketSttOptions:
    # WebSocket STT 엔드포인트의 기본 설정.
    path: str = "/stt/ws"
    default_encoding: AudioEncoding = "pcm16"


def create_stt_router(
    service_factory: Callable[[], STTService] | None = None,
    *,
    options: WebSocketSttOptions | None = None,
) -> APIRouter:
    # STT 전용 WebSocket 라우터를 만든다.
    router = APIRouter()
    opts = options or WebSocketSttOptions()

    def _make_service() -> STTService:
        if service_factory is not None:
            return service_factory()
        return STTService()

    @router.websocket(opts.path)
    async def stt_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        service = _make_service()
        connection_state: dict[str, str] = {"encoding": opts.default_encoding}

        try:
            await _send_ack(websocket, "ready", "STT websocket connected")

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if message.get("text") is not None:
                    try:
                        await _handle_text_message(
                            websocket=websocket,
                            service=service,
                            raw_text=message["text"],
                            connection_state=connection_state,
                        )
                    except Exception as exc:
                        await _send_error(websocket, str(exc))
                    continue

                if message.get("bytes") is not None:
                    payload = message["bytes"]
                    if service.state is None:
                        service.start_session()

                    try:
                        samples = _decode_audio_payload(payload, connection_state["encoding"])
                        result = service.ingest_audio(samples, received_at=time.time())
                        await _emit_result(websocket, result.events)
                    except Exception as exc:
                        await _send_error(websocket, str(exc))
                    continue

                # FastAPI/Starlette가 ping/pong 프레임을 직접 노출하지 않는 경우를 대비한 방어 코드.
                await _send_error(websocket, "Unsupported websocket message")
        except WebSocketDisconnect:
            return

    return router


async def _handle_text_message(
    *,
    websocket: WebSocket,
    service: STTService,
    raw_text: str,
    connection_state: dict[str, str],
) -> None:
    payload = _parse_control_message(raw_text)
    message_type = payload.get("type")

    if message_type == "start":
        config = _extract_config(payload)
        if config is not None:
            _validate_session_config(config, service.config)

        session_id = payload.get("session_id")
        service.start_session(session_id=session_id if isinstance(session_id, str) else None)

        encoding = payload.get("encoding")
        if isinstance(encoding, str) and encoding in ("pcm16", "float32"):
            connection_state["encoding"] = encoding

        await _send_ack(websocket, "started", "STT session started", service.state.session_id if service.state else None)
        return

    if message_type == "finalize":
        result = service.finalize_session(finalized_at=time.time())
        await _emit_result(websocket, result.events)
        await _send_ack(websocket, "finalized", "STT session finalized", service.state.session_id if service.state else None)
        return

    if message_type == "reset":
        service.reset_session()
        await _send_ack(websocket, "reset", "STT session reset")
        return

    if message_type == "ping":
        await _send_ack(websocket, "pong", "pong")
        return

    await _send_error(websocket, f"Unknown control message type: {message_type!r}")


def _parse_control_message(raw_text: str) -> dict:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"type": raw_text.strip()}


def _extract_config(payload: dict) -> SttConfig | None:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        return None

    return SttConfig(
        sample_rate=int(raw_config.get("sample_rate", 16000)),
        channels=int(raw_config.get("channels", 1)),
        chunk_secs=float(raw_config.get("chunk_secs", 2.0)),
        left_context_secs=float(raw_config.get("left_context_secs", 10.0)),
        right_context_secs=float(raw_config.get("right_context_secs", 2.0)),
        min_partial_audio_secs=float(raw_config.get("min_partial_audio_secs", 1.0)),
        partial_update_interval_secs=float(raw_config.get("partial_update_interval_secs", 1.0)),
        max_utterance_secs=float(raw_config.get("max_utterance_secs", 30.0)),
        batch_size=int(raw_config.get("batch_size", 1)),
        model_name=str(raw_config.get("model_name", "")),
        model_path=str(raw_config.get("model_path", "")),
        language=str(raw_config.get("language", "en")),
        use_timestamps=bool(raw_config.get("use_timestamps", False)),
    )


def _validate_session_config(requested: SttConfig, actual: SttConfig) -> None:
    # 현재 구현은 서버 설정을 기준으로 동작한다.
    # 샘플레이트나 채널이 다르면 모델 입력이 깨지므로 여기서 막는다.
    if requested.sample_rate != actual.sample_rate:
        raise ValueError(
            f"sample_rate mismatch: requested={requested.sample_rate}, actual={actual.sample_rate}"
        )
    if requested.channels != actual.channels:
        raise ValueError(f"channels mismatch: requested={requested.channels}, actual={actual.channels}")


async def _emit_result(websocket: WebSocket, events: list[TranscriptSegment]) -> None:
    for event in events:
        await websocket.send_json(
            {
                "type": event.kind.value,
                "session_id": event.session_id,
                "revision": event.revision,
                "text": event.text,
                "created_at": event.created_at,
                "confidence": event.confidence,
            }
        )


async def _send_ack(
    websocket: WebSocket,
    kind: str,
    message: str,
    session_id: str | None = None,
) -> None:
    payload = {"type": kind, "message": message}
    if session_id is not None:
        payload["session_id"] = session_id
    await websocket.send_json(payload)


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


def _decode_audio_payload(payload: bytes, encoding: AudioEncoding) -> np.ndarray:
    if not payload:
        return np.array([], dtype=np.float32)

    if encoding == "float32":
        return np.frombuffer(payload, dtype=np.float32).astype(np.float32, copy=False)

    if encoding == "pcm16":
        return (np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(f"Unsupported audio encoding: {encoding}")
