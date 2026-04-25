from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from module.llm import QwenChat
from module.tts import KokoroTTS
from stt.audio import decode_audio_payload
from stt.config import SttConfig
from stt.domain import TranscriptKind, TranscriptSegment
from stt.engine import NemoAsrAdapter
from stt.service import STTService


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = SttConfig.from_env()
    app.state.stt_config = config
    app.state.stt_adapter = NemoAsrAdapter(config)
    app.state.llm = QwenChat()
    app.state.tts = KokoroTTS(
        lang_code="a",
        voice="af_heart",
        output_dir="tts_out",
    )
    yield


app = FastAPI(title="STT LLM TTS Server", lifespan=lifespan)


@app.websocket("/stt/ws")
async def stt_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    config: SttConfig = websocket.app.state.stt_config
    adapter: NemoAsrAdapter = websocket.app.state.stt_adapter
    llm: QwenChat = websocket.app.state.llm
    tts: KokoroTTS = websocket.app.state.tts
    service = STTService(config=config, adapter=adapter)
    connection_state: dict[str, str] = {"encoding": "pcm16"}

    try:
        await _send_ack(websocket, "ready", "STT websocket connected")

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                await _handle_text_message(
                    websocket=websocket,
                    service=service,
                    raw_text=message["text"],
                    connection_state=connection_state,
                    actual_config=config,
                )
                continue

            if message.get("bytes") is not None:
                if service.state is None:
                    service.start_session()
                samples = decode_audio_payload(message["bytes"], connection_state["encoding"])
                result = service.ingest_audio(samples, received_at=time.time())
                await _emit_result(websocket, result.events)
                await _handle_final_pipeline(websocket, result.events, llm=llm, tts=tts)
                continue

            await _send_error(websocket, "Unsupported websocket message")
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _send_error(websocket, str(exc))


async def _handle_text_message(
    *,
    websocket: WebSocket,
    service: STTService,
    raw_text: str,
    connection_state: dict[str, str],
    actual_config: SttConfig,
) -> None:
    payload = _parse_control_message(raw_text)
    message_type = payload.get("type")

    if message_type == "start":
        requested_config = _extract_config(payload)
        if requested_config is not None:
            _validate_session_config(requested=requested_config, actual=actual_config)

        session_id = payload.get("session_id")
        service.start_session(session_id=session_id if isinstance(session_id, str) else None)

        encoding = payload.get("encoding")
        if isinstance(encoding, str) and encoding in ("pcm16", "float32"):
            connection_state["encoding"] = encoding

        await _send_ack(
            websocket,
            "started",
            "STT session started",
            service.state.session_id if service.state else None,
        )
        return

    if message_type == "reset":
        service.reset_session()
        await _send_ack(websocket, "reset", "STT session reset")
        return

    if message_type == "ping":
        await _send_ack(websocket, "pong", "pong")
        return

    await _send_error(websocket, f"Unknown control message type: {message_type!r}")


async def _handle_final_pipeline(
    websocket: WebSocket,
    events: list[TranscriptSegment],
    *,
    llm: QwenChat,
    tts: KokoroTTS,
) -> None:
    for event in events:
        if event.kind is not TranscriptKind.FINAL:
            continue

        llm_text = await asyncio.to_thread(llm.ask, event.text)
        print(f"[LLM] {llm_text}", flush=True)
        await websocket.send_json(
            {
                "type": "llm",
                "session_id": event.session_id,
                "source_text": event.text,
                "text": llm_text,
            }
        )

        wav_name = f"{event.session_id}_{event.revision}.wav"
        wav_path = await asyncio.to_thread(tts.speak, llm_text, wav_name)
        print(f"[TTS] {wav_path}", flush=True)
        await websocket.send_json(
            {
                "type": "tts",
                "session_id": event.session_id,
                "text": llm_text,
                "wav_path": str(Path(wav_path).resolve()),
            }
        )


async def _emit_result(websocket: WebSocket, events: list[TranscriptSegment]) -> None:
    for event in events:
        print(f"[{event.kind.value.upper()}] {event.text}", flush=True)
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
        batch_size=int(raw_config.get("batch_size", 1)),
        model_name=str(raw_config.get("model_name", "")),
        model_path=str(raw_config.get("model_path", "")),
        language=str(raw_config.get("language", "en")),
        use_timestamps=bool(raw_config.get("use_timestamps", False)),
        speech_rms_threshold=float(raw_config.get("speech_rms_threshold", 0.01)),
        pause_timeout_secs=float(raw_config.get("pause_timeout_secs", 1.0)),
        partial_repeat_threshold=int(raw_config.get("partial_repeat_threshold", 2)),
    )


def _validate_session_config(*, requested: SttConfig, actual: SttConfig) -> None:
    if requested.sample_rate != actual.sample_rate:
        raise ValueError(
            f"sample_rate mismatch: requested={requested.sample_rate}, actual={actual.sample_rate}"
        )
    if requested.channels != actual.channels:
        raise ValueError(f"channels mismatch: requested={requested.channels}, actual={actual.channels}")


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


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
