from __future__ import annotations

from dataclasses import dataclass
import time
import uuid

import numpy as np

from .audio import AudioBuffer, StreamingAudioWindow
from .config import SttConfig
from .domain import SttSessionState, TranscriptKind, TranscriptSegment
from .engine import TranscriptAdapter


@dataclass(frozen=True, slots=True)
class STTEmitResult:
    events: list[TranscriptSegment]
    has_more: bool = False


class STTService:
    def __init__(
        self,
        config: SttConfig | None = None,
        adapter: TranscriptAdapter | None = None,
    ) -> None:
        self.config = config or SttConfig.from_env()
        self.adapter = adapter
        self.buffer = AudioBuffer(self.config)
        self.state: SttSessionState | None = None

    def start_session(self, session_id: str | None = None, *, started_at: float | None = None) -> SttSessionState:
        self.buffer.reset()
        self.state = SttSessionState(
            session_id=session_id or str(uuid.uuid4()),
            started_at=started_at if started_at is not None else time.time(),
        )
        return self.state

    def ensure_session(self) -> SttSessionState:
        if self.state is None:
            return self.start_session()
        return self.state

    def ingest_audio(
        self,
        samples: np.ndarray | bytes | bytearray | memoryview,
        *,
        received_at: float | None = None,
    ) -> STTEmitResult:
        state = self.ensure_session()
        now = received_at if received_at is not None else time.time()
        state.last_audio_at = now

        self.buffer.append(samples)

        events: list[TranscriptSegment] = []
        while self.buffer.can_emit_window():
            window = self.buffer.pop_window()
            if window is None:
                break
            event = self._build_event(
                state=state,
                text=self._transcribe_window(window),
                kind=TranscriptKind.PARTIAL,
                created_at=now,
            )
            if event is not None:
                state.unstable_text = event.text
                events.append(event)

        return STTEmitResult(events=events, has_more=self.buffer.can_emit_window())

    def finalize_session(self, *, finalized_at: float | None = None) -> STTEmitResult:
        state = self.ensure_session()
        now = finalized_at if finalized_at is not None else time.time()
        state.ended_at = now

        events: list[TranscriptSegment] = []
        window = self.buffer.pop_final_window()
        if window is not None:
            event = self._build_event(
                state=state,
                text=self._transcribe_window(window),
                kind=TranscriptKind.FINAL,
                created_at=now,
            )
            if event is not None:
                state.stable_text = event.text
                state.unstable_text = ""
                events.append(event)

        self.buffer.reset()
        return STTEmitResult(events=events, has_more=False)

    def reset_session(self) -> None:
        self.buffer.reset()
        self.state = None

    def _transcribe_window(self, window: StreamingAudioWindow) -> str:
        if self.adapter is None:
            return ""
        return self.adapter.transcribe_window(window, self.config).strip()

    def _build_event(
        self,
        *,
        state: SttSessionState,
        text: str,
        kind: TranscriptKind,
        created_at: float,
    ) -> TranscriptSegment | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        return TranscriptSegment(
            session_id=state.session_id,
            kind=kind,
            text=cleaned,
            created_at=created_at,
            revision=state.next_revision(),
        )
