from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import time
import uuid

import numpy as np

from .audio_buffer import AudioBuffer, StreamingAudioWindow
from .config import SttConfig
from .types import SttSessionState, TranscriptKind, TranscriptSegment


class TranscriptAdapter(Protocol):
    # STT 엔진 어댑터가 지켜야 할 최소 인터페이스.
    # 나중에 NeMo, Whisper, 다른 엔진으로 바뀌어도 service는 이 계약만 보면 된다.
    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str: ...


@dataclass(frozen=True, slots=True)
class STTEmitResult:
    # ingest/finalize 결과를 한번에 다루기 위한 반환값.
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
        # 새로운 스트리밍 세션을 시작한다.
        self.buffer.reset()
        self.state = SttSessionState(
            session_id=session_id or str(uuid.uuid4()),
            started_at=started_at if started_at is not None else time.time(),
        )
        return self.state

    def ensure_session(self) -> SttSessionState:
        # 아직 세션이 없으면 자동으로 만든다.
        if self.state is None:
            return self.start_session()
        return self.state

    def ingest_audio(
        self,
        samples: np.ndarray | bytes | bytearray | memoryview,
        *,
        received_at: float | None = None,
    ) -> STTEmitResult:
        # 외부에서 들어온 오디오를 버퍼에 넣고, partial/final 이벤트를 생성한다.
        state = self.ensure_session()
        now = received_at if received_at is not None else time.time()
        state.last_audio_at = now

        self.buffer.append(samples)

        events: list[TranscriptSegment] = []
        while self.buffer.can_emit_window():
            window = self.buffer.pop_window()
            if window is None:
                break
            text = self._transcribe_window(window)
            event = self._make_segment(state, text, kind=TranscriptKind.PARTIAL, created_at=now)
            if event is not None:
                state.unstable_text = event.text
                events.append(event)

        return STTEmitResult(events=events, has_more=self.buffer.can_emit_window())

    def finalize_session(self, *, finalized_at: float | None = None) -> STTEmitResult:
        # 발화가 끝났을 때 남아 있는 오디오까지 마지막으로 처리한다.
        state = self.ensure_session()
        now = finalized_at if finalized_at is not None else time.time()
        state.ended_at = now

        events: list[TranscriptSegment] = []
        window = self.buffer.pop_final_window()
        if window is not None:
            text = self._transcribe_window(window)
            event = self._make_segment(state, text, kind=TranscriptKind.FINAL, created_at=now)
            if event is not None:
                state.stable_text = event.text
                state.unstable_text = ""
                events.append(event)

        self.buffer.reset()
        return STTEmitResult(events=events, has_more=False)

    def reset_session(self) -> None:
        # 현재 세션을 버리고 초기 상태로 만든다.
        self.buffer.reset()
        self.state = None

    def _transcribe_window(self, window: StreamingAudioWindow) -> str:
        if self.adapter is None:
            # 아직 추론기 어댑터가 연결되지 않았으면 빈 문자열을 돌려준다.
            # service 구조를 먼저 잡고 adapter는 나중에 붙일 수 있게 하기 위함이다.
            return ""

        return self.adapter.transcribe_window(window, self.config).strip()

    def _make_segment(
        self,
        state: SttSessionState,
        text: str,
        *,
        kind: TranscriptKind,
        created_at: float,
    ) -> TranscriptSegment | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        revision = state.next_revision()
        return TranscriptSegment(
            session_id=state.session_id,
            kind=kind,
            text=cleaned,
            created_at=created_at,
            revision=revision,
        )
