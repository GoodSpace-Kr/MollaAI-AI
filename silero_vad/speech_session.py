from collections import deque

import numpy as np

from settings import (
    END_GRACE_SECONDS,
    MAX_UTTERANCE_SECONDS,
    MIN_PARTIAL_AUDIO_SECONDS,
    PARTIAL_UPDATE_INTERVAL_SECONDS,
    SAMPLE_RATE,
)
from transcription_runtime import TranscriptionRequest


class SpeechSession:
    def __init__(self, pre_roll_chunks: int) -> None:
        self.pre_roll = deque(maxlen=pre_roll_chunks)
        self.current_chunks: list[np.ndarray] = []
        self.active = False
        self.current_samples = 0
        self.speech_id = 0
        self.revision = 0
        self.next_partial_at = 0.0
        self.pending_end_at: float | None = None
        self.max_utterance_samples = MAX_UTTERANCE_SECONDS * SAMPLE_RATE

    def push_chunk(
        self,
        chunk: np.ndarray,
        speech_event: dict | None,
        now: float,
    ) -> tuple[list[TranscriptionRequest], bool, bool]:
        is_start = bool(speech_event and "start" in speech_event)
        is_end = bool(speech_event and "end" in speech_event)
        requests: list[TranscriptionRequest] = []

        if self.active and self.pending_end_at is not None and now >= self.pending_end_at:
            requests.extend(self._finalize())

        if self.active:
            self.current_chunks.append(chunk)
            self.current_samples += chunk.size

            should_emit_partial = (
                self.current_samples >= int(MIN_PARTIAL_AUDIO_SECONDS * SAMPLE_RATE)
                and now >= self.next_partial_at
                and self.pending_end_at is None
            )
            if should_emit_partial:
                self.revision += 1
                requests.append(
                    TranscriptionRequest(
                        speech_id=self.speech_id,
                        revision=self.revision,
                        kind="partial",
                        audio=self.snapshot(),
                    )
                )
                self.next_partial_at = now + PARTIAL_UPDATE_INTERVAL_SECONDS
        elif is_start:
            self._start_new_utterance(chunk, now)
        else:
            self.pre_roll.append(chunk)

        if self.active and self.current_samples >= self.max_utterance_samples:
            is_end = True

        if self.active and is_start:
            self.pending_end_at = None

        if self.active and is_end:
            if self.current_samples >= self.max_utterance_samples:
                requests.extend(self._finalize())
            else:
                self.pending_end_at = now + END_GRACE_SECONDS

        return requests, is_start, is_end

    def _start_new_utterance(self, chunk: np.ndarray, now: float) -> None:
        self.active = True
        self.speech_id += 1
        self.revision = 0
        self.current_chunks = list(self.pre_roll)
        self.current_chunks.append(chunk)
        self.current_samples = sum(item.size for item in self.current_chunks)
        self.next_partial_at = now + PARTIAL_UPDATE_INTERVAL_SECONDS
        self.pending_end_at = None

    def _finalize(self) -> list[TranscriptionRequest]:
        self.revision += 1
        request = TranscriptionRequest(
            speech_id=self.speech_id,
            revision=self.revision,
            kind="final",
            audio=self.snapshot(),
        )
        self.reset_after_finalize()
        return [request]

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.current_chunks) if self.current_chunks else np.array([], dtype=np.float32)

    def reset_after_finalize(self) -> None:
        self.current_chunks = []
        self.active = False
        self.current_samples = 0
        self.pre_roll.clear()
        self.next_partial_at = 0.0
        self.pending_end_at = None
