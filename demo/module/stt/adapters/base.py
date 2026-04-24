from __future__ import annotations

from typing import Protocol

from ..audio.buffer import StreamingAudioWindow
from ..config import SttConfig


class TranscriptAdapter(Protocol):
    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str: ...
