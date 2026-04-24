from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SttSessionState:
    session_id: str
    started_at: float
    last_audio_at: float | None = None
    ended_at: float | None = None
    chunk_index: int = 0
    revision: int = 0
    stable_text: str = ""
    unstable_text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def next_chunk_index(self) -> int:
        self.chunk_index += 1
        return self.chunk_index

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision
