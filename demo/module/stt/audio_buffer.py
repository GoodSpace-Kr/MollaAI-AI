from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from .config import SttConfig
from .types import AudioChunk


@dataclass(frozen=True, slots=True)
class StreamingAudioWindow:
    # 모델에 넘길 하나의 스트리밍 입력 창.
    # left_context: 이전 문맥, chunk: 이번에 실제로 추론할 구간, right_context: 미래 문맥
    left_context: np.ndarray
    chunk: np.ndarray
    right_context: np.ndarray
    left_padding_samples: int = 0
    right_padding_samples: int = 0

    @property
    def samples(self) -> np.ndarray:
        # NeMo에 넣을 최종 오디오 배열.
        parts = [self.left_context, self.chunk, self.right_context]
        if not any(part.size for part in parts):
            return np.array([], dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)


class AudioBuffer:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        # 아직 추론에 쓰지 않은 오디오 조각들을 순서대로 쌓아둔다.
        self._frames: Deque[np.ndarray] = deque()
        # 버퍼 안에서 현재 저장된 데이터가 시작하는 절대 샘플 위치.
        self._base_sample_index = 0
        # 현재 버퍼에 남아 있는 총 샘플 수.
        self._total_samples = 0
        # 다음에 추론용으로 내보낼 chunk의 시작 샘플 위치.
        self._next_emit_sample = 0

    @property
    def chunk_samples(self) -> int:
        return max(1, int(self.config.chunk_secs * self.config.sample_rate))

    @property
    def left_context_samples(self) -> int:
        return max(0, int(self.config.left_context_secs * self.config.sample_rate))

    @property
    def right_context_samples(self) -> int:
        return max(0, int(self.config.right_context_secs * self.config.sample_rate))

    @property
    def available_samples(self) -> int:
        return self._total_samples

    @property
    def available_seconds(self) -> float:
        return self._total_samples / float(self.config.sample_rate)

    def reset(self) -> None:
        # 새 세션이 시작되면 버퍼 상태를 전부 초기화한다.
        self._frames.clear()
        self._base_sample_index = 0
        self._total_samples = 0
        self._next_emit_sample = 0

    def append(self, samples: np.ndarray | bytes | bytearray | memoryview) -> int:
        # 들어온 오디오 프레임을 float32 1차원 배열로 정리해서 추가한다.
        frame = self._normalize_samples(samples)
        if frame.size == 0:
            return 0

        self._frames.append(frame)
        self._total_samples += int(frame.size)
        return int(frame.size)

    def can_emit_window(self) -> bool:
        # chunk + right context까지 확보되어야 안정적으로 partial 추론을 할 수 있다.
        buffer_end = self._base_sample_index + self._total_samples
        required = self._next_emit_sample + self.chunk_samples + self.right_context_samples
        return buffer_end >= required

    def pop_window(self) -> StreamingAudioWindow | None:
        # 다음 chunk를 잘라서 반환한다. right context까지 확보된 경우에만 가능하다.
        if not self.can_emit_window():
            return None

        window = self._build_window(
            chunk_start=self._next_emit_sample,
            chunk_end=self._next_emit_sample + self.chunk_samples,
            pad_right=False,
        )
        self._next_emit_sample += self.chunk_samples
        self._discard_obsolete_prefix()
        return window

    def pop_final_window(self) -> StreamingAudioWindow | None:
        # 발화 종료 시 남아 있는 오디오를 마지막 window로 반환한다.
        buffer_end = self._base_sample_index + self._total_samples
        if buffer_end <= self._next_emit_sample:
            return None

        window = self._build_window(
            chunk_start=self._next_emit_sample,
            chunk_end=buffer_end,
            pad_right=True,
        )
        self._next_emit_sample = buffer_end
        self._discard_obsolete_prefix()
        return window

    def snapshot_chunk(self, start_index: int, end_index: int) -> AudioChunk:
        # 디버깅이나 외부 전달용으로 특정 범위의 오디오를 AudioChunk 형태로 감싼다.
        audio = self._slice_absolute(start_index, end_index)
        return AudioChunk(
            session_id="",
            chunk_index=0,
            samples=audio.tobytes(),
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            created_at=0.0,
        )

    def _normalize_samples(self, samples: np.ndarray | bytes | bytearray | memoryview) -> np.ndarray:
        # 입력 형식이 bytes든 ndarray든, 내부에서는 항상 float32 1차원 배열로 통일한다.
        if isinstance(samples, (bytes, bytearray, memoryview)):
            frame = np.frombuffer(samples, dtype=np.float32)
        else:
            frame = np.asarray(samples, dtype=np.float32)
        return np.ascontiguousarray(frame.reshape(-1))

    def _build_window(
        self,
        *,
        chunk_start: int,
        chunk_end: int,
        pad_right: bool,
    ) -> StreamingAudioWindow:
        # chunk 앞뒤로 context를 붙인 스트리밍 window를 만든다.
        left_start = max(0, chunk_start - self.left_context_samples)
        left_context = self._slice_absolute(left_start, chunk_start)
        chunk = self._slice_absolute(chunk_start, chunk_end)

        if pad_right:
            right_context = np.array([], dtype=np.float32)
            right_padding_samples = max(0, self.right_context_samples - max(0, self._total_samples - chunk_end))
            if right_padding_samples:
                right_context = np.zeros(right_padding_samples, dtype=np.float32)
        else:
            right_end = chunk_end + self.right_context_samples
            right_context = self._slice_absolute(chunk_end, right_end)
            right_padding_samples = max(0, self.right_context_samples - int(right_context.size))
            if right_padding_samples:
                right_context = np.concatenate(
                    [right_context, np.zeros(right_padding_samples, dtype=np.float32)]
                )

        left_padding_samples = max(0, self.left_context_samples - int(left_context.size))
        if left_padding_samples:
            left_context = np.concatenate([np.zeros(left_padding_samples, dtype=np.float32), left_context])

        return StreamingAudioWindow(
            left_context=left_context,
            chunk=chunk,
            right_context=right_context,
            left_padding_samples=left_padding_samples,
            right_padding_samples=right_padding_samples,
        )

    def _slice_absolute(self, start: int, end: int) -> np.ndarray:
        # 절대 샘플 위치 기준으로 버퍼 일부를 잘라낸다.
        if end <= start:
            return np.array([], dtype=np.float32)

        relative_start = max(0, start - self._base_sample_index)
        relative_end = max(0, end - self._base_sample_index)
        if relative_start >= self._total_samples:
            return np.array([], dtype=np.float32)

        relative_end = min(relative_end, self._total_samples)
        if relative_end <= relative_start:
            return np.array([], dtype=np.float32)

        parts: list[np.ndarray] = []
        current_index = 0
        wanted_start = relative_start
        wanted_end = relative_end

        for frame in self._frames:
            frame_end = current_index + int(frame.size)
            if frame_end <= wanted_start:
                current_index = frame_end
                continue
            if current_index >= wanted_end:
                break

            local_start = max(0, wanted_start - current_index)
            local_end = min(int(frame.size), wanted_end - current_index)
            if local_end > local_start:
                parts.append(frame[local_start:local_end])

            current_index = frame_end

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32, copy=False)

    def _discard_obsolete_prefix(self) -> None:
        # left context로 재사용할 범위를 남기고, 완전히 지난 오래된 샘플은 버린다.
        keep_from = max(0, self._next_emit_sample - self.left_context_samples)
        if keep_from <= self._base_sample_index:
            return

        drop_count = keep_from - self._base_sample_index
        while self._frames and drop_count > 0:
            frame = self._frames[0]
            if drop_count >= int(frame.size):
                drop_count -= int(frame.size)
                self._base_sample_index += int(frame.size)
                self._total_samples -= int(frame.size)
                self._frames.popleft()
                continue

            self._frames[0] = frame[drop_count:]
            self._base_sample_index += drop_count
            self._total_samples -= drop_count
            drop_count = 0
