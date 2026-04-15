import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from settings import (
    CHANNELS,
    DEFAULT_LANGUAGE,
    SAMPLE_RATE,
    STABLE_WORD_AGE_SECONDS,
    STABLE_WORD_COUNT_THRESHOLD,
    WHISPER_CPP_BIN,
    WHISPER_MODEL_PATH,
    WHISPER_NO_GPU,
)


@dataclass
class TranscriptionRequest:
    speech_id: int
    revision: int
    kind: str
    audio: np.ndarray


@dataclass
class WordSlot:
    word: str
    seen_count: int
    first_seen_at: float


@dataclass
class TranscriptSegments:
    stable_text: str
    unstable_text: str


def _resolve_whisper_bin() -> str:
    candidates = []
    if WHISPER_CPP_BIN:
        candidates.append(WHISPER_CPP_BIN)
    candidates.extend(["whisper-cpp", "whisper-cli", "main"])

    for candidate in candidates:
        resolved = shutil.which(candidate) if candidate else None
        if resolved:
            return resolved

    raise RuntimeError(
        "whisper.cpp 실행 파일을 찾을 수 없습니다. "
        "WHISPER_CPP_BIN 환경변수로 경로를 지정하세요."
    )


def _write_wav(path: Path, audio: np.ndarray) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())


class WhisperCppTranscriber:
    def __init__(self, model_path: str, language: str = DEFAULT_LANGUAGE) -> None:
        if not model_path:
            raise RuntimeError(
                "WHISPER_MODEL_PATH 환경변수가 비어 있습니다. "
                "whisper.cpp 모델 파일 경로를 지정하세요."
            )

        self.model_path = model_path
        self.language = language
        self.bin_path = _resolve_whisper_bin()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        audio = np.asarray(audio, dtype=np.float32).flatten()

        with tempfile.TemporaryDirectory(prefix="whisper_cpp_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            wav_path = tmpdir_path / "segment.wav"
            out_prefix = tmpdir_path / "segment"
            _write_wav(wav_path, audio)

            cmd = [self.bin_path]
            if WHISPER_NO_GPU:
                cmd.append("-ng")

            cmd.extend(
                [
                    "-m",
                    self.model_path,
                    "-f",
                    str(wav_path),
                    "--output-txt",
                    "--output-file",
                    str(out_prefix),
                    "--no-prints",
                    "--no-timestamps",
                    "-l",
                    self.language,
                ]
            )

            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            txt_path = out_prefix.with_suffix(".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    return text

            combined_output = "\n".join(
                line.strip()
                for line in (completed.stdout, completed.stderr)
                if line and line.strip()
            ).strip()
            if completed.returncode != 0:
                raise RuntimeError(
                    f"whisper.cpp 실행 실패 (code={completed.returncode}): {combined_output}"
                )

            return combined_output


class TranscriptionDispatcher:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: TranscriptionRequest | None = None
        self._latest_revision: dict[int, int] = {}
        self._closed = False

    def submit(self, request: TranscriptionRequest) -> None:
        with self._condition:
            self._latest_revision[request.speech_id] = request.revision
            self._pending = request
            self._condition.notify()

    def latest_revision(self, speech_id: int) -> int:
        with self._condition:
            return self._latest_revision.get(speech_id, -1)

    def get(self) -> TranscriptionRequest | None:
        with self._condition:
            while self._pending is None and not self._closed:
                self._condition.wait(timeout=0.1)

            if self._pending is not None:
                request = self._pending
                self._pending = None
                return request

            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class StreamingTranscriptRenderer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stable_words: list[str] = []
        self._unstable_slots: list[WordSlot] = []
        self._rendered_lines = 0

    def _write_block(self, lines: list[str]) -> None:
        if self._rendered_lines:
            sys.stdout.write(f"\x1b[{self._rendered_lines}A")

        for line in lines:
            sys.stdout.write("\r\x1b[2K" + line + "\n")

        self._rendered_lines = len(lines)
        sys.stdout.flush()

    def _split_words(self, text: str) -> list[str]:
        return [word for word in text.strip().split() if word]

    def get_segments(self) -> TranscriptSegments:
        with self._lock:
            return TranscriptSegments(
                stable_text=" ".join(self._stable_words).strip(),
                unstable_text=" ".join(slot.word for slot in self._unstable_slots).strip(),
            )

    def get_segment_texts(self) -> tuple[str, str]:
        segments = self.get_segments()
        return segments.stable_text, segments.unstable_text

    def _render(self) -> None:
        segments = self.get_segments()
        lines = [
            f"[STABLE] {segments.stable_text}".rstrip(),
            f"[UNSTABLE] {segments.unstable_text}".rstrip(),
        ]
        self._write_block(lines)

    def render_partial(self, text: str, now: float) -> None:
        with self._lock:
            words = self._split_words(text)
            if not words and not self._unstable_slots:
                return

            if self._stable_words and words[: len(self._stable_words)] == self._stable_words:
                suffix_words = words[len(self._stable_words) :]
            else:
                suffix_words = words

            current_words = [slot.word for slot in self._unstable_slots]
            common_prefix_len = 0
            for old_word, new_word in zip(current_words, suffix_words):
                if old_word != new_word:
                    break
                common_prefix_len += 1

            preserved_slots = self._unstable_slots[:common_prefix_len]
            updated_slots = [
                WordSlot(
                    word=slot.word,
                    seen_count=slot.seen_count + 1,
                    first_seen_at=slot.first_seen_at,
                )
                for slot in preserved_slots
            ]

            for word in suffix_words[common_prefix_len:]:
                updated_slots.append(WordSlot(word=word, seen_count=1, first_seen_at=now))

            self._unstable_slots = updated_slots
            self._promote_stable_words(now)
            self._render()

    def _promote_stable_words(self, now: float) -> None:
        stable_count = 0
        for slot in self._unstable_slots:
            if slot.seen_count >= STABLE_WORD_COUNT_THRESHOLD or (
                now - slot.first_seen_at >= STABLE_WORD_AGE_SECONDS
            ):
                stable_count += 1
            else:
                break

        if stable_count == 0:
            return

        self._stable_words.extend(slot.word for slot in self._unstable_slots[:stable_count])
        self._unstable_slots = self._unstable_slots[stable_count:]

    def render_final(self, text: str) -> None:
        with self._lock:
            self._stable_words = self._split_words(text)
            self._unstable_slots = []
            self._render()

    def reset(self) -> None:
        with self._lock:
            self._stable_words = []
            self._unstable_slots = []
            self._rendered_lines = 0
