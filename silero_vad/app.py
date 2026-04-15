import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

# 1. 환경 설정
SAMPLE_RATE = 16000
VAD_WINDOW = 512  # 약 32ms (Silero VAD 권장 사이즈)
CHANNELS = 1
PRE_ROLL_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 30
PARTIAL_UPDATE_INTERVAL_SECONDS = float(os.getenv("PARTIAL_UPDATE_INTERVAL_SECONDS", "1.0"))
MIN_PARTIAL_AUDIO_SECONDS = float(os.getenv("MIN_PARTIAL_AUDIO_SECONDS", "1.0"))
END_GRACE_SECONDS = float(os.getenv("END_GRACE_SECONDS", "1.2"))
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")
WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "")
WHISPER_NO_GPU = os.getenv("WHISPER_NO_GPU", "1") != "0"


# 2. Silero VAD 모델 로드
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


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
    def __init__(self, model_path: str, language: str = "ko") -> None:
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


@dataclass
class TranscriptionRequest:
    speech_id: int
    revision: int
    kind: str
    audio: np.ndarray


class SpeechSession:
    def __init__(self, pre_roll_chunks: int, max_utterance_samples: int) -> None:
        self.pre_roll = deque(maxlen=pre_roll_chunks)
        self.current_chunks: list[np.ndarray] = []
        self.active = False
        self.current_samples = 0
        self.max_utterance_samples = max_utterance_samples
        self.speech_id = 0
        self.revision = 0
        self.next_partial_at = 0.0
        self.pending_end_at: float | None = None

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
            self.revision += 1
            requests.append(
                TranscriptionRequest(
                    speech_id=self.speech_id,
                    revision=self.revision,
                    kind="final",
                    audio=self.snapshot(),
                )
            )
            self.reset_after_finalize()

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
            self.active = True
            self.speech_id += 1
            self.revision = 0
            self.current_chunks = list(self.pre_roll)
            self.current_chunks.append(chunk)
            self.current_samples = sum(item.size for item in self.current_chunks)
            self.next_partial_at = now + PARTIAL_UPDATE_INTERVAL_SECONDS
            self.pending_end_at = None
        else:
            self.pre_roll.append(chunk)

        if self.active and self.current_samples >= self.max_utterance_samples:
            is_end = True

        if self.active and is_start:
            self.pending_end_at = None

        if self.active and is_end:
            if self.current_samples >= self.max_utterance_samples:
                self.revision += 1
                requests.append(
                    TranscriptionRequest(
                        speech_id=self.speech_id,
                        revision=self.revision,
                        kind="final",
                        audio=self.snapshot(),
                    )
                )
                self.reset_after_finalize()
            else:
                self.pending_end_at = now + END_GRACE_SECONDS

        return requests, is_start, is_end

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.current_chunks) if self.current_chunks else np.array([], dtype=np.float32)

    def reset_after_finalize(self) -> None:
        self.current_chunks = []
        self.active = False
        self.current_samples = 0
        self.pre_roll.clear()
        self.next_partial_at = 0.0
        self.pending_end_at = None


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

            if self._closed:
                return None

            request = self._pending
            self._pending = None
            return request

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class TranscriptRenderer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_text = ""
        self._current_width = 0

    def _write_line(self, prefix: str, text: str, finalize: bool) -> None:
        line = f"{prefix} {text}".rstrip()
        clear_width = max(0, self._current_width - len(line))
        sys.stdout.write("\r" + line + (" " * clear_width))
        if finalize:
            sys.stdout.write("\n")
            self._current_text = ""
            self._current_width = 0
        else:
            self._current_text = text
            self._current_width = len(line)
        sys.stdout.flush()

    def render_partial(self, text: str) -> None:
        with self._lock:
            if text == self._current_text:
                return
            self._write_line("[STT]", text, finalize=False)

    def render_final(self, text: str) -> None:
        with self._lock:
            self._write_line("[STT]", text, finalize=True)


audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
stop_event = threading.Event()


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    chunk = indata.copy().reshape(-1)
    try:
        audio_queue.put_nowait(chunk)
    except queue.Full:
        print("오디오 큐가 가득 차서 프레임을 버렸습니다.")


# 실시간 처리를 위한 이터레이터 객체 생성
# threshold: 민감도 (0.5 기본, 높을수록 더 확실한 목소리만 감지)
vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=SAMPLE_RATE)


def vad_worker(session: SpeechSession, dispatcher: TranscriptionDispatcher):
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.monotonic()
        audio_tensor = torch.from_numpy(chunk)
        speech_event = vad_iterator(audio_tensor)
        requests, is_start, is_end = session.push_chunk(chunk, speech_event, now)

        if is_start:
            print("\n🎤 [말 시작됨]")

        for request in requests:
            dispatcher.submit(request)

        if is_end:
            print("\n😶 [말 끝남]")


def transcription_worker(
    transcriber: WhisperCppTranscriber,
    dispatcher: TranscriptionDispatcher,
    renderer: TranscriptRenderer,
):
    while True:
        request = dispatcher.get()
        if request is None:
            break

        try:
            text = transcriber.transcribe(request.audio)
            if dispatcher.latest_revision(request.speech_id) != request.revision:
                continue

            if not text:
                continue

            if request.kind == "partial":
                renderer.render_partial(text)
            else:
                renderer.render_final(text)
        except Exception as exc:
            print(f"\n[STT 오류] {exc}")


def main():
    pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS * SAMPLE_RATE / VAD_WINDOW))
    max_utterance_samples = MAX_UTTERANCE_SECONDS * SAMPLE_RATE
    session = SpeechSession(pre_roll_chunks=pre_roll_chunks, max_utterance_samples=max_utterance_samples)
    transcriber = WhisperCppTranscriber(
        model_path=WHISPER_MODEL_PATH,
        language=DEFAULT_LANGUAGE,
    )
    dispatcher = TranscriptionDispatcher()
    renderer = TranscriptRenderer()

    vad_thread = threading.Thread(target=vad_worker, args=(session, dispatcher), daemon=True)
    stt_thread = threading.Thread(
        target=transcription_worker,
        args=(transcriber, dispatcher, renderer),
        daemon=True,
    )
    vad_thread.start()
    stt_thread.start()

    print("VAD 감지 시작 (512 샘플 / 32ms 단위)...")
    print(f"Whisper 모델: {WHISPER_MODEL_PATH}")
    print(f"Whisper 실행 파일: {transcriber.bin_path}")
    print(f"Partial 업데이트 간격: {PARTIAL_UPDATE_INTERVAL_SECONDS:.1f}s")
    print(f"End 유예 시간: {END_GRACE_SECONDS:.1f}s")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_WINDOW,  # 여기서 32ms 단위로 자릅니다.
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        stop_event.set()
        dispatcher.close()


if __name__ == "__main__":
    main()
