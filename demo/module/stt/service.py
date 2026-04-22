from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import torch

from .settings import (
    CHANNELS,
    DEFAULT_LANGUAGE,
    MAX_UTTERANCE_SECONDS,
    PARTIAL_UPDATE_INTERVAL_SECONDS,
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    VAD_WINDOW,
    WHISPER_MODEL_PATH,
)
from .speech_session import SpeechSession
from .transcription_runtime import (
    StreamingTranscriptRenderer,
    TranscriptionDispatcher,
    WhisperCppTranscriber,
)

# Silero VAD 모델 로드
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
_, _, _, VADIterator, _ = utils


class STTService:
    def __init__(self) -> None:
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=SAMPLE_RATE)
        self._threads: list[threading.Thread] = []
        self._dispatcher: TranscriptionDispatcher | None = None

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        chunk = indata.copy().reshape(-1)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            print("오디오 큐가 가득 차서 프레임을 버렸습니다.")

    def vad_worker(
        self,
        session: SpeechSession,
        dispatcher: TranscriptionDispatcher,
        renderer: StreamingTranscriptRenderer,
    ) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.monotonic()
            speech_event = self.vad_iterator(torch.from_numpy(chunk))
            requests, is_start, is_end = session.push_chunk(chunk, speech_event, now)

            if is_start:
                renderer.reset()
                print("\n🎤 [말 시작됨]")

            for request in requests:
                dispatcher.submit(request)

            if is_end:
                print("\n😶 [말 끝남]")

    def end_timer_worker(
        self,
        session: SpeechSession,
        dispatcher: TranscriptionDispatcher,
    ) -> None:
        while not self.stop_event.is_set():
            for request in session.poll_due_final(time.monotonic()):
                dispatcher.submit(request)
            time.sleep(0.05)

    def transcription_worker(
        self,
        transcriber: WhisperCppTranscriber,
        dispatcher: TranscriptionDispatcher,
        renderer: StreamingTranscriptRenderer,
        on_final_text: Callable[[str], None] | None = None,
    ) -> None:
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
                    renderer.render_partial(text, time.monotonic())
                else:
                    renderer.render_final(text)
                    if on_final_text is not None:
                        on_final_text(text)
            except Exception as exc:
                print(f"\n[STT 오류] {exc}")

    def stop(self) -> None:
        self.stop_event.set()
        if self._dispatcher is not None:
            self._dispatcher.close()

    def run(self, on_final_text: Callable[[str], None] | None = None) -> None:
        pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS * SAMPLE_RATE / VAD_WINDOW))
        session = SpeechSession(pre_roll_chunks=pre_roll_chunks)
        transcriber = WhisperCppTranscriber(model_path=WHISPER_MODEL_PATH, language=DEFAULT_LANGUAGE)
        dispatcher = TranscriptionDispatcher()
        renderer = StreamingTranscriptRenderer()
        self._dispatcher = dispatcher

        vad_thread = threading.Thread(
            target=self.vad_worker,
            args=(session, dispatcher, renderer),
            daemon=True,
        )
        end_timer_thread = threading.Thread(
            target=self.end_timer_worker,
            args=(session, dispatcher),
            daemon=True,
        )
        stt_thread = threading.Thread(
            target=self.transcription_worker,
            args=(transcriber, dispatcher, renderer, on_final_text),
            daemon=True,
        )
        self._threads = [vad_thread, end_timer_thread, stt_thread]
        for thread in self._threads:
            thread.start()

        print("VAD 감지 시작 (512 샘플 / 32ms 단위)...")
        print(f"Whisper 모델: {transcriber.model_path}")
        print(f"Whisper 실행 파일: {transcriber.bin_path}")
        print(f"Partial 업데이트 간격: {PARTIAL_UPDATE_INTERVAL_SECONDS:.1f}s")
        print(f"최대 발화 길이: {MAX_UTTERANCE_SECONDS:.1f}s")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=VAD_WINDOW,
                callback=self.callback,
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n종료합니다.")
        finally:
            self.stop()
            for thread in self._threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
