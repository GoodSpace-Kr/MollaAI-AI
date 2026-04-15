import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from settings import (
    CHANNELS,
    DEFAULT_LANGUAGE,
    MAX_UTTERANCE_SECONDS,
    PARTIAL_UPDATE_INTERVAL_SECONDS,
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    VAD_WINDOW,
    WHISPER_MODEL_PATH,
)
from speech_session import SpeechSession
from transcription_runtime import (
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
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
stop_event = threading.Event()
vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=SAMPLE_RATE)


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    chunk = indata.copy().reshape(-1)
    try:
        audio_queue.put_nowait(chunk)
    except queue.Full:
        print("오디오 큐가 가득 차서 프레임을 버렸습니다.")


def vad_worker(
    session: SpeechSession,
    dispatcher: TranscriptionDispatcher,
    renderer: StreamingTranscriptRenderer,
):
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.monotonic()
        speech_event = vad_iterator(torch.from_numpy(chunk))
        requests, is_start, is_end = session.push_chunk(chunk, speech_event, now)

        if is_start:
            renderer.reset()
            print("\n🎤 [말 시작됨]")

        for request in requests:
            dispatcher.submit(request)

        if is_end:
            print("\n😶 [말 끝남]")


def transcription_worker(
    transcriber: WhisperCppTranscriber,
    dispatcher: TranscriptionDispatcher,
    renderer: StreamingTranscriptRenderer,
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
                renderer.render_partial(text, time.monotonic())
            else:
                renderer.render_final(text)
        except Exception as exc:
            print(f"\n[STT 오류] {exc}")


def main():
    pre_roll_chunks = max(1, int(PRE_ROLL_SECONDS * SAMPLE_RATE / VAD_WINDOW))
    session = SpeechSession(pre_roll_chunks=pre_roll_chunks)
    transcriber = WhisperCppTranscriber(model_path=WHISPER_MODEL_PATH, language=DEFAULT_LANGUAGE)
    dispatcher = TranscriptionDispatcher()
    renderer = StreamingTranscriptRenderer()

    vad_thread = threading.Thread(
        target=vad_worker,
        args=(session, dispatcher, renderer),
        daemon=True,
    )
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
    print(f"최대 발화 길이: {MAX_UTTERANCE_SECONDS:.1f}s")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_WINDOW,
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
