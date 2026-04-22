import queue
import signal
import threading
import time

from module.llm import QwenChat
from module.stt import STT
from module.tts import KokoroTTS

def main():
    chat = QwenChat()
    speaker = KokoroTTS(
        lang_code="a",
        voice="af_heart",
        output_dir="tts_out",
    )
    stt = STT()

    shutdown_event = threading.Event()
    transcript_queue: "queue.Queue[str]" = queue.Queue()

    def request_shutdown(signum, frame):
        shutdown_event.set()
        stt.stop()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    vad_thread = threading.Thread(
        target=stt.run,
        kwargs={"on_final_text": transcript_queue.put},
        daemon=False,
    )
    vad_thread.start()

    print("음성 입력을 기다리는 중입니다. 'q', 'quit', 'exit'가 인식되면 종료합니다.")

    try:
        while not shutdown_event.is_set():
            try:
                query = transcript_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            query = query.strip()
            if not query:
                continue

            print("STT: ", query)

            if query.lower() in ["q", "quit", "exit"]:
                print("종료합니다.")
                break

            start_llm_time = time.time()
            answer = chat.ask(query)
            print("LLM: ", answer)
            print("LLM 소요 시간: ", time.time() - start_llm_time)

            try:
                wav_path = speaker.speak(answer, filename="reply.wav")
                print("재생 완료:", wav_path)
            except Exception as e:
                print("TTS/재생 오류:", e)
    except KeyboardInterrupt:
        shutdown_event.set()
        print("\n종료합니다.")
    finally:
        stt.stop()
        vad_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
