import time

from module.llm import QwenChat
from module.tts import KokoroTTS


def main():
    chat = QwenChat()
    speaker = KokoroTTS(
        lang_code="a",
        voice="af_heart",
        output_dir="tts_out",
    )

    try:
        print("텍스트를 입력하세요. 'q', 'quit', 'exit'로 종료합니다.")
        while True:
            query = input("> ").strip()
            if not query:
                continue

            if query.lower() in ["q", "quit", "exit"]:
                print("종료합니다.")
                break

            print("USER: ", query)

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
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
