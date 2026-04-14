import os
import sys
import time

try:
    import pyaudio
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pyaudio. Install it with: python -m pip install pyaudio"
    ) from exc

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: openai. Install it with: python -m pip install openai"
    ) from exc


MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_SAMPLE_RATE = 24_000
TTS_CHANNELS = 1


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running: "
            "export OPENAI_API_KEY='your_api_key_here'"
        )
    return api_key


client = OpenAI(api_key=require_api_key())


def call_gpt_stream(prompt: str) -> str:
    start = time.perf_counter()
    first_token_at = None
    chunks: list[str] = []
    actual_model = None

    print("LLM stream 시작", flush=True)
    with client.responses.stream(
        model=MODEL,
        input=(
            "Talk like an English conversation teacher. "
            "Keep the answer concise and natural.\n"
            "IMPORTANT: Keep your response very short, maximum 3 sentences."
            f"User prompt: {prompt}"
        ),
        max_output_tokens=100,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    print(
                        f"LLM first token latency: {first_token_at - start:.2f}s",
                        flush=True,
                    )
                if event.delta:
                    chunks.append(event.delta)
                    print(event.delta, end="", flush=True)
            elif event.type == "response.completed":
                actual_model = event.response.model
                break

    total = time.perf_counter() - start
    print()
    if actual_model is None:
        actual_model = MODEL
    print(f"LLM model from response: {actual_model}", flush=True)
    print(f"LLM total time: {total:.2f}s", flush=True)

    text = "".join(chunks).strip()
    print(f"LLM output length: {len(text)} chars", flush=True)
    return text


def play_pcm_stream_from_tts(text: str) -> None:
    start = time.perf_counter()
    first_audio_at = None

    print("TTS stream 시작", flush=True)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=TTS_CHANNELS,
        rate=TTS_SAMPLE_RATE,
        output=True,
    )

    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="pcm",
        ) as response:
            tts_model = (
                response.headers.get("openai-model")
                or response.headers.get("x-openai-model")
                or response.headers.get("openai-model-id")
                or "unknown"
            )
            print(f"TTS model from response: {tts_model}", flush=True)
            print(f"TTS voice: {TTS_VOICE}", flush=True)
            print(f"TTS response content-type: {response.headers.get('content-type', 'unknown')}", flush=True)
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                    print(
                        f"TTS first audio chunk latency: {first_audio_at - start:.2f}s",
                        flush=True,
                    )
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def main() -> int:
    prompt = (
        "hello hi there. i am joshua, i am from korea. i wanted to discuss "
        "about twerking. do you have any idea about it?"
    )

    try:
        text = call_gpt_stream(prompt)
        print("\n--- TTS input text ---")
        print(text, flush=True)
        print("--- TTS input text end ---\n")
        play_pcm_stream_from_tts(text)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        raise SystemExit(f"Execution failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
