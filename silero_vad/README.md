# silero_vad

실시간 마이크 입력을 Silero VAD로 32ms 단위로 자르고, 발화 중에는 누적 버퍼를 주기적으로 Whisper.cpp에 넘겨 부분 전사를 갱신합니다. `VAD end`가 오면 최종 전사를 한 번 더 보내서 화면을 확정합니다.

## 동작 방식

1. `sounddevice`가 마이크 입력을 `512 samples / 32ms` 단위로 받습니다.
2. `VADIterator`가 음성 시작과 종료를 감지합니다.
3. 발화 중인 동안에는 오디오 프레임을 누적 버퍼에 계속 쌓습니다.
4. 일정 간격마다 지금까지 쌓인 전체 버퍼를 다시 Whisper에 넘겨 부분 전사를 갱신합니다.
5. `VAD end`가 오면 최종 버퍼를 다시 전사하고, 화면의 이전 결과를 새 결과로 덮어씁니다.

## 환경 변수

- `WHISPER_MODEL_PATH`
  - whisper.cpp 모델 파일 경로
  - 예: `/Users/ralph/BackEnd/whisper.cpp/models/ggml-tiny.bin`
- `WHISPER_CPP_BIN`
  - whisper.cpp 실행 파일 경로
  - 기본값은 `whisper-cpp`, `whisper-cli`, `main` 순서로 탐색합니다.
- `WHISPER_LANGUAGE`
  - 인식 언어
  - 기본값은 `ko`
  - 영어만 쓸 경우 `en`
- `WHISPER_NO_GPU`
  - `whisper.cpp`를 CPU 모드로 강제 실행할지 여부
  - 기본값은 `1`이며, macOS/로컬 빌드에서 `failed to initialize whisper context`가 뜨면 이 값이 켜진 상태로 두세요
- `PARTIAL_UPDATE_INTERVAL_SECONDS`
  - 발화 중 부분 전사를 다시 보내는 간격
  - 기본값은 `1.0`
- `MIN_PARTIAL_AUDIO_SECONDS`
  - 부분 전사를 시작하기 위한 최소 누적 길이
  - 기본값은 `1.0`
- `END_GRACE_SECONDS`
  - `VAD end` 이후 최종 전사를 확정하기 전에 더 기다리는 시간
  - 기본값은 `0.6`
- `STABLE_WORD_COUNT_THRESHOLD`
  - 같은 단어가 연속 partial에서 몇 번 유지되면 stable로 넘길지
  - 기본값은 `3`
- `STABLE_WORD_AGE_SECONDS`
  - 같은 단어가 이 시간 이상 안 바뀌면 stable로 넘길지
  - 기본값은 `0.3`

## 실행

```bash
export WHISPER_MODEL_PATH="/absolute/path/to/your/model"
export WHISPER_CPP_BIN="/absolute/path/to/whisper-cli"
export WHISPER_LANGUAGE="en"
export WHISPER_NO_GPU="1"
export PARTIAL_UPDATE_INTERVAL_SECONDS="1.0"
export END_GRACE_SECONDS="0.6"
python3 app.py
```

## 참고

- 코드 구조
  - `app.py`: 오디오 스트림, VAD, worker thread를 연결하는 진입점
  - `speech_session.py`: 발화 누적, partial/final 요청 생성
  - `transcription_runtime.py`: whisper.cpp 실행, partial/final 렌더링, stable/unstable 상태 보관
  - `settings.py`: 환경변수와 공통 설정
- 출력 흐름
  - `WhisperCppTranscriber.transcribe()`가 whisper.cpp 결과를 문자열 `text`로 반환합니다.
  - `text`는 `transcription_worker()`에서 받아서 `render_partial()` 또는 `render_final()`로 전달됩니다.
  - 화면의 `[STABLE]` / `[UNSTABLE]`는 `text`를 다시 분해해서 만든 렌더러 내부 상태입니다.
- `app.py`는 콜백에서 무거운 STT 처리를 하지 않고, 별도 worker thread에서 VAD와 Whisper를 처리합니다.
- 부분 전사는 두 줄로 갱신됩니다.
- `[STABLE] ...`
- `[UNSTABLE] ...`
- 렌더러 내부 상태는 `get_segments()`로 `stable_text`, `unstable_text`를 따로 꺼낼 수 있습니다.
- `VAD end` 직후 바로 확정하지 않고 `END_GRACE_SECONDS`만큼 더 기다렸다가, 다시 말이 이어지면 같은 문장으로 계속 이어갑니다.
- `VAD end` 이후 final 제출은 별도 타이머 스레드가 처리합니다.
- whisper.cpp 바이너리의 옵션은 현재 코드가 공용 플래그를 사용하도록 맞춰 두었습니다. 사용 중인 빌드의 인자 형식이 다르면 `WHISPER_CPP_BIN`으로 실제 실행 파일을 지정하세요.
- `failed to initialize whisper context`가 나오면 먼저 `WHISPER_NO_GPU=1` 상태인지 확인하세요. 이 프로젝트의 현재 `whisper-cli`는 `-ng` 옵션으로 정상 동작했습니다.
