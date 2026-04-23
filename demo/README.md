# Demo

이 디렉토리는 마이크로 입력된 음성을 받아 STT로 텍스트를 만들고, LLM 응답을 생성한 뒤, TTS로 다시 음성으로 읽어주는 음성 대화 데모입니다.

`main.py`가 전체 실행 흐름을 담당하고, 기능은 다음처럼 나뉩니다.

- `module/stt/`: 마이크 입력, VAD, Whisper 기반 음성 인식
- `module/llm.py`: Qwen LLM으로 응답 생성
- `module/tts.py`: Kokoro TTS로 음성 생성 및 재생

## 동작 방식

1. `STTService`가 마이크 입력을 `sounddevice.InputStream`으로 계속 수집합니다.
2. Silero VAD가 발화 시작/종료를 감지합니다.
3. 발화가 진행되는 동안 Whisper.cpp로 partial transcript를 주기적으로 갱신합니다.
4. 발화가 끝나면 최종 transcript를 확정합니다.
5. `main.py`는 최종 transcript를 받아 LLM에 전달합니다.
6. `QwenChat`이 사용자의 문장을 영어 대화용 프롬프트로 감싸서 응답을 생성합니다.
7. 생성된 응답은 `KokoroTTS`가 `tts_out/reply.wav`로 저장하고, `ffplay`로 재생합니다.

종료 문구로 `q`, `quit`, `exit`를 말하면 프로그램이 종료됩니다. `Ctrl+C`도 지원합니다.

## 파일별 역할

### `main.py`

- `QwenChat`, `STT`, `KokoroTTS`를 생성합니다.
- STT 스레드를 시작하고, 최종 인식 결과를 큐로 받습니다.
- 큐에서 텍스트를 꺼내 LLM 응답을 만들고 TTS로 재생합니다.
- `SIGINT`, `SIGTERM` 시 STT를 중지하고 스레드를 정리합니다.

### `module/llm.py`

- 기본 모델은 `Qwen/Qwen3-4B`입니다.
- `BitsAndBytesConfig`를 사용해 4bit 로딩을 시도합니다.
- 프롬프트는 "영어 회화 선생님" 역할로 고정됩니다.
- 응답 생성 후 `</think>`나 assistant 태그 같은 불필요한 부분을 제거합니다.

### `module/stt/`

- `service.py`: 마이크 입력, VAD, Whisper.cpp 호출, partial/final transcript 렌더링
- `speech_session.py`: 발화 단위 버퍼링, pre-roll, partial/final 요청 생성
- `transcription_runtime.py`: Whisper.cpp 실행, transcription dispatcher, 터미널 출력 렌더링
- `settings.py`: 샘플링 레이트, VAD 설정, Whisper 환경변수 로딩

### `module/tts.py`

- `kokoro.KPipeline`을 사용해 텍스트를 음성으로 변환합니다.
- 기본 음성은 `af_heart`이고, 언어 코드는 `a`입니다.
- 결과 wav는 `tts_out/reply.wav`에 저장됩니다.
- 재생은 `ffplay`가 설치되어 있어야 합니다.

## 실행 전 준비

### 1. Python 의존성

`demo/my_requirements.txt`에 주요 의존성이 정리되어 있습니다.

```bash
pip install -r my_requirements.txt
```

STT 전용으로는 `module/stt/requirements.txt`도 있습니다.

### 2. `whisper.cpp`

이 프로젝트는 Python Whisper가 아니라, 로컬에 빌드된 `whisper.cpp` 실행 파일을 사용합니다.

기본 탐색 순서는 다음과 같습니다.

- `WHISPER_CPP_BIN` 환경변수
- `whisper-cpp`
- `whisper-cli`
- `main`
- `./whisper.cpp/build/bin/whisper-cli`
- `./build/bin/whisper-cli`
- `./whisper.cpp/main`
- `./main`

모델 파일은 기본적으로 `whisper.cpp/models/ggml-tiny.bin`을 찾습니다.
별도 모델을 쓰려면 `WHISPER_MODEL_PATH`를 지정하세요.

### 3. 오디오 재생 도구

TTS 재생은 `ffplay`를 사용합니다. `ffplay`가 없으면 재생 단계에서 오류가 납니다.

## 환경 변수

`module/stt/settings.py`는 `.env` 파일을 자동으로 읽습니다.

- `WHISPER_CPP_BIN`: `whisper.cpp` 실행 파일 경로
- `WHISPER_MODEL_PATH`: Whisper 모델 파일 경로
- `WHISPER_LANGUAGE`: Whisper 인식 언어, 기본값 `ko`
- `WHISPER_NO_GPU`: `1`이면 `-ng` 옵션 사용, `0`이면 GPU 사용 시도
- `PARTIAL_UPDATE_INTERVAL_SECONDS`: partial transcript 갱신 간격, 기본값 `1.0`
- `MIN_PARTIAL_AUDIO_SECONDS`: partial 인식 시작 전 최소 오디오 길이, 기본값 `1.0`
- `END_GRACE_SECONDS`: 발화 종료 후 final 확정까지 유예 시간, 기본값 `0.6`
- `STABLE_WORD_COUNT_THRESHOLD`: 안정화된 단어로 간주하는 최소 반복 횟수, 기본값 `3`
- `STABLE_WORD_AGE_SECONDS`: 일정 시간 이상 유지되면 안정 단어로 승격, 기본값 `0.3`

## 실행 방법

프로젝트 루트에서 아래처럼 실행합니다.

```bash
python main.py
```

실행하면 다음 메시지를 보게 됩니다.

- 마이크 입력 대기
- STT 결과 출력
- LLM 응답 출력
- TTS 생성 및 재생

## 출력 결과

- 인식된 문장은 터미널에 `STT:`로 출력됩니다.
- LLM 응답은 `LLM:`로 출력됩니다.
- TTS wav 파일은 `tts_out/reply.wav`에 저장됩니다.
- STT 스트리밍 중에는 터미널에 `[STABLE]`, `[UNSTABLE]` 라인이 갱신됩니다.

## 주의 사항

- `Qwen/Qwen3-4B`는 메모리 요구량이 큽니다. 4bit 로딩이 가능해야 합니다.
- `whisper.cpp` 실행 파일과 모델 파일이 없으면 STT가 시작되지 않습니다.
- 마이크 입력은 기본 입력 장치를 사용합니다.
- `sounddevice`, `ffplay`, `whisper.cpp` 조합이 맞아야 정상 동작합니다.

## 요약

이 데모는 "음성 입력 -> STT -> LLM -> TTS -> 음성 출력"을 한 번에 연결한 파이프라인입니다.
핵심 실행 파일은 `main.py`이며, 나머지 모듈은 각각 STT, LLM, TTS 단계의 구현을 담당합니다.
