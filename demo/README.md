# Demo

이 디렉토리는 텍스트 입력을 받아 LLM 응답을 생성한 뒤, TTS로 다시 음성으로 읽어주는 데모입니다.

`main.py`가 전체 실행 흐름을 담당하고, 기능은 다음처럼 나뉩니다.

- `module/llm.py`: Qwen LLM으로 응답 생성
- `module/tts.py`: Kokoro TTS로 음성 생성 및 재생

## 동작 방식

1. 사용자가 터미널에 텍스트를 입력합니다.
2. `QwenChat`이 사용자의 문장을 영어 대화용 프롬프트로 감싸서 응답을 생성합니다.
3. 생성된 응답은 `KokoroTTS`가 `tts_out/reply.wav`로 저장하고, `ffplay`로 재생합니다.

종료 문구로 `q`, `quit`, `exit`를 입력하면 프로그램이 종료됩니다. `Ctrl+C`도 지원합니다.

## 파일별 역할

### `main.py`

- `QwenChat`, `KokoroTTS`를 생성합니다.
- 터미널 입력을 받아 LLM 응답을 만들고 TTS로 재생합니다.
- `Ctrl+C`로 종료합니다.

### `module/llm.py`

- 기본 모델은 `Qwen/Qwen3-4B`입니다.
- `BitsAndBytesConfig`를 사용해 4bit 로딩을 시도합니다.
- 프롬프트는 "영어 회화 선생님" 역할로 고정됩니다.
- 응답 생성 후 `</think>`나 assistant 태그 같은 불필요한 부분을 제거합니다.

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

### 2. 오디오 재생 도구

TTS 재생은 `ffplay`를 사용합니다. `ffplay`가 없으면 재생 단계에서 오류가 납니다.

### 3. STT 테스트 클라이언트

마이크 오디오를 WebSocket으로 보내는 테스트 클라이언트가 있습니다.

먼저 STT 서버를 띄웁니다.

```bash
python -m uvicorn stt_app:app --reload
```

그 다음 다른 터미널에서 클라이언트를 실행합니다.

```bash
python stt_client.py
```

기본 연결 주소는 `ws://127.0.0.1:8000/stt/ws`입니다.

### 4. STT 환경변수

협업을 위해 STT 설정은 로컬 shell export보다 `.env` 파일로 관리하는 걸 권장합니다.

- `.env.example`을 `.env`로 복사한 뒤
- 필요한 값만 각자 로컬에서 채우면 됩니다.

특히 아래 둘 중 하나는 꼭 필요합니다.

- `STT_MODEL_NAME`: NeMo pretrained model name
- `STT_MODEL_PATH`: local `.nemo` file path

## STT 구조

STT는 동작 경로를 유지한 채 내부 책임만 다시 나눴습니다.

- `module/stt/domain/`
  - 세션 상태와 전사 이벤트 같은 순수 데이터 모델
- `module/stt/audio/`
  - 오디오 디코딩, 버퍼링, 스트리밍 window 생성
- `module/stt/adapters/`
  - 외부 STT 엔진 연동
  - 현재는 `NemoAsrAdapter`
- `module/stt/services/`
  - 세션 시작, 오디오 ingest, finalize 같은 오케스트레이션
- `module/stt/transport/`
  - websocket 프로토콜 처리

기존 `module/stt/api.py`, `service.py`, `audio_buffer.py`, `nemo_adapter.py`, `types.py`는
호환용 진입점으로 남아 있으므로 기존 import 경로는 계속 사용할 수 있습니다.

## 실행 방법

프로젝트 루트에서 아래처럼 실행합니다.

```bash
python main.py
```

실행하면 다음 메시지를 보게 됩니다.

- 텍스트 입력 대기
- LLM 응답 출력
- TTS 생성 및 재생

## 출력 결과

- 입력 문장은 터미널에 `USER:`로 출력됩니다.
- LLM 응답은 `LLM:`로 출력됩니다.
- TTS wav 파일은 `tts_out/reply.wav`에 저장됩니다.

## 주의 사항

- `Qwen/Qwen3-4B`는 메모리 요구량이 큽니다. 4bit 로딩이 가능해야 합니다.
- `ffplay`가 없으면 재생 단계에서 오류가 납니다.

## 요약

이 데모는 "텍스트 입력 -> LLM -> TTS -> 음성 출력"을 한 번에 연결한 파이프라인입니다.
핵심 실행 파일은 `main.py`이며, 나머지 모듈은 각각 LLM, TTS 단계의 구현을 담당합니다.
