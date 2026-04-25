# Demo

이 디렉토리는 `main.py`가 websocket 서버를 띄우고, STT 결과를 LLM과 TTS까지 연결하는 데모입니다.

`main.py`가 전체 실행 흐름을 담당하고, 기능은 다음처럼 나뉩니다.

- `stt/`: STT 기능 모듈
- `module/llm.py`: Qwen LLM으로 응답 생성
- `module/tts.py`: Kokoro TTS로 음성 생성 및 재생

## 동작 방식

1. `main.py`가 websocket 서버를 실행합니다.
2. `stt.client`가 마이크 오디오를 websocket으로 보냅니다.
3. `stt/service.py`가 `partial`과 `final` 전사를 만듭니다.
4. `main.py`가 `final`을 `QwenChat`에 넘겨 응답을 생성합니다.
5. 생성된 응답은 `KokoroTTS`가 `tts_out/*.wav`로 저장하고 재생합니다.

## 파일별 역할

### `main.py`

- FastAPI websocket 서버를 실행합니다.
- STT/LLM/TTS 모듈을 조립합니다.
- `final` 전사를 받아 LLM과 TTS를 호출합니다.

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

STT 코드는 이제 `demo/stt/` 아래 하나의 루트 패키지로 정리되어 있습니다.

먼저 서버를 띄웁니다.

```bash
python main.py
```

그 다음 다른 터미널에서 클라이언트를 실행합니다.

```bash
python -m stt.client
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

STT는 `demo/stt/` 아래 하나의 루트 디렉토리로 정리했습니다.

- `stt/client.py`
  - 마이크 입력용 CLI 클라이언트
- `stt/config.py`
  - `.env` 로딩과 STT 설정
- `stt/domain.py`
  - 세션 상태와 전사 이벤트 모델
- `stt/audio.py`
  - 오디오 디코딩, 버퍼링, 스트리밍 window 생성
- `stt/engine.py`
  - 외부 STT 엔진 연동
  - 현재는 `NemoAsrAdapter`
- `stt/service.py`
  - 오디오 ingest와 `partial`/`final` 전사 생성

## 실행 방법

`demo/` 디렉토리에서 아래처럼 실행합니다.

```bash
python main.py
```

실행하면 다음 메시지를 보게 됩니다.

- websocket 서버 시작
- STT `partial` / `final` 출력
- LLM 응답 출력
- TTS 생성 및 재생

## 출력 결과

- STT 결과는 `PARTIAL`, `FINAL`로 출력됩니다.
- LLM 응답은 `LLM:`로 출력됩니다.
- TTS wav 파일은 `tts_out/*.wav`에 저장됩니다.

## 주의 사항

- `Qwen/Qwen3-4B`는 메모리 요구량이 큽니다. 4bit 로딩이 가능해야 합니다.
- `ffplay`가 없으면 재생 단계에서 오류가 납니다.

## 요약

이 데모는 "오디오 입력 -> STT -> LLM -> TTS -> 음성 출력"을 한 번에 연결한 파이프라인입니다.
핵심 실행 파일은 `main.py`이며, `stt`, `module/llm.py`, `module/tts.py`는 기능 모듈로 동작합니다.
