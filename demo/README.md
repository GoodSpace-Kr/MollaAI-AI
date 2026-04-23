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
