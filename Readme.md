# MollaAI API Test

이 저장소는 실시간 STT 결과를 LLM과 TTS까지 연결하는 실험용 프로젝트입니다.
현재 기준의 주요 실행 흐름은 `demo/` 디렉토리 아래에 정리되어 있습니다.

## 현재 구성

- `demo/`: websocket 기반 STT -> LLM -> TTS 데모
- `speech.mp3`: 현재 저장소에 포함된 음성 관련 파일
- `tts_debug.pcm`: 이전 오디오 테스트 과정에서 생성된 디버그 PCM 파일

## 실행 전 확인 사항

- 실제 실행 방법과 의존성은 [demo/README.md](/Users/ralph/BackEnd/MollaAI_api_test/demo/README.md) 기준으로 확인하는 것을 권장합니다.
- 루트 디렉토리의 예전 OpenAI 단독 테스트 스크립트는 제거되었습니다.

## 작업 기록

앞으로 사용자가 요청하는 작업 단위로 이 섹션에 변경 사항을 누적 기록합니다.

### 2026-04-27

- 현재 코드에서 참조되지 않던 `openAI.py`를 삭제함
- 루트 `Readme.md`를 현재 `demo/` 중심 구조에 맞게 정리함

### 2026-04-14

- `Readme.md` 파일을 새로 생성함
- 이후 작업마다 이 파일에 변경 내용을 기록하도록 구조를 추가함
- `openAI.py`에 타이밍 측정 기준을 추가함
- LLM 첫 출력 지연은 스트리밍 시작부터 첫 텍스트 델타까지로 측정함
- LLM 첫 단어 지연은 첫 텍스트 델타 이후 첫 공백이 확인되는 시점까지로 별도 측정함
- TTS 첫 재생 지연은 `play_pcm_stream_from_tts()` 진입부터 첫 PCM 청크 재생까지로 측정하도록 정리함
- `LLM to TTS handoff latency` 출력은 제거하고, 의미가 명확한 지표만 남김
- TTS PCM 재생 시 2바이트 경계 정렬을 추가함
- 디버그용 원시 PCM 파일 `tts_debug.pcm` 저장을 추가함
- TTS 출력 설정과 PCM 청크 바이트 길이 로그를 추가함
- `Pipeline to first TTS audio latency`를 추가해 `LLM 입력 시작 -> TTS 첫 오디오 출력` 시간을 바로 볼 수 있게 함
