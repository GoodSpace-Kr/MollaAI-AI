from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .audio_buffer import StreamingAudioWindow
from .config import SttConfig


class NemoAsrAdapter:
    # NeMo 모델 로딩과 추론만 담당하는 어댑터.
    # service.py는 이 클래스를 통해서만 STT 엔진에 접근한다.
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._model = self._load_model()

    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str:
        # 스트리밍 window를 임시 wav로 저장한 뒤 NeMo의 transcribe()에 넘긴다.
        audio = np.asarray(window.samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""

        with tempfile.TemporaryDirectory(prefix="nemo_stt_") as tmpdir:
            wav_path = Path(tmpdir) / "window.wav"
            self._write_wav(wav_path, audio, config.sample_rate, config.channels)
            result = self._model.transcribe(
                [str(wav_path)],
                batch_size=max(1, config.batch_size),
            )

        return self._normalize_transcript(result)

    def _load_model(self) -> Any:
        # 모델은 로컬 .nemo 파일 또는 NGC/HF pretrained 이름으로 로드한다.
        import nemo.collections.asr as nemo_asr

        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"STT_MODEL_PATH not found: {path}")

            model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path))
        elif self.config.model_name:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model_name)
        else:
            raise ValueError(
                "STT_MODEL_PATH 또는 STT_MODEL_NAME 중 하나는 필요합니다."
            )

        # 스트리밍 inference에서 불필요한 로그나 텐서 반환을 줄이기 위한 기본 설정.
        if hasattr(model, "eval"):
            model.eval()

        return model

    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
        # NeMo가 읽을 수 있도록 float32 오디오를 16-bit PCM wav로 저장한다.
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(max(1, channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

    def _normalize_transcript(self, result: Any) -> str:
        # NeMo 버전에 따라 list[str], list[list[str]], dict 유사 형태가 올 수 있어
        # 가능한 한 안전하게 문자열로 정리한다.
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, (list, tuple)):
            if not result:
                return ""

            first = result[0]
            if isinstance(first, str):
                return first.strip()

            if isinstance(first, dict):
                for key in ("text", "transcript", "prediction"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

            if isinstance(first, (list, tuple)) and first:
                nested = first[0]
                if isinstance(nested, str):
                    return nested.strip()

        if isinstance(result, dict):
            for key in ("text", "transcript", "prediction"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return str(result).strip()
