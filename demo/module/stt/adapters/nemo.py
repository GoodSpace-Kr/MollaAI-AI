from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.buffer import StreamingAudioWindow
from ..config import SttConfig


class NemoAsrAdapter:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._model = self._load_model()

    def transcribe_window(self, window: StreamingAudioWindow, config: SttConfig) -> str:
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
        import nemo.collections.asr as nemo_asr

        if self.config.model_path:
            path = Path(self.config.model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"STT_MODEL_PATH not found: {path}")

            model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path))
        elif self.config.model_name:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model_name)
        else:
            raise ValueError("STT_MODEL_PATH 또는 STT_MODEL_NAME 중 하나는 필요합니다.")

        if hasattr(model, "eval"):
            model.eval()

        return model

    def _write_wav(self, path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(max(1, channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

    def _normalize_transcript(self, result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if hasattr(result, "text"):
            text = getattr(result, "text")
            if isinstance(text, str) and text.strip():
                return text.strip()

        if isinstance(result, (list, tuple)):
            if not result:
                return ""

            first = result[0]
            if isinstance(first, str):
                return first.strip()

            if hasattr(first, "text"):
                text = getattr(first, "text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

            if isinstance(first, dict):
                for key in ("text", "transcript", "prediction"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

            if isinstance(first, (list, tuple)) and first:
                nested = first[0]
                if isinstance(nested, str):
                    return nested.strip()
                if hasattr(nested, "text"):
                    text = getattr(nested, "text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

        if isinstance(result, dict):
            for key in ("text", "transcript", "prediction"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return str(result).strip()
