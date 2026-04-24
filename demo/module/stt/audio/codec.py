from __future__ import annotations

from typing import Literal

import numpy as np


AudioEncoding = Literal["pcm16", "float32"]


def decode_audio_payload(payload: bytes, encoding: AudioEncoding) -> np.ndarray:
    if not payload:
        return np.array([], dtype=np.float32)

    if encoding == "float32":
        return np.frombuffer(payload, dtype=np.float32).astype(np.float32, copy=False)

    if encoding == "pcm16":
        return (np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(f"Unsupported audio encoding: {encoding}")
