import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if value.startswith(("'", '"')):
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            return key, value[1:]
        return key, value[1:end]

    if "#" in value:
        value = value.split("#", 1)[0].rstrip()

    return key, value


def _load_dotenv() -> None:
    project_root = Path(__file__).resolve().parents[2]
    candidate_paths = [Path.cwd() / ".env", project_root / ".env"]

    seen: set[Path] = set()
    for env_path in candidate_paths:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue

            key, value = parsed
            os.environ.setdefault(key, value)


_load_dotenv()

SAMPLE_RATE = 16000
VAD_WINDOW = 512  # 약 32ms (Silero VAD 권장 사이즈)
CHANNELS = 1
PRE_ROLL_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 30
PARTIAL_UPDATE_INTERVAL_SECONDS = float(os.getenv("PARTIAL_UPDATE_INTERVAL_SECONDS", "1.0"))
MIN_PARTIAL_AUDIO_SECONDS = float(os.getenv("MIN_PARTIAL_AUDIO_SECONDS", "1.0"))
END_GRACE_SECONDS = float(os.getenv("END_GRACE_SECONDS", "0.6"))
STABLE_WORD_COUNT_THRESHOLD = int(os.getenv("STABLE_WORD_COUNT_THRESHOLD", "3"))
STABLE_WORD_AGE_SECONDS = float(os.getenv("STABLE_WORD_AGE_SECONDS", "0.3"))
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")
WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "")
WHISPER_NO_GPU = os.getenv("WHISPER_NO_GPU", "1") != "0"
