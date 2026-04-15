import time
import numpy as np
import sounddevice as sd
import torch

# 1. 환경 설정
SAMPLE_RATE = 16000
VAD_WINDOW = 512  # 약 32ms (Silero VAD 권장 사이즈)
CHANNELS = 1

# 2. Silero VAD 모델 로드
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# 실시간 처리를 위한 이터레이터 객체 생성
# threshold: 민감도 (0.5 기본, 높을수록 더 확실한 목소리만 감지)
vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=SAMPLE_RATE)

def callback(indata, frames, time_info, status):
    if status:
        print(status)

    # indata는 numpy 배열입니다. 이를 torch tensor로 변환합니다.
    audio_chunk = torch.from_numpy(indata.copy()).flatten()

    # 3. VAD에게 한 조각(32ms)을 던져줍니다.
    # 결과값(speech_dict)은 목소리가 시작되거나 끝날 때만 반환됩니다.
    speech_dict = vad_iterator(audio_chunk)

    if speech_dict:
        if "start" in speech_dict:
            print("\n🎤 [말 시작됨]")
        if "end" in speech_dict:
            print("\n😶 [말 끝남]")

# 실행부
def main():
    print("VAD 감지 시작 (512 샘플 / 32ms 단위)...")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_WINDOW, # 여기서 32ms 단위로 자릅니다.
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료합니다.")

if __name__ == "__main__":
    main()