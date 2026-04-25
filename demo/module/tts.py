import os
import platform
import shutil
import subprocess
import soundfile as sf
from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, lang_code="a", voice="af_heart", sample_rate=24000, output_dir="tts_out"):
        self.pipeline = KPipeline(lang_code=lang_code, device="cpu")
        self.voice = voice
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def text_to_wav(self, text: str, filename="reply.wav") -> str:
        generator = self.pipeline(text, voice=self.voice)

        final_audio = None
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"[TTS chunk {i}]")
            final_audio = audio

        if final_audio is None:
            raise RuntimeError("오디오 생성 실패")

        wav_path = os.path.join(self.output_dir, filename)
        sf.write(wav_path, final_audio, self.sample_rate)
        return wav_path

    def play_wav(self, wav_path: str):
        if shutil.which("ffplay"):
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if platform.system() == "Darwin" and shutil.which("afplay"):
            subprocess.run(
                ["afplay", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if shutil.which("mpv"):
            subprocess.run(
                ["mpv", "--no-video", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if shutil.which("aplay"):
            subprocess.run(
                ["aplay", wav_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        raise RuntimeError("오디오 플레이어가 없습니다. aplay, ffplay, mpv 중 하나를 설치하세요.")

    def speak(self, text: str, filename="reply.wav") -> str:
        import time
        start_time = time.time()
        wav_path = self.text_to_wav(text, filename=filename)
        print("TTS 소요 시간: ", time.time() - start_time)
        self.play_wav(wav_path)
        return wav_path
