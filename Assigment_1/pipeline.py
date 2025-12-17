import time
import requests
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
import whisper
from diffusers import StableDiffusionPipeline
AUDIO_FILE = "input.wav"
DURATION = 10
SAMPLE_RATE = 16000

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_KEEP_ALIVE = 0

SD_MODEL = "runwayml/stable-diffusion-v1-5"
OUT_FILE = "output.png"

WIDTH = 512
HEIGHT = 512
STEPS = 25
GUIDANCE = 7.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_session = requests.Session()
_whisper_model = None
_sd_pipe = None

def record_audio(path=AUDIO_FILE, duration=DURATION, fs=SAMPLE_RATE):
    print("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    audio = audio.squeeze()
    wav.write(path, fs, audio)
    print(f"Saved audio to {path}")

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def speech_to_text(path=AUDIO_FILE):
    print("Transcribing...")
    model = get_whisper_model()
    result = model.transcribe(path)
    text = (result.get("text") or "").strip()
    print("You said:", text)
    return text

def llm_generate_prompt(user_text: str):
    print("Generating image prompt (Ollama)...")
    system_prompt = (
        "Convert the user text into ONE Stable Diffusion prompt.\n"
        "Make it visual: subject, setting, lighting, mood, camera, style.\n"
        "No explanations; output only the prompt text."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\nUser text:\n{user_text}\n",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    r = _session.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    prompt = (r.json().get("response") or "").strip()
    if not prompt:
        raise RuntimeError("Ollama returned an empty response.")
    print("Prompt:", prompt)
    return prompt

def get_sd_pipe():
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    print("Loading Stable Diffusion pipeline (one-time)...")

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=DTYPE,
        use_safetensors=True,
        safety_checker=None,
    )

    if DEVICE == "cuda":

        pipe.enable_sequential_cpu_offload()
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        pipe = pipe.to("cpu")

    _sd_pipe = pipe
    return _sd_pipe

def generate_image(prompt: str, out_path=OUT_FILE):
    print("Generating image...")
    pipe = get_sd_pipe()

    generator = None
    if DEVICE == "cuda":
        generator = torch.Generator("cuda").manual_seed(int(time.time()) % 2**32)

    with torch.inference_mode():
        result = pipe(
            prompt,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        )

    image = result.images[0]
    image.save(out_path)
    print(f"Saved image to {out_path}")

def main():
    record_audio()
    text = speech_to_text()
    if not text:
        print("No speech detected; exiting.")
        return
    image_prompt = llm_generate_prompt(text)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    generate_image(image_prompt)

if __name__ == "__main__":
    main()
