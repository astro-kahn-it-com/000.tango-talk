import os
import sys
from datetime import datetime
import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = MODELS_DIR
os.environ["TORCH_HOME"] = MODELS_DIR

from tangoflux import TangoFluxInference

def parse_prompt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing prompt file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    duration = 10
    steps = 50
    prompt_lines = []
    current_section = None

    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("[") and line_clean.endswith("]"):
            current_section = line_clean[1:-1].upper()
        elif line_clean:
            if current_section == "DURATION":
                try:
                    duration = int(line_clean)
                except ValueError:
                    duration = 10
            elif current_section == "STEPS":
                try:
                    steps = int(line_clean)
                except ValueError:
                    steps = 50
            elif current_section == "PROMPT":
                prompt_lines.append(line_clean)

    return duration, steps, " ".join(prompt_lines)

def main():
    print("=" * 60)
    print(" TangoFlux Foley Generation Engine")
    print("=" * 60)

    duration, steps, prompt_text = parse_prompt(PROMPT_FILE)

    print(f"[Directing] Duration : {duration} seconds")
    print(f"[Directing] Steps    : {steps}")
    print(f"[Prompt]    Foley    : {prompt_text}")
    print("-" * 60)

    print("[Loader] Loading TangoFlux onto GPU...")
    model = TangoFluxInference(name='declare-lab/TangoFlux')

    print("[Pipeline] Synthesizing audio...")
    audio = model.generate(prompt_text, steps=steps, duration=duration)

    # Convert tensor to flat 1D numpy array so soundfile can format the WAV container
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.array(audio)

    audio_np = np.squeeze(audio_np)
    if audio_np.ndim > 1:
        audio_np = np.ravel(audio_np)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = os.path.join(OUTPUT_DIR, f"tangoflux_{timestamp}.wav")

    sf.write(out_file, audio_np, 44100, subtype="FLOAT")

    print("-" * 60)
    print(f"[SUCCESS] Audio generated and saved to:")
    print(f"          -> {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()