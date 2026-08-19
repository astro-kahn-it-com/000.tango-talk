import os
import time
import torchaudio

# Force cache directories to point to the local models/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
os.environ["HF_HOME"] = models_dir
os.environ["TORCH_HOME"] = models_dir

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(script_dir, "output"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "cache", "temp"), exist_ok=True)

from tangoflux import TangoFluxInference

def parse_prompt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('[')

    duration = 10
    steps = 50
    prompt = ""

    for part in parts:
        if part.startswith('DURATION]'):
            val = part.replace('DURATION]', '').strip()
            if val:
                duration = min(int(val), 30)
        elif part.startswith('STEPS]'):
            val = part.replace('STEPS]', '').strip()
            if val:
                steps = int(val)
        elif part.startswith('PROMPT]'):
            prompt = part.replace('PROMPT]', '').strip()

    return duration, steps, prompt

def main():
    prompt_path = os.path.join(script_dir, 'prompt.txt')
    duration, steps, prompt = parse_prompt(prompt_path)

    print(f"Parsed Duration: {duration}")
    print(f"Parsed Steps: {steps}")
    print(f"Parsed Prompt: '{prompt}'")

    print("Initializing TangoFluxInference...")
    model = TangoFluxInference(name='declare-lab/TangoFlux')

    print("Generating audio...")
    audio = model.generate(prompt, steps=steps, duration=duration)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.wav"
    output_path = os.path.join(script_dir, 'output', output_filename)

    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, audio, 44100)
    print("Generation complete!")

if __name__ == "__main__":
    main()
