import os
import datetime
import torchaudio

# Force cache directories to point to the local models/ directory
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')
os.environ['HF_HOME'] = models_dir
os.environ['TORCH_HOME'] = models_dir

from tangoflux import TangoFluxInference

def parse_prompt_file(filepath):
    duration = None
    steps = None
    prompt = None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    current_section = None
    prompt_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == '[DURATION]':
            current_section = 'DURATION'
        elif stripped == '[STEPS]':
            current_section = 'STEPS'
        elif stripped == '[PROMPT]':
            current_section = 'PROMPT'
        elif stripped:
            if current_section == 'DURATION':
                duration = int(stripped)
                # Ensure duration is up to 30
                duration = min(duration, 30)
                current_section = None
            elif current_section == 'STEPS':
                steps = int(stripped)
                current_section = None
            elif current_section == 'PROMPT':
                prompt_lines.append(stripped)

    if prompt_lines:
        prompt = ' '.join(prompt_lines)

    return duration, steps, prompt

def main():
    prompt_file = os.path.join(base_dir, 'prompt.txt')
    duration, steps, prompt = parse_prompt_file(prompt_file)

    if duration is None or steps is None or prompt is None:
        raise ValueError("Could not fully parse prompt.txt. Make sure it contains [DURATION], [STEPS], and [PROMPT].")

    print(f"Parsed -> Duration: {duration}, Steps: {steps}, Prompt: {prompt}")

    # Initialize the model
    print("Initializing TangoFlux model...")
    model = TangoFluxInference(name='declare-lab/TangoFlux')

    # Synthesize the audio
    print("Generating audio...")
    audio = model.generate(prompt, steps=steps, duration=duration)

    # Save the resulting .wav file
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.wav"
    output_path = os.path.join(output_dir, filename)

    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, audio, sample_rate=44100)
    print("Done!")

if __name__ == "__main__":
    main()
