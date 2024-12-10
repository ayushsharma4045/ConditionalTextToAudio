import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
model = MusicGen.get_pretrained('facebook/musicgen-medium', device=device)

# Set the model parameters
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=10  # Generate 10 seconds of audio
)

# Generate music
descriptions = ["tip tip barsa paani paani me aag lagau ravina tandon akshay kumar"]
wav = model.generate(descriptions)

# Save the generated audio
for i, one_wav in enumerate(wav):
    audio_write(f'generated_music_{i}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

print("Music generation complete. Check the generated_music_0.wav file.")