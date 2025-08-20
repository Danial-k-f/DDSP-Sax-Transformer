
import torch
import torchaudio
from loudness_extractor import LoudnessExtractor

device = "cpu"
wav_path = r"E:\ddsp-pytorch-master\ckpt\torchscript_model\input.wav"
output_path = r"E:\ddsp-pytorch-master\ckpt\torchscript_model\loudness.pt"

# Load audio
waveform, sr = torchaudio.load(wav_path)  # [1, T]
assert sr == 16000, "Expected 16kHz audio"

# Truncate to mono
audio = waveform.mean(dim=0, keepdim=True)

# Instantiate extractor
extractor = LoudnessExtractor(sr=16000, device=device)
extractor = extractor.to(device)

# Create input dict
input_dict = {"audio": audio.to(device)}

# Compute loudness
with torch.no_grad():
    loudness = extractor(input_dict)  # [1, T']

# Save
torch.jit.save(torch.jit.script(torch.nn.Identity().to(device)), output_path)  # dummy model holder
torch.save(loudness.cpu(), output_path)
print(f"Loudness saved to {output_path}, shape = {loudness.shape}")
