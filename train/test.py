"""
args : 
--input : input wav
--output : output wav path
--ckpt : pretrained weight file
--config : network-corresponding yaml config file
--wave_length : wave length in format 
    (default : 0, which means all)
    WARNING : gpu memory might be not enough.
"""

import os
import sys
import torch
import torchaudio
import argparse
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from network.autoencoder.autoencoder import AutoEncoder

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".wav", help="Input wav file")
parser.add_argument("--output", default="output.wav", help="Output wav path")
parser.add_argument("--ckpt", default=".pth", help="Pretrained model weights (.pth)")
parser.add_argument("--config", default=".yaml", help="Model config YAML file")
parser.add_argument("--wave_length", type=int, default=16000, help="Wave length (0 = all)")
args = parser.parse_args()

# ----------------- Load Input WAV -----------------
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

if args.wave_length == 0:
    y, sr = torchaudio.load(args.input)
else:
    y, sr = torchaudio.load(args.input, num_frames=args.wave_length)
print(f"[INFO] Loaded input file: {args.input} (Sample Rate: {sr}, Shape: {y.shape})")

# ----------------- Load Config -----------------
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config file not found: {args.config}")

config = OmegaConf.load(args.config)

# Resample if needed
if sr != config.sample_rate:
    print(f"[INFO] Resampling from {sr} Hz to {config.sample_rate} Hz")
    resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
    y = resampler(y)

# Ensure y is in correct shape (Batch x Samples)
if y.ndim == 1:
    y = y.unsqueeze(0)  # (1, Samples)
elif y.ndim == 2 and y.shape[0] != 1:
    print(f"[WARNING] Input has multiple channels ({y.shape[0]}), using first channel only")
    y = y[0:1, :]  # Take first channel only

# ----------------- Load Network -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AutoEncoder(config).to(device)

if not os.path.exists(args.ckpt):
    raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

checkpoint = torch.load(args.ckpt, map_location=device)
net.load_state_dict(checkpoint['model'])  # Extract model weights
net.eval()

print(f"[INFO] Model loaded from {args.ckpt}")

# ----------------- Forward Pass -----------------
with torch.no_grad():
    y = y.to(device)
    recon = net.reconstruction(y)

# ----------------- Save Outputs -----------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Synthesized audio
dereverb = recon["audio_synth"].cpu()
if dereverb.ndim == 1:
    dereverb = dereverb.unsqueeze(0)  # Ensure shape: (1, samples)
torchaudio.save(
    os.path.splitext(args.output)[0] + "_synth.wav",
    dereverb,
    sample_rate=config.sample_rate,
)
print(f"[INFO] Saved: {os.path.splitext(args.output)[0]}_synth.wav")

# Reverb-added audio (optional)
if config.get("use_reverb", False):
    recon_add_reverb = recon["audio_reverb"].cpu()
    if recon_add_reverb.ndim == 1:
        recon_add_reverb = recon_add_reverb.unsqueeze(0)
    torchaudio.save(
        os.path.splitext(args.output)[0] + "_reverb.wav",
        recon_add_reverb,
        sample_rate=config.sample_rate,
    )
    print(f"[INFO] Saved: {os.path.splitext(args.output)[0]}_reverb.wav")

print("[INFO] Done.")
