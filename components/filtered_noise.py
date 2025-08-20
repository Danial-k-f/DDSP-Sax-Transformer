import numpy as np
import torch
import torch.nn as nn

class FilteredNoise(nn.Module):
    def __init__(self, frame_length=64, attenuate_gain=1e-2, device='cuda'):
        super(FilteredNoise, self).__init__()
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain

    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varying filter banks) and create filtered noise by overlap-add.
        Args:
            z['H']: filter coefficients (batch_num, frame_num, filter_coeff_length)
        """
        batch_num, frame_num, filter_coeff_length = z['H'].shape
        self.filter_window = nn.Parameter(
            torch.hann_window(filter_coeff_length * 2 - 1, dtype=torch.float32),
            requires_grad=False
        ).to(self.device)

        # Frequency domain representation (complex valued)
        INPUT_FILTER_COEFFICIENT = z['H']
        ZERO_PHASE_FR_BANK = torch.complex(INPUT_FILTER_COEFFICIENT, torch.zeros_like(INPUT_FILTER_COEFFICIENT))
        ZERO_PHASE_FR_BANK = ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length)

        zero_phase_ir_bank = torch.fft.irfft(ZERO_PHASE_FR_BANK, n=filter_coeff_length * 2 - 1, dim=-1)

        # Linear phase, windowed, zero-padded
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, dims=1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(
            windowed_linear_phase_ir_bank, (0, self.frame_length - 1)
        )

        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.fft.rfft(zero_paded_windowed_linear_phase_ir_bank, dim=1)

        # Generate white noise, pad, FFT
        noise = (torch.rand(batch_num, frame_num, self.frame_length, dtype=torch.float32).to(self.device) * 2 - 1)
        noise = noise.view(-1, self.frame_length)
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.fft.rfft(zero_paded_noise, dim=1)

        # Convolution in frequency domain (complex multiplication)
        FILTERED_NOISE = ZERO_PADED_NOISE * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK

        # Back to time domain
        filtered_noise = torch.fft.irfft(FILTERED_NOISE, n=FILTERED_NOISE.shape[1], dim=1)
        filtered_noise = filtered_noise.view(batch_num, frame_num, -1) * self.attenuate_gain

        # Overlap-add to reconstruct time-varying filtered noise
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad=False).unsqueeze(1).to(self.device)
        output_signal = nn.functional.conv_transpose1d(
            filtered_noise.transpose(1, 2),
            overlap_add_filter,
            stride=self.frame_length,
            
            padding=0
        ).squeeze(1)

        return output_signal
