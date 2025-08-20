import numpy as np
import torch
import torch.nn as nn

class TrainableFIRReverb(nn.Module):
    def __init__(self, reverb_length=48000, device="cuda"):

        super(TrainableFIRReverb, self).__init__()

        self.reverb_length = reverb_length
        self.device = device

        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1,
            requires_grad=True,
        )

        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

    def forward(self, z):
        """
        Compute FIR Reverb
        Input:
            z['audio_synth'] : batch of time-domain signals
        Output:
            output_signal : batch of reverberated signals
        """

        input_signal = z["audio_synth"]
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))

        # ✅ New rfft API
        INPUT_SIGNAL = torch.fft.rfft(zero_pad_input_signal, dim=1)

        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device)
        )
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        )
        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))

        # ✅ New rfft API
        FIR = torch.fft.rfft(zero_pad_final_fir, dim=1)

        # ✅ Complex multiply (using complex tensors)
        OUTPUT_SIGNAL = INPUT_SIGNAL * FIR

        # ✅ New irfft API
        output_signal = torch.fft.irfft(OUTPUT_SIGNAL, n=zero_pad_input_signal.shape[-1], dim=1)

        return output_signal
