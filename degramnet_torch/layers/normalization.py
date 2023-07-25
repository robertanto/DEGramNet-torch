import torch
import torch.nn as nn


class SpectrogramNormalization(nn.Module):

    def __init__(
        self,
        std_norm=True,
        freq_norm=True,
        eps=1e-07,
    ):

        super().__init__()

        self.std_norm = std_norm
        self.freq_norm = freq_norm
        self.eps = eps

    def forward(self, x):

        # x -> [batch, channel, frequency, time]

        mean = torch.mean(x, [1, 3] if self.freq_norm else [
                          1, 2, 3], keepdim=True)
        y = torch.sub(x, mean)

        if self.std_norm:
            std = torch.add(torch.std(x, [1, 3] if self.freq_norm else [
                            1, 2, 3], keepdim=True), self.eps)
            y = torch.div(y, std)

        return y
