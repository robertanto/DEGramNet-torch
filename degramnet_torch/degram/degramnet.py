import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..layers.degramnet import SincGram, ChannelSpatialSELayer
from ..layers.normalization import SpectrogramNormalization


class DEGramBasedModel(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        sincgram=True,
        attention=True,
        filters=64,
        order=4,
        initializer='mel',
        num_spectrogram_bins=256,
        sample_rate=16000,
        lower_edge_hertz=0.0,
        upper_edge_hertz=None,
        eps=1e-7,
        reduction_ratio=1/16,
        strategy='maxout',
        normalization=True,
        std_norm=True,
        freq_norm=True,
        device='cpu'
    ) -> None:
        super().__init__()

        if not sincgram and not attention:
            raise ValueError(
                'At least one between sincgram and attention arguments must be True.')

        self.sincgram = SincGram(
            filters,
            order,
            initializer,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            eps,
            device
        ).to(device) if sincgram else None

        self.de = ChannelSpatialSELayer(
            reduction_ratio, 64, strategy
        ).to(device) if attention else None

        self.backbone = model

        self.normalization = SpectrogramNormalization(
            std_norm,
            freq_norm,
            eps
        ) if normalization else None

    def forward(self, x: Tensor) -> Tensor:
        representation = x

        if self.sincgram:
            representation = self.sincgram(representation)

        if self.de:
            representation = self.de(representation)

        if self.normalization:
            representation = self.normalization(representation)

        return self.backbone(representation)
