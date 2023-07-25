import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


## SE Base
class ChannelSELayer(nn.Module): 
    def __init__(self, channel, ratio=1): 
        super().__init__() 
        #self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.network = nn.Sequential(
            nn.Linear(channel, int(channel*ratio), bias=False),
            nn.ReLU(inplace=True), 
            nn.Linear(int(channel*ratio), channel, bias=False),
            nn.Sigmoid()
        ) 
    def forward(self, inputs): 
        b, _ , c, _ = inputs.shape 

        x = torch.mean(inputs,3)
        x = x.view(b, c)
        x = self.network(x) 
        x = x.view(b, 1, c, 1) 
        x = inputs * x 
        return x 


## sSE
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv1d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, freq, time = input_tensor.size()

        x = torch.mean(input_tensor,3)
        x = x.view(batch_size, channel, freq)

        if weights:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)
        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, freq, 1))
        return output_tensor

## csSE
class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, reduction_ratio=1/16, num_channels=64, strategy='maxout', data_format="channels_last"):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self._cSE = ChannelSELayer(num_channels, reduction_ratio)
        self._sSE = SpatialSELayer(1)
        self._strategy = self._get_strategy(strategy)
        self.channel_axis = 1 if data_format == "channels_first" else -1

    # apply strategy
    def _get_strategy(self, strategy):
        if strategy == "maxout":
            return torch.max

        elif strategy == "add":
            return torch.add

        elif strategy == "multiply":
            return torch.multiply

        elif strategy == "concat":
            return torch.cat
        else:
            raise ValueError(
                "strategy must be one of ['maxout','concat','multiply','add']")

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        cse = self._cSE(input_tensor)
        sse = self._sSE(input_tensor)
        output_tensor = self._strategy(cse, sse)
        return output_tensor