import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class SincGram(nn.Module):  
    def hertz_to_mel(self, x):
        return (2595 * np.log10(1 + x / 700))

    def mel_to_hertz(self, x):
        return (700 * (10**(x / 2595) - 1))

    def weights_to_hertz(self, freq_centers_w, bands_w):
        freq_centers_hz = (freq_centers_w + 1.0)*self.sample_rate/4.0

        # unscale and uncenter
        beta_w = self.weight_to_beta(bands_w)

        # un transpose and unscale
        bands_hz = beta_w*self.sample_rate/2.0

        return freq_centers_hz, bands_hz

    def hertz_to_weights(self, freq_centers_hz, bands_hz):
        freq_centers_w = 4.0*freq_centers_hz/self.sample_rate - 1.0

        # scale in the range 0-2 and transpose for weight beta
        beta_w = 2.0*bands_hz/self.sample_rate

        # CENTER AND SCALING
        bands_w = self.beta_to_weight(beta_w)

        return freq_centers_w, bands_w

    def beta_to_weight(self, beta_w):
        mu = 0.25-2.0/self.num_spectrogram_bins

        # scaling beta obtaining final weights
        bands_w = beta_w*2.0/mu - 1

        return bands_w

    def weight_to_beta(self, bands_w):
        mu = 0.25-2.0/self.num_spectrogram_bins

        # uncenter
        beta_w = (bands_w+1.0)*mu/2.0

        return beta_w
    
    def __init__(self, filters=64, order=5, initializer='mel', num_spectrogram_bins=256, sample_rate=16000, lower_edge_hertz=0.0, upper_edge_hertz=None, eps=0.0, device='cpu',**kwargs):
        super().__init__()
        self.filters = filters
        self.initializer = initializer
        self.order = order*2
        self.num_spectrogram_bins = num_spectrogram_bins
        self.sample_rate = sample_rate
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz if upper_edge_hertz is not None else self.sample_rate/2
        self.eps = eps
        self.device=device
      
        self.freq_centers = torch.ones([1, self.filters], dtype=torch.float64, device=self.device)
        self.bands = torch.ones([1, self.filters], dtype=torch.float64, device=self.device)
        torch.nn.init.normal_(self.freq_centers, mean=0.0, std=1.0)        
        torch.nn.init.normal_(self.bands, mean=0.0, std=1.0)
        
        # Initialize bands and freq centers according to the mel scale
        if self.initializer == 'mel':
            low_freq_mel = self.hertz_to_mel(self.lower_edge_hertz)
            high_freq_mel = self.hertz_to_mel(self.upper_edge_hertz)

            # Equally spaced centers in the Mel scale
            # End point of triangular filters = center frequency before and after
            # Linspace takes into account also the endpoints: REMEMBER TO REMOVE THEM!
            f_centers_mel = np.linspace(
                low_freq_mel, high_freq_mel, self.filters+2)

            # Equivalent points in linear scale
            f_centers_hz = self.mel_to_hertz(f_centers_mel)[
                1:-1]  # remove endpoints

            # Bands
            bands_hz = self.mel_to_hertz(f_centers_mel)[
                2:] - self.mel_to_hertz(f_centers_mel)[:-2]  # remove endpoints

            # Scaling and centering for better learning
            f_centers_w, bands_w = self.hertz_to_weights(
                f_centers_hz, bands_hz)


            # The filters are trainable parameters.
            self.freq_centers = nn.Parameter(torch.from_numpy(np.expand_dims(f_centers_w, 0)))
            self.bands = nn.Parameter(torch.from_numpy(np.expand_dims(bands_w, 0)))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

    def get_linear_to_sincgram_matrix(self):
        # shape (n_bins,n_filters)
        np_array = np.linspace([-1.0]*self.filters, [1.0]*self.filters, self.num_spectrogram_bins)
        x = torch.from_numpy(np_array).to(self.device)

        beta = self.weight_to_beta(self.bands[0])
        y = (x - self.freq_centers[0])/(beta+self.eps)

        weights_matrix = 1.0/torch.sqrt(1.0+(y)**(self.order))
        
        return weights_matrix

    def forward(self, x):
        # state_dict = self.state_dict()
        weights_matrix = self.get_linear_to_sincgram_matrix()

        ret = torch.matmul(x.transpose(2,3), weights_matrix.float())
        # norma = torch.linalg.vector_norm(self.freq_centers, ord=2)
        return ret.transpose(2,3)