import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils

from pystoi import stoi
from pesq import pesq
from spectral_normalization import SpectralNorm


class MetricSTOI(nn.Module):
    '''range: [0, 1]'''
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self._device = torch.device('cpu')
        
    def forward(self, clean, enhanced):
        assert len(clean) == len(enhanced)
        scores = []
        for c, e in zip(clean, enhanced):
            q = stoi(c, e, self.sr)
            scores.append(q)
        return torch.tensor(scores, device=self._device)
    
    def to_origin_range(v):
        return v

    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self._device = args[0]
        return r
            

class MetricPESQ(nn.Module):
    '''range: [-0.5, 4.5]'''
    def __init__(self, sr, mode='wb'):
        super().__init__()
        self.sr = sr
        self.mode = mode
        self._device = torch.device('cpu')

    def forward(self, clean, enhanced):
        assert len(clean) == len(enhanced)
        clean, enhanced = np.array(clean), np.array(enhanced)
        scores = []
        for c, e in zip(clean, enhanced):
            q = pesq(self.sr, c, e, mode=self.mode)
            q = (q+0.5)/(4.5+0.5)  # (q-min)/(max-min)
            scores.append(q)
        return torch.tensor(scores, device=self._device)
    
    def to_origin_range(self, v):
        return v*(4.5+0.5)-0.5
    
    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self._device = args[0]
        return r

    
def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight_bar.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight_bar.data)

        
class Generator_Sigmoid_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(257, 300, num_layers=2, bidirectional=True, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(300*2, 300),
            nn.LeakyReLU(0.2, True),
            nn.Linear(300, 257),
            nn.Sigmoid()
        )

    def forward(self, spec, spec_normalized):
        spec_normalized = spec_normalized.transpose(1, 2)  # swap dims to match (batch, time, freq)
        out_rnn, h = self.rnn(spec_normalized)
        mask = self.fc_layers(out_rnn)
        mask = mask.transpose(2, 1)  # swap back
        enhanced = mask * spec
        return enhanced, mask


class Discriminator_Stride2_SN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(2, 10, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(10, 20, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(20, 40, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(40, 80, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            SpectralNorm(nn.Linear(80, 40)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(40, 1)),
        )

    def forward(self, x, y):
        # conditional GAN
        xy = torch.stack([x,y], dim=1)  # to shape (batch, channel, H, W)
        return self.layers(xy)

    
class Generator_Tanh_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(257, 300, num_layers=2, bidirectional=True, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(300*2, 300),
            nn.LeakyReLU(0.2, True),
            nn.Linear(300, 257),
            nn.Tanh()
        )

    def forward(self, x, empty_arg=None):
        assert empty_arg is None, 'This model wants dB 1-input preprocessing'
        x = x.transpose(1, 2)  # swap dims to match (batch, time, freq)
        out_rnn, h = self.rnn(x)
        mask = self.fc_layers(out_rnn)
        enhanced = mask + x
        enhanced.transpose_(2, 1)  # swap back
        return enhanced, mask


class Discriminator_Stride1_SN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            SpectralNorm(nn.Conv2d(2, 10, 5)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(10, 20, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(20),
            SpectralNorm(nn.Conv2d(20, 40, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(40, 80, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(80),
            nn.AdaptiveAvgPool2d(1),  # (b, 80, 1, 1)
            nn.Flatten(),  # (b, 80)
            SpectralNorm(nn.Linear(80, 40)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(40, 10)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(10, 1))
        )

    def forward(self, x, y):
        # conditional GAN
        xy = torch.stack([x,y], dim=1)  # to shape (batch, channel, H, W)
        return self.layers(xy)

    
class Discriminator_Stride1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 10, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(10, 20, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 40, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(40, 80, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(80),
            nn.AdaptiveAvgPool2d(1),  # (b, 80, 1, 1)
            nn.Flatten(),  # (b, 80)
            nn.Linear(80, 40),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(40, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        )

    def forward(self, x, y):
        # conditional GAN
        xy = torch.stack([x,y], dim=1)  # to shape (batch, channel, H, W)
        return self.layers(xy)