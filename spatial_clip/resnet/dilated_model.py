from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip.model import AttentionPool2d


class DenseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 1x1 filter; no dilation.
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride > 1:
            self.avgpool = AvgFilter(planes, filter_size=stride, padding=0,
                                     padding_mode='constant', dilation=dilation)
        else:
            self.avgpool = nn.Identity()

        # 1x1 filter; no dilation.
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        self.padding = padding

        if stride > 1 or inplanes != planes * DenseBottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", AvgFilter(inplanes, filter_size=stride, padding=0,
                                 padding_mode='constant', dilation=dilation) if stride > 1 else nn.Identity()
                 ),
                ("0", nn.Conv2d(inplanes, planes * self.expansion,
                                1, stride=1, bias=False)),  # 1x1; no dilation
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AvgFilter(nn.Module):
    def __init__(self, in_channels, filter_size,
                 padding, padding_mode, dilation=1):
        super().__init__()
        self.filter_size = filter_size
        self.padding_mode = padding_mode
        self.padding = padding
        self.channels = in_channels
        self.dilation = dilation
        a = np.array([[1., ]*filter_size])
        filt = a * a.T
        filt = torch.Tensor(filt/np.sum(filt)).to(torch.float16)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat(self.channels, 1, 1, 1))

    def extra_repr(self):
        return ("in_channels={in_channels}, filter_size={filter_size}, padding={padding}, "
                "padding_mode={padding_mode}, dilation={dilation}".format(in_channels=self.channels,
                                                                          filter_size=self.filter_size, padding=self.padding, padding_mode=self.padding_mode,
                                                                          dilation=self.dilation)
                )

    def forward(self, inp):
        return F.conv2d(inp, self.filt, groups=inp.shape[1], padding=self.padding, dilation=self.dilation)


class ModifiedSpatialResNetDilated(nn.Module):
    """Dilated version of ModifiedResNet. Replacing strides with dilation.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3,
                               dilation=2,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3,
                               dilation=2,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = AvgFilter(width, 2, padding=0,
                                 padding_mode='constant', dilation=2)  # Avg. pool no padding.
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        # Dilation 4
        self.layer1 = self._make_layer(width, layers[0], padding=4, dilation=4)
        # Dilation 4
        self.layer2 = self._make_layer(
            width * 2, layers[1], stride=2, padding=4, dilation=4)
        # Dilation 8
        self.layer3 = self._make_layer(
            width * 4, layers[2], stride=2, padding=8, dilation=8)
        # Dilation 16
        self.layer4 = self._make_layer(
            width * 8, layers[3], stride=2, padding=16, dilation=16)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionSpatial2d(
            input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, padding=1, dilation=1):
        layers = [DenseBottleneck(self._inplanes, planes,
                                  stride, padding=padding, dilation=dilation)]

        self._inplanes = planes * DenseBottleneck.expansion

        if stride > 1:
            dilation *= 2
            padding *= 2
        for _ in range(1, blocks):
            layers.append(DenseBottleneck(self._inplanes, planes,
                                          padding=padding, dilation=dilation))
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


def linear(x, weight, bias):
    x = x.matmul(weight.t())
    x += bias
    return x


class AttentionSpatial2d(AttentionPool2d):
    def __init__(self, spacial_dim: int, embed_dim: int,
                 num_heads: int, output_dim: int = None, use_pos_enc: bool = False):
        super().__init__(spacial_dim, embed_dim, num_heads, output_dim)
        self.use_pos_enc = use_pos_enc
        assert not use_pos_enc  # TODO: add support for positional encoding.

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] *
                      x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # Without Positional Encoding.
        x = linear(x, self.v_proj.weight, self.v_proj.bias)
        x = linear(x, self.c_proj.weight, self.c_proj.bias)
        x = x.permute(1, 2, 0).reshape(n, -1, h, w)
        return x
