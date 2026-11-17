"""
MobileNetV2 (Sandler et al., CVPR 2018) in PyTorch.

Architecture definition follows the layout of the open-source implementation
at https://github.com/d-li14/mobilenetv2.pytorch (Apache-2.0), so that its
ImageNet-pretrained weights (pretrained/mobilenetv2_1.0-0c6065bc.pth,
72.19% top-1) can be loaded directly. Weights are loaded with
torch.load(weights_only=True), i.e. as plain tensors only.
"""
import math

import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_bn(inp, oup, k, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """Inverted residual block: 1x1 expand -> 3x3 depthwise -> 1x1 linear projection."""

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(inp, hidden, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
                   nn.ReLU6(inplace=True),
                   nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.identity else self.conv(x)


class MobileNetV2(nn.Module):
    CFGS = [  # t (expansion), c (channels), n (repeats), s (stride)
        [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
        [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        inp = _make_divisible(32 * width_mult, 8)
        layers = [_conv_bn(3, inp, 3, 2)]
        for t, c, n, s in self.CFGS:
            out = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(InvertedResidual(inp, out, s if i == 0 else 1, t))
                inp = out
        self.features = nn.Sequential(*layers)
        last = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = _conv_bn(inp, last, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last, num_classes)
        self._init_weights()

    def forward(self, x):
        x = self.avgpool(self.conv(self.features(x))).flatten(1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def build_mobilenetv2(num_classes, pretrained_path=None, dropout=0.2):
    """MobileNetV2 with ImageNet weights and a new classification head."""
    model = MobileNetV2(1000)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
    return model
