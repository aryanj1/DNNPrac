import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.net(x)


class Generator(nn.Module):
    def __init__(self, channels=32, blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, channels, 7),
                  nn.InstanceNorm2d(channels), nn.ReLU(True)]
        c = channels
        for _ in range(2):
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1),
                       nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        layers += [Residual(c) for _ in range(blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN with least-squares adversarial loss."""
    def __init__(self, channels=32):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = channels
        for stride in (2, 2, 1):
            layers += [nn.Conv2d(c, c * 2, 4, stride, 1),
                       nn.InstanceNorm2d(c * 2), nn.LeakyReLU(0.2, True)]
            c *= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
