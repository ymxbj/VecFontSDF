import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """3x3 conv residual block with a projection shortcut when the channel count
    changes."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0),
                nn.BatchNorm2d(out_channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.in_channels != self.out_channels:
            y = y + self.shortcut(x)
        return F.relu(y)


class VecFontSDF(nn.Module):
    """Class-conditional reconstruction model.

    A grayscale glyph image is encoded by a small ResNet, concatenated with a
    one-hot character label, and decoded into (v_dim * p_dim, 6) parabolic curve
    parameters (k, p, q, d, e, f).

    The submodule names (layer0, block1..5, fc_layer1, fc_layer2) are kept stable
    so released checkpoints load with strict=True.
    """

    def __init__(self, fc_channel: int, v_dim: int, p_dim: int,
                 char_categories: int):
        super().__init__()
        self.v_dim = v_dim
        self.p_dim = p_dim
        self.char_categories = char_categories
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (2, 2), padding=1), nn.BatchNorm2d(64))
        self.block1 = ResBlock(64, 128, (3, 3), stride=(2, 2))
        self.block2 = ResBlock(128, 256, (3, 3), stride=(2, 2))
        self.block3 = ResBlock(256, 512, (3, 3), stride=(2, 2))
        self.block4 = ResBlock(512, 512, (3, 3), stride=(2, 2))
        self.block5 = ResBlock(512, 512, (3, 3), stride=(2, 2))
        self.fc_layer1 = nn.Linear(512 + char_categories, fc_channel)
        self.fc_layer2 = nn.Linear(fc_channel, v_dim * p_dim * 6)
        for m in self.layer0.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, val=0.0)
        for fc in (self.fc_layer1, self.fc_layer2):
            nn.init.normal_(fc.weight, mean=0.0, std=0.02)
            nn.init.constant_(fc.bias, val=0.0)

    def forward(self, image: torch.Tensor, clss: torch.Tensor) -> torch.Tensor:
        """image: [B, 1, H, W] grayscale in [0, 1]; clss: [B, char_categories]
        one-hot. Returns [B, v_dim * p_dim, 6]."""
        y = F.relu(self.layer0(image))
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = F.adaptive_avg_pool2d(y, [1, 1]).flatten(1)        # [B, 512]
        y = torch.cat((y, clss.float()), dim=1)                # [B, 512 + cc]
        y = self.fc_layer1(y)
        y = self.fc_layer2(y)
        return y.view(-1, self.v_dim * self.p_dim, 6)
