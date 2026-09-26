import torch
import torch.nn as nn


class StagedGlobalAvgPool(nn.Module):
    def __init__(self, stage1_kernel=4):
        super().__init__()
        self.stage1 = nn.AvgPool2d(kernel_size=stage1_kernel, stride=stage1_kernel, ceil_mode=True)
        self.stage2 = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.stage2(self.stage1(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = StagedGlobalAvgPool(stage1_kernel=4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.gate = nn.Hardsigmoid()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_h = torch.max(x, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_h, dim=3, keepdim=True)[0]
        max_out = self.mlp(max_pool)
        return x * self.gate(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.channel_reduce_avg = nn.Conv2d(channels, 1, 1, bias=False)
        self.channel_reduce_max = nn.Conv2d(channels, 1, 1, bias=False)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Hardsigmoid()
    def forward(self, x):
        avg_out = self.channel_reduce_avg(x)
        max_out = self.channel_reduce_max(x)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        return x * self.gate(self.conv(spatial_input))


class LCAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(channels, spatial_kernel)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
