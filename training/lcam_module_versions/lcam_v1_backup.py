"""
LCAM - Lightweight Channel and Spatial Attention Module
Modified to be Vitis AI 2.5 quantizer compatible.
- Replaced AdaptiveMaxPool2d with torch.amax (DPU-friendly)
- All other operations are standard ops supported by DPU
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention: identifies WHICH feature channels are important.
    Uses both avg pooling and max-reduction over spatial dims.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # NO max_pool - we use torch.amax in forward() instead
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling (DPU supports this)
        avg_out = self.mlp(self.avg_pool(x))
        
        # Max over spatial dims using torch.amax
        # Keeps same shape as adaptive max pool would
        max_pooled = torch.amax(x, dim=(2, 3), keepdim=True)
        max_out = self.mlp(max_pooled)
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention: identifies WHERE in the image is important.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average (DPU supports torch.mean)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Channel-wise max using torch.amax (DPU supports this)
        max_out = torch.amax(x, dim=1, keepdim=True)
        
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(spatial_input))
        return x * attention


class LCAM(nn.Module):
    """Lightweight Channel + Spatial Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
