import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class Spatial_Self_Attention(nn.Module):
    def __init__(self, inplanes, ratio=0.25, pooling_type='att', fusion_types=('channel_add',)):
        super(Spatial_Self_Attention, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class Channel_Self_Attention(nn.Module):
    def __init__(self, in_channels):
        super(Channel_Self_Attention, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_attn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        query = self.conv_q(x)
        key = self.conv_k(x)
        value = self.conv_v(x)
        gap_q = F.adaptive_avg_pool2d(query, (1, 1))
        gap_k = F.adaptive_avg_pool2d(key, (1, 1))
        attn_input = torch.cat([gap_q, gap_k], dim=1)
        attn_map = self.conv_attn(attn_input)
        attn_map = self.sigmoid(attn_map)
        out = value * attn_map
        return out


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Bimodal_Foreground_Enhancement_Module(nn.Module):
    def __init__(self, in_channels):
        super(Bimodal_Foreground_Enhancement_Module, self).__init__()
        self.sfa = Spatial_Self_Attention(inplanes=in_channels)
        self.cfa = Channel_Self_Attention(in_channels=in_channels)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.mlp = MLP(in_channels=in_channels * 2, hidden_channels=64, out_channels=2)

    def forward(self, Feature_RGB, Feature_Depth):
        Feature_Depth_Enhanced = self.sfa(Feature_Depth)
        global_feature_rgb = F.adaptive_avg_pool2d(Feature_RGB, (1, 1)).view(Feature_RGB.size(0), -1)
        global_feature_depth = F.adaptive_avg_pool2d(Feature_Depth_Enhanced, (1, 1)).view(
            Feature_Depth_Enhanced.size(0), -1)
        global_features = torch.cat([global_feature_rgb, global_feature_depth], dim=1)
        weights = self.mlp(global_features)
        weights = F.softmax(weights, dim=1)
        weight_rgb, weight_depth = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        weight_rgb = weight_rgb.view(-1, 1, 1, 1)
        weight_depth = weight_depth.view(-1, 1, 1, 1)
        F_Enhanced = weight_rgb * Feature_RGB + weight_depth * Feature_Depth_Enhanced
        foreground_mask = torch.sigmoid(self.conv_mask(F_Enhanced))
        # self.visualize_foreground_mask(foreground_mask)
        F_Enhanced = F_Enhanced * foreground_mask
        F_Fusion = self.cfa(F_Enhanced)
        return F_Fusion


if __name__ == '__main__':
    F_RGB = torch.randn(1, 16, 64, 64)
    F_Depth = torch.randn(1, 16, 64, 64)
    model = Bimodal_Foreground_Enhancement_Module(in_channels=16)
    print(model)
    output_feature = model(F_RGB, F_Depth)
    print(output_feature.shape)
