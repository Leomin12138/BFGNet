import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
from vmamba.vmamba import VSSBlock,SS2D
from BFEM import Dual_Modality_Foreground_Enhancement_Module as DMFEM


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = SS2D(
            d_model=input_dim // 4,  # 主维度大小
            d_state=16,  # 状态维度大小
            ssm_ratio=2,  # 缩放比例
            dt_rank="auto",  # 时间步秩
            act_layer=nn.SiLU,  # 激活函数
            d_conv=3,  # 卷积核大小
            conv_bias=True,  # 是否使用卷积偏置
            dropout=0.0,  # Dropout 比率
            bias=False,  # 是否使用线性层偏置
            dt_min=0.001,  # 时间步最小值
            dt_max=0.1,  # 时间步最大值
            dt_init="random",  # 时间步初始化方式
            dt_scale=1.0,  # 时间步缩放因子
            dt_init_floor=1e-4,  # 时间步初始化下限
            initialize="v0",  # 初始化方式
            forward_type="v05_noz",  # 前向传播类型
            channel_first=False,  # 是否使用通道优先格式
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x = x.permute(0, 2, 3, 1)  # 将通道维度移动到最后

        x_norm = self.norm(x)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=3)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=3)

        x_mamba = self.norm(x_mamba)  # 对 x_mamba 进行归一化
        x_mamba = self.proj(x_mamba)

        x_mamba = x_mamba.permute(0, 3, 1, 2)  # 恢复到原始形状
        out = x_mamba
        return out


def make_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
    )


class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=4, c_list=[32, 64, 128, 128, 128, 128],
                 bridge=True):
        super().__init__()

        self.bridge = bridge

        # 为 RGB 通道设计的卷积层
        self.encoder1_rgb = nn.Sequential(
            nn.Conv2d(input_channels-1, c_list[0], 3, stride=2, padding=1),  # 输入通道为 3，输出通道为 c_list[0]
            nn.Conv2d(c_list[0], c_list[0], 1)  # 逐点卷积
        )

        # 为深度通道设计的卷积层
        self.encoder1_depth = nn.Sequential(
            nn.Conv2d(input_channels-3, c_list[0], 3, stride=2, padding=1),  # 输入通道为 1，输出通道为 c_list[0]
            nn.Conv2d(c_list[0], c_list[0], 1)  # 逐点卷积
        )

        self.encoder2_rgb = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[0], 3, stride=2, padding=1, groups=c_list[0]),
            nn.Conv2d(c_list[0], c_list[1], 1),
        )

        self.encoder2_depth = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[0], 3, stride=2, padding=1, groups=c_list[0]),
            nn.Conv2d(c_list[0], c_list[1], 1),
        )
        self.encoder3_rgb = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[1], 3, stride=2, padding=1, groups=c_list[1]),
            nn.Conv2d(c_list[1], c_list[2], 1),
        )
        self.encoder3_depth = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[1], 3, stride=2, padding=1, groups=c_list[1]),
            nn.Conv2d(c_list[1], c_list[2], 1),
        )

        # 替换 Encoder 中的 PVMLayers
        self.encoder4 = make_conv_block(c_list[2], c_list[3])
        self.encoder5 = make_conv_block(c_list[3], c_list[4])
        self.encoder6 = make_conv_block(c_list[4], c_list[5])

        # 替换 Decoder 中的 PVMLayers
        self.decoder1 = make_conv_block(c_list[5], c_list[4])
        self.decoder2 = make_conv_block(c_list[4], c_list[3])
        self.decoder3 = make_conv_block(c_list[3], c_list[2])

        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[2], 3, stride=1, padding=1, groups=c_list[2]),
            nn.Conv2d(c_list[2], c_list[1], 1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[1], 3, stride=1, padding=1, groups=c_list[1]),
            nn.Conv2d(c_list[1], c_list[0], 1),
        )

        self.FFM_1 = DMFEM(in_channels=32)
        self.FFM_2 = DMFEM(in_channels=64)
        self.FFM_3 = DMFEM(in_channels=128)

        self.conv_skip5 = nn.Sequential(
            nn.Conv2d(c_list[4] * 2, c_list[4], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[4]),
            nn.GELU()
        )
        self.conv_skip4 = nn.Sequential(
            nn.Conv2d(c_list[3] * 2, c_list[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[3]),
            nn.GELU()
        )
        self.conv_skip3 = nn.Sequential(
            nn.Conv2d(c_list[2] * 2, c_list[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[2]),
            nn.GELU()
        )
        self.conv_skip2 = nn.Sequential(
            nn.Conv2d(c_list[1] * 2, c_list[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[1]),
            nn.GELU()
        )
        self.conv_skip1 = nn.Sequential(
            nn.Conv2d(c_list[0] * 2, c_list[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(c_list[0]),
            nn.GELU()
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])


        # Final layer
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        Feature_RGB = x[:, :3, :, :]  # 选择前三个通道作为 RGB 特征
        Feature_Depth = x[:, 3:, :, :]  # 选择最后一个通道作为 D 特征

        # Encoder
        F1_RGB = F.gelu(self.ebn1(self.encoder1_rgb(Feature_RGB)))
        F1_Depth = F.gelu(self.ebn1(self.encoder1_depth(Feature_Depth)))
        x1 = self.FFM_1(F1_RGB, F1_Depth)

        F2_RGB = F.gelu(self.ebn2(self.encoder2_rgb(F1_RGB)))
        F2_Depth = F.gelu(self.ebn2(self.encoder2_depth(F1_Depth)))
        x2 = self.FFM_2(F2_RGB, F2_Depth)

        F3_RGB = F.gelu(self.ebn3(self.encoder3_rgb(F2_RGB)))
        F3_Depth = F.gelu(self.ebn3(self.encoder3_depth(F2_Depth)))
        x3 = self.FFM_3(F3_RGB, F3_Depth)
        # 直接拼接 消融实验1 不使用DMFEM
        # F1_RGB = F.gelu(self.ebn1(self.encoder1_rgb(Feature_RGB)))
        # F1_Depth = F.gelu(self.ebn1(self.encoder1_depth(Feature_Depth)))
        # x1 = F1_RGB + F1_Depth
        #
        # F2_RGB = F.gelu(self.ebn2(self.encoder2_rgb(F1_RGB)))
        # F2_Depth = F.gelu(self.ebn2(self.encoder2_depth(F1_Depth)))
        # x2 = F2_RGB + F2_Depth
        #
        # F3_RGB = F.gelu(self.ebn3(self.encoder3_rgb(F2_RGB)))
        # F3_Depth = F.gelu(self.ebn3(self.encoder3_depth(F2_Depth)))
        # x3 = F3_RGB + F3_Depth

        x4 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(x3)), 2, 2))
        x5 = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(x4)), 2, 2))

        x6 = F.gelu(self.encoder6(x5))  # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(x6)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, x5)  # b, c4, H/32, W/32
        out5 = self.conv_skip5(torch.cat([out5, x5], dim=1))  # b, 2*c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, x4)  # b, c3, H/16, W/16
        out4 = self.conv_skip4(torch.cat([out4, x4], dim=1))  # b, 2*c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, x3)  # b, c2, H/8, W/8
        out3 = self.conv_skip3(torch.cat([out3, x3], dim=1))  # b, 2*c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, x2)  # b, c1, H/4, W/4
        out2 = self.conv_skip2(torch.cat([out2, x2], dim=1))  # b, 2*c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, x1)  # b, c0, H/2, W/2
        out1 = self.conv_skip1(torch.cat([out1, x1], dim=1))  # b, 2*c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return out0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DM_UNet()
    model = model.to(device)

    input_tensor = torch.randn(1, 4, 224, 224).to(device)

    from thop import profile

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.2f} G")  # 转换为Giga FLOPs
    print(f"Parameters: {params / 1e6:.2f} M")  # 转换为Million参数