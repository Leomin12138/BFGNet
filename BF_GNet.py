import torch.nn as nn
import torch

# 假设 VSSM 已经正确定义并能返回合适的特征图
from inference.models.grasp_model import GraspModel
from UltraLight_BF_GNet import UltraLight_BF_UNet


class BF_GNet(GraspModel):
    def __init__(self, input_channels=1,output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(BF_GNet, self).__init__()

        # VSSM 模块
        self.VM_GNet = UltraLight_BF_UNet(input_channels=input_channels, num_classes = channel_size)

        # 输出层定义
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=1)

        # Dropout 模块
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:  # 初始化 bias
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # VSSM 模块前向传播
        x = self.VM_GNet(x)

        # print("VSSM Output Shape:", x.shape)  # 打印输出的形状

        # 根据 dropout 标志选择是否应用 Dropout
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        # 返回多个输出
        return pos_output, cos_output, sin_output, width_output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BF_GNet(input_channels=4)
    model = model.to(device)

    input_tensor = torch.randn(1, 4, 224, 224).to(device)

