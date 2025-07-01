import torch
import torch.nn as nn
import math
from vmamba.vmamba import VSSBlock,SS2D

# FLOPs计算函数
def compute_conv2d_flops(layer, input_tensor):
    # 计算Conv2d层的FLOPs
    out_channels, in_channels, kernel_height, kernel_width = layer.weight.shape
    batch_size, in_channels, input_height, input_width = input_tensor.shape

    # 计算输出特征图的高度和宽度
    output_height = (input_height + 2 * layer.padding[0] - kernel_height) // layer.stride[0] + 1
    output_width = (input_width + 2 * layer.padding[1] - kernel_width) // layer.stride[1] + 1

    # 每个卷积操作需要in_channels * kernel_height * kernel_width次乘法
    flops_per_output = in_channels * kernel_height * kernel_width

    # 每个输出像素有out_channels个输出通道
    total_flops = flops_per_output * output_height * output_width * out_channels * batch_size
    return total_flops

def compute_linear_flops(layer, input_tensor):
    # 计算Linear层的FLOPs
    batch_size = input_tensor.shape[0]
    in_features = layer.in_features
    out_features = layer.out_features

    # 每个元素都需要in_features次乘法
    flops_per_sample = in_features * out_features

    # 总的FLOPs
    total_flops = flops_per_sample * batch_size
    return total_flops

def compute_activation_flops(x):
    # 激活函数的FLOPs通常与输入tensor的元素数成正比
    return x.numel()

def calculate_flops(model, input_tensor):
    total_flops = 0

    # 遍历每一层，计算对应的FLOPs
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            total_flops += compute_conv2d_flops(layer, input_tensor)
        elif isinstance(layer, nn.Linear):
            total_flops += compute_linear_flops(layer, input_tensor)
        elif isinstance(layer, nn.SiLU) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.GELU):
            total_flops += compute_activation_flops(input_tensor)

    return total_flops

# 示例代码
if __name__ == "__main__":
    # 模型实例化
    model = SS2D(
            d_model=96,  # 主维度大小
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

    # 假设输入张量的维度为(batch_size=1, d_model=96, height=32, width=32)
    input_tensor = torch.randn(1, model.d_model, 32, 32)

    # 计算FLOPs
    flops = calculate_flops(model, input_tensor)
    print(f"Total FLOPs: {flops}")
