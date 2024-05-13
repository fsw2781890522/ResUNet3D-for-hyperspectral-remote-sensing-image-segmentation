import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_tensor_shape(shape: int, padding: int, kernel_size: int, stride: int):
    return np.floor((shape + (2 * padding) - kernel_size) / stride) + 1


def align_tensor_shape(origin: torch.tensor, target: torch.tensor):
    """
    Modify the shape of a tensor for skip connection.(tensors with different shape cannot be sum up or concatenated.)
    nn.Conv, nn.ConvTranspose, nn.Upsample or F.interpolate not always output tensor with expected shape
    even though hyperparameters like 'scale_factor' have been appointed, maybe because of the rounding strategy.
    Output shape can be directly assigned in nn.Upsample and F.interpolate (using param 'size'),
    but unable in convolution related layers,
    where output shape is depend on kernel size, dilation, padding, stride.
    """

    if origin.shape != target.shape:
        diffD = target.size()[2] - origin.size()[2]
        diffH = target.size()[3] - origin.size()[3]
        diffW = target.size()[4] - origin.size()[4]

        aligned = F.pad(origin, [diffW // 2, diffW - diffW // 2,
                                 diffH // 2, diffH - diffH // 2,
                                 diffD // 2, diffD - diffD // 2]
                        # mode='zeros'
                        )
        return aligned

    return origin


class ResBlock(nn.Module):
    """
    Residual block with double convolution.
    Activate the output after shortcut connection.
    Output with the same height and width as input tensor.
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        """
        when kernel size = 3 and stride = 1, let padding = dilation to keep the original shape
        """
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation,
                               padding_mode='zeros', bias=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation,
                               padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, connect=True, activate=True):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out = align_tensor_shape(out, shortcut)

        if connect:
            """shortcut connection"""
            out += shortcut

        if activate:
            out = self.relu2(out)

        return out


class MultiBlock(nn.Module):
    """extract features with multiscale receptive fields"""

    def __init__(self, in_channels, out_channels):
        super(MultiBlock, self).__init__()

        self.block1 = ResBlock(in_channels, out_channels, dilation=1)
        self.block2 = ResBlock(in_channels, out_channels, dilation=2)
        self.block3 = ResBlock(in_channels, out_channels, dilation=3)
        self.relu = nn.ReLU(inplace=True)
        self.reduce = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        # self.dropout = nn.Dropout3d(p=0.1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x1 = self.block1(x, connect=False, activate=False)
        # x2 = self.block2(x, connect=False, activate=False)
        # x3 = self.block3(x, connect=False, activate=False)

        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)

        shortcut = self.shortcut(x)

        out = torch.cat([x1, x2, x3], dim=1)
        # out = x1 + x2 + x3
        out = self.reduce(out)
        # out = self.dropout(out)
        out += shortcut

        # out = self.relu(addition)

        return out


class MixedPool3d(nn.Module):
    """trainable mixed pooling"""

    def __init__(self, kernel_size):
        super(MixedPool3d, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size)
        self.max_pool = nn.MaxPool3d(kernel_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始设定加权系数为0.5，可以通过训练进行调整

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        mixed_out = self.alpha * avg_out + (1 - self.alpha) * max_out  # 线性加权

        return mixed_out


class Down(nn.Module):
    """
    Down sample by mixed pooling
    """

    def __init__(self, in_channels, out_channels, multi_field=False, mix_pool=False):
        super(Down, self).__init__()
        if mix_pool:
            self.pool = MixedPool3d(2)
        else:
            self.pool = nn.MaxPool3d(2)

        if multi_field:
            self.conv = MultiBlock(in_channels, out_channels)
        else:
            self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Up sample by conv transpose (more trainable parameters than interpolation)
    """

    def __init__(self, in_channels, out_channels, multi_field=False):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if multi_field:
            self.conv = MultiBlock(in_channels, out_channels)
        else:
            self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x, counterpart):
        x = self.up(x)

        x = align_tensor_shape(x, counterpart)

        """skip connection"""
        x = torch.cat([x, counterpart], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, num_bands, num_classes):
        super(OutConv, self).__init__()

        """reduce the channels to 1"""
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.5)  # prevent from overfitting

        """conv multi-bands to multi-classes"""
        self.reduce = nn.Conv2d(num_bands, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)

        """
        remove the channels dimension then depth (actually nBands) will be channels
        (N, 1, D, H, W) -> (N, C, H, W), with D -> C
        """
        x = x.squeeze(1)

        return self.reduce(x)


class ResUNet3D(nn.Module):
    """
    Simultaneously extract joint spatial-spectral features from hyperspectral samples by applying 3D convolution.

    Features:
        1. 3D convolution for hyperspectral samples
        2. Multiscale receptive fields (optional)
        3. Residual shortcut connection
        4. Trainable mixed pooling (optional)
        5. Skip connection between encoder and decoder

    Note:
        A 3D kernel in PyTorch only work at (D, H, W), D (depth) dimension is designed for videos, time series, etc.
        Hence, spectral features won't be extracted if we set the number of bands as channels.
        A trick is to regard a hyperspectral image as a sequence stacked by single-channel images,
        then the dimension 'depth' represents num of bands so that we can apply 3D convolution.
        In the last layer of this network, we reduce the number of channels to 1,
        then remove the channels dimension, the depth (actually nBands) will be channels, like this:

            expand the dimension of input samples:
                (N, C, H, W) -> (N, C, D, H, W)
            but set the number of channels as depth to participate in 3D convolution
            so that spectral features can be extracted,
            and let channels = 1
            e.g. (10, 188, 16, 16) -> (10, 1, 188, 16, 16)
                   |   |    |   |       |  |   |   |   |
                 (N,   C,   H,  W) -> ( N, C,  D,  H,  W)

            after feature extraction:
                (N, 1, D, H, W) -> (N, C, H, W), with D -> C

    """

    def __init__(self, num_bands, num_classes):
        super(ResUNet3D, self).__init__()

        self.pre = MultiBlock(1, 64)

        self.down1 = Down(64, 128, multi_field=True, mix_pool=False)
        self.down2 = Down(128, 256, multi_field=True, mix_pool=False)
        self.down3 = Down(256, 512, multi_field=True, mix_pool=False)
        # self.down4 = Down(512, 1024)
        # self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256, multi_field=True)
        self.up3 = Up(256, 128, multi_field=True)
        self.up4 = Up(128, 64, multi_field=True)

        self.outc = OutConv(64, num_bands=num_bands, num_classes=num_classes)

    def forward(self, x):
        # preprocess
        """(N, C, H, W) -> (N, 1, C->D, H, W)"""
        if len(x.size()) < 5:
            x = x.unsqueeze(1)
        x1 = self.pre(x)

        # encoder
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # decoder
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # segmentation
        out = self.outc(x)

        return F.softmax(out, dim=1)

        # # encoder
        # x1 = self.conv1(x)
        # x2 = self.pool1(x1)
        # x3 = self.conv2(x2)
        # x4 = self.pool2(x3)
        # x5 = self.conv3(x4)
        # x6 = self.pool3(x5)
        # x7 = self.conv4(x6)
        # x8 = self.pool4(x7)
        # x9 = self.conv5(x8)
        #
        # # decoder
        # x = self.up1(x9)
        # # x = F.interpolate(x9, size=x7.size()[2:], mode='nearest')
        # x = torch.cat([x, x7], dim=1)
        # x = self.conv6(x)
        # x = self.up2(x)
        # # x = F.interpolate(x, size=x5.size()[2:], mode='nearest')
        # x = torch.cat([x, x5], dim=1)
        # x = self.conv7(x)
        # x = self.up3(x)
        # # x = F.interpolate(x, size=x3.size()[2:], mode='nearest')
        # x = torch.cat([x, x3], dim=1)
        # x = self.conv8(x)
        # x = self.up4(x)
        # # x = F.interpolate(x, size=x1.size()[2:], mode='nearest')
        # x = torch.cat([x, x1], dim=1)
        # x = self.conv9(x)

# self.conv1 = ResBlock(1, 64)
# self.pool1 = nn.MaxPool3d(2)
# self.conv2 = ResBlock(64, 128)
# self.pool2 = nn.MaxPool3d(2)
# self.conv3 = ResBlock(128,256)
# self.pool3 = nn.MaxPool3d(2)
# self.conv4 = ResBlock(256, 512)
# self.pool4 = nn.MaxPool3d(2)
# self.conv5 = ResBlock(512, 1024)
# self.up1 = nn.ConvTranspose3d(1024, 1024, kernel_size=2, stride=2)
# self.conv6 = ResBlock(1024 + 512, 512)
# self.up2 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
# self.conv7 = ResBlock(512 + 256, 256)
# self.up3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
# self.conv8 = ResBlock(256 + 128, 128)
# self.up4 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
# self.conv9 = ResBlock(128 + 64, 64)


#
#
# class Up(nn.Module):
#     """
#     Up sample by nearest
#     up sample -> skip connection -> residual block
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super(Up, self).__init__()
#         self.conv = ResBlock(in_channels, out_channels)
#
#     def forward(self, x, counterpart: torch.tensor):
#         """up sample on D, H, W"""
#         x = F.interpolate(x, size=counterpart.size()[2:], mode='nearest')
#         """Concatenate with skip connection"""
#         x = torch.cat([x, counterpart], dim=1)
#         return self.conv(x)


# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(out_channels)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)  # 添加残差连接的卷积层
#
#     def forward(self, x):
#         residual = self.residual(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual  # 将残差连接添加到卷积结果中
#         out = self.relu2(out)
#         return out
