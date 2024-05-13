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

        diffH = target.size()[2] - origin.size()[2]
        diffW = target.size()[3] - origin.size()[3]

        aligned = F.pad(origin, [diffW // 2, diffW - diffW // 2,
                                 diffH // 2, diffH - diffH // 2])
        return aligned

    return origin


class ResBlock(nn.Module):
    """
    Residual block with double convolution.
    Activate the output after shortcut connection.
    Output with the same height and width as input tensor.

    Use LeakyReLU to activate feature map because there are many negative values in PCA samples.
    For this condition, ReLU who converts all negative values to 0 may lose much information.
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activate1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activate2 = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, connect=True, activate=True):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out = align_tensor_shape(out, shortcut)

        if connect:
            """shortcut connection"""
            out += shortcut

        if activate:
            out = self.activate2(out)

        return out


class MultiBlock(nn.Module):
    """extract features with multiscale receptive fields"""

    def __init__(self, in_channels, out_channels):
        super(MultiBlock, self).__init__()

        self.block1 = ResBlock(in_channels, out_channels, dilation=1)
        self.block2 = ResBlock(in_channels, out_channels, dilation=2)
        self.block3 = ResBlock(in_channels, out_channels, dilation=3)
        self.reduce = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.activate = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x, connect=False, activate=False)
        x2 = self.block2(x, connect=False, activate=False)
        x3 = self.block3(x, connect=False, activate=False)
        shortcut = self.shortcut(x)

        addition = torch.cat([x1, x2, x3], dim=1)
        addition = self.reduce(addition)
        addition += shortcut

        out = self.activate(addition)

        return out


class Down(nn.Module):
    """
    Down sample by max pooling
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = MultiBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Up sample by conv transpose (more trainable parameters than interpolation)
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = MultiBlock(in_channels, out_channels)

    def forward(self, x, counterpart):
        x = self.up(x)

        x = align_tensor_shape(x, counterpart)

        """skip connection"""
        x = torch.cat([x, counterpart], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()

        self.dropout = nn.Dropout2d(p=0.5)  # prevent from overfitting

        """conv multi-bands to multi-classes"""
        self.reduce = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):

        x = self.dropout(x)
        x = self.reduce(x)

        return x


class ResUNet2D(nn.Module):
    """
    Extract spatial features from low-channels samples by applying 2d convolution.
    """

    def __init__(self, num_bands, num_classes):
        super(ResUNet2D, self).__init__()

        self.init = MultiBlock(num_bands, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, num_classes=num_classes)

    def forward(self, x):
        # encoder
        x1 = self.init(x)
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

