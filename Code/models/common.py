from torch import nn


def conv3x3(input_channels, out_channels, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(input_channels, out_channels, stride=1):
    """
    1x1 convolution
    """
    return nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=stride, bias=False)
