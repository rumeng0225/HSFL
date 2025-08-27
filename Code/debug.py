import torch
from torch import nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

norm_layer = nn.BatchNorm2d
in_channels = 64
net = nn.Sequential(
    nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
    norm_layer(in_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
    norm_layer(in_channels),
    nn.ReLU(inplace=True),
    # output channle = inplanes * 2
    nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
    norm_layer(in_channels * 2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

dummy_tensor = torch.rand(128, 3, 64, 64)
print(net)
