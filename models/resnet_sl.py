import torch
from torch import nn

from models.common import conv1x1
from models.resnet import Bottleneck, BasicBlock


class ResNet_client(nn.Module):
    def __init__(self, block, layers, num_classes=100, zero_init_residual=True, train_layer=1):
        super(ResNet_client, self).__init__()
        self.input_channels = 64

        # for training cifar, change the kernel_size=7 -> kernel_size=3 with stride=1
        self.layer0 = nn.Sequential(
            # nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = None
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        if train_layer < 2:
            self.layer1 = None
        if train_layer < 3:
            self.layer2 = None
        if train_layer < 4:
            self.layer3 = None
        if train_layer < 5:
            self.layer4 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    conv1x1(self.input_channels, out_channels * block.expansion, stride),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    # Ref:
                    # Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
                    # https://arxiv.org/abs/1812.01187
                    # https://github.com/rwightman/pytorch-image-models/blob
                    # /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
                    # nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, padding=0, count_include_pad=False),
                    conv1x1(self.input_channels, out_channels * block.expansion, stride=1),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

        layers = []
        layers.append(
            block(
                input_channels=self.input_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample))

        self.input_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.input_channels,
                    out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer0(x)
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x


class ResNet_server(nn.Module):
    def __init__(self, block, layers, num_classes=100, zero_init_residual=True, dropout_p=None, train_layer=1):
        super(ResNet_server, self).__init__()
        self.input_channels = 64

        self.layer0 = nn.Sequential(
            # nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layer0 = None
        if train_layer >= 2:
            self.layer1 = None
        if train_layer >= 3:
            self.layer2 = None
        if train_layer >= 4:
            self.layer3 = None
        if train_layer >= 5:
            self.layer4 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    conv1x1(self.input_channels, out_channels * block.expansion, stride),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    # Ref:
                    # Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
                    # https://arxiv.org/abs/1812.01187
                    # https://github.com/rwightman/pytorch-image-models/blob
                    # /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
                    # nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, padding=0, count_include_pad=False),
                    conv1x1(self.input_channels, out_channels * block.expansion, stride=1),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

        layers = []
        layers.append(
            block(
                input_channels=self.input_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample))

        self.input_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.input_channels,
                    out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


def ResNet18_client(num_classes=1000, train_layer=1):
    return ResNet_client(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, train_layer=train_layer)


def ResNet18_server(num_classes=1000, train_layer=1):
    return ResNet_server(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, train_layer=train_layer)


def ResNet34_client(num_classes=1000, train_layer=1):
    return ResNet_client(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, train_layer=train_layer)


def ResNet34_server(num_classes=1000, train_layer=1):
    return ResNet_client(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, train_layer=train_layer)
