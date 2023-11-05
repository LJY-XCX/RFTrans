import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class DecodeBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DecodeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.upsample(x)
        out = self.relu(self.bn(self.conv(out)))
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.en_layer1 = self.make_encode_layer(ResidualBlock, 64, 2, stride=1)
        self.en_layer2 = self.make_encode_layer(ResidualBlock, 128, 2, stride=2)
        self.en_layer3 = self.make_encode_layer(ResidualBlock, 256, 2, stride=2)
        self.en_layer4 = self.make_encode_layer(ResidualBlock, 512, 2, stride=2)
        self.de_layer1 = self.make_decode_layer(DecodeBlock, 256)
        self.de_layer2 = self.make_decode_layer(DecodeBlock, 128)
        self.de_layer3 = self.make_decode_layer(DecodeBlock, 64)
        self.de_layer4 = self.make_decode_layer(DecodeBlock, 3)

    def make_encode_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def make_decode_layer(self, block, channels):
        layer = block(self.inchannel, channels)
        self.inchannel = channels
        return layer


    def forward(self, x):
        out = self.conv1(x)
        out = self.en_layer1(out)
        out = self.en_layer2(out)
        out = self.en_layer3(out)
        out = self.en_layer4(out)
        out = F.avg_pool2d(out, 2)
        out = self.de_layer1(out)
        out = self.de_layer2(out)
        out = self.de_layer3(out)
        out = self.de_layer4(out)
        return out

if __name__ == '__main__':
    resnet = ResNet()
    input = torch.rand((16, 3, 512, 512))
    output = resnet(input)
    print(output.shape)
