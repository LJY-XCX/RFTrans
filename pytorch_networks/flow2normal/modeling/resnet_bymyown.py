import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, filter_sz, state_num, in_channel):
        super(ResNetBlock, self).__init__()
        self.shortcut = nn.Identity() if filter_sz == 1 else nn.Sequential(nn.AvgPool2d(1, 1))
        self.pad = (filter_sz - 1) // 2
        self.conv1 = nn.Conv2d(in_channel, state_num, filter_sz, stride=1, padding=self.pad)
        self.bn1 = nn.BatchNorm2d(state_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(state_num, state_num, filter_sz, stride=1, padding=self.pad)
        self.bn2 = nn.BatchNorm2d(state_num)
        self.add = nn.Sequential(
            nn.Conv2d(in_channel, state_num, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(state_num)
        )
        
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out3 = self.shortcut(x)
        out3 = self.add(out3)
        out = torch.add(out2, out3)
        out = nn.ReLU(inplace=True)(out)
        return out


def createResNet(filter_sz, dim_list):

    model = nn.Sequential()
    for i in range(len(dim_list)-1):
        model.add_module('resNetBlock_{}'.format(i+1), ResNetBlock(filter_sz, dim_list[i+1], dim_list[i]))
    return model