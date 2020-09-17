import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  padding=1):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(planes)
        self.resample = None
        if inplanes != planes:
            self.resample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.resample is not None:
            identity = self.resample(x)
        out += identity
        out = self.relu(out)

        return out

class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        self.layer1 = BasicBlock(1, 64)
        self.layer2 = BasicBlock(64, 128)
        self.layer3 = BasicBlock(128, 256)
        self.layer4 = BasicBlock(256, 361)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return x.view(-1, 361)

def test_model():
    net = GoNet()
    x = torch.randn(2, 1, 19, 19)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test_model()
