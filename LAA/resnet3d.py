import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from LAA.RGA import RGA_Module


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     # stride=(1,stride,stride),
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     # stride=(1,stride,stride),
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 rga_mode=False,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,

                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = 32
        self.no_max_pool = no_max_pool
        self.resize = 128
        self.rga_mode=rga_mode
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], shortcut_type)
        self.rga1 = RGA_Module(128, (self.resize // 4) * (self.resize // 4), s_ratio=8, d_ratio=8)
        self.layer2 = self._make_layer(block, 64, layers[1], shortcut_type, stride=2)
        self.rga2 = RGA_Module(256, (self.resize // 8) * (self.resize // 8), s_ratio=8, d_ratio=8)
        self.layer3 = self._make_layer(block, 128, layers[2], shortcut_type, stride=2)
        self.rga3 = RGA_Module(512, (self.resize // 16) * (self.resize // 16), s_ratio=8, d_ratio=8)
        self.layer4 = self._make_layer(block, 256, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, 1)
        #self.dropout = nn.Dropout(0.15)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.rga_mode:
            x=self.rga1(x)
            if debug:
                print('rga1',x.shape)
        if debug:
            print(x.shape)
        x = self.layer2(x)
        if self.rga_mode:
            x=self.rga2(x)
            if debug:
                print('rga2', x.shape)
        if debug:
            print(x.shape)
        x = self.layer3(x)
        if self.rga_mode:
            x=self.rga3(x)
            if debug:
                print('rga3', x.shape)
        if debug:
            print(x.shape)
        x = self.layer4(x)
        if debug:
            print(x.shape)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #x = self.dropout(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model


# def generate_model(model_depth, **kwargs):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]
#
#     if model_depth == 10:
#         model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
#     elif model_depth == 18:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
#     elif model_depth == 34:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 50:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 101:
#         model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
#     elif model_depth == 152:
#         model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
#     elif model_depth == 200:
#         model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
#
#     return model

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    input_tensor = torch.autograd.Variable(torch.rand(16, 3, 16, 128, 128))
    input_tensor = input_tensor.to(device)
    model = resnet50(rga_mode=True).to(device)
    model.to(device)
    
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)
    output = model(input_tensor)
    print(output.shape)
    # model = MTD([3,4,14,3]).to(device)
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, f'model_dict_best910.pt')
