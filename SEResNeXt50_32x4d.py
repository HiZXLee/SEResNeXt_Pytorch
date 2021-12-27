"""
References:
1. https://github.com/moskomule/senet.pytorch
2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                    stride=stride, padding=dilation, 
                    groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
 
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                     stride=stride, bias=False)

########################################################################################################################################################

class SELayer(nn.Module):
    
    def __init__(
        self,
        channels,
        r = 16
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias = False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

########################################################################################################################################################


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

########################################################################################################################################################

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None):
    
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
   
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                              stride=2, padding=3,
                              bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ###############
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                           self.groups, self.base_width, previous_dilation,
                           norm_layer))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                               base_width=self.base_width,
                               dilation=self.dilation,
                               norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

#########################################################################################################################################

def seresnext50_32x4d(**kwargs):
    
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model

########################################################################################################################################

def run_test(batch_size=8, img_channel=3, img_h=256, img_w=256, n_classes=2):
    m = nn.Softmax(dim=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((batch_size, img_channel, img_h, img_w)).to(device)
    model = seresnext50_32x4d(num_classes=n_classes).to(device)
    output = model(x)
    
    print(f'Output: \n{output}\n')
    print(f'Output Softmax: \n{m(output)}\n')
    print(f'Output Shape: \n{output.shape}')
    return

run_test()
