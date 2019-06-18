import torch
import torch.nn as nn
import torch.nn.functional as F

from aspp import build_aspp
from decoder import build_decoder
from aspp import ASPPModule
from resnet18layers import ResNet18
from deeplab_resnet import ResNet101

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=21):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.ResNet = ResNet18()
        self.aspp = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

    def forward(self, input):
        x, low_level_feat = self.ResNet(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        return x



class InstanceBranch(nn.Module):
    def __init__(self, output_stride=16):
        super(InstanceBranch, self).__init__()
        BatchNorm = nn.BatchNorm2d
        inplanes = 2048

        self.ResNet = ResNet101()
        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=18, dilation=18, BatchNorm=BatchNorm)
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(256, 56, 1, bias=False)     # the final layer has 56 channels, which means 56 neighbor points
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        x,_ = self.ResNet(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1) 

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sigmoid(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x
     


