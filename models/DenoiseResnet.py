import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as st
import numpy as np
from torch.nn import init

from .non_local_embedded_gaussian import NONLocalBlock2D


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, denoisemean, latent_size=512, denoise=[True,True,True,True,True]):
        super(ResNet, self).__init__()
        self.denoise = denoise
        self.in_planes = 64

        print('Latent_size of Encoder: {:.1f}'.format(latent_size))
        print('denoisemean is : {}'.format(denoisemean))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.denoise[0]:
            self.layer0denoise = NONLocalBlock2D(64, bn_layer=True)
        if self.denoise[1]:
            self.layer1denoise = NONLocalBlock2D(64, bn_layer=True)
        if self.denoise[2]:
            self.layer2denoise = NONLocalBlock2D(128, bn_layer=True)
        if self.denoise[3]:
            self.layer3denoise = NONLocalBlock2D(256, bn_layer=True)
        if self.denoise[4]:
            self.layer4denoise = NONLocalBlock2D(512, bn_layer=True)


        self.fc1 = nn.Sequential(nn.Linear(512*block.expansion, latent_size),
                                 nn.BatchNorm1d(latent_size),
                                 nn.ReLU(True)
                                )

    def init_nonlocal(self):
        if self.denoise[0]:
            nn.init.constant_(self.layer0denoise.W[1].weight, 0)
            nn.init.constant_(self.layer0denoise.W[1].bias, 0)

            nn.init.normal_(self.layer0denoise.theta.weight, 0.0, 0.01)
            nn.init.constant_(self.layer0denoise.theta.bias, 0.0)

            nn.init.normal_(self.layer0denoise.phi.weight, 0.0, 0.01)
            nn.init.constant_(self.layer0denoise.phi.bias, 0.0)

        if self.denoise[1]:
            nn.init.constant_(self.layer1denoise.W[1].weight, 0)
            nn.init.constant_(self.layer1denoise.W[1].bias, 0)

            nn.init.normal_(self.layer1denoise.theta.weight, 0.0, 0.01)
            nn.init.constant_(self.layer1denoise.theta.bias, 0.0)

            nn.init.normal_(self.layer1denoise.phi.weight, 0.0, 0.01)
            nn.init.constant_(self.layer1denoise.phi.bias, 0.0)

        if self.denoise[2]:
            nn.init.constant_(self.layer2denoise.W[1].weight, 0)
            nn.init.constant_(self.layer2denoise.W[1].bias, 0)

            nn.init.normal_(self.layer2denoise.theta.weight, 0.0, 0.01)
            nn.init.constant_(self.layer2denoise.theta.bias, 0.0)

            nn.init.normal_(self.layer2denoise.phi.weight, 0.0, 0.01)
            nn.init.constant_(self.layer2denoise.phi.bias, 0.0)

        if self.denoise[3]:
            nn.init.constant_(self.layer3denoise.W[1].weight, 0)
            nn.init.constant_(self.layer3denoise.W[1].bias, 0)

            nn.init.normal_(self.layer3denoise.theta.weight, 0.0, 0.01)
            nn.init.constant_(self.layer3denoise.theta.bias, 0.0)

            nn.init.normal_(self.layer3denoise.phi.weight, 0.0, 0.01)
            nn.init.constant_(self.layer3denoise.phi.bias, 0.0)

        if self.denoise[4]:
            nn.init.constant_(self.layer4denoise.W[1].weight, 0)
            nn.init.constant_(self.layer4denoise.W[1].bias, 0)

            nn.init.normal_(self.layer4denoise.theta.weight, 0.0, 0.01)
            nn.init.constant_(self.layer4denoise.theta.bias, 0.0)

            nn.init.normal_(self.layer4denoise.phi.weight, 0.0, 0.01)
            nn.init.constant_(self.layer4denoise.phi.bias, 0.0)




    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.denoise[0]:
            out = self.layer0denoise(out)

        out = self.layer1(out)
        if self.denoise[1]:
            out = self.layer1denoise(out)

        out = self.layer2(out)
        if self.denoise[2]:
            out = self.layer2denoise(out)

        out = self.layer3(out)
        if self.denoise[3]:
            out = self.layer3denoise(out)    

        out = self.layer4(out)
        if self.denoise[4]:
            out = self.layer4denoise(out)
  
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)

        embed_feat = self.fc1(out)

        return embed_feat



def ResnetEncoder(denoisemean, latent_size, denoise):
    return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], denoisemean=denoisemean, latent_size=latent_size, denoise=denoise)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])




class ResnetDecoder(nn.Module):
    def __init__(self, latent_size=512, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)

        self.Deconv = nn.Sequential(
            nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),  

            nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # nn.ConvTranspose2d(   128,      128, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1),
            nn.Tanh()        
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.fc1(x)
        x = x.resize(batch_size, 512, 2, 2)
        x = self.Deconv(x)

        return x


class NorClassifier(nn.Module):
    def __init__(self, latent_size=512, num_classes=10):
        super(self.__class__, self).__init__()

        self.norcls = nn.Linear(latent_size, num_classes)
                        

    def forward(self, x):
        out = self.norcls(x)
        return out



class SSDClassifier(nn.Module):
    def __init__(self, latent_size=512):
        super(self.__class__, self).__init__()

        self.rotcls = nn.Linear(latent_size, 4)
                            
                             

    def forward(self, x):
        out = self.rotcls(x)
        return out
