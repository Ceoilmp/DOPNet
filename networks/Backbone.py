import torch.nn as nn
from torchvision import models

class backbone(nn.Module):
    def __init__(self, name='vgg16_bn', pretrained=True):
        super(backbone, self).__init__()

        if name == 'vgg16_bn':
            vgg = models.vgg16_bn(pretrained=pretrained)
        elif name == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)

        features = list(vgg.features.children())
        # get each stage of the backbone
        if name == 'vgg16_bn':
            self.features1 = nn.Sequential(*features[0:23])
            self.features2 = nn.Sequential(*features[23:33])
            self.features3 = nn.Sequential(*features[33:43])
        elif name == 'vgg16':
            self.features1 = nn.Sequential(*features[0:16])
            self.features2 = nn.Sequential(*features[16:23])
            self.features3 = nn.Sequential(*features[23:30])

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        return [x1, x2, x3]

def build_backbone(name='vgg16', pretrained=True):
    dppnet_backbone = backbone(name, pretrained)
    return dppnet_backbone
