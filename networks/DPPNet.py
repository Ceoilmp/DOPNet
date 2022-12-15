import torch.nn as nn
from .Backbone import build_backbone
from .Decoder import Decoder
from .Head import DPPNet_head

class DPPNet(nn.Module):
    def __init__(self, feature_size=128, num_classes=2, backbone='vgg16', deformable=True):
        super(DPPNet, self).__init__()
        self.backbone = build_backbone(name=backbone)
        self.decoder = Decoder(in_channels=[256, 512, 512], feature_size=feature_size, deformable=deformable)
        self.head_4 = DPPNet_head(in_channels=feature_size, num_classes=num_classes, deformable=deformable)
        self.head_8 = DPPNet_head(in_channels=feature_size, num_classes=num_classes, deformable=deformable)
        self.head_16 = DPPNet_head(in_channels=feature_size, num_classes=num_classes, deformable=deformable)

    def forward(self, x):
        features = self.backbone(x)
        features, L3_offset1, L4_offset1, L5_offset1 = self.decoder(features)
        L3_out, L3_offset2 = self.head_4(features[0])
        L4_out, L4_offset2 = self.head_8(features[1])
        L5_out, L5_offset2 = self.head_16(features[2])
        return [L3_out, L4_out, L5_out], [L3_offset1 * 4, L4_offset1 * 8, L5_offset1 * 16], [L3_offset2 * 4, L4_offset2 * 8, L5_offset2 * 16]