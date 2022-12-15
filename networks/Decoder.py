import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from mmcv.cnn.utils.weight_init import kaiming_init

class Decoder(nn.Module):
    def __init__(self, deformable=True, in_channels=[256, 512, 512], feature_size=128):
        super(Decoder, self).__init__()
        self.deformable = deformable
        self.conv_sub_L5 = nn.Conv2d(in_channels[2], feature_size, kernel_size=1, stride=1)
        self.conv_sub_L4 = nn.Conv2d(in_channels[1] + feature_size, feature_size, kernel_size=1, stride=1)
        self.conv_sub_L3 = nn.Conv2d(in_channels[0] + feature_size, feature_size, kernel_size=1, stride=1)

        self.conv_sub_L5_dilation = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=2, dilation=2)
        self.conv_sub_L4_dilation = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=2, dilation=2)
        self.conv_sub_L3_dilation = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=2, dilation=2)

        self.L4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.L5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.offset_L3 = nn.Conv2d(feature_size , 18, kernel_size=3, stride=1, padding=1)
        self.L3_conv = DeformConv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.offset_L4 = nn.Conv2d(feature_size , 18, kernel_size=3, stride=1, padding=1)
        self.L4_conv = DeformConv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.offset_L5 = nn.Conv2d(feature_size , 18, kernel_size=3, stride=1, padding=1)
        self.L5_conv = DeformConv2d(feature_size, feature_size, kernel_size=3, padding=1)
        

        self.init_weights()
    
    def forward(self, inputs):
        L3, L4, L5 = inputs

        L5_out = self.conv_sub_L5(L5)
        L5_cat = self.L5_upsampled(L5_out)
        L5_out = self.conv_sub_L5_dilation(L5_out)
        if self.deformable:
            L5_offset = self.offset_L5(L5_out)
            L5_out = self.L5_conv(L5_out, L5_offset)
        else:
            L5_out = self.L5_conv(L5_out)
        
        L4 = torch.cat([L5_cat, L4], dim=1)
        L4_out = self.conv_sub_L4(L4)
        L4_cat = self.L4_upsampled(L4_out)
        L4_out = self.conv_sub_L4_dilation(L4_out)
        if self.deformable:
            L4_offset = self.offset_L4(L4_out)
            L4_out = self.L4_conv(L4_out, L4_offset)
        else:
            L4_out = self.L4_conv(L4_out)

        L3 = torch.cat([L4_cat, L3], dim=1)
        L3_out = self.conv_sub_L3(L3)
        L3_out = self.conv_sub_L3_dilation(L3_out)
        if self.deformable:
            L3_offset = self.offset_L3(L3_out)
            L3_out = self.L3_conv(L3_out, L3_offset)
        else:
            L3_out = self.L3_conv(L3_out)

        if self.deformable:
            return [L3_out, L4_out, L5_out], L3_offset, L4_offset, L5_offset
        else:
            return [L3_out, L4_out, L5_out]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def init_offset(self):
        self.offset_L3.weight.data.zero_()
        self.offset_L3.bias.data.zero_()
        self.offset_L4.weight.data.zero_()
        self.offset_L4.bias.data.zero_()
        self.offset_L5.weight.data.zero_()
        self.offset_L5.bias.data.zero_()
        