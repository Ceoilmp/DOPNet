import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d

class ClassificationModel(nn.Module):
    def __init__(self, in_channels, num_classes, num_points=1, softmax=True):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.cls_conv = nn.Conv2d(in_channels, num_classes * num_points, kernel_size=1)

        if softmax:
            self.output_act = nn.Softmax(dim=1)
        else:
            self.output_act = nn.Sigmoid()
    
    def forward(self, x):
        cls = self.cls_conv(x)
        cls = self.output_act(cls)
        
        return cls

class RegressionModel(nn.Module):
    def __init__(self, in_channels, num_classes, num_points=1):
        super(RegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points

        self.output_conf = nn.Conv2d(in_channels, 1 * num_points, kernel_size=1)
        self.output_conf_act = nn.Sigmoid()
        self.output_pos = nn.Conv2d(in_channels, 2 * num_points, kernel_size=1)
        self.output_pos_act = nn.Sigmoid()
    
    def forward(self, x):
        out_conf = self.output_conf(x)
        out_conf = self.output_conf_act(out_conf)
        out_pos = self.output_pos(x)
        out_pos = self.output_pos_act(out_pos)
        return out_conf, out_pos

class DPPNet_head(nn.Module):
    def __init__(self, in_channels, num_classes=2, deformable=True):
        super(DPPNet_head, self).__init__()
        self.deformable = deformable
        self.classificationBranch = ClassificationModel(in_channels, num_classes)
        self.regressionBranch = RegressionModel(in_channels, num_classes)

        self.RepPoints2_offsets = nn.Conv2d(in_channels, 18, kernel_size=3, stride=1, padding=1)
        self.RepPoints2 = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, input_feature):

        cls_result = self.classificationBranch(input_feature)
        
        offset2 = self.RepPoints2_offsets(input_feature)
        
        conf_result, pos_result = self.regressionBranch(input_feature)

        result = torch.cat([conf_result, cls_result, pos_result], dim=1)
        return result, offset2
