import torch
from torch import nn
from mmcv.ops import box_iou_rotated, min_area_polygons, points_in_polygons

class Post_processing(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, offset1, offset_fin):
        outputs, offset1, offset_fin = self.outputs_decoder(outputs, offset1, offset_fin)

        offset_fin = offset_fin + offset1
        cnt, result, rect = self.get_cnt_nms(outputs,offset_fin)
        return cnt, result, rect

    @torch.no_grad()
    def get_cnt_nms(self, outputs, offset_fin, conf_threshold=0.7):
        outputs, rect = self.get_final_result(outputs, offset_fin, conf_threshold)

        cls = outputs[:,1:self.num_classes+1]
        
        max_cls = cls.max(dim=1).values
        mask = []
        result = []
        cnt = []

        for i in range(self.num_classes):
            mask_ = cls[:,i] == max_cls
            mask.append(mask_)
            result.append(outputs[mask_][:,self.num_classes+1:])
            cnt.append(outputs[mask_].shape[0])

        return cnt, result, rect

    def get_final_result(self, outputs, offset_fin, conf_threshold): # NMS
        
        x = outputs[:,self.num_classes+1].unsqueeze(1)
        y = outputs[:,self.num_classes+2].unsqueeze(1)
        offset_add = torch.cat([  y,x, y,x, y,x, 
                                    y,x, y,x, y,x, 
                                    y,x, y,x, y,x], dim=1)
        offset_fin = offset_fin + offset_add

        # decode offset_fin
        offset_fin_x_max, _ = offset_fin[:,1:18:2].max(dim=1)
        offset_fin_x_min, _ = offset_fin[:,1:18:2].min(dim=1)
        offset_fin_y_max, _ = offset_fin[:,0:18:2].max(dim=1)
        offset_fin_y_min, _ = offset_fin[:,0:18:2].min(dim=1)
        offset_area = (offset_fin_x_max - offset_fin_x_min) * (offset_fin_y_max - offset_fin_y_min) / 40

        conf_mask = outputs[:,0] > (conf_threshold * torch.sigmoid(offset_area))
        outputs = outputs[conf_mask]
        conf = outputs[:,0]
        position = outputs[:,self.num_classes+1:]
        offset_fin = offset_fin[conf_mask]
        if offset_fin.shape[0] != 0:
            rect = min_area_polygons(offset_fin)
        else:
            rect = None

        _, order = conf.sort(0, descending=True)

        keep = []

        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)
            
            point1 = offset_fin[i]
            points = position[order[1:]]
            dis = self.cal_min_area_point(point1, points)
            dis_mask = (dis == 0)
            point1 = self.box_decoder(rect[i].unsqueeze(0))
            points = self.box_decoder(rect[order[1:]])
            iou = self.cal_iou(point1, points)
            iou_mask = (iou < 0.2)
            dis_mask[~iou_mask] = False
            order = order[1:]
            order = order[dis_mask]

        if offset_fin.shape[0] != 0:
            return outputs[keep,:], rect[keep,:]
        else:
            return outputs[keep,:], rect

    def outputs_decoder(self, outputs, offset1, offset_fin):
        L3_output = outputs[0].squeeze()
        L3_offset1 = offset1[0].squeeze()
        L3_offset_fin = offset_fin[0].squeeze()
        c, h, w = L3_output.shape
        w3 = torch.arange(0,4*w,4).unsqueeze(0).repeat(h,1).cuda()
        h3 = torch.arange(0,4*h,4).unsqueeze(1).repeat(1,w).cuda()
        L3_output[self.num_classes + 1,:,:] = L3_output[self.num_classes + 1,:,:] * 4 + w3
        L3_output[self.num_classes + 2,:,:] = L3_output[self.num_classes + 2,:,:] * 4 + h3
        
        L4_output = outputs[1].squeeze()
        L4_offset1 = offset1[1].squeeze()
        L4_offset_fin = offset_fin[1].squeeze()
        c, h, w = L4_output.shape
        w4 = torch.arange(0,8*w,8).unsqueeze(0).repeat(h,1).cuda()
        h4 = torch.arange(0,8*h,8).unsqueeze(1).repeat(1,w).cuda()
        L4_output[self.num_classes + 1,:,:] = L4_output[self.num_classes + 1,:,:] * 8 + w4
        L4_output[self.num_classes + 2,:,:] = L4_output[self.num_classes + 2,:,:] * 8 + h4

        L5_output = outputs[2].squeeze()
        L5_offset1 = offset1[2].squeeze()
        L5_offset_fin = offset_fin[2].squeeze()
        c, h, w = L5_output.shape
        w5 = torch.arange(0,16*w,16).unsqueeze(0).repeat(h,1).cuda()
        h5 = torch.arange(0,16*h,16).unsqueeze(1).repeat(1,w).cuda()
        L5_output[self.num_classes + 1,:,:] = L5_output[self.num_classes + 1,:,:] * 16 + w5
        L5_output[self.num_classes + 2,:,:] = L5_output[self.num_classes + 2,:,:] * 16 + h5
        
        L3_output = L3_output.view(self.num_classes + 3,-1).T.contiguous()
        L4_output = L4_output.view(self.num_classes + 3,-1).T.contiguous()
        L5_output = L5_output.view(self.num_classes + 3,-1).T.contiguous()

        L3_offset1 = L3_offset1.view(18,-1).T.contiguous()
        L3_offset_fin = L3_offset_fin.view(18,-1).T.contiguous()

        L4_offset1 = L4_offset1.view(18,-1).T.contiguous()
        L4_offset_fin = L4_offset_fin.view(18,-1).T.contiguous()

        L5_offset1 = L5_offset1.view(18,-1).T.contiguous()
        L5_offset_fin = L5_offset_fin.view(18,-1).T.contiguous()

        result = torch.cat([L3_output, L4_output, L5_output], dim=0)
        offset1 = torch.cat([L3_offset1, L4_offset1, L5_offset1], dim=0)
        offset_fin = torch.cat([L3_offset_fin, L4_offset_fin, L5_offset_fin], dim=0)
        return result, offset1, offset_fin

    def cal_iou(self, box1, boxs):
        iou = box_iou_rotated(box1, boxs).squeeze()
        return iou

    def cal_min_area_point(self, point1, points):
        point1 = point1.unsqueeze(0)
        offset_fin_box = min_area_polygons(point1) # return n * 8

        points = torch.cat([points[:,1].unsqueeze(1), points[:,0].unsqueeze(1)], dim=1)
        mask = points_in_polygons(points, offset_fin_box)
        return mask.squeeze()

    def box_decoder(self, box, y_first=True):
        # box n * 8
        width = torch.sqrt((box[:, 0] - box[:, 2]) * (box[:, 0] - box[:, 2]) + (box[:, 1] - box[:, 3]) * (box[:, 1] - box[:, 3]))
        height = torch.sqrt((box[:, 4] - box[:, 2]) * (box[:, 4] - box[:, 2]) + (box[:, 5] - box[:, 3]) * (box[:, 5] - box[:, 3]))
        center_x = box[:, 1:8:2].sum(dim=1) / 4
        center_y = box[:, 0:8:2].sum(dim=1) / 4
        if y_first:
            angle = torch.atan((box[:, 0] - box[:, 2])/(box[:, 1] - box[:, 3]))
            return torch.cat([center_x.unsqueeze(1), center_y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1), angle.unsqueeze(1)], dim=1)
        else:
            angle = torch.atan((box[:, 1] - box[:, 3])/(box[:, 0] - box[:, 2]))
            return torch.cat([center_y.unsqueeze(1), center_x.unsqueeze(1), height.unsqueeze(1), width.unsqueeze(1), angle.unsqueeze(1)], dim=1)