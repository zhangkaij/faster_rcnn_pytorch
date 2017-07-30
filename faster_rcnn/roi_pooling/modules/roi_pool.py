import torch
from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)


def data_select(input, dim, pooled_len, data_len):
    """
    
    """
    shift = 1.0 * data_len / pooled_len * torch.arange(0, pooled_len+1)
    shift_start = torch.floor(shift).long()[:-1]
    shift_end   = torch.ceil(shift).long()[1:]
    shift_stride = shift_end - shift_start + 1
    stride_max = torch.max(shift_stride)
    
    shift_index = torch.arange(0, stride_max).long().unsqueeze(0).expand(pooled_len, stride_max).contiguous().view(1, -1).squeeze()
    index_ceil = shift_stride.unsqueeze(2-dim).expand(pooled_len, stride_max).contiguous().view(1, -1).squeeze()
    shift_index[shift_index.ge(index_ceil)] = 0
    
    index_start = shift_start.unsqueeze(1).expand(pooled_len, stride_max).contiguous().view(1, -1).squeeze()
    index = index_start + shift_index
    index = torch.clamp(index, max=(data_len-1))
    #没有排除 整个m*n区域中为相同的值，且为边界处的值
    output = torch.index_select(input, dim, index)
    return output, stride_max

def max_pool(input, w_stride, h_stride):

    C, H, W = input.size()

    output = input.view(C, H, int(W/w_stride), w_stride).permute(0, 2, 1, 3).contiguous().view(-1, h_stride, w_stride)\
            .contiguous().view(-1, h_stride*w_stride)
    output, _ = torch.max(output, 1)
    output = output.view(C, int(W/w_stride), int(H/h_stride)).permute(0, 2, 1)
    
    return output
  

def RoIPool_test(input, rois, pooled_height=7, pooled_width=7, spatial_scale=1/16.0):
    
    rois = torch.round(rois*spatial_scale)
    rois_num = rois.size(0)
    _, C, _, _ = input.size()
    input = input.squeeze(0)

    pooled_features = torch.zeros(rois_num, C, pooled_height, pooled_width)
    for i, roi in enumerate(rois):
        
        roi_start_w = roi[1]
        roi_start_h = roi[2]
        roi_end_w = roi[3]
        roi_end_h = roi[4]
        W = torch.clamp(roi_end_w-roi_start_w+1, min=1)
        H = torch.clamp(roi_end_h-roi_start_h+1, min=1)
        index_column = torch.arange(roi_start_w, roi_end_w+1).long()
        data = torch.index_select(input, dim=1, index=index_column)
        index_row = torch.arange(roi_start_h, roi_end_h+1).long()
        data = torch.index_select(data, dim=0, index=index_row)
        # 先选择列
        output_col, w_stride= data_select(data, dim=2, pooled_len=pooled_width, data_len=W)
        # 选择行
        outpu_cr, h_stride = data_select(output_col, dim=1, pooled_len=pooled_height, data_len=H)
        output[i] = max_pool(output, w_stride, h_stride)

    return output





