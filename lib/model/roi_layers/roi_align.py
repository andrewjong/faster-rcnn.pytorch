# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from model import _C

import pdb

class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply

count = 0

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        print("DEBUG: INIT ROIALIGN OBJECT")
        print(f"DEBUG: OUTPUT SIZE: {output_size}")
        print(f"DEBUG: SPATIAL_SCALE: {spatial_scale}")
        print(f"DEBUG: SAMPLING_RATIO: {sampling_ratio}")
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        global count
        if count % 10 == 0:
            print("DEBUG: INPUT SHAPE:", input.shape)
            print("DEBUG: INPUT DTYPE:", input.dtype)
            print("DEBUG: INPUT MIN:", input.min())
            print("DEBUG: INPUT MAX:", input.max())
            print("DEBUG: INPUT IS_CUDA:", input.is_cuda)
            print("DEBUG: ROIS SHAPE:", rois.shape)
            print("DEBUG: FIRST 2 ROIS:", rois[:2])
            print("DEBUG: ROIS IDS:", rois[:, 0])

        count+=1
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
