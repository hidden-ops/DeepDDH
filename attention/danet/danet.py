from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import upsample, normalize
from attention.attention import CAM_Module, PAM_Module, FeatureFusion
from attention.seg_opr.seg_oprs import ConvBnRelu
from ddhnet_model.attention_module import PAM_SE_Module, CAM_SE_Module

from attention.danet.base import BaseNet

__all__ = ['DANet', 'get_danet']


class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    Reference:
        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015
    """

    def __init__(self, n_classes, n_channels, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(n_classes, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, n_classes, norm_layer)
        self.n_classes = n_classes
        self.n_channels = n_channels

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)

class danet_DDNet(BaseNet):

    def __init__(self, n_classes, n_channels, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(danet_DDNet, self).__init__(n_classes, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        #self.head = DANetHead(2048, n_classes, norm_layer)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.refine3x3 = ConvBnRelu(2048, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        self.refine1x1 = ConvBnRelu(512, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        # self.fpa = FPA(channels=512)
        self.CA = CAM_SE_Module(in_dim=512, dropout=1.0)  # CAM_Module(in_dim=512)
        self.PA = PAM_Module(in_dim=512)
        # self.final_FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.FFM = FeatureFusion(in_planes=1024, out_planes=1024)
        self.low_FFM = FeatureFusion(in_planes=1280, out_planes=512)
        self.output_head = CoordHead(514, n_classes, 4, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        refine = self.refine3x3(c4)
        refine = self.refine1x1(refine)
        # FPA = self.fpa(refine)
        # final_fm = self.final_FFM(refine, FPA)
        ca = self.CA(refine)
        pa = self.PA(refine)
        ffm = self.FFM(ca, pa)
        fm = F.interpolate(ffm, size=c1.shape[2:], mode="bilinear", align_corners=True)
        # h = torch.cat((fm, context_blocks[2]), dim=1)
        h = self.low_FFM(fm, c1)
        h = self.output_head(h)
        return h

class CoordHead(nn.Module):
    def __init__(self, in_planes, n_classes, scale, norm_layer=nn.BatchNorm2d):
        super(CoordHead, self).__init__()
        self.conv1_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

        self.conv1_1x1 = nn.Conv2d(256, n_classes, kernel_size=1,
                                      stride=1, padding=0)

        self.scale = scale

    def forward(self, x):
        x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
        y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
        Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
        Y = Y.expand([x.shape[0], 1, -1, -1])
        X = X.expand([x.shape[0], 1, -1, -1])
        coord_feat_1 = torch.cat([X, Y], 1)
        fm = torch.cat([x, coord_feat_1], 1)
        fm = self.conv1_3x3(fm)
        output = self.conv1_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


if __name__ == "__main__":
    model = DANet(n_classes=8, n_channels=3, backbone='resnet50')
    ddhnet_model = danet_DDNet(n_classes=8, n_channels=3, backbone='resnet50')
    image = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 8, 256, 256)
    main_pred = model(image)
    ddh_pred = ddhnet_model(image)
    print(model)
    print(main_pred[0].size(), main_pred[1].size())
    print('ddhnet:', ddh_pred.size())
