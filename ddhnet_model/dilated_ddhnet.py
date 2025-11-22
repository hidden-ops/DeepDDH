# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.seg_opr.seg_oprs import ConvBnRelu
from attention.attention import CAM_Module, PAM_Module, FeatureFusion
from ddhnet_model.attention_module import PAM_SE_Module, CAM_SE_Module
from ddhnet_model.dilated_resnet import resnet50

num_classes = 8
bn_eps = 1e-5
bn_momentum = 0.1


def get():
    return dilated_DDHNet(num_classes, None, None)

class dilated_DDHNet(nn.Module):
    def __init__(self, n_classes, n_channels, pretrained_model=True, dilated=True, norm_layer=nn.BatchNorm2d,
                 root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(dilated_DDHNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.context_path = resnet50(pretrained=pretrained_model, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)

        self.refine3x3 = ConvBnRelu(2048, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        self.refine1x1 = ConvBnRelu(512, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        # self.fpa = FPA(channels=512)
        self.CA = CAM_SE_Module(in_dim=512, dropout=1.0)#CAM_Module(in_dim=512)
        self.PA = PAM_Module(in_dim=512)
        # self.final_FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.FFM = FeatureFusion(in_planes=1024, out_planes=1024)
        self.low_FFM = FeatureFusion(in_planes=1280, out_planes=512)
        self.output_head = CoordHead(514, n_classes, 4, norm_layer)

    def forward(self, data):
        # spatial_out = self.spatial_path(data)
        x = self.context_path.conv1(data)
        x = self.context_path.bn1(x)
        x = self.context_path.relu(x)
        x = self.context_path.maxpool(x)
        c1 = self.context_path.layer1(x)
        c2 = self.context_path.layer2(c1)
        c3 = self.context_path.layer3(c2)
        c4 = self.context_path.layer4(c3)

        context_blocks = [c1, c2, c3, c4]
        context_blocks.reverse()

        refine = self.refine3x3(context_blocks[0])
        refine = self.refine1x1(refine)
        # FPA = self.fpa(refine)
        # final_fm = self.final_FFM(refine, FPA)
        ca = self.CA(refine)
        pa = self.PA(refine)
        ffm = self.FFM(ca, pa)
        fm = F.interpolate(ffm, size=context_blocks[3].shape[2:], mode="bilinear", align_corners=True)
        # h = torch.cat((fm, context_blocks[2]), dim=1)
        h = self.low_FFM(fm, context_blocks[3])
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
        x_range = torch.linspace(-1, 1, x.shape[-1])
        y_range = torch.linspace(-1, 1, x.shape[-2])
        Y, X = torch.meshgrid(y_range, x_range)
        Y = Y.expand([x.shape[0], 1, -1, -1])
        X = X.expand([x.shape[0], 1, -1, -1])
        coord_feat_1 = torch.cat([X, Y], 1).cuda()
        fm = torch.cat([x, coord_feat_1], 1).cuda()
        fm = self.conv1_3x3(fm)
        output = self.conv1_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = dilated_DDHNet(n_classes=8, n_channels=3, pretrained_model=True)
    image = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 8, 256, 256)
    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()
        label = label.cuda()
    pred = model(image)
    print(pred.size())
