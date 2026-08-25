import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
#torch_ver = torch.__version__[:3]


class PAM_SE_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, height, width, dropout, cuda=False,):
        super(PAM_SE_Module, self).__init__()
        self.chanel_in = in_dim
        self.is_cuda = cuda

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.squeeze = Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        #dim_se = height * width
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        pos_out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        pos_out = pos_out.view(m_batchsize, C, height, width)

        se_attention = self.squeeze(x)
        se_attention = self.flatten(se_attention)
        se_attention = self.excitation(se_attention)
        se_attention = se_attention.view(m_batchsize, 1, height, width)
        out = pos_out * se_attention.expand_as(x)
        return out

class CAM_SE_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, dropout, cuda=False):
        super(CAM_SE_Module, self).__init__()
        self.chanel_in = in_dim
        self.is_cuda = cuda

        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # self.flatten = torch.flatten()
        #self.flatten = nn.Flatten()
        #dim_se = height * width
        self.excitation = nn.Sequential(
            ConvBnRelu(in_dim, in_dim//8, 1, 1, 0,
                       has_bn=False, norm_layer=nn.BatchNorm2d,
                       has_relu=True, has_bias=False),
            ConvBnRelu(in_dim//8, in_dim, 1, 1, 0,
                       has_bn=False, norm_layer=nn.BatchNorm2d,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        ch_out = torch.bmm(attention, proj_value)
        ch_out = ch_out.view(m_batchsize, C, height, width)

        se_attention = self.squeeze(x)
        se_attention = se_attention.view(m_batchsize, C, 1, 1)
        se_attention = self.excitation(se_attention)
        #se_attention = se_attention.view(m_batchsize, C, 1, 1)
        out = ch_out * se_attention.expand_as(x)
        return out

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
if __name__ == "__main__":
    image = torch.randn(1, 320, 64, 64)
    #PA_model = PAM_SE_Module(in_dim=320, height=64, width=64, dropout=0.8)
    CA_model = CAM_SE_Module(in_dim=320, dropout=0.8)
    image = torch.randn(1, 320, 64, 64)
    #p_out = PA_model(image)
    c_out = CA_model(image)
    print(image.shape)
    print("input:", image.shape)
    #mini_mask, mask = model(image)
    #print("output:", mini_mask.shape)
    ##print('output:', mask.shape)
#    print("p_output:", p_out.shape)
    print("c_output:", c_out.shape)