import numpy as np
import os
import torch
import logging
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN, dcn_v2_conv
import math

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ConvAWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma', torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta', torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = weight / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return super().conv2d_forward(x, weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight_gamma.data.fill_(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            return
        weight = self.weight.data
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                       keepdim=True).mean(dim=3, keepdim=True)
        self.weight_beta.data.copy_(weight_mean)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)

class SAConv2d(ConvAWS2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 # groups=1,
                 bias=True,
                 use_deform=False):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            # groups=groups,
            bias=bias)
        self.use_deform = use_deform
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.weight_diff.data.zero_()
        self.bias.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        if self.use_deform:
            self.offset_s = torch.nn.Conv2d(
                self.in_channels,
                27,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_l = torch.nn.Conv2d(
                self.in_channels,
                27,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_s.weight.data.fill_(0)
            self.offset_s.bias.data.fill_(0)
            self.offset_l.weight.data.fill_(0)
            self.offset_l.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            o1, o2, mask = torch.chunk(offset, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            out_s = dcn_v2_conv(
                    x,
                    offset, mask,
                    weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    # self.groups,
                    1)
        else:
            out_s = super().conv2d_forward(x, weight)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        if self.use_deform:
            offset = self.offset_l(avg_x)
            o1, o2, mask = torch.chunk(offset, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            out_l = dcn_v2_conv(
                    x,
                    offset, mask,
                    weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    # self.groups,
                    1)
        else:
            out_l = super().conv2d_forward(x, weight)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return SAConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, dilation=1,
                    bias=True, use_deform=False)

class sacConv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(sacConv_BN_LeakyReLU, self).__init__()
 
        self.convs = nn.Sequential(
             conv3x3(in_channels, out_channels, stride=stride),
             nn.BatchNorm2d(out_channels),
             nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        return self.convs(x)


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
             nn.BatchNorm2d(out_channels),
             nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch // 2, 1),
                Conv_BN_LeakyReLU(ch // 2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x

class sacresblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch // 2, 1),
                sacConv_BN_LeakyReLU(ch // 2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class DarkNet_Tiny(nn.Module):
    def __init__(self, heads, head_conv):
        super(DarkNet_Tiny, self).__init__()
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.heads = heads
        self.deconv_with_bias = False
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            Conv_BN_LeakyReLU(32, 32, 3, padding=1, stride=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            Conv_BN_LeakyReLU(64, 64, 3, padding=1, stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 128, 3, padding=1, stride=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 256, 3, padding=1, stride=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 512, 3, padding=1, stride=2),
        )
        self.reshape_layer1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=8)
        )

        self.reshape_layer2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=4)
        )

        self.reshape_layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2)
        )
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        self.deconv_layer1 = self.deconv_layers[:4]
        self.deconv_layer2 = self.deconv_layers[4:8]
        self.deconv_layer3 = self.deconv_layers[8:12]

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(planes * 2, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.Upsample(scale_factor=2)
            # up = nn.ConvTranspose2d(
            #     in_channels=planes,
            #     out_channels=planes,
            #     kernel_size=kernel,
            #     stride=2,
            #     padding=padding,
            #     output_padding=output_padding,
            #     bias=self.deconv_with_bias)
            # fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(up)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))
            # self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x0 = x
        x = self.conv_3(x)
        x1 = x
        x = self.conv_4(x)
        x2 = x
        x = self.conv_5(x)
        r1 = self.reshape_layer1(x)
        x = self.deconv_layer1(x)
        # print(x.shape, x2.shape)
        x = x + x2
        r2 = self.reshape_layer2(x)
        x = self.deconv_layer2(x)
        x = x + x1
        r3 = self.reshape_layer3(x)
        x = self.deconv_layer3(x)

        x = x + x0
        x = x + r1 + r2 + r3
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return [ret]

    def init_weights(self, num_layers):
        if 1:
            # url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(torch.load('models/darknet_tiny_63.50_85.06.pth', map_location='cuda'),
                                 strict=False)
            print("Initializing the darknet_tiny network ......")
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class DarkNet_19(nn.Module):
    def __init__(self):
        print("Initializing the darknet19 network ......")

        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, 1000, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(self.maxpool_4(x))
        x = self.conv_6(self.maxpool_5(x))

        x = self.avgpool(x)
        x = self.conv_7(x)
        x = x.view(x.size(0), -1)
        return x


class DarkNet_53(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """

    def __init__(self, heads, head_conv):
        super(DarkNet_53, self).__init__()
        self.heads = heads
        self.deconv_with_bias = False

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

        self.refine_layer1 = nn.Sequential(
            DCN(1024, 128, kernel_size=(3, 3), stride=1,
                padding=1, dilation=1, deformable_groups=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128,
                               kernel_size=16, stride=8, padding=4),
        )

        self.refine_layer2 = nn.Sequential(
            DCN(512, 128, kernel_size=(3, 3), stride=1,
                padding=1, dilation=1, deformable_groups=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128,
                               kernel_size=8, stride=4, padding=2),
        )

        self.refine_layer3 = nn.Sequential(
            DCN(256, 128, kernel_size=(3, 3), stride=1,
                padding=1, dilation=1, deformable_groups=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 128,
                               kernel_size=4, stride=2, padding=1),
        )
        
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [512, 256, 128],
            [4, 4, 4],
        )
        self.deconv_layer1 = self.deconv_layers[:6]
        self.deconv_layer2 = self.deconv_layers[6:12]
        self.deconv_layer3 = self.deconv_layers[12:18]
        
	    # ASFF
        self.weight_level_0 = add_conv(512*2, 2, 1, 1, leaky=False)
        self.weight_level_1 = add_conv(256*2, 2, 1, 1, leaky=False)
        self.weight_level_2 = add_conv(128*2, 2, 1, 1, leaky=False)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(128, classes,
                              kernel_size=3, padding=1, bias=True),
                    #nn.ReLU(inplace=True),
                    #nn.Conv2d(head_conv, classes,
                    #          kernel_size=1, stride=1,
                    #          padding=0, bias=True)
                    )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(128, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(planes*2, planes,
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, targets=None):
        x = self.layer_1(x)

        x = self.layer_2(x)
        x0 = x
        x = self.layer_3(x)
        x1 = x
        x = self.layer_4(x)
        x2 = x
        x = self.layer_5(x)
        r2 = self.refine_layer1(x)

        x = self.deconv_layer1(x)
        levels_weight_0 = torch.cat((x, x2), dim=1)
        levels_weight = self.weight_level_0(levels_weight_0)
        levels_weight = F.softmax(levels_weight, dim=1) * 2
        x = x * (levels_weight[:, 0:1, :, :]) + x2 * (levels_weight[:, 1:, :, :])
        # x = x + x2
        r1 = self.refine_layer2(x)
        x = self.deconv_layer2(x)
        levels_weight_1 = torch.cat((x, x1), dim=1)
        levels_weight = self.weight_level_1(levels_weight_1)
        levels_weight = F.softmax(levels_weight, dim=1) * 2
        x = x * (levels_weight[:, 0:1, :, :]) + x1 * (levels_weight[:, 1:, :, :])
        # x = x + x1
        r0 = self.refine_layer3(x)
        x = self.deconv_layer3(x)

        levels_weight_2 = torch.cat((x, x0), dim=1)
        levels_weight = self.weight_level_2(levels_weight_2)
        levels_weight = F.softmax(levels_weight, dim=1) * 2
        x = x * (levels_weight[:, 0:1, :, :]) + x0 * (levels_weight[:, 1:, :, :])
        # x = x + x0
        x = x + r2 + r1 + r0
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            # url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(torch.load('/home/wb/dark/darknet53_hr_77.76.pth', map_location='cuda'), strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


def get_darknet(num_layers, heads, head_conv=64):
  if num_layers == 10:
      model = DarkNet_Tiny(heads, head_conv=head_conv)
  if num_layers == 19:
      model = DarkNet_19(heads, head_conv=head_conv)
  if num_layers == 53:
      model = DarkNet_53(heads, head_conv=head_conv)

  model.init_weights(num_layers)
  return model

# def darknet19(pretrained=False, **kwargs):
#     """Constructs a darknet-19 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = DarkNet_19()
#     if pretrained:
#         path_to_dir = os.path.dirname(os.path.abspath(__file__))
#         print('Loading the darknet19 ...')
#         model.load_state_dict(torch.load(path_to_dir + '/weights/darknet19_72.96.pth', map_location='cuda'),
#                               strict=False)
#     return model
#
#
# def darknet53(pretrained=False, **kwargs):
#     """Constructs a darknet-53 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = DarkNet_53()
#     if pretrained:
#         path_to_dir = os.path.dirname(os.path.abspath(__file__))
#         print('Loading the darknet53 ...')
#         model.load_state_dict(torch.load(path_to_dir + '/weights/darknet53_75.42.pth', map_location='cuda'),
#                               strict=False)
#     return model
