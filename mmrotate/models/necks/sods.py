import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import warnings
import e2cnn.nn as enn
from mmcv.runner import BaseModule, auto_fp16
from ..builder import ROTATED_NECKS
# from ..utils import ConvModule
from mmcv.cnn import ConvModule
from ..utils import (build_enn_feature, build_enn_norm_layer, ennConv,
                     ennInterpolate, ennMaxPool, ennReLU)

# class ConvModule(enn.EquivariantModule):
#     """ConvModule.
#
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale).
#         kernel_size (int, optional): The size of kernel.
#         stride (int, optional): Stride of the convolution. Default: 1.
#         padding (int or tuple): Zero-padding added to both sides of the input.
#             Default: 0.
#         dilation (int or tuple): Spacing between kernel elements. Default: 1.
#         groups (int): Number of blocked connections from input.
#             channels to output channels. Default: 1.
#         bias (bool): If True, adds a learnable bias to the output.
#             Default: False.
#         conv_cfg (dict, optional): Config dict for convolution layer.
#             Default: None.
#         norm_cfg (dict, optional): Config dict for normalization layer.
#             Default: None.
#         activation (str, optional): Activation layer in ConvModule.
#             Default: None.
#         inplace (bool): can optionally do the operation in-place.
#         order (tuple[str]): The order of conv/norm/activation layers. It is a
#             sequence of "conv", "norm" and "act". Common examples are
#             ("conv", "norm", "act") and ("act", "conv", "norm").
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias='auto',
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  activation='relu',
#                  inplace=False,
#                  order=('conv', 'norm', 'act')):
#         super(ConvModule, self).__init__()
#         assert conv_cfg is None or isinstance(conv_cfg, dict)
#         assert norm_cfg is None or isinstance(norm_cfg, dict)
#         self.in_type = build_enn_feature(in_channels)
#         self.out_type = build_enn_feature(out_channels)
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.activation = activation
#         self.inplace = inplace
#         self.order = order
#         assert isinstance(self.order, tuple) and len(self.order) == 3
#         assert set(order) == set(['conv', 'norm', 'act'])
#
#         self.with_norm = norm_cfg is not None
#         self.with_activatation = activation is not None
#         # if the conv layer is before a norm layer, bias is unnecessary.
#         if bias == 'auto':
#             bias = False if self.with_norm else True
#         self.with_bias = bias
#
#         if self.with_norm and self.with_bias:
#             warnings.warn('ConvModule has norm and bias at the same time')
#         # build convolution layer
#         self.conv = ennConv(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias)
#         # export the attributes of self.conv to a higher level for convenience
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.transposed = False
#         self.output_padding = padding
#         self.groups = groups
#
#         # build normalization layers
#         if self.with_norm:
#             # norm layer is after conv layer
#             if order.index('norm') > order.index('conv'):
#                 norm_channels = out_channels
#             else:
#                 norm_channels = in_channels
#             if conv_cfg is not None and conv_cfg['type'] == 'ORConv':
#                 norm_channels = int(norm_channels * 8)
#             self.norm_name, norm = build_enn_norm_layer(norm_channels)
#             self.add_module(self.norm_name, norm)
#
#         # build activation layer
#         if self.with_activatation:
#             # TODO: introduce `act_cfg` and supports more activation layers
#             if self.activation not in ['relu']:
#                 raise ValueError(
#                     f'{self.activation} is currently not supported.')
#             if self.activation == 'relu':
#                 self.activate = ennReLU(out_channels)
#
#         # Use msra init by default
#         self.init_weights()
#
#     @property
#     def norm(self):
#         """Get normalizion layer's name."""
#         return getattr(self, self.norm_name)
#
#     def init_weights(self):
#         """Initialize weights of the head."""
#         nonlinearity = 'relu' if self.activation is None \
#             else self.activation  # noqa: F841
#
#     def forward(self, x, activate=True, norm=True):
#         """Forward function of ConvModule."""
#         for layer in self.order:
#             if layer == 'conv':
#                 x = self.conv(x)
#             elif layer == 'norm' and norm and self.with_norm:
#                 x = self.norm(x)
#             elif layer == 'act' and activate and self.with_activatation:
#                 x = self.activate(x)
#         return x
#
#     def evaluate_output_shape(self, input_shape):
#         """Evaluate output shape."""
#         return input_shape

class SE_Block(nn.Module):
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


class SE_ASPP(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.senet=SE_Block(in_planes=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             #加入通道注意力机制
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        result = self.conv_cat(se_feature_cat)
        # print('result:',result.shape)
        return result

def HEAT(feature,str_dir):

    img_name = "P0024__1024__524___2096"
    img_path = '/media/yun/4t/work4t/scc/data/split-heat-map-images/test/images/' + img_name + '.png'
    # heat 为某层的特征图，自己手动获取
    # heat = feature.data.cpu().numpy()       # 将tensor格式的feature map转为numpy格式
    # heat = np.squeeze(heat, 0)           # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    channel = [0,32,64,96,128,160,192,224,256]
    channel = [0, 16, 32, 48, 64, 80, 96,112, 128,144, 160,176, 192,208, 224,240, 256]
    list = []
    for i in range(256):
        flag = 32*i
        if flag > 256:
            break
        list.append(flag)
    # channel_end = 64
    for i in range(len(list)-1):
        heat = feature.data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
        heat = np.squeeze(heat, 0)
        channel_start = list[i]
        channel_end = list[i+1]
        heat = heat[channel_start:channel_end,:]       # 切片获取某几个通道的特征图
        heatmap = np.maximum(heat, 0)        # heatmap与0比较
        heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
        heatmap /= np.max(heatmap)          # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        # plt.matshow(heatmap)               # 可以通过 plt.matshow 显示热力图
        # plt.show()

        # 用cv2加载原始图像
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
        heat_img = cv2.addWeighted(img, 0.3, heatmap, 0.7, 0)     # 将伪彩色图与原始图片融合
        #heat_img = heatmap * 0.5 + img 　　　　　　        　　　 # 也可以用这种方式融合
        results_dir = "/media/yun/4t/work4t/scc/mmrotate-heat/work_dirs/heat-maps-ours-v100/"+img_name+str_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_path = results_dir + '_'  + str(channel_start) + "-" + str(channel_end)  + "_CAM" + ".png"
        cv2.imwrite(results_path, heat_img)

@ROTATED_NECKS.register_module()
class SODS(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(SODS, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        # self.activation = activation
        self.act_cfg = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        # self.aspp = False
        # self.ASPP = SE_ASPP(dim_in=in_channels, dim_out=out_channels)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # add scale sequence
        self.conv3D_bn_activation = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(self.out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        num_outs_3D = 4
        self.average_pooling = nn.AvgPool3d(kernel_size=(num_outs_3D, 1, 1), stride=(num_outs_3D, 1, 1), padding=(0, 0, 0))
        self.refine_convolution = nn.Conv2d(self.out_channels * 2, self.out_channels, 1, 1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        # if self.aspp:
        #     inputs = self.ASPP(inputs)
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # size [1,256,56,56], [1,512,28,28], [1,1024,14,14],[1,2048,7,7]
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # size [1,256,56,56], [1,256,28,28], [1,256,14,14],[1,256,7,7]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # multi-scale feature stream
        # 我们从第一个输出开始处理
        final_output = outs[0]
        processed_layers = []
        # 获取最终输出的特征尺寸
        feature_map_size = final_output.size()[-2:]

        # 将outs中的每个输出添加到processed_layers列表zhong
        processed_layers.append(final_output)
        processed_layers.append(outs[1])
        processed_layers.append(outs[2])
        processed_layers.append(outs[3])
        # heat_map
        # HEAT(outs[0], 'dir_0')
        # HEAT(outs[1], 'dir_1')
        # HEAT(outs[2], 'dir_2')
        # HEAT(outs[3], 'dir_3)

        # 对每个输出进行尺寸调整并插值，增加time维度
        adjusted_scales = [
            torch.unsqueeze(
                F.interpolate(layer, size=feature_map_size,model='nearest'), dim=2
            ) for layer in processed_layers
        ]

        # 将调整的特征图沿着time维度拼接起来
        combined_feature = torch.cat(adjusted_scales, dim=2)
        # 处理特征图
        processed_feature = self.conv3D_bn_activation(combined_feature)

        pooled_features = self.average_pooling(processed_feature)
        compressed_fearures = torch.squeeze(pooled_features, dim=2)

        # 将处理后的特征图与P2最大尺度特征图进行合并
        combined_feature_final = torch.cat([compressed_fearures, final_output], dim=1)
        # 融合特征，降维
        refined_output = self.refine_convolution(combined_feature_final)

        # heat_map
        #HEAT(refined_output, 'dir_sods')

        outs[0] = refined_output

        return tuple(outs)


