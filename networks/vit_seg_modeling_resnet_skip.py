import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class GroupNorm(nn.GroupNorm):
        
    def forward(self,x,fast_weights=None,layer_idx=None):
        if fast_weights==None:
            x=F.group_norm(x,self.num_groups,self.weight,self.bias,self.eps)
        else:
            weights_adapt = fast_weights[layer_idx+'.weight']
            if self.bias is not None:
                bias_adapt = fast_weights[layer_idx+'.bias']
            else:
                bias_adapt=self.bias
            x=F.group_norm(x,self.num_groups,weights_adapt,bias_adapt,self.eps)
        return x
class StdConv2d(nn.Conv2d):

    def forward(self, x,fast_weights=None,layer_idx=None):
        if fast_weights==None:
            w = self.weight
            v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
            w = (w - m) / torch.sqrt(v + 1e-5)
            return F.conv2d(x, w, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            w = fast_weights[layer_idx+'.weight']
            v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
            w = (w - m) / torch.sqrt(v + 1e-5)
            
            if self.bias is not None:
                bias_adapt = fast_weights[layer_idx+'.bias']
               
            else:
                bias_adapt=self.bias
            return F.conv2d(x,w,bias_adapt,self.stride,self.padding,self.dilation,self.groups)
            

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1,block=None,unit=None):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4
        self.block=block
        self.unit=unit
        self.gn1 = GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = GroupNorm(cout, cout)

    def forward(self, input):
        x,fast_weights=input
        residual = x
        # Residual branch
        if fast_weights==None:
            if hasattr(self, 'downsample'):
                residual = self.downsample(x)
                residual = self.gn_proj(residual)

            # Unit's branch
            y = self.relu(self.gn1(self.conv1(x)))
            y = self.relu(self.gn2(self.conv2(y)))
            y = self.gn3(self.conv3(y))

            y = self.relu(residual + y)
        else:
            if hasattr(self, 'downsample'):
                residual = self.downsample(x,fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.downsample'.format(self.block,self.unit))
                residual = self.gn_proj(residual,fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.gn_proj'.format(self.block,self.unit))
            # Unit's branch
            y = self.relu(self.gn1(self.conv1(x,fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.conv1'.format(self.block,self.unit)),fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.gn1'.format(self.block,self.unit)))
            y = self.relu(self.gn2(self.conv2(y,fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.conv2'.format(self.block,self.unit)),fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.gn2'.format(self.block,self.unit)))
            y = self.gn3(self.conv3(y,fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.conv3'.format(self.block,self.unit)),fast_weights,'transformer.embeddings.hybrid_model.body.{}.{}.gn3'.format(self.block,self.unit))

            y = self.relu(residual + y)
        return (y,fast_weights)

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.stdconv2d=StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)
        self.groupnorm=GroupNorm(32, width, eps=1e-6)
        self.relu=nn.ReLU(inplace=True)
        '''
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        '''

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width,block='block1',unit='unit1'))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width,block='block1',unit=f'unit{i:d}')) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2,block='block2',unit='unit1'))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2,block='block2',unit=f'unit{i:d}')) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2,block='block3',unit='unit1'))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4,block='block3',unit=f'unit{i:d}')) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x,fast_weights=None):
        features = []
        b, c, in_size, _ = x.size()
        x = self.stdconv2d(x,fast_weights,'transformer.embeddings.hybrid_model.stdconv2d')
        x = self.relu(self.groupnorm(x,fast_weights,'transformer.embeddings.hybrid_model.groupnorm'))

        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x,_ = self.body[i]([x,fast_weights])
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x,_ = self.body[-1]([x,fast_weights])
        return x, features[::-1]
