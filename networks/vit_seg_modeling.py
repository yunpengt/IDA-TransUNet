# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from tkinter import CENTER
from .simam_module import simam_module as Simam
import copy
import logging
import math
import torch.nn.functional as F
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from functools import partial
class BatchNorm2d(nn.BatchNorm2d):
    def forward(self,x,fast_weights,layer_id):
        if fast_weights==None:
            return F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        else:
            weight_adapt = fast_weights[layer_id+'.weight']
            if self.bias is not None:
               
                bias_adapt = fast_weights[layer_id+'.bias']
                
            else:
                bias_adapt=self.bias
            return F.batch_norm(x,self.running_mean,self.running_var,weight_adapt,bias_adapt,self.training,self.momentum,self.eps)

class Conv2d(nn.Conv2d):
    def forward(self,x,fast_weights,layer_id):
        if fast_weights==None:
            return F.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
        else:
            weight_adapt  = fast_weights[layer_id+'.weight']
            if self.bias is not None:
              
                bias_adapt = fast_weights[layer_id+'.bias']
                
            else:
                bias_adapt=self.bias
            return F.conv2d(x,weight_adapt,bias_adapt,self.stride,self.padding,self.dilation,self.groups)

class LayerNorm(nn.LayerNorm):
    def forward(self,x,fast_weights,layer_id):
        if fast_weights==None:
            return F.layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps)
        else:
            weights_adapt = fast_weights[layer_id+'.weight']
            if self.bias is not None:
                
                bias_adapt = fast_weights[layer_id+'.bias']
              
              
            else:
                bias_adapt=self.bias
            return F.layer_norm(x,self.normalized_shape,weights_adapt,bias_adapt,self.eps)

class Linear(nn.Linear):
    def forward(self,x,fast_weight=None,layer_id=None):
        if fast_weight==None:
            return F.linear(x,self.weight,self.bias)
        else:
            weight_adapt = fast_weight[layer_id+'.weight']
            if self.bias is not None:
                bias_adapt = fast_weight[layer_id+'.bias']
               
            else:
                bias_adapt=self.bias
            x=F.linear(x,weight_adapt,bias_adapt)
            
            return x
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
     
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.simam = Simam()
        self.softmax = Softmax(dim=-1)
        self.con1x1  = Conv2d(2,1,1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,fast_weights,layer_id):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        sim_x = hidden_states.permute(0, 2, 1)
        sim_x = sim_x.contiguous().view(B, hidden, h, w)
        sim_output=self.simam(sim_x)
        sim_output = sim_output.flatten(2)
        sim_output = sim_output.transpose(-1, -2)

        mixed_query_layer = self.query(hidden_states,fast_weights,layer_id+'.query')
        mixed_key_layer = self.key(hidden_states,fast_weights,layer_id+'.key')
        mixed_value_layer = self.value(hidden_states,fast_weights,layer_id+'.value')

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer,fast_weights,layer_id+'.out')
        attention_output = self.proj_dropout(attention_output)
        
        output=torch.cat([sim_output.unsqueeze(1),attention_output.unsqueeze(1)],dim=1)
        output = self.con1x1(output,fast_weights,layer_id+'.con1x1').squeeze(1)
        return output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x,fast_weights,layer_id):
        x = self.fc1(x,fast_weights,layer_id+'.fc1')
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x,fast_weights,layer_id+'.fc2')
        x = self.dropout(x)
        return x

nonlinearity = partial(F.relu, inplace=True)

class  Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
     
    def forward(self, x,fast_weights=None):
        if self.hybrid:
            x, features = self.hybrid_model(x,fast_weights)
        else:
            features = None
       
        x = self.patch_embeddings(x,fast_weights,'transformer.embeddings.patch_embeddings')  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        if fast_weights==None:
            embeddings = x + self.position_embeddings
        else:
            
            emgrad=fast_weights['transformer.embeddings.position_embeddings']
            embeddings=x+emgrad
           
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config,num):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.num=num
    def forward(self, x,fast_weights):
        h = x
        x = self.attention_norm(x,fast_weights,'transformer.encoder.layer.{}.attention_norm'.format(self.num))
        x= self.attn(x,fast_weights,'transformer.encoder.layer.{}.attn'.format(self.num))
        x = x + h

        h = x
        x = self.ffn_norm(x,fast_weights,'transformer.encoder.layer.{}.ffn_norm'.format(self.num))
        x = self.ffn(x,fast_weights,'transformer.encoder.layer.{}.ffn'.format(self.num))
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
      
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config,num=_)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states,fast_weights):
        attn_weights = []
        for index,layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states,fast_weights)
            if (index+1)%4==0:
                attn_weights.append(hidden_states.unsqueeze(1))
     
        encoded = self.encoder_norm(hidden_states,fast_weights,'transformer.encoder.encoder_norm')
        return encoded,attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids,fast_weights=None):
        embedding_output, features = self.embeddings(input_ids,fast_weights)
        encoded,attn= self.encoder(embedding_output,fast_weights)  # (B, n_patch, hidden)
        return encoded, attn,features


class Conv2dReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        self.relu = nn.ReLU(inplace=True)

        self.bn = BatchNorm2d(out_channels)
    def forward(self,x,fast_weights,layer_id):
        x=self.conv(x,fast_weights,layer_id+'.conv')
        x=self.bn(x,fast_weights,layer_id+'.bn')
        x=self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None,fast_weights=None,layer_id=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x,fast_weights,layer_id+'.conv1')
        x = self.conv2(x,fast_weights,layer_id+'.conv2')
        return x


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super().__init__()
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
    def forward(self,x,fast_weights=None):
        x=self.conv2d(x,fast_weights,'segmentation_head.conv2d')
        x=self.upsampling(x)
        return x
        
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
     
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
    
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None, fast_weights=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x,fast_weights,'decoder.conv_more')
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip,fast_weights=fast_weights,layer_id='decoder.blocks.{}'.format(i))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False,weights=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        '''
        self.fc1 = Linear(150528,388)
        self.fc2 = Linear(388,66)
        self.fc3 = Linear(66,18)
        '''
        self.fc1 = Linear(150528,388)
        self.fc2 = Linear(388,140)
        self.fc3 = Linear(140,70)
        
        self.simam=Simam()
        self.conv1x1=Conv2d(3,1,1)
    def forward(self, img,fast_weights=None):
        if img.size()[1] == 1:
            img = img.repeat(1,3,1,1)
      
       
        x, attn_weights, features = self.transformer(img,fast_weights)
       
        fuse_attention=self.simam(torch.cat(attn_weights,dim=1))
        fuse_attention=self.conv1x1(fuse_attention,fast_weights,'conv1x1')
        fuse_attention=fuse_attention.squeeze(1)
        

        c=x.flatten(1)
        c=F.relu(self.fc1(c,fast_weights,'fc1'))
        c=F.relu(self.fc2(c,fast_weights,'fc2'))
        c=self.fc3(c,fast_weights,'fc3')
        #c=F.softmax(c)  # (B, n_patch, hidden)
        
        x  = self.decoder(fuse_attention,features,fast_weights)
      
        logits = self.segmentation_head(x,fast_weights)
        return logits,c

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    try:
                    
                        unit.load_from(weights, n_block=uname)
                    except:
                        print(uname,unit)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.stdconv2d.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.groupnorm.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.groupnorm.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


