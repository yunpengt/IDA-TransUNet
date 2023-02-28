import torch.nn.functional as F 
import torch
import torch.nn as nn
class GroupNorms(nn.Module):
    def __init__(self,num_groups,num_channels,eps,weights=None):
        super(GroupNorms,self).__init__()
        self.num_groups=num_groups
        self.num_channels=num_channels
        self.eps=eps
        self.gn=nn.GroupNorm(num_groups=num_groups,num_channels=num_channels,eps=eps)
    def forward(self,input):
        if len(input)==1:
            x=self.gn(input[0])
        else:
            weight=input[1]['transformer.embeddings.hybrid_model.root.conv.weight']
            bias=input[1]['transformer.embeddings.hybrid_model.root.gn.weight']
            x=F.group_norm(input[0],self.num_groups,weight,bias,self.eps)
        return x