

from __future__ import division
from __future__ import absolute_import
import torch
#overrides
from allennlp.modules.attention.legacy_attention import Attention


class CosineAttention(Attention):
    u"""
    Computes attention between a vector and a matrix using cosine similarity.
    """
    #overrides
    def _forward_internal(self, vector              , matrix              )                :
        a_norm = vector / (vector.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = matrix / (matrix.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)

CosineAttention = Attention.register(u"cosine")(CosineAttention)
