
from __future__ import absolute_import
import torch
#overrides
from allennlp.modules.attention.legacy_attention import Attention


class DotProductAttention(Attention):
    u"""
    Computes attention between a vector and a matrix using dot product.
    """
    #overrides
    def _forward_internal(self, vector              , matrix              )                :
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)

DotProductAttention = Attention.register(u"dot_product")(DotProductAttention)
