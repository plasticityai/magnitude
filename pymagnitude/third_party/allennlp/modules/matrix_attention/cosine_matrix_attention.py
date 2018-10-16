

from __future__ import division
from __future__ import absolute_import
import torch
#overrides

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class CosineMatrixAttention(MatrixAttention):
    u"""
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
    similarity.
    """

    #overrides
    def forward(self, matrix_1              , matrix_2              )                :
        a_norm = matrix_1 / (matrix_1.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = matrix_2 / (matrix_2.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))

CosineMatrixAttention = MatrixAttention.register(u"cosine")(CosineMatrixAttention)
