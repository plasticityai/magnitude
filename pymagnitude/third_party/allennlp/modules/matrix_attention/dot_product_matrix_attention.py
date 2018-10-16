
from __future__ import absolute_import
import torch
#overrides

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class DotProductMatrixAttention(MatrixAttention):
    u"""
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product.
    """

    #overrides
    def forward(self, matrix_1              , matrix_2              )                :
        return matrix_1.bmm(matrix_2.transpose(2, 1))

DotProductMatrixAttention = MatrixAttention.register(u"dot_product")(DotProductMatrixAttention)
