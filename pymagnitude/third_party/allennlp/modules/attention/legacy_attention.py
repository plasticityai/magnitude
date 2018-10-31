

from __future__ import absolute_import
import torch

#overrides
from allennlp.modules.attention.attention import Attention
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction


class LegacyAttention(Attention):
    u"""
    Computes attention between a vector and a matrix using a similarity function.
    This should be considered deprecated, as it consumes more memory than the specialized attention modules.
    """

    def __init__(self,
                 similarity_function                     = None,
                 normalize       = True)        :
        super(LegacyAttention, self).__init__(normalize)
        self._similarity_function = similarity_function or DotProductSimilarity()

    #overrides
    def _forward_internal(self, vector              , matrix              )                :
        tiled_vector = vector.unsqueeze(1).expand(vector.size()[0],
                                                  matrix.size()[1],
                                                  vector.size()[1])
        return self._similarity_function(tiled_vector, matrix)

LegacyAttention = Attention.register(u"legacy")(LegacyAttention)
