

from __future__ import division
from __future__ import absolute_import
#overrides
import torch

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction


class CosineSimilarity(SimilarityFunction):
    u"""
    This similarity function simply computes the cosine similarity between each pair of vectors.  It has
    no parameters.
    """
    #overrides
    def forward(self, tensor_1              , tensor_2              )                :
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

CosineSimilarity = SimilarityFunction.register(u"cosine")(CosineSimilarity)
