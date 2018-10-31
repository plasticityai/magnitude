
from __future__ import absolute_import
import torch

from allennlp.nn.regularizers.regularizer import Regularizer


class L1Regularizer(Regularizer):
    u"""Represents a penalty proportional to the sum of the absolute values of the parameters"""

    def __init__(self, alpha        = 0.01)        :
        self.alpha = alpha

    def __call__(self, parameter              )                :
        return self.alpha * torch.sum(torch.abs(parameter))


L1Regularizer = Regularizer.register(u"l1")(L1Regularizer)

class L2Regularizer(Regularizer):
    u"""Represents a penalty proportional to the sum of squared values of the parameters"""

    def __init__(self, alpha        = 0.01)        :
        self.alpha = alpha

    def __call__(self, parameter              )                :
        return self.alpha * torch.sum(torch.pow(parameter, 2))

L2Regularizer = Regularizer.register(u"l2")(L2Regularizer)
