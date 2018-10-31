
from __future__ import absolute_import
import torch

from allennlp.common import Registrable

class Regularizer(Registrable):
    u"""
    An abstract class representing a regularizer. It must implement
    call, returning a scalar tensor.
    """
    default_implementation = u'l2'

    def __call__(self, parameter              )                :
        raise NotImplementedError
