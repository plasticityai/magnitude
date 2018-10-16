
from __future__ import absolute_import
#typing
import torch

from allennlp.common.registrable import Registrable


class Metric(Registrable):
    u"""
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self,
                 predictions              ,
                 gold_labels              ,
                 mask                        ):
        u"""
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset      )                                                     :
        u"""
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self)        :
        u"""
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors              ):
        u"""
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)
