

from __future__ import division
from __future__ import absolute_import
#typing
#overrides

import torch

from allennlp.training.metrics.metric import Metric
try:
    from itertools import izip
except:
    izip = zip



class MentionRecall(Metric):
    def __init__(self)        :
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

    #overrides
    def __call__(self,  # type: ignore
                 batched_top_spans              ,
                 batched_metadata                      ):
        for top_spans, metadata in izip(batched_top_spans.data.tolist(), batched_metadata):

            gold_mentions                       = set([mention for cluster in metadata[u"clusters"]
                                                   for mention in cluster])
            predicted_spans                       = set((span[0], span[1]) for span in top_spans)
            self._num_gold_mentions += len(gold_mentions)
            self._num_recalled_mentions += len(gold_mentions & predicted_spans)

    #overrides
    def get_metric(self, reset       = False)         :
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions/float(self._num_gold_mentions)
        if reset:
            self.reset()
        return recall

    #overrides
    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

MentionRecall = Metric.register(u"mention_recall")(MentionRecall)
