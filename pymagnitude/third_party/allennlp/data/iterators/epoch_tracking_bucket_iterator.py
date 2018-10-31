
from __future__ import absolute_import
import logging
#typing
import warnings

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EpochTrackingBucketIterator(BucketIterator):
    u"""
    This is essentially a :class:`allennlp.data.iterators.BucketIterator` with just one difference.
    It keeps track of the epoch number, and adds that as an additional meta field to each instance.
    That way, ``Model.forward`` will have access to this information. We do this by keeping track of
    epochs globally, and incrementing them whenever the iterator is called. However, the iterator is
    called both for training and validation sets. So, we keep a dict of epoch numbers, one key per
    dataset.

    Parameters
    ----------
    See :class:`BucketIterator`.
    """
    def __init__(self,
                 sorting_keys                       ,
                 padding_noise        = 0.1,
                 biggest_batch_first       = False,
                 batch_size      = 32,
                 instances_per_epoch      = None,
                 max_instances_in_memory      = None,
                 cache_instances       = False)        :
        super(EpochTrackingBucketIterator, self).__init__(sorting_keys=sorting_keys,
                         padding_noise=padding_noise,
                         biggest_batch_first=biggest_batch_first,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         track_epoch=True,
                         cache_instances=cache_instances)
        warnings.warn(u"EpochTrackingBucketIterator is deprecated, "
                      u"please just use BucketIterator with track_epoch=True",
                      DeprecationWarning)

EpochTrackingBucketIterator = DataIterator.register(u"epoch_tracking_bucket")(EpochTrackingBucketIterator)
