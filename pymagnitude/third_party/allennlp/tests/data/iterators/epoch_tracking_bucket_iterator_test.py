
from __future__ import absolute_import
from allennlp.data.iterators import EpochTrackingBucketIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class EpochTrackingBucketIteratorTest(IteratorTest):
    def setUp(self):
        # The super class creates a self.instances field and populates it with some instances with
        # TextFields.
        super(EpochTrackingBucketIteratorTest, self).setUp()
        self.iterator = EpochTrackingBucketIterator(sorting_keys=[[u"text", u"num_tokens"]])
        # We'll add more to create a second dataset.
        self.more_instances = [
                self.create_instance([u"this", u"is", u"a", u"sentence"]),
                self.create_instance([u"this", u"is", u"in", u"the", u"second", u"dataset"]),
                self.create_instance([u"so", u"is", u"this", u"one"])
                ]

    def test_iterator_tracks_epochs_per_dataset(self):
        generated_dataset1 = list(self.iterator(self.instances, num_epochs=2))
        generated_dataset2 = list(self.iterator(self.more_instances, num_epochs=2))

        # First dataset has five sentences. See ``IteratorTest.setUp``
        assert generated_dataset1[0][u"epoch_num"] == [0, 0, 0, 0, 0]
        assert generated_dataset1[1][u"epoch_num"] == [1, 1, 1, 1, 1]
        # Second dataset has three sentences.
        assert generated_dataset2[0][u"epoch_num"] == [0, 0, 0]
        assert generated_dataset2[1][u"epoch_num"] == [1, 1, 1]
