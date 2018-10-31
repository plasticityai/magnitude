# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestDataset(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace(u"this")
        self.vocab.add_token_to_namespace(u"is")
        self.vocab.add_token_to_namespace(u"a")
        self.vocab.add_token_to_namespace(u"sentence")
        self.vocab.add_token_to_namespace(u".")
        self.token_indexer = {u"tokens": SingleIdTokenIndexer()}
        self.instances = self.get_instances()
        super(TestDataset, self).setUp()

    def test_instances_must_have_homogeneous_fields(self):
        instance1 = Instance({u"tag": (LabelField(1, skip_indexing=True))})
        instance2 = Instance({u"words": TextField([Token(u"hello")], {})})
        with pytest.raises(ConfigurationError):
            _ = Batch([instance1, instance2])

    def test_padding_lengths_uses_max_instance_lengths(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        padding_lengths = dataset.get_padding_lengths()
        assert padding_lengths == {u"text1": {u"num_tokens": 5}, u"text2": {u"num_tokens": 6}}

    def test_as_tensor_dict(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        padding_lengths = dataset.get_padding_lengths()
        tensors = dataset.as_tensor_dict(padding_lengths)
        text1 = tensors[u"text1"][u"tokens"].detach().cpu().numpy()
        text2 = tensors[u"text2"][u"tokens"].detach().cpu().numpy()

        numpy.testing.assert_array_almost_equal(text1, numpy.array([[2, 3, 4, 5, 6],
                                                                    [1, 3, 4, 5, 6]]))
        numpy.testing.assert_array_almost_equal(text2, numpy.array([[2, 3, 4, 1, 5, 6],
                                                                    [2, 3, 1, 0, 0, 0]]))

    def get_instances(self):
        field1 = TextField([Token(t) for t in [u"this", u"is", u"a", u"sentence", u"."]],
                           self.token_indexer)
        field2 = TextField([Token(t) for t in [u"this", u"is", u"a", u"different", u"sentence", u"."]],
                           self.token_indexer)
        field3 = TextField([Token(t) for t in [u"here", u"is", u"a", u"sentence", u"."]],
                           self.token_indexer)
        field4 = TextField([Token(t) for t in [u"this", u"is", u"short"]],
                           self.token_indexer)
        instances = [Instance({u"text1": field1, u"text2": field2}),
                     Instance({u"text1": field3, u"text2": field4})]
        return instances
