# pylint: disable=no-self-use,invalid-name,abstract-method

from __future__ import absolute_import
#typing

from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list


class LazyDatasetReader(DatasetReader):
    def __init__(self, instances                , lazy      )        :
        super(LazyDatasetReader, self).__init__()
        self.lazy = lazy
        self._instances = instances
        self.num_reads = 0

    def _read(self, _     )                      :
        self.num_reads += 1
        return (instance for instance in self._instances)


class TestLazyDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super(TestLazyDatasetReader, self).setUp()
        token_indexer = {u"tokens": SingleIdTokenIndexer()}

        field1 = TextField([Token(t) for t in [u"this", u"is", u"a", u"sentence", u"."]],
                           token_indexer)
        field2 = TextField([Token(t) for t in [u"this", u"is", u"a", u"different", u"sentence", u"."]],
                           token_indexer)
        field3 = TextField([Token(t) for t in [u"here", u"is", u"a", u"sentence", u"."]],
                           token_indexer)
        field4 = TextField([Token(t) for t in [u"this", u"is", u"short"]],
                           token_indexer)
        self.instances = [Instance({u"text1": field1, u"text2": field2}),
                          Instance({u"text1": field3, u"text2": field4})]

    def test_lazy(self):
        reader = LazyDatasetReader(self.instances, lazy=True)
        assert reader.num_reads == 0

        instances = reader.read(u'path/to/file')

        for _ in range(10):
            _instances = (i for i in instances)
            assert ensure_list(_instances) == self.instances

        assert reader.num_reads == 10

    def test_non_lazy(self):
        reader = LazyDatasetReader(self.instances, lazy=False)
        assert reader.num_reads == 0

        instances = reader.read(u'path/to/file')

        for _ in range(10):
            _instances = (i for i in instances)
            assert ensure_list(_instances) == self.instances

        assert reader.num_reads == 1
