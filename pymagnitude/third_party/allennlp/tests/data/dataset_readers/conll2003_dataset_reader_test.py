# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestConll2003Reader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    @pytest.mark.parametrize(u"coding_scheme", (u'IOB1', u'BIOUL'))
    def test_read_from_file(self, lazy, coding_scheme):
        conll_reader = Conll2003DatasetReader(lazy=lazy, coding_scheme=coding_scheme)
        instances = conll_reader.read(unicode(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'conll2003.txt'))
        instances = ensure_list(instances)

        if coding_scheme == u'IOB1':
            expected_labels = [u'I-ORG', u'O', u'I-PER', u'O', u'O', u'I-LOC', u'O']
        else:
            expected_labels = [u'U-ORG', u'O', u'U-PER', u'O', u'O', u'U-LOC', u'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u'U.N.', u'official', u'Ekeus', u'heads', u'for', u'Baghdad', u'.']
        assert fields[u"tags"].labels == expected_labels

        fields = instances[1].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u'AI2', u'engineer', u'Joel', u'lives', u'in', u'Seattle', u'.']
        assert fields[u"tags"].labels == expected_labels
