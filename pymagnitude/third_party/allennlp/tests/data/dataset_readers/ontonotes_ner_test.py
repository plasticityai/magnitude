# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestOntonotesNamedEntityRecognitionReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = OntonotesNamedEntityRecognition(lazy=lazy)
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'conll_2012' / u'subdomain')
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u"Mali", u"government", u"officials", u"say", u"the", u"woman", u"'s",
                          u"confession", u"was", u"forced", u"."]
        assert fields[u"tags"].labels == [u'B-GPE', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u'The', u'prosecution', u'rested', u'its', u'case', u'last', u'month', u'after',
                          u'four', u'months', u'of', u'hearings', u'.']
        assert fields[u"tags"].labels == [u'O', u'O', u'O', u'O', u'O', u'B-DATE', u'I-DATE', u'O',
                                         u'B-DATE', u'I-DATE', u'O', u'O', u'O']

        fields = instances[2].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u"Denise", u"Dillon", u"Headline", u"News", u"."]
        assert fields[u"tags"].labels == [u'B-PERSON', u'I-PERSON', u'B-WORK_OF_ART', u'I-WORK_OF_ART', u'O']

    def test_ner_reader_can_filter_by_domain(self):
        conll_reader = OntonotesNamedEntityRecognition(domain_identifier=u"subdomain2")
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'conll_2012')
        instances = ensure_list(instances)
        assert len(instances) == 1
