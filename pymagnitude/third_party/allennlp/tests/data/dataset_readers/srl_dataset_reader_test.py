# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestSrlReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = SrlReader(lazy=lazy)
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'conll_2012' / u'subdomain')
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u"Mali", u"government", u"officials", u"say", u"the", u"woman", u"'s",
                          u"confession", u"was", u"forced", u"."]
        assert fields[u"verb_indicator"].labels[3] == 1
        assert fields[u"tags"].labels == [u'B-ARG0', u'I-ARG0', u'I-ARG0', u'B-V', u'B-ARG1',
                                         u'I-ARG1', u'I-ARG1', u'I-ARG1', u'I-ARG1', u'I-ARG1', u'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u"Mali", u"government", u"officials", u"say", u"the", u"woman", u"'s",
                          u"confession", u"was", u"forced", u"."]
        assert fields[u"verb_indicator"].labels[8] == 1
        assert fields[u"tags"].labels == [u'O', u'O', u'O', u'O', u'B-ARG1', u'I-ARG1',
                                         u'I-ARG1', u'I-ARG1', u'B-V', u'B-ARG2', u'O']

        fields = instances[2].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u'The', u'prosecution', u'rested', u'its', u'case', u'last', u'month', u'after',
                          u'four', u'months', u'of', u'hearings', u'.']
        assert fields[u"verb_indicator"].labels[2] == 1
        assert fields[u"tags"].labels == [u'B-ARG0', u'I-ARG0', u'B-V', u'B-ARG1', u'I-ARG1', u'B-ARGM-TMP',
                                         u'I-ARGM-TMP', u'B-ARGM-TMP', u'I-ARGM-TMP', u'I-ARGM-TMP',
                                         u'I-ARGM-TMP', u'I-ARGM-TMP', u'O']

        fields = instances[3].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u'The', u'prosecution', u'rested', u'its', u'case', u'last', u'month', u'after',
                          u'four', u'months', u'of', u'hearings', u'.']
        assert fields[u"verb_indicator"].labels[11] == 1
        assert fields[u"tags"].labels == [u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'B-V', u'O']

        # Tests a sentence with no verbal predicates.
        fields = instances[4].fields
        tokens = [t.text for t in fields[u'tokens'].tokens]
        assert tokens == [u"Denise", u"Dillon", u"Headline", u"News", u"."]
        assert fields[u"verb_indicator"].labels == [0, 0, 0, 0, 0]
        assert fields[u"tags"].labels == [u'O', u'O', u'O', u'O', u'O']

    def test_srl_reader_can_filter_by_domain(self):

        conll_reader = SrlReader(domain_identifier=u"subdomain2")
        instances = conll_reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'conll_2012')
        instances = ensure_list(instances)
        # If we'd included the folder, we'd have 9 instances.
        assert len(instances) == 2
