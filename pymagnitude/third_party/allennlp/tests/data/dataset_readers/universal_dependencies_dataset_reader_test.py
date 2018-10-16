# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import absolute_import
from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / u"data" / u"dependencies.conllu"

    def test_read_from_file(self):
        reader = UniversalDependenciesDatasetReader()
        instances = list(reader.read(unicode(self.data_path)))

        instance = instances[0]
        fields = instance.fields
        assert [t.text for t in fields[u"words"].tokens] == [u'What', u'if', u'Google',
                                                            u'Morphed', u'Into', u'GoogleOS', u'?']

        assert fields[u"pos_tags"].labels == [u'PRON', u'SCONJ', u'PROPN', u'VERB', u'ADP', u'PROPN', u'PUNCT']
        assert fields[u"head_tags"].labels == [u'root', u'mark', u'nsubj', u'advcl',
                                              u'case', u'obl', u'punct']
        assert fields[u"head_indices"].labels == [0, 4, 4, 1, 6, 4, 4]

        instance = instances[1]
        fields = instance.fields
        assert [t.text for t in fields[u"words"].tokens] == [u'What', u'if', u'Google', u'expanded', u'on',
                                                            u'its', u'search', u'-', u'engine', u'(', u'and',
                                                            u'now', u'e-mail', u')', u'wares', u'into', u'a',
                                                            u'full', u'-', u'fledged', u'operating', u'system', u'?']

        assert fields[u"pos_tags"].labels == [u'PRON', u'SCONJ', u'PROPN', u'VERB', u'ADP', u'PRON', u'NOUN',
                                             u'PUNCT', u'NOUN', u'PUNCT', u'CCONJ', u'ADV', u'NOUN', u'PUNCT', u'NOUN',
                                             u'ADP', u'DET', u'ADV', u'PUNCT', u'ADJ', u'NOUN', u'NOUN', u'PUNCT']
        assert fields[u"head_tags"].labels == [u'root', u'mark', u'nsubj', u'advcl', u'case', u'nmod:poss',
                                              u'compound', u'punct', u'compound', u'punct', u'cc', u'advmod',
                                              u'conj', u'punct', u'obl', u'case', u'det', u'advmod', u'punct',
                                              u'amod', u'compound', u'obl', u'punct']
        assert fields[u"head_indices"].labels == [0, 4, 4, 1, 15, 15, 9, 9, 15, 9, 13, 13,
                                                 9, 15, 4, 22, 22, 20, 20, 22, 22, 4, 4]

        instance = instances[2]
        fields = instance.fields
        assert [t.text for t in fields[u"words"].tokens] == [u'[', u'via', u'Microsoft', u'Watch',
                                                            u'from', u'Mary', u'Jo', u'Foley', u']']
        assert fields[u"pos_tags"].labels == [u'PUNCT', u'ADP', u'PROPN', u'PROPN', u'ADP',
                                             u'PROPN', u'PROPN', u'PROPN', u'PUNCT']
        assert fields[u"head_tags"].labels == [u'punct', u'case', u'compound', u'root', u'case',
                                              u'nmod', u'flat', u'flat', u'punct']
        assert fields[u"head_indices"].labels == [4, 4, 4, 0, 6, 4, 6, 6, 4]

        # This instance tests specifically for filtering of elipsis:
        # http://universaldependencies.org/u/overview/specific-syntax.html#ellipsis
        # The original sentence is:
        # "Over 300 Iraqis are reported dead and 500 [reported] wounded in Fallujah alone."
        # But the second "reported" is elided, and as such isn't included in the syntax tree.
        instance = instances[3]
        fields = instance.fields
        assert [t.text for t in fields[u"words"].tokens] == [u'Over', u'300', u'Iraqis', u'are',
                                                            u'reported', u'dead', u'and', u'500', u'wounded',
                                                            u'in', u'Fallujah', u'alone', u'.']
        assert fields[u"pos_tags"].labels == [u'ADV', u'NUM', u'PROPN', u'AUX', u'VERB', u'ADJ',
                                             u'CCONJ', u'NUM', u'ADJ', u'ADP', u'PROPN', u'ADV', u'PUNCT']
        assert fields[u"head_tags"].labels == [u'advmod', u'nummod', u'nsubj:pass', u'aux:pass',
                                              u'root', u'xcomp', u'cc', u'conj', u'orphan', u'case', u'obl',
                                              u'advmod', u'punct']
        assert fields[u"head_indices"].labels == [2, 3, 5, 5, 0, 5, 8, 5, 8, 11, 5, 11, 5]
