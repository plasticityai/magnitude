# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestSequenceTaggingDatasetReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_default_format(self, lazy):
        reader = SequenceTaggingDatasetReader(lazy=lazy)
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 4
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"cats", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"dogs", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"snakes", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[3].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"birds", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]

    def test_brown_corpus_format(self):
        reader = SequenceTaggingDatasetReader(word_tag_delimiter=u'/')
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'brown_corpus.txt')
        instances = ensure_list(instances)

        assert len(instances) == 4
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"cats", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"dogs", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"snakes", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
        fields = instances[3].fields
        assert [t.text for t in fields[u"tokens"].tokens] == [u"birds", u"are", u"animals", u"."]
        assert fields[u"tags"].labels == [u"N", u"V", u"N", u"N"]
