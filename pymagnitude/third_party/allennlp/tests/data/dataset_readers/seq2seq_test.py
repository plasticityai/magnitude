# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestSeq2SeqDatasetReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_default_format(self, lazy):
        reader = Seq2SeqDatasetReader(lazy=lazy)
        instances = reader.read(unicode(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'seq2seq_copy.tsv'))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields[u"source_tokens"].tokens] == [u"@start@", u"this", u"is",
                                                                    u"a", u"sentence", u"@end@"]
        assert [t.text for t in fields[u"target_tokens"].tokens] == [u"@start@", u"this", u"is",
                                                                    u"a", u"sentence", u"@end@"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"source_tokens"].tokens] == [u"@start@", u"this", u"is",
                                                                    u"another", u"@end@"]
        assert [t.text for t in fields[u"target_tokens"].tokens] == [u"@start@", u"this", u"is",
                                                                    u"another", u"@end@"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"source_tokens"].tokens] == [u"@start@", u"all", u"these", u"sentences",
                                                                    u"should", u"get", u"copied", u"@end@"]
        assert [t.text for t in fields[u"target_tokens"].tokens] == [u"@start@", u"all", u"these", u"sentences",
                                                                    u"should", u"get", u"copied", u"@end@"]

    def test_source_add_start_token(self):
        reader = Seq2SeqDatasetReader(source_add_start_token=False)
        instances = reader.read(unicode(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'seq2seq_copy.tsv'))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields[u"source_tokens"].tokens] == [u"this", u"is", u"a", u"sentence", u"@end@"]
        assert [t.text for t in fields[u"target_tokens"].tokens] == [u"@start@", u"this", u"is",
                                                                    u"a", u"sentence", u"@end@"]
