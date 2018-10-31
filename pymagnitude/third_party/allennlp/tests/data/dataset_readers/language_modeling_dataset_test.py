# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestLanguageModelingDatasetReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = LanguageModelingReader(tokens_per_instance=3, lazy=lazy)

        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'language_modeling.txt'))
        # The last potential instance is left out, which is ok, because we don't have an end token
        # in here, anyway.
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields[u"input_tokens"].tokens] == [u"This", u"is", u"a"]
        assert [t.text for t in instances[0].fields[u"output_tokens"].tokens] == [u"is", u"a", u"sentence"]

        assert [t.text for t in instances[1].fields[u"input_tokens"].tokens] == [u"sentence", u"for", u"language"]
        assert [t.text for t in instances[1].fields[u"output_tokens"].tokens] == [u"for", u"language", u"modelling"]

        assert [t.text for t in instances[2].fields[u"input_tokens"].tokens] == [u"modelling", u".", u"Here"]
        assert [t.text for t in instances[2].fields[u"output_tokens"].tokens] == [u".", u"Here", u"'s"]

        assert [t.text for t in instances[3].fields[u"input_tokens"].tokens] == [u"'s", u"another", u"one"]
        assert [t.text for t in instances[3].fields[u"output_tokens"].tokens] == [u"another", u"one", u"for"]

        assert [t.text for t in instances[4].fields[u"input_tokens"].tokens] == [u"for", u"extra", u"language"]
        assert [t.text for t in instances[4].fields[u"output_tokens"].tokens] == [u"extra", u"language", u"modelling"]
