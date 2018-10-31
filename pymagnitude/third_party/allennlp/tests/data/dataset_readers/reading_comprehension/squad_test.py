# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import SquadReader
from allennlp.common.testing import AllenNlpTestCase

class TestSquadReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SquadReader(lazy=lazy)
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'squad.json'))
        assert len(instances) == 5

        assert [t.text for t in instances[0].fields[u"question"].tokens[:3]] == [u"To", u"whom", u"did"]
        assert [t.text for t in instances[0].fields[u"passage"].tokens[:3]] == [u"Architecturally", u",", u"the"]
        assert [t.text for t in instances[0].fields[u"passage"].tokens[-3:]] == [u"of", u"Mary", u"."]
        assert instances[0].fields[u"span_start"].sequence_index == 102
        assert instances[0].fields[u"span_end"].sequence_index == 104

        assert [t.text for t in instances[1].fields[u"question"].tokens[:3]] == [u"What", u"sits", u"on"]
        assert [t.text for t in instances[1].fields[u"passage"].tokens[:3]] == [u"Architecturally", u",", u"the"]
        assert [t.text for t in instances[1].fields[u"passage"].tokens[-3:]] == [u"of", u"Mary", u"."]
        assert instances[1].fields[u"span_start"].sequence_index == 17
        assert instances[1].fields[u"span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert ([t.text for t in instances[3].fields[u"question"].tokens[:3]] ==
                [u"Which", u"individual", u"worked"])
        assert [t.text for t in instances[3].fields[u"passage"].tokens[:3]] == [u"In", u"1882", u","]
        assert [t.text for t in instances[3].fields[u"passage"].tokens[-3:]] == [u"Nuclear", u"Astrophysics", u"."]
        span_start = instances[3].fields[u"span_start"].sequence_index
        span_end = instances[3].fields[u"span_end"].sequence_index
        answer_tokens = instances[3].fields[u"passage"].tokens[span_start:(span_end + 1)]
        expected_answer_tokens = [u"Father", u"Julius", u"Nieuwland"]
        assert [t.text for t in answer_tokens] == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == u'WordTokenizer'
        assert reader._token_indexers[u"tokens"].__class__.__name__ == u'SingleIdTokenIndexer'
