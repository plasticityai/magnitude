# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers import TriviaQaReader
from allennlp.common.testing import AllenNlpTestCase

class TestTriviaQaReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read(self, lazy):
        params = Params({
                u'base_tarball_path': unicode(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'triviaqa-sample.tgz'),
                u'lazy': lazy
                })
        reader = TriviaQaReader.from_params(params)
        instances = reader.read(u'web-train.json')
        instances = ensure_list(instances)
        assert len(instances) == 3

        assert [t.text for t in instances[0].fields[u"question"].tokens[:3]] == [u"Which", u"American", u"-"]
        assert [t.text for t in instances[0].fields[u"passage"].tokens[:3]] == [u"The", u"Nobel", u"Prize"]
        url = u"http://www.nobelprize.org/nobel_prizes/literature/laureates/1930/"
        assert [t.text for t in instances[0].fields[u"passage"].tokens[-3:]] == [u"<", url, u">"]
        assert instances[0].fields[u"span_start"].sequence_index == 12
        assert instances[0].fields[u"span_end"].sequence_index == 13

        assert [t.text for t in instances[1].fields[u"question"].tokens[:3]] == [u"Which", u"American", u"-"]
        assert [t.text for t in instances[1].fields[u"passage"].tokens[:3]] == [u"Why", u"Do", u"n’t"]
        assert [t.text for t in instances[1].fields[u"passage"].tokens[-3:]] == [u"adults", u",", u"and"]
        assert instances[1].fields[u"span_start"].sequence_index == 38
        assert instances[1].fields[u"span_end"].sequence_index == 39

        assert [t.text for t in instances[2].fields[u"question"].tokens[:3]] == [u"Where", u"in", u"England"]
        assert [t.text for t in instances[2].fields[u"passage"].tokens[:3]] == [u"Judi", u"Dench", u"-"]
        assert [t.text for t in instances[2].fields[u"passage"].tokens[-3:]] == [u")", u"(", u"special"]
        assert instances[2].fields[u"span_start"].sequence_index == 16
        assert instances[2].fields[u"span_end"].sequence_index == 16
