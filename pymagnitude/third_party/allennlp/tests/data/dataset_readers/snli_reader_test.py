# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import SnliReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestSnliReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SnliReader(lazy=lazy)
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'snli.jsonl')
        instances = ensure_list(instances)

        instance1 = {u"premise": [u"A", u"person", u"on", u"a", u"horse", u"jumps", u"over", u"a", u"broken",
                                 u"down", u"airplane", u"."],
                     u"hypothesis": [u"A", u"person", u"is", u"training", u"his", u"horse", u"for", u"a",
                                    u"competition", u"."],
                     u"label": u"neutral"}

        instance2 = {u"premise": [u"A", u"person", u"on", u"a", u"horse", u"jumps", u"over", u"a", u"broken",
                                 u"down", u"airplane", u"."],
                     u"hypothesis": [u"A", u"person", u"is", u"at", u"a", u"diner", u",", u"ordering", u"an",
                                    u"omelette", u"."],
                     u"label": u"contradiction"}
        instance3 = {u"premise": [u"A", u"person", u"on", u"a", u"horse", u"jumps", u"over", u"a", u"broken",
                                 u"down", u"airplane", u"."],
                     u"hypothesis": [u"A", u"person", u"is", u"outdoors", u",", u"on", u"a", u"horse", u"."],
                     u"label": u"entailment"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields[u"premise"].tokens] == instance1[u"premise"]
        assert [t.text for t in fields[u"hypothesis"].tokens] == instance1[u"hypothesis"]
        assert fields[u"label"].label == instance1[u"label"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"premise"].tokens] == instance2[u"premise"]
        assert [t.text for t in fields[u"hypothesis"].tokens] == instance2[u"hypothesis"]
        assert fields[u"label"].label == instance2[u"label"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"premise"].tokens] == instance3[u"premise"]
        assert [t.text for t in fields[u"hypothesis"].tokens] == instance3[u"hypothesis"]
        assert fields[u"label"].label == instance3[u"label"]
