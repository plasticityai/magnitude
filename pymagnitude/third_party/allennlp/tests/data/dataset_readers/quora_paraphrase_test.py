# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import QuoraParaphraseDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
try:
    from itertools import izip
except:
    izip = zip



class TestQuoraParaphraseReader(object):
    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = QuoraParaphraseDatasetReader(lazy=lazy)
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'quora_paraphrase.tsv')
        instances = ensure_list(instances)

        instance1 = {u"premise": u"What should I do to avoid sleeping in class ?".split(),
                     u"hypothesis": u"How do I not sleep in a boring class ?".split(),
                     u"label": u"1"}

        instance2 = {u"premise": u"Do women support each other more than men do ?".split(),
                     u"hypothesis": u"Do women need more compliments than men ?".split(),
                     u"label": u"0"}

        instance3 = {u"premise": u"How can one root android devices ?".split(),
                     u"hypothesis": u"How do I root an Android device ?".split(),
                     u"label": u"1"}

        assert len(instances) == 3

        for instance, expected_instance in izip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [t.text for t in fields[u"premise"].tokens] == expected_instance[u"premise"]
            assert [t.text for t in fields[u"hypothesis"].tokens] == expected_instance[u"hypothesis"]
            assert fields[u"label"].label == expected_instance[u"label"]
