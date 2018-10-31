# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestStanfordSentimentTreebankReader(object):
    sst_path = AllenNlpTestCase.FIXTURES_ROOT / u"data" / u"sst.txt"

    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = StanfordSentimentTreeBankDatasetReader(lazy=lazy)
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {u"tokens": [u"The", u"actors", u"are", u"fantastic", u"."],
                     u"label": u"4"}
        instance2 = {u"tokens": [u"It", u"was", u"terrible", u"."],
                     u"label": u"0"}
        instance3 = {u"tokens": [u"Chomp", u"chomp", u"!"],
                     u"label": u"2"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance1[u"tokens"]
        assert fields[u"label"].label == instance1[u"label"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance2[u"tokens"]
        assert fields[u"label"].label == instance2[u"label"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance3[u"tokens"]
        assert fields[u"label"].label == instance3[u"label"]

    def test_use_subtrees(self):
        reader = StanfordSentimentTreeBankDatasetReader(use_subtrees=True)
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {u"tokens": [u"The", u"actors", u"are", u"fantastic", u"."],
                     u"label": u"4"}
        instance2 = {u"tokens": [u"The", u"actors"],
                     u"label": u"2"}
        instance3 = {u"tokens": [u"The"],
                     u"label": u"2"}

        assert len(instances) == 21
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance1[u"tokens"]
        assert fields[u"label"].label == instance1[u"label"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance2[u"tokens"]
        assert fields[u"label"].label == instance2[u"label"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance3[u"tokens"]
        assert fields[u"label"].label == instance3[u"label"]

    def test_3_class(self):
        reader = StanfordSentimentTreeBankDatasetReader(granularity=u"3-class")
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {u"tokens": [u"The", u"actors", u"are", u"fantastic", u"."],
                     u"label": u"2"}
        instance2 = {u"tokens": [u"It", u"was", u"terrible", u"."],
                     u"label": u"0"}
        instance3 = {u"tokens": [u"Chomp", u"chomp", u"!"],
                     u"label": u"1"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance1[u"tokens"]
        assert fields[u"label"].label == instance1[u"label"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance2[u"tokens"]
        assert fields[u"label"].label == instance2[u"label"]
        fields = instances[2].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance3[u"tokens"]
        assert fields[u"label"].label == instance3[u"label"]

    def test_2_class(self):
        reader = StanfordSentimentTreeBankDatasetReader(granularity=u"2-class")
        instances = reader.read(self.sst_path)
        instances = ensure_list(instances)

        instance1 = {u"tokens": [u"The", u"actors", u"are", u"fantastic", u"."],
                     u"label": u"1"}
        instance2 = {u"tokens": [u"It", u"was", u"terrible", u"."],
                     u"label": u"0"}

        assert len(instances) == 2
        fields = instances[0].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance1[u"tokens"]
        assert fields[u"label"].label == instance1[u"label"]
        fields = instances[1].fields
        assert [t.text for t in fields[u"tokens"].tokens] == instance2[u"tokens"]
        assert fields[u"label"].label == instance2[u"label"]

    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({u"use_subtrees": True, u"granularity": u"5-class"})
        reader = StanfordSentimentTreeBankDatasetReader.from_params(params)
        assert reader._use_subtrees is True
        assert reader._granularity == u"5-class"
