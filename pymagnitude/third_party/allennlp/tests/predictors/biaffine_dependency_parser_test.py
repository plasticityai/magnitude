# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBiaffineDependencyParser(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"sentence": u"Please could you parse this sentence?",
        }

        archive = load_archive(self.FIXTURES_ROOT / u'biaffine_dependency_parser'
                               / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'biaffine-dependency-parser')

        result = predictor.predict_json(inputs)

        words = result.get(u"words")
        predicted_heads = result.get(u"predicted_heads")
        assert len(predicted_heads) == len(words)

        predicted_dependencies = result.get(u"predicted_dependencies")
        assert len(predicted_dependencies) == len(words)
        assert isinstance(predicted_dependencies, list)
        assert all(isinstance(x, unicode) for x in predicted_dependencies)

        assert result.get(u"loss") is not None
        assert result.get(u"arc_loss") is not None
        assert result.get(u"tag_loss") is not None

        hierplane_tree = result.get(u"hierplane_tree")
        hierplane_tree.pop(u"nodeTypeToStyle")
        hierplane_tree.pop(u"linkToPosition")
        # pylint: disable=line-too-long,bad-continuation
        assert result.get(u"hierplane_tree") == {u'text': u'Please could you parse this sentence ?',
                                                u'root': {u'word': u'Please', u'nodeType': u'det', u'attributes': [u'UH'], u'link': u'det', u'spans': [{u'start': 0, u'end': 7}],
                                                    u'children': [
                                                            {u'word': u'could', u'nodeType': u'nummod', u'attributes': [u'MD'], u'link': u'nummod', u'spans': [{u'start': 7, u'end': 13}]},
                                                            {u'word': u'you', u'nodeType': u'nummod', u'attributes': [u'PRP'], u'link': u'nummod', u'spans': [{u'start': 13, u'end': 17}]},
                                                            {u'word': u'parse', u'nodeType': u'nummod', u'attributes': [u'VB'], u'link': u'nummod', u'spans': [{u'start': 17, u'end': 23}]},
                                                            {u'word': u'this', u'nodeType': u'nummod', u'attributes': [u'DT'], u'link': u'nummod', u'spans': [{u'start': 23, u'end': 28}]},
                                                            {u'word': u'sentence', u'nodeType': u'nummod', u'attributes':[u'NN'], u'link': u'nummod', u'spans': [{u'start': 28, u'end': 37}]},
                                                            {u'word': u'?', u'nodeType': u'nummod', u'attributes': [u'.'], u'link': u'nummod', u'spans': [{u'start': 37, u'end': 39}]}
                                                            ]
                                                        }
                                               }
        # pylint: enable=line-too-long,bad-continuation
    def test_batch_prediction(self):
        inputs = [
                {
                        u"sentence": u"What kind of test succeeded on its first attempt?",
                },
                {
                        u"sentence": u"What kind of test succeeded on its first attempt at batch processing?",
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / u'biaffine_dependency_parser'
                               / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'biaffine-dependency-parser')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            sequence_length = len(result.get(u"words"))
            predicted_heads = result.get(u"predicted_heads")
            assert len(predicted_heads) == sequence_length

            predicted_dependencies = result.get(u"predicted_dependencies")
            assert len(predicted_dependencies) == sequence_length
            assert isinstance(predicted_dependencies, list)
            assert all(isinstance(x, unicode) for x in predicted_dependencies)
