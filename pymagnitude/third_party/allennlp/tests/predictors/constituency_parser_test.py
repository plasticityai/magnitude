# pylint: disable=no-self-use,invalid-name,protected-access


from __future__ import division
from __future__ import absolute_import
from nltk import Tree

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.predictors.constituency_parser import LINK_TO_LABEL, NODE_TYPE_TO_STYLE


class TestConstituencyParserPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"sentence": u"What a great test sentence.",
        }

        archive = load_archive(self.FIXTURES_ROOT / u'constituency_parser' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'constituency-parser')

        result = predictor.predict_json(inputs)

        assert len(result[u"spans"]) == 21 # number of possible substrings of the sentence.
        assert len(result[u"class_probabilities"]) == 21
        assert result[u"tokens"] == [u"What", u"a", u"great", u"test", u"sentence", u"."]
        assert isinstance(result[u"trees"], unicode)

        for class_distribution in result[u"class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

    def test_batch_prediction(self):
        inputs = [
                {u"sentence": u"What a great test sentence."},
                {u"sentence": u"Here's another good, interesting one."}
        ]

        archive = load_archive(self.FIXTURES_ROOT / u'constituency_parser' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'constituency-parser')
        results = predictor.predict_batch_json(inputs)

        result = results[0]
        assert len(result[u"spans"]) == 21 # number of possible substrings of the sentence.
        assert len(result[u"class_probabilities"]) == 21
        assert result[u"tokens"] == [u"What", u"a", u"great", u"test", u"sentence", u"."]
        assert isinstance(result[u"trees"], unicode)

        for class_distribution in result[u"class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

        result = results[1]

        assert len(result[u"spans"]) == 36 # number of possible substrings of the sentence.
        assert len(result[u"class_probabilities"]) == 36
        assert result[u"tokens"] == [u"Here", u"'s", u"another", u"good", u",", u"interesting", u"one", u"."]
        assert isinstance(result[u"trees"], unicode)

        for class_distribution in result[u"class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

    def test_build_hierplane_tree(self):
        tree = Tree.fromstring(u"(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        archive = load_archive(self.FIXTURES_ROOT / u'constituency_parser' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'constituency-parser')

        hierplane_tree = predictor._build_hierplane_tree(tree, 0, is_root=True)

        # pylint: disable=bad-continuation
        correct_tree = {
                u'text': u'the dog chased the cat',
                u"linkNameToLabel": LINK_TO_LABEL,
                u"nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                u'root': {
                        u'word': u'the dog chased the cat',
                        u'nodeType': u'S',
                        u'attributes': [u'S'],
                        u'link': u'S',
                        u'children': [{
                                u'word': u'the dog',
                                u'nodeType': u'NP',
                                u'attributes': [u'NP'],
                                u'link': u'NP',
                                u'children': [{
                                        u'word': u'the',
                                        u'nodeType': u'D',
                                        u'attributes': [u'D'],
                                        u'link': u'D'
                                        },
                                        {
                                        u'word': u'dog',
                                        u'nodeType': u'N',
                                        u'attributes': [u'N'],
                                        u'link': u'N'}
                                        ]
                                },
                                {
                                u'word': u'chased the cat',
                                u'nodeType': u'VP',
                                u'attributes': [u'VP'],
                                u'link': u'VP',
                                u'children': [{
                                    u'word': u'chased',
                                    u'nodeType': u'V',
                                    u'attributes': [u'V'],
                                    u'link': u'V'
                                    },
                                    {
                                    u'word':
                                    u'the cat',
                                    u'nodeType': u'NP',
                                    u'attributes': [u'NP'],
                                    u'link': u'NP',
                                    u'children': [{
                                            u'word': u'the',
                                            u'nodeType': u'D',
                                            u'attributes': [u'D'],
                                            u'link': u'D'
                                            },
                                            {
                                            u'word': u'cat',
                                            u'nodeType': u'N',
                                            u'attributes': [u'N'],
                                            u'link': u'N'}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
        # pylint: enable=bad-continuation
        assert correct_tree == hierplane_tree
