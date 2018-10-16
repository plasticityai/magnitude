

from __future__ import division
from __future__ import absolute_import
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestNlvrParserPredictor(AllenNlpTestCase):
    def setUp(self):
        super(TestNlvrParserPredictor, self).setUp()
        self.inputs = {u'worlds': [[[{u'y_loc': 80, u'type': u'triangle', u'color': u'#0099ff', u'x_loc': 80,
                                     u'size': 20}],
                                   [{u'y_loc': 80, u'type': u'square', u'color': u'Yellow', u'x_loc': 13,
                                     u'size': 20}],
                                   [{u'y_loc': 67, u'type': u'triangle', u'color': u'Yellow', u'x_loc': 35,
                                     u'size': 10}]],
                                  [[{u'y_loc': 8, u'type': u'square', u'color': u'Yellow', u'x_loc': 57,
                                     u'size': 30}],
                                   [{u'y_loc': 43, u'type': u'square', u'color': u'#0099ff', u'x_loc': 70,
                                     u'size': 30}],
                                   [{u'y_loc': 59, u'type': u'square', u'color': u'Yellow', u'x_loc': 47,
                                     u'size': 10}]]],
                       u'identifier': u'fake_id',
                       u'sentence': u'Each grey box contains atleast one yellow object touching the edge'}

    def test_predictor_with_coverage_parser(self):
        archive_dir = self.FIXTURES_ROOT / u'semantic_parsing' / u'nlvr_coverage_semantic_parser' / u'serialization'
        archive = load_archive(os.path.join(archive_dir, u'model.tar.gz'))
        predictor = Predictor.from_archive(archive, u'nlvr-parser')

        result = predictor.predict_json(self.inputs)
        assert u'logical_form' in result
        assert u'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result[u'denotations'][0]) == 2  # Because there are two worlds in the input.

    def test_predictor_with_direct_parser(self):
        archive_dir = self.FIXTURES_ROOT / u'semantic_parsing' / u'nlvr_direct_semantic_parser' / u'serialization'
        archive = load_archive(os.path.join(archive_dir, u'model.tar.gz'))
        predictor = Predictor.from_archive(archive, u'nlvr-parser')

        result = predictor.predict_json(self.inputs)
        assert u'logical_form' in result
        assert u'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result[u'denotations'][0]) == 2  # Because there are two worlds in the input.
