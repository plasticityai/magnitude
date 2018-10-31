# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class TestPredictor(AllenNlpTestCase):
    def test_from_archive_does_not_consume_params(self):
        archive = load_archive(self.FIXTURES_ROOT / u'bidaf' / u'serialization' / u'model.tar.gz')
        Predictor.from_archive(archive, u'machine-comprehension')

        # If it consumes the params, this will raise an exception
        Predictor.from_archive(archive, u'machine-comprehension')
