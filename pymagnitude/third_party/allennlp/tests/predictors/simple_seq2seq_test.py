# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSimpleSeq2SeqPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"source": u"What kind of test succeeded on its first attempt?",
        }

        archive = load_archive(self.FIXTURES_ROOT / u'encoder_decoder' / u'simple_seq2seq' /
                               u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'simple_seq2seq')

        result = predictor.predict_json(inputs)

        predicted_tokens = result.get(u"predicted_tokens")
        assert predicted_tokens is not None
        assert isinstance(predicted_tokens, list)
        assert all(isinstance(x, unicode) for x in predicted_tokens)
