# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBidafPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"question": u"What kind of test succeeded on its first attempt?",
                u"passage": u"One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive(self.FIXTURES_ROOT / u'bidaf' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'machine-comprehension')

        result = predictor.predict_json(inputs)

        best_span = result.get(u"best_span")
        assert best_span is not None
        assert isinstance(best_span, list)
        assert len(best_span) == 2
        assert all(isinstance(x, int) for x in best_span)
        assert best_span[0] <= best_span[1]

        best_span_str = result.get(u"best_span_str")
        assert isinstance(best_span_str, unicode)
        assert best_span_str != u""

        for probs_key in (u"span_start_probs", u"span_end_probs"):
            probs = result.get(probs_key)
            assert probs is not None
            assert all(isinstance(x, float) for x in probs)
            assert sum(probs) == approx(1.0)

    def test_batch_prediction(self):
        inputs = [
                {
                        u"question": u"What kind of test succeeded on its first attempt?",
                        u"passage": u"One time I was writing a unit test, and it succeeded on the first attempt."
                },
                {
                        u"question": u"What kind of test succeeded on its first attempt at batch processing?",
                        u"passage": u"One time I was writing a unit test, and it always failed!"
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / u'bidaf' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'machine-comprehension')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            best_span = result.get(u"best_span")
            best_span_str = result.get(u"best_span_str")
            start_probs = result.get(u"span_start_probs")
            end_probs = result.get(u"span_end_probs")
            assert best_span is not None
            assert isinstance(best_span, list)
            assert len(best_span) == 2
            assert all(isinstance(x, int) for x in best_span)
            assert best_span[0] <= best_span[1]

            assert isinstance(best_span_str, unicode)
            assert best_span_str != u""

            for probs in (start_probs, end_probs):
                assert probs is not None
                assert all(isinstance(x, float) for x in probs)
                assert sum(probs) == approx(1.0)
