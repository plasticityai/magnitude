# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import math

from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
try:
    from itertools import izip
except:
    izip = zip



class TestDecomposableAttentionPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"premise": u"I always write unit tests for my code.",
                u"hypothesis": u"One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / u'decomposable_attention' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'textual-entailment')
        result = predictor.predict_json(inputs)

        # Label probs should be 3 floats that sum to one
        label_probs = result.get(u"label_probs")
        assert label_probs is not None
        assert isinstance(label_probs, list)
        assert len(label_probs) == 3
        assert all(isinstance(x, float) for x in label_probs)
        assert all(x >= 0 for x in label_probs)
        assert sum(label_probs) == approx(1.0)

        # Logits should be 3 floats that softmax to label_probs
        label_logits = result.get(u"label_logits")
        assert label_logits is not None
        assert isinstance(label_logits, list)
        assert len(label_logits) == 3
        assert all(isinstance(x, float) for x in label_logits)

        exps = [math.exp(x) for x in label_logits]
        sumexps = sum(exps)
        for e, p in izip(exps, label_probs):
            assert e / sumexps == approx(p)

    def test_batch_prediction(self):
        batch_inputs = [
                {
                        u"premise": u"I always write unit tests for my code.",
                        u"hypothesis": u"One time I didn't write any unit tests for my code."
                },
                {
                        u"premise": u"I also write batched unit tests for throughput!",
                        u"hypothesis": u"Batch tests are slower."
                },
        ]

        archive = load_archive(self.FIXTURES_ROOT / u'decomposable_attention' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'textual-entailment')
        results = predictor.predict_batch_json(batch_inputs)
        print(results)
        assert len(results) == 2

        for result in results:
            # Logits should be 3 floats that softmax to label_probs
            label_logits = result.get(u"label_logits")
            # Label probs should be 3 floats that sum to one
            label_probs = result.get(u"label_probs")
            assert label_probs is not None
            assert isinstance(label_probs, list)
            assert len(label_probs) == 3
            assert all(isinstance(x, float) for x in label_probs)
            assert all(x >= 0 for x in label_probs)
            assert sum(label_probs) == approx(1.0)

            assert label_logits is not None
            assert isinstance(label_logits, list)
            assert len(label_logits) == 3
            assert all(isinstance(x, float) for x in label_logits)

            exps = [math.exp(x) for x in label_logits]
            sumexps = sum(exps)
            for e, p in izip(exps, label_probs):
                assert e / sumexps == approx(p)
