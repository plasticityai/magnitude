# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestCorefPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {u"document": u"This is a single string document about a test. Sometimes it "
                              u"contains coreferent parts."}
        archive = load_archive(self.FIXTURES_ROOT / u'coref' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'coreference-resolution')

        result = predictor.predict_json(inputs)

        document = result[u"document"]
        assert document == [u'This', u'is', u'a', u'single', u'string',
                            u'document', u'about', u'a', u'test', u'.', u'Sometimes',
                            u'it', u'contains', u'coreferent', u'parts', u'.']

        clusters = result[u"clusters"]
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, list)
            for mention in cluster:
                # Spans should be integer indices.
                assert isinstance(mention[0], int)
                assert isinstance(mention[1], int)
                # Spans should be inside document.
                assert 0 < mention[0] <= len(document)
                assert 0 < mention[1] <= len(document)
