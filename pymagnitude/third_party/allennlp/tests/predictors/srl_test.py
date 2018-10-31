# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSrlPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                u"sentence": u"The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        archive = load_archive(self.FIXTURES_ROOT / u'srl' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'semantic-role-labeling')

        result = predictor.predict_json(inputs)

        words = result.get(u"words")
        assert words == [u"The", u"squirrel", u"wrote", u"a", u"unit", u"test",
                         u"to", u"make", u"sure", u"its", u"nuts", u"worked", u"as", u"designed", u"."]
        num_words = len(words)

        verbs = result.get(u"verbs")
        assert verbs is not None
        assert isinstance(verbs, list)

        assert any(v[u"verb"] == u"wrote" for v in verbs)
        assert any(v[u"verb"] == u"make" for v in verbs)
        assert any(v[u"verb"] == u"worked" for v in verbs)

        for verb in verbs:
            tags = verb.get(u"tags")
            assert tags is not None
            assert isinstance(tags, list)
            assert all(isinstance(tag, unicode) for tag in tags)
            assert len(tags) == num_words

    def test_batch_prediction(self):
        inputs = {
                u"sentence": u"The squirrel wrote a unit test to make sure its nuts worked as designed."
        }
        archive = load_archive(self.FIXTURES_ROOT / u'srl' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'semantic-role-labeling')
        result = predictor.predict_batch_json([inputs, inputs])
        assert result[0] == result[1]

    def test_prediction_with_no_verbs(self):

        input1 = {u"sentence": u"Blah no verb sentence."}
        archive = load_archive(self.FIXTURES_ROOT / u'srl' / u'serialization' / u'model.tar.gz')
        predictor = Predictor.from_archive(archive, u'semantic-role-labeling')
        result = predictor.predict_json(input1)
        assert result == {u'words': [u'Blah', u'no', u'verb', u'sentence', u'.'], u'verbs': []}

        input2 = {u"sentence": u"This sentence has a verb."}
        results = predictor.predict_batch_json([input1, input2])
        assert results[0] == {u'words': [u'Blah', u'no', u'verb', u'sentence', u'.'], u'verbs': []}
        assert results[1] == {u'words': [u'This', u'sentence', u'has', u'a', u'verb', u'.'],
                              u'verbs': [{u'verb': u'has', u'description': u'This sentence has a verb .',
                                         u'tags': [u'O', u'O', u'O', u'O', u'O', u'O']}]}
