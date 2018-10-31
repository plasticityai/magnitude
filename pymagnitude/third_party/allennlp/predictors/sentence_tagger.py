
from __future__ import absolute_import
#overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


class SentenceTaggerPredictor(Predictor):
    u"""
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model       , dataset_reader               )        :
        super(SentenceTaggerPredictor, self).__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=u'en_core_web_sm', pos_tags=True)

    def predict(self, sentence     )            :
        return self.predict_json({u"sentence" : sentence})

    #overrides
    def _json_to_instance(self, json_dict          )            :
        u"""
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict[u"sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)

SentenceTaggerPredictor = Predictor.register(u'sentence-tagger')(SentenceTaggerPredictor)
