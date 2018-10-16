
from __future__ import absolute_import
#overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

class SimpleSeq2SeqPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    def predict(self, source     )            :
        return self.predict_json({u"source" : source})

    #overrides
    def _json_to_instance(self, json_dict          )            :
        u"""
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict[u"source"]
        return self._dataset_reader.text_to_instance(source)

SimpleSeq2SeqPredictor = Predictor.register(u'simple_seq2seq')(SimpleSeq2SeqPredictor)
