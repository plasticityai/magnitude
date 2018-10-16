
from __future__ import absolute_import
#overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

class BidafPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question     , passage     )            :
        u"""
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({u"passage" : passage, u"question" : question})

    #overrides
    def _json_to_instance(self, json_dict          )            :
        u"""
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict[u"question"]
        passage_text = json_dict[u"passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

BidafPredictor = Predictor.register(u'machine-comprehension')(BidafPredictor)
