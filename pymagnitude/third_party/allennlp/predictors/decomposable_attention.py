

from __future__ import absolute_import
#overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


class DecomposableAttentionPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise     , hypothesis     )            :
        u"""
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({u"premise" : premise, u"hypothesis": hypothesis})

    #overrides
    def _json_to_instance(self, json_dict          )            :
        u"""
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json_dict[u"premise"]
        hypothesis_text = json_dict[u"hypothesis"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

DecomposableAttentionPredictor = Predictor.register(u'textual-entailment')(DecomposableAttentionPredictor)
