
from __future__ import absolute_import
import json

#overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


class NlvrParserPredictor(Predictor):
    #overrides
    def _json_to_instance(self, json_dict          )            :
        sentence = json_dict[u'sentence']
        if u'worlds' in json_dict:
            # This is grouped data
            worlds = json_dict[u'worlds']
        else:
            worlds = [json_dict[u'structured_rep']]
        identifier = json_dict[u'identifier'] if u'identifier' in json_dict else None
        instance = self._dataset_reader.text_to_instance(sentence=sentence,  # type: ignore
                                                         structured_representations=worlds,
                                                         identifier=identifier)
        return instance

    #overrides
    def dump_line(self, outputs          )       :  # pylint: disable=no-self-use
        if u"identifier" in outputs:
            # Returning CSV lines for official evaluation
            identifier = outputs[u"identifier"]
            denotation = outputs[u"denotations"][0][0]
            return "{identifier},{denotation}\n"
        else:
            return json.dumps(outputs) + u"\n"

NlvrParserPredictor = Predictor.register(u'nlvr-parser')(NlvrParserPredictor)
