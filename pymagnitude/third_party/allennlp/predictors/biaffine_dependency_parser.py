
from __future__ import absolute_import
#typing

#overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = {}

NODE_TYPE_TO_STYLE[u"root"] = [u"color5", u"strong"]
NODE_TYPE_TO_STYLE[u"dep"] = [u"color5", u"strong"]

# Arguments
NODE_TYPE_TO_STYLE[u"nsubj"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"nsubjpass"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"csubj"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"csubjpass"] = [u"color1"]

# Complements
NODE_TYPE_TO_STYLE[u"pobj"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"dobj"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"iobj"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"mark"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"pcomp"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"xcomp"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"ccomp"] = [u"color2"]
NODE_TYPE_TO_STYLE[u"acomp"] = [u"color2"]

# Modifiers
NODE_TYPE_TO_STYLE[u"aux"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"cop"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"det"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"conj"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"cc"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"prep"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"number"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"possesive"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"poss"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"discourse"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"expletive"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"prt"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"advcl"] = [u"color3"]

NODE_TYPE_TO_STYLE[u"mod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"amod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"tmod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"quantmod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"npadvmod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"infmod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"advmod"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"appos"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"nn"] = [u"color4"]

NODE_TYPE_TO_STYLE[u"neg"] = [u"color0"]
NODE_TYPE_TO_STYLE[u"punct"] = [u"color0"]


LINK_TO_POSITION = {}
# Put subjects on the left
LINK_TO_POSITION[u"nsubj"] = u"left"
LINK_TO_POSITION[u"nsubjpass"] = u"left"
LINK_TO_POSITION[u"csubj"] = u"left"
LINK_TO_POSITION[u"csubjpass"] = u"left"

# Put arguments and some clauses on the right
LINK_TO_POSITION[u"pobj"] = u"right"
LINK_TO_POSITION[u"dobj"] = u"right"
LINK_TO_POSITION[u"iobj"] = u"right"
LINK_TO_POSITION[u"pcomp"] = u"right"
LINK_TO_POSITION[u"xcomp"] = u"right"
LINK_TO_POSITION[u"ccomp"] = u"right"
LINK_TO_POSITION[u"acomp"] = u"right"

class BiaffineDependencyParserPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model       , dataset_reader               )        :
        super(BiaffineDependencyParserPredictor, self).__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyWordSplitter(language=u'en_core_web_sm', pos_tags=True)

    def predict(self, sentence     )            :
        u"""
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({u"sentence" : sentence})

    #overrides
    def _json_to_instance(self, json_dict          )            :
        u"""
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        spacy_tokens = self._tokenizer.split_words(json_dict[u"sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        pos_tags = [token.tag_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    #overrides
    def predict_instance(self, instance          )            :
        outputs = self._model.forward_on_instance(instance)

        words = outputs[u"words"]
        pos = outputs[u"pos"]
        heads = outputs[u"predicted_heads"]
        tags = outputs[u"predicted_dependencies"]
        outputs[u"hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    #overrides
    def predict_batch_instance(self, instances                )                  :
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            words = output[u"words"]
            pos = output[u"pos"]
            heads = output[u"predicted_heads"]
            tags = output[u"predicted_dependencies"]
            output[u"hierplane_tree"] = self._build_hierplane_tree(words, heads, tags, pos)
        return sanitize(outputs)

    @staticmethod
    def _build_hierplane_tree(words           ,
                              heads           ,
                              tags           ,
                              pos           )                  :
        u"""
        Returns
        -------
        A JSON dictionary render-able by Hierplane for the given tree.
        """

        word_index_to_cumulative_indices                             = {}
        cumulative_index = 0
        for i, word in enumerate(words):
            word_length = len(word) + 1
            word_index_to_cumulative_indices[i] = (cumulative_index, cumulative_index + word_length)
            cumulative_index += word_length

        def node_constuctor(index     ):
            children = []
            for next_index, child in enumerate(heads):
                if child == index + 1:
                    children.append(node_constuctor(next_index))

            # These are the icons which show up in the bottom right
            # corner of the node.
            attributes = [pos[index]]
            start, end = word_index_to_cumulative_indices[index]

            hierplane_node = {
                    u"word": words[index],
                    # The type of the node - all nodes with the same
                    # type have a unified colour.
                    u"nodeType": tags[index],
                    # Attributes of the node.
                    u"attributes": attributes,
                    # The link between  the node and it's parent.
                    u"link": tags[index],
                    u"spans": [{u"start": start, u"end": end}]
            }
            if children:
                hierplane_node[u"children"] = children
            return hierplane_node
        # We are guaranteed that there is a single word pointing to
        # the root index, so we can find it just by searching for 0 in the list.
        root_index = heads.index(0)
        hierplane_tree = {
                u"text": u" ".join(words),
                u"root": node_constuctor(root_index),
                u"nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                u"linkToPosition": LINK_TO_POSITION
        }
        return hierplane_tree

BiaffineDependencyParserPredictor = Predictor.register(u'biaffine-dependency-parser')(BiaffineDependencyParserPredictor)
