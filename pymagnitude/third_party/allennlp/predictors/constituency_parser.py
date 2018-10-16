
from __future__ import absolute_import
#typing

#overrides
from nltk import Tree
from spacy.lang.en.tag_map import TAG_MAP

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


# Make the links to POS tag nodes render as "pos",
# to distinguish them from constituency tags. The
# actual tag is still visible within the node.
LINK_TO_LABEL = dict((x, u"pos") for x in TAG_MAP)

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = dict((x, [u"color0"]) for x in TAG_MAP)

# Verb and Noun phrases get their own colour.
NODE_TYPE_TO_STYLE[u"NP"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"NX"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"QP"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"NAC"] = [u"color1"]
NODE_TYPE_TO_STYLE[u"VP"] = [u"color2"]

# Clause level fragments
NODE_TYPE_TO_STYLE[u"S"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"SQ"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"SBAR"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"SBARQ"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"SINQ"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"FRAG"] = [u"color3"]
NODE_TYPE_TO_STYLE[u"X"] = [u"color3"]

# Wh-phrases.
NODE_TYPE_TO_STYLE[u"WHADVP"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"WHADJP"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"WHNP"] = [u"color4"]
NODE_TYPE_TO_STYLE[u"WHPP"] = [u"color4"]

# Prepositional Phrases get their own colour because
# they are linguistically interesting.
NODE_TYPE_TO_STYLE[u"PP"] = [u"color6"]

# Everything else.
NODE_TYPE_TO_STYLE[u"ADJP"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"ADVP"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"CONJP"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"INTJ"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"LST"] = [u"color5", u"seq"]
NODE_TYPE_TO_STYLE[u"PRN"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"PRT"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"RRC"] = [u"color5"]
NODE_TYPE_TO_STYLE[u"UCP"] = [u"color5"]


class ConstituencyParserPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.SpanConstituencyParser` model.
    """
    def __init__(self, model       , dataset_reader               )        :
        super(ConstituencyParserPredictor, self).__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=u'en_core_web_sm', pos_tags=True)

    def predict(self, sentence     )            :
        u"""
        Predict a constituency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the constituency tree.
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

        # format the NLTK tree as a string on a single line.
        tree = outputs.pop(u"trees")
        outputs[u"hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
        outputs[u"trees"] = tree.pformat(margin=1000000)
        return sanitize(outputs)

    #overrides
    def predict_batch_instance(self, instances                )                  :
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            # format the NLTK tree as a string on a single line.
            tree = output.pop(u"trees")
            output[u"hierplane_tree"] = self._build_hierplane_tree(tree, 0, is_root=True)
            output[u"trees"] = tree.pformat(margin=1000000)
        return sanitize(outputs)


    def _build_hierplane_tree(self, tree      , index     , is_root      )            :
        u"""
        Recursively builds a JSON dictionary from an NLTK ``Tree`` suitable for
        rendering trees using the `Hierplane library<https://allenai.github.io/hierplane/>`.

        Parameters
        ----------
        tree : ``Tree``, required.
            The tree to convert into Hierplane JSON.
        index : int, required.
            The character index into the tree, used for creating spans.
        is_root : bool
            An indicator which allows us to add the outer Hierplane JSON which
            is required for rendering.

        Returns
        -------
        A JSON dictionary render-able by Hierplane for the given tree.
        """
        children = []
        for child in tree:
            if isinstance(child, Tree):
                # If the child is a tree, it has children,
                # as NLTK leaves are just strings.
                children.append(self._build_hierplane_tree(child, index, is_root=False))
            else:
                # We're at a leaf, so add the length of
                # the word to the character index.
                index += len(child)

        label = tree.label()
        span = u" ".join(tree.leaves())
        hierplane_node = {
                u"word": span,
                u"nodeType": label,
                u"attributes": [label],
                u"link": label
        }
        if children:
            hierplane_node[u"children"] = children
        # TODO(Mark): Figure out how to span highlighting to the leaves.
        if is_root:
            hierplane_node = {
                    u"linkNameToLabel": LINK_TO_LABEL,
                    u"nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                    u"text": span,
                    u"root": hierplane_node
            }
        return hierplane_node

ConstituencyParserPredictor = Predictor.register(u'constituency-parser')(ConstituencyParserPredictor)
