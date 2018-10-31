
from __future__ import absolute_import
import logging
#typing

#overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NerTagIndexer(TokenIndexer):
    u"""
    This :class:`TokenIndexer` represents tokens by their entity type (i.e., their NER tag), as
    determined by the ``ent_type_`` field on ``Token``.

    Parameters
    ----------
    namespace : ``str``, optional (default=``ner_tags``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace      = u'ner_tags')        :
        self._namespace = namespace

    #overrides
    def count_vocab_items(self, token       , counter                           ):
        tag = token.ent_type_
        if not tag:
            tag = u'NONE'
        counter[self._namespace][tag] += 1

    #overrides
    def tokens_to_indices(self,
                          tokens             ,
                          vocabulary            ,
                          index_name     )                        :
        tags = [u'NONE' if token.ent_type_ is None else token.ent_type_ for token in tokens]

        return {index_name: [vocabulary.get_token_index(tag, self._namespace) for tag in tags]}

    #overrides
    def get_padding_token(self)       :
        return 0

    #overrides
    def get_padding_lengths(self, token     )                  :  # pylint: disable=unused-argument
        return {}

    #overrides
    def pad_token_sequence(self,
                           tokens                      ,
                           desired_num_tokens                ,
                           padding_lengths                )                        :  # pylint: disable=unused-argument
        return dict((key, pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in list(tokens.items()))

NerTagIndexer = TokenIndexer.register(u"ner_tag")(NerTagIndexer)
