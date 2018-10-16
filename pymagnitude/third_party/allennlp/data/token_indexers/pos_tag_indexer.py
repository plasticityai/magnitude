
from __future__ import absolute_import
import logging
#typing

#overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PosTagIndexer(TokenIndexer):
    u"""
    This :class:`TokenIndexer` represents tokens by their part of speech tag, as determined by
    the ``pos_`` or ``tag_`` fields on ``Token`` (corresponding to spacy's coarse-grained and
    fine-grained POS tags, respectively).

    Parameters
    ----------
    namespace : ``str``, optional (default=``pos_tags``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    coarse_tags : ``bool``, optional (default=``False``)
        If ``True``, we will use coarse POS tags instead of the default fine-grained POS tags.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace      = u'pos_tags', coarse_tags       = False)        :
        self._namespace = namespace
        self._coarse_tags = coarse_tags
        self._logged_errors = set()

    #overrides
    def count_vocab_items(self, token       , counter                           ):
        if self._coarse_tags:
            tag = token.pos_
        else:
            tag = token.tag_
        if not tag:
            if token.text not in self._logged_errors:
                logger.warning(u"Token had no POS tag: %s", token.text)
                self._logged_errors.add(token.text)
            tag = u'NONE'
        counter[self._namespace][tag] += 1

    #overrides
    def tokens_to_indices(self,
                          tokens             ,
                          vocabulary            ,
                          index_name     )                        :
        tags            = []

        for token in tokens:
            if self._coarse_tags:
                tag = token.pos_
            else:
                tag = token.tag_
            if tag is None:
                tag = u'NONE'

            tags.append(tag)

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

PosTagIndexer = TokenIndexer.register(u"pos_tag")(PosTagIndexer)
