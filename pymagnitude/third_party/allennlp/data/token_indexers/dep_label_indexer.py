
from __future__ import absolute_import
import logging
#typing

#overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DepLabelIndexer(TokenIndexer):
    u"""
    This :class:`TokenIndexer` represents tokens by their syntactic dependency label, as determined
    by the ``dep_`` field on ``Token``.

    Parameters
    ----------
    namespace : ``str``, optional (default=``dep_labels``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace      = u'dep_labels')        :
        self.namespace = namespace
        self._logged_errors = set()

    #overrides
    def count_vocab_items(self, token       , counter                           ):
        dep_label = token.dep_
        if not dep_label:
            if token.text not in self._logged_errors:
                logger.warning(u"Token had no dependency label: %s", token.text)
                self._logged_errors.add(token.text)
            dep_label = u'NONE'
        counter[self.namespace][dep_label] += 1

    #overrides
    def tokens_to_indices(self,
                          tokens             ,
                          vocabulary            ,
                          index_name     )                        :
        dep_labels = [token.dep_ or u'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(dep_label, self.namespace) for dep_label in dep_labels]}

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

DepLabelIndexer = TokenIndexer.register(u"dependency_label")(DepLabelIndexer)
