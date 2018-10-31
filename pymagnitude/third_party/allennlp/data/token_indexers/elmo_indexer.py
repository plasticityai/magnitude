
from __future__ import absolute_import
#typing

#overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary


def _make_bos_eos(
        character     ,
        padding_character     ,
        beginning_of_word_character     ,
        end_of_word_character     ,
        max_word_length     
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids

class ELMoCharacterMapper(object):
    u"""
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = u'<S>'
    eos_token = u'</S>'

    @staticmethod
    def convert_word_to_char_ids(word     )             :
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode(u'utf-8', u'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                if not(isinstance(chr_id, int)):
                    char_ids[k] = ord(chr_id)
                else:
                    char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]


class ELMoTokenCharactersIndexer(TokenIndexer):
    u"""
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace      = u'elmo_characters')        :
        self._namespace = namespace

    #overrides
    def count_vocab_items(self, token       , counter                           ):
        pass

    #overrides
    def tokens_to_indices(self,
                          tokens             ,
                          vocabulary            ,
                          index_name     )                              :
        # pylint: disable=unused-argument
        texts = [token.text for token in tokens]

        if any(text is None for text in texts):
            raise ConfigurationError(u'ELMoTokenCharactersIndexer needs a tokenizer '
                                     u'that retains text')
        return {index_name: [ELMoCharacterMapper.convert_word_to_char_ids(text) for text in texts]}

    #overrides
    def get_padding_lengths(self, token           )                  :
        # pylint: disable=unused-argument
        return {}

    #overrides
    def get_padding_token(self)             :
        return []

    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    #overrides
    def pad_token_sequence(self,
                           tokens                            ,
                           desired_num_tokens                ,
                           padding_lengths                )                              :
        # pylint: disable=unused-argument
        return dict((key, pad_sequence_to_length(val, desired_num_tokens[key],
                                            default_value=self._default_value_for_padding))
                for key, val in list(tokens.items()))

ELMoTokenCharactersIndexer = TokenIndexer.register(u"elmo_characters")(ELMoTokenCharactersIndexer)
