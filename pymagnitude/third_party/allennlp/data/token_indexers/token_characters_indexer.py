
from __future__ import absolute_import
#typing
import itertools

#overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
try:
    from itertools import izip
except:
    izip = zip



class TokenCharactersIndexer(TokenIndexer):
    u"""
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace      = u'token_characters',
                 character_tokenizer                     = CharacterTokenizer())        :
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

    #overrides
    def count_vocab_items(self, token       , counter                           ):
        if token.text is None:
            raise ConfigurationError(u'TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(character, u'text_id', None) is None:
                counter[self._namespace][character.text] += 1

    #overrides
    def tokens_to_indices(self,
                          tokens             ,
                          vocabulary            ,
                          index_name     )                              :
        indices                  = []
        for token in tokens:
            token_indices            = []
            if token.text is None:
                raise ConfigurationError(u'TokenCharactersIndexer needs a tokenizer that retains text')
            for character in self._character_tokenizer.tokenize(token.text):
                if getattr(character, u'text_id', None) is not None:
                    # `text_id` being set on the token means that we aren't using the vocab, we just
                    # use this id instead.
                    index = character.text_id
                else:
                    index = vocabulary.get_token_index(character.text, self._namespace)
                token_indices.append(index)
            indices.append(token_indices)
        return {index_name: indices}

    #overrides
    def get_padding_lengths(self, token           )                  :
        return {u'num_token_characters': len(token)}

    #overrides
    def get_padding_token(self)             :
        return []

    #overrides
    def pad_token_sequence(self,
                           tokens                            ,
                           desired_num_tokens                ,
                           padding_lengths                )                              :
        # Pad the tokens.
        # tokens has only one key...
        key = list(tokens.keys())[0]

        padded_tokens = pad_sequence_to_length(
                tokens[key], desired_num_tokens[key],
                default_value=self.get_padding_token
        )

        # Pad the characters within the tokens.
        desired_token_length = padding_lengths[u'num_token_characters']
        longest_token            = max(tokens[key], key=len, default=[])
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(izip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return {key: [list(token[:desired_token_length]) for token in padded_tokens]}

TokenCharactersIndexer = TokenIndexer.register(u"characters")(TokenCharactersIndexer)
