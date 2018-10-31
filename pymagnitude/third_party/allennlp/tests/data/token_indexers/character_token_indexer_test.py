# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer


class CharacterTokenIndexerTest(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = TokenCharactersIndexer(u"characters")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token(u"Hello"), counter)
        indexer.count_vocab_items(Token(u"hello"), counter)
        assert counter[u"characters"] == {u"h": 1, u"H": 1, u"e": 2, u"l": 4, u"o": 2}

        indexer = TokenCharactersIndexer(u"characters", CharacterTokenizer(lowercase_characters=True))
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token(u"Hello"), counter)
        indexer.count_vocab_items(Token(u"hello"), counter)
        assert counter[u"characters"] == {u"h": 2, u"e": 2, u"l": 4, u"o": 2}

    def test_as_array_produces_token_sequence(self):
        indexer = TokenCharactersIndexer(u"characters")
        padded_tokens = indexer.pad_token_sequence({u'k': [[1, 2, 3, 4, 5], [1, 2, 3], [1]]},
                                                   desired_num_tokens={u'k': 4},
                                                   padding_lengths={u"num_token_characters": 10})
        assert padded_tokens == {u'k': [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                       [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}

    def test_tokens_to_indices_produces_correct_characters(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"A", namespace=u'characters')
        vocab.add_token_to_namespace(u"s", namespace=u'characters')
        vocab.add_token_to_namespace(u"e", namespace=u'characters')
        vocab.add_token_to_namespace(u"n", namespace=u'characters')
        vocab.add_token_to_namespace(u"t", namespace=u'characters')
        vocab.add_token_to_namespace(u"c", namespace=u'characters')

        indexer = TokenCharactersIndexer(u"characters")
        indices = indexer.tokens_to_indices([Token(u"sentential")], vocab, u"char")
        assert indices == {u"char": [[3, 4, 5, 6, 4, 5, 6, 1, 1, 1]]}
