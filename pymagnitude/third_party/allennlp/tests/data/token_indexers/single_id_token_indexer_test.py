# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSingleIdTokenIndexer(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = SingleIdTokenIndexer(u"words")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token(u"Hello"), counter)
        indexer.count_vocab_items(Token(u"hello"), counter)
        assert counter[u"words"] == {u"hello": 1, u"Hello": 1}

        indexer = SingleIdTokenIndexer(u"words", lowercase_tokens=True)
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token(u"Hello"), counter)
        indexer.count_vocab_items(Token(u"hello"), counter)
        assert counter[u"words"] == {u"hello": 2}

    def test_as_array_produces_token_sequence(self):
        indexer = SingleIdTokenIndexer(u"words")
        padded_tokens = indexer.pad_token_sequence({u'key': [1, 2, 3, 4, 5]}, {u'key': 10}, {})
        assert padded_tokens == {u'key': [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]}
