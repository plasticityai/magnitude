# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import NerTagIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestNerTagIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestNerTagIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(ner=True)

    def test_count_vocab_items_uses_ner_tags(self):
        tokens = self.tokenizer.split_words(u"Larry Page is CEO of Google.")
        tokens = [Token(u"<S>")] + [t for t in tokens] + [Token(u"</S>")]
        indexer = NerTagIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter[u"ner_tags"] == {u'PERSON': 2, u'ORG': 1, u'NONE': 6}

    def test_tokens_to_indices_uses_ner_tags(self):
        tokens = self.tokenizer.split_words(u"Larry Page is CEO of Google.")
        tokens = [t for t in tokens] + [Token(u"</S>")]
        vocab = Vocabulary()
        person_index = vocab.add_token_to_namespace(u'PERSON', namespace=u'ner_tags')
        none_index = vocab.add_token_to_namespace(u'NONE', namespace=u'ner_tags')
        vocab.add_token_to_namespace(u'ORG', namespace=u'ner_tags')
        indexer = NerTagIndexer()
        assert indexer.tokens_to_indices([tokens[1]], vocab, u"tokens1") == {u"tokens1": [person_index]}
        assert indexer.tokens_to_indices([tokens[-1]], vocab, u"tokens-1") == {u"tokens-1": [none_index]}

    def test_padding_functions(self):
        indexer = NerTagIndexer()
        assert indexer.get_padding_token() == 0
        assert indexer.get_padding_lengths(0) == {}

    def test_as_array_produces_token_sequence(self):
        indexer = NerTagIndexer()
        padded_tokens = indexer.pad_token_sequence({u'key': [1, 2, 3, 4, 5]}, {u'key': 10}, {})
        assert padded_tokens == {u'key': [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]}
