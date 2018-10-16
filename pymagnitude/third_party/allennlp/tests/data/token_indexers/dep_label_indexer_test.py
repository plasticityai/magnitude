# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import DepLabelIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestDepLabelIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestDepLabelIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(parse=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.split_words(u"This is a sentence.")
        tokens = [Token(u"<S>")] + [t for t in tokens] + [Token(u"</S>")]
        indexer = DepLabelIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)

        assert counter[u"dep_labels"] == {u"ROOT": 1, u"nsubj": 1,
                                         u"det": 1, u"NONE": 2, u"attr": 1, u"punct": 1}

    def test_tokens_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.split_words(u"This is a sentence.")
        tokens = [t for t in tokens] + [Token(u"</S>")]
        vocab = Vocabulary()
        root_index = vocab.add_token_to_namespace(u'ROOT', namespace=u'dep_labels')
        none_index = vocab.add_token_to_namespace(u'NONE', namespace=u'dep_labels')
        indexer = DepLabelIndexer()
        assert indexer.tokens_to_indices([tokens[1]], vocab, u"tokens1") == {u"tokens1": [root_index]}
        assert indexer.tokens_to_indices([tokens[-1]], vocab, u"tokens-1") == {u"tokens-1": [none_index]}

    def test_padding_functions(self):
        indexer = DepLabelIndexer()
        assert indexer.get_padding_token() == 0
        assert indexer.get_padding_lengths(0) == {}

    def test_as_array_produces_token_sequence(self):
        indexer = DepLabelIndexer()
        padded_tokens = indexer.pad_token_sequence({u'key': [1, 2, 3, 4, 5]}, {u'key': 10}, {})
        assert padded_tokens == {u'key': [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]}
