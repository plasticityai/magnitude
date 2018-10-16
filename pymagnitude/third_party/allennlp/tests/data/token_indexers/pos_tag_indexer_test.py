# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PosTagIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestPosTagIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestPosTagIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(pos_tags=True)

    def test_count_vocab_items_uses_pos_tags(self):
        tokens = self.tokenizer.split_words(u"This is a sentence.")
        tokens = [Token(u"<S>")] + [t for t in tokens] + [Token(u"</S>")]
        indexer = PosTagIndexer()
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter[u"pos_tags"] == {u'DT': 2, u'VBZ': 1, u'.': 1, u'NN': 1, u'NONE': 2}

        indexer._coarse_tags = True  # pylint: disable=protected-access
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)
        assert counter[u"pos_tags"] == {u'VERB': 1, u'PUNCT': 1, u'DET': 2, u'NOUN': 1, u'NONE': 2}

    def test_tokens_to_indices_uses_pos_tags(self):
        tokens = self.tokenizer.split_words(u"This is a sentence.")
        tokens = [t for t in tokens] + [Token(u"</S>")]
        vocab = Vocabulary()
        verb_index = vocab.add_token_to_namespace(u'VERB', namespace=u'pos_tags')
        cop_index = vocab.add_token_to_namespace(u'VBZ', namespace=u'pos_tags')
        none_index = vocab.add_token_to_namespace(u'NONE', namespace=u'pos_tags')
        # Have to add other tokens too, since we're calling `tokens_to_indices` on all of them
        vocab.add_token_to_namespace(u'DET', namespace=u'pos_tags')
        vocab.add_token_to_namespace(u'NOUN', namespace=u'pos_tags')
        vocab.add_token_to_namespace(u'PUNCT', namespace=u'pos_tags')

        indexer = PosTagIndexer(coarse_tags=True)

        indices = indexer.tokens_to_indices(tokens, vocab, u"tokens")
        assert len(indices) == 1
        assert u"tokens" in indices
        assert indices[u"tokens"][1] == verb_index
        assert indices[u"tokens"][-1] == none_index

        indexer._coarse_tags = False  # pylint: disable=protected-access
        assert indexer.tokens_to_indices([tokens[1]], vocab, u"coarse") == {u"coarse": [cop_index]}

    def test_padding_functions(self):
        indexer = PosTagIndexer()
        assert indexer.get_padding_token() == 0
        assert indexer.get_padding_lengths(0) == {}

    def test_as_array_produces_token_sequence(self):
        indexer = PosTagIndexer()
        padded_tokens = indexer.pad_token_sequence({u'key': [1, 2, 3, 4, 5]}, {u'key': 10}, {})
        assert padded_tokens == {u'key': [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]}
