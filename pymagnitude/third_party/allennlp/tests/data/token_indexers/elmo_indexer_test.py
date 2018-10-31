# pylint: disable=no-self-use

from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer


class TestELMoTokenCharactersIndexer(AllenNlpTestCase):
    def test_bos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token(u'<S>')], Vocabulary(), u"test-elmo")
        expected_indices = [259, 257, 260, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == {u"test-elmo": [expected_indices]}

    def test_eos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token(u'</S>')], Vocabulary(), u"test-eos")
        expected_indices = [259, 258, 260, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == {u"test-eos": [expected_indices]}

    def test_unicode_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token(unichr(256) + u't')], Vocabulary(), u"test-unicode")
        expected_indices = [259, 197, 129, 117, 260, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == {u"test-unicode": [expected_indices]}

    def test_elmo_as_array_produces_token_sequence(self): # pylint: disable=invalid-name
        indexer = ELMoTokenCharactersIndexer()
        tokens = [Token(u'Second'), Token(u'.')]
        indices = indexer.tokens_to_indices(tokens, Vocabulary(), u"test-elmo")[u"test-elmo"]
        padded_tokens = indexer.pad_token_sequence({u'test-elmo': indices},
                                                   desired_num_tokens={u'test-elmo': 3},
                                                   padding_lengths={})
        expected_padded_tokens = [[259, 84, 102, 100, 112, 111, 101, 260, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261],
                                  [259, 47, 260, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0]]

        assert padded_tokens[u'test-elmo'] == expected_padded_tokens
