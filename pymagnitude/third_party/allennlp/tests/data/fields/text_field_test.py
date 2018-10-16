# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict
#typing

import pytest
import numpy

from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, TokenIndexer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length


class DictReturningTokenIndexer(TokenIndexer):
    u"""
    A stub TokenIndexer that returns multiple arrays of different lengths.
    """
    def count_vocab_items(self, token       , counter                           ):
        pass

    def tokens_to_indices(self, tokens             ,
                          vocabulary            ,
                          index_name     )                        : # pylint: disable=unused-argument
        return {
                u"token_ids": [10, 15] +\
                         [vocabulary.get_token_index(token.text, u'words') for token in tokens] +\
                         [25],
                u"additional_key": [22, 29]
        }

    def get_padding_token(self)       :
        return 0

    def get_padding_lengths(self, token     )                  :  # pylint: disable=unused-argument
        return {}

    def pad_token_sequence(self,
                           tokens                      ,
                           desired_num_tokens                ,
                           padding_lengths                )                        :  # pylint: disable=unused-argument
        return dict((key, pad_sequence_to_length(val, desired_num_tokens[key])) for key, val in list(tokens.items()))

    def get_keys(self, index_name     )             :
        # pylint: disable=unused-argument,no-self-use
        return [u"token_ids", u"additional_key"]


class TestTextField(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace(u"sentence", namespace=u'words')
        self.vocab.add_token_to_namespace(u"A", namespace=u'words')
        self.vocab.add_token_to_namespace(u"A", namespace=u'characters')
        self.vocab.add_token_to_namespace(u"s", namespace=u'characters')
        self.vocab.add_token_to_namespace(u"e", namespace=u'characters')
        self.vocab.add_token_to_namespace(u"n", namespace=u'characters')
        self.vocab.add_token_to_namespace(u"t", namespace=u'characters')
        self.vocab.add_token_to_namespace(u"c", namespace=u'characters')
        super(TestTextField, self).setUp()

    def test_field_counts_vocab_items_correctly(self):
        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words")})
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts[u"words"][u"This"] == 1
        assert namespace_token_counts[u"words"][u"is"] == 1
        assert namespace_token_counts[u"words"][u"a"] == 1
        assert namespace_token_counts[u"words"][u"sentence"] == 1
        assert namespace_token_counts[u"words"][u"."] == 1
        assert list(namespace_token_counts.keys()) == [u"words"]

        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"characters": TokenCharactersIndexer(u"characters")})
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts[u"characters"][u"T"] == 1
        assert namespace_token_counts[u"characters"][u"h"] == 1
        assert namespace_token_counts[u"characters"][u"i"] == 2
        assert namespace_token_counts[u"characters"][u"s"] == 3
        assert namespace_token_counts[u"characters"][u"a"] == 1
        assert namespace_token_counts[u"characters"][u"e"] == 3
        assert namespace_token_counts[u"characters"][u"n"] == 2
        assert namespace_token_counts[u"characters"][u"t"] == 1
        assert namespace_token_counts[u"characters"][u"c"] == 1
        assert namespace_token_counts[u"characters"][u"."] == 1
        assert list(namespace_token_counts.keys()) == [u"characters"]

        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words"),
                                          u"characters": TokenCharactersIndexer(u"characters")})
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)
        assert namespace_token_counts[u"characters"][u"T"] == 1
        assert namespace_token_counts[u"characters"][u"h"] == 1
        assert namespace_token_counts[u"characters"][u"i"] == 2
        assert namespace_token_counts[u"characters"][u"s"] == 3
        assert namespace_token_counts[u"characters"][u"a"] == 1
        assert namespace_token_counts[u"characters"][u"e"] == 3
        assert namespace_token_counts[u"characters"][u"n"] == 2
        assert namespace_token_counts[u"characters"][u"t"] == 1
        assert namespace_token_counts[u"characters"][u"c"] == 1
        assert namespace_token_counts[u"characters"][u"."] == 1
        assert namespace_token_counts[u"words"][u"This"] == 1
        assert namespace_token_counts[u"words"][u"is"] == 1
        assert namespace_token_counts[u"words"][u"a"] == 1
        assert namespace_token_counts[u"words"][u"sentence"] == 1
        assert namespace_token_counts[u"words"][u"."] == 1
        assert set(namespace_token_counts.keys()) == set([u"words", u"characters"])

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        sentence_index = vocab.add_token_to_namespace(u"sentence", namespace=u'words')
        capital_a_index = vocab.add_token_to_namespace(u"A", namespace=u'words')
        capital_a_char_index = vocab.add_token_to_namespace(u"A", namespace=u'characters')
        s_index = vocab.add_token_to_namespace(u"s", namespace=u'characters')
        e_index = vocab.add_token_to_namespace(u"e", namespace=u'characters')
        n_index = vocab.add_token_to_namespace(u"n", namespace=u'characters')
        t_index = vocab.add_token_to_namespace(u"t", namespace=u'characters')
        c_index = vocab.add_token_to_namespace(u"c", namespace=u'characters')

        field = TextField([Token(t) for t in [u"A", u"sentence"]],
                          {u"words": SingleIdTokenIndexer(namespace=u"words")})
        field.index(vocab)
        # pylint: disable=protected-access
        assert field._indexed_tokens[u"words"] == [capital_a_index, sentence_index]

        field1 = TextField([Token(t) for t in [u"A", u"sentence"]],
                           {u"characters": TokenCharactersIndexer(namespace=u"characters")})
        field1.index(vocab)
        assert field1._indexed_tokens[u"characters"] == [[capital_a_char_index],
                                                        [s_index, e_index, n_index, t_index,
                                                         e_index, n_index, c_index, e_index]]
        field2 = TextField([Token(t) for t in [u"A", u"sentence"]],
                           token_indexers={u"words": SingleIdTokenIndexer(namespace=u"words"),
                                           u"characters": TokenCharactersIndexer(namespace=u"characters")})
        field2.index(vocab)
        assert field2._indexed_tokens[u"words"] == [capital_a_index, sentence_index]
        assert field2._indexed_tokens[u"characters"] == [[capital_a_char_index],
                                                        [s_index, e_index, n_index, t_index,
                                                         e_index, n_index, c_index, e_index]]
        # pylint: enable=protected-access

    def test_get_padding_lengths_raises_if_no_indexed_tokens(self):

        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words")})
        with pytest.raises(ConfigurationError):
            field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {u"num_tokens": 5}

        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"characters": TokenCharactersIndexer(u"characters")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {u"num_tokens": 5, u"num_token_characters": 8}

        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"characters": TokenCharactersIndexer(u"characters"),
                                          u"words": SingleIdTokenIndexer(u"words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {u"num_tokens": 5, u"num_token_characters": 8}

    def test_as_tensor_handles_words(self):
        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"].detach().cpu().numpy(),
                                                numpy.array([1, 1, 1, 2, 1]))

    def test_as_tensor_handles_longer_lengths(self):
        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths[u"num_tokens"] = 10
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"].detach().cpu().numpy(),
                                                numpy.array([1, 1, 1, 2, 1, 0, 0, 0, 0, 0]))

    def test_as_tensor_handles_characters(self):
        field = TextField([Token(t) for t in [u"This", u"is", u"a", u"sentence", u"."]],
                          token_indexers={u"characters": TokenCharactersIndexer(u"characters")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        expected_character_array = numpy.array([[1, 1, 1, 3, 0, 0, 0, 0],
                                                [1, 3, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 0],
                                                [3, 4, 5, 6, 4, 5, 7, 4],
                                                [1, 0, 0, 0, 0, 0, 0, 0]])
        numpy.testing.assert_array_almost_equal(tensor_dict[u"characters"].detach().cpu().numpy(),
                                                expected_character_array)

    def test_as_tensor_handles_words_and_characters_with_longer_lengths(self):
        field = TextField([Token(t) for t in [u"a", u"sentence", u"."]],
                          token_indexers={u"words": SingleIdTokenIndexer(u"words"),
                                          u"characters": TokenCharactersIndexer(u"characters")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths[u"num_tokens"] = 5
        padding_lengths[u"num_token_characters"] = 10
        tensor_dict = field.as_tensor(padding_lengths)

        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"].detach().cpu().numpy(),
                                                numpy.array([1, 2, 1, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"characters"].detach().cpu().numpy(),
                                                numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [3, 4, 5, 6, 4, 5, 7, 4, 0, 0],
                                                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_printing_doesnt_crash(self):
        field = TextField([Token(t) for t in [u"A", u"sentence"]],
                          {u"words": SingleIdTokenIndexer(namespace=u"words")})
        print(field)

    def test_token_embedder_returns_dict(self):
        field = TextField([Token(t) for t in [u"A", u"sentence"]],
                          token_indexers={u"field_with_dict": DictReturningTokenIndexer(),
                                          u"words": SingleIdTokenIndexer(u"words"),
                                          u"characters": TokenCharactersIndexer(u"characters")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
                u'token_ids': 5,
                u'additional_key': 2,
                u'words': 2,
                u'characters': 2,
                u'num_token_characters': 8
        }
        padding_lengths[u'token_ids'] = 7
        padding_lengths[u'additional_key'] = 3
        padding_lengths[u'words'] = 4
        padding_lengths[u'characters'] = 4
        tensors = field.as_tensor(padding_lengths)
        assert list(tensors[u'token_ids'].shape) == [7]
        assert list(tensors[u'additional_key'].shape) == [3]
        assert list(tensors[u'words'].shape) == [4]
        assert list(tensors[u'characters'].shape) == [4, 8]
