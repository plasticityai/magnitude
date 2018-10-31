# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from __future__ import print_function
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, LabelField, ListField, IndexField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer


class TestListField(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace(u"this", u"words")
        self.vocab.add_token_to_namespace(u"is", u"words")
        self.vocab.add_token_to_namespace(u"a", u"words")
        self.vocab.add_token_to_namespace(u"sentence", u'words')
        self.vocab.add_token_to_namespace(u"s", u'characters')
        self.vocab.add_token_to_namespace(u"e", u'characters')
        self.vocab.add_token_to_namespace(u"n", u'characters')
        self.vocab.add_token_to_namespace(u"t", u'characters')
        self.vocab.add_token_to_namespace(u"c", u'characters')
        for label in [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k']:
            self.vocab.add_token_to_namespace(label, u'labels')

        self.word_indexer = {u"words": SingleIdTokenIndexer(u"words")}
        self.words_and_characters_indexers = {u"words": SingleIdTokenIndexer(u"words"),
                                              u"characters": TokenCharactersIndexer(u"characters")}
        self.field1 = TextField([Token(t) for t in [u"this", u"is", u"a", u"sentence"]],
                                self.word_indexer)
        self.field2 = TextField([Token(t) for t in [u"this", u"is", u"a", u"different", u"sentence"]],
                                self.word_indexer)
        self.field3 = TextField([Token(t) for t in [u"this", u"is", u"another", u"sentence"]],
                                self.word_indexer)

        self.empty_text_field = self.field1.empty_field()
        self.index_field = IndexField(1, self.field1)
        self.empty_index_field = self.index_field.empty_field()
        self.sequence_label_field = SequenceLabelField([1, 1, 0, 1], self.field1)
        self.empty_sequence_label_field = self.sequence_label_field.empty_field()

        super(TestListField, self).setUp()

    def test_get_padding_lengths(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        lengths = list_field.get_padding_lengths()
        assert lengths == {u"num_fields": 3, u"list_num_tokens": 5}

    def test_list_field_can_handle_empty_text_fields(self):
        list_field = ListField([self.field1, self.field2, self.empty_text_field])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor_dict[u"words"].detach().cpu().numpy(),
                                         numpy.array([[2, 3, 4, 5, 0],
                                                      [2, 3, 4, 1, 5],
                                                      [0, 0, 0, 0, 0]]))

    def test_list_field_can_handle_empty_index_fields(self):
        list_field = ListField([self.index_field, self.index_field, self.empty_index_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.detach().cpu().numpy(), numpy.array([[1], [1], [-1]]))

    def test_list_field_can_handle_empty_sequence_label_fields(self):
        list_field = ListField([self.sequence_label_field,
                                self.sequence_label_field,
                                self.empty_sequence_label_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.detach().cpu().numpy(),
                                         numpy.array([[1, 1, 0, 1],
                                                      [1, 1, 0, 1],
                                                      [0, 0, 0, 0]]))

    def test_all_fields_padded_to_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][0].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][1].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][2].detach().cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0]))

    def test_nested_list_fields_are_padded_correctly(self):
        nested_field1 = ListField([LabelField(c) for c in [u'a', u'b', u'c', u'd', u'e']])
        nested_field2 = ListField([LabelField(c) for c in [u'f', u'g', u'h', u'i', u'j', u'k']])
        list_field = ListField([nested_field1.empty_field(), nested_field1, nested_field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        assert padding_lengths == {u'num_fields': 3, u'list_num_fields': 6}
        tensor = list_field.as_tensor(padding_lengths).detach().cpu().numpy()
        numpy.testing.assert_almost_equal(tensor, [[-1, -1, -1, -1, -1, -1],
                                                   [0, 1, 2, 3, 4, -1],
                                                   [5, 6, 7, 8, 9, 10]])

    def test_fields_can_pad_to_greater_than_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        padding_lengths[u"list_num_tokens"] = 7
        padding_lengths[u"num_fields"] = 5
        tensor_dict = list_field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][0].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][1].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][2].detach().cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][3].detach().cpu().numpy(),
                                                numpy.array([0, 0, 0, 0, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict[u"words"][4].detach().cpu().numpy(),
                                                numpy.array([0, 0, 0, 0, 0, 0, 0]))

    def test_as_tensor_can_handle_multiple_token_indexers(self):
        # pylint: disable=protected-access
        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict[u"words"].detach().cpu().numpy()
        characters = tensor_dict[u"characters"].detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(words, numpy.array([[2, 3, 4, 5, 0],
                                                                    [2, 3, 4, 1, 5],
                                                                    [2, 3, 1, 5, 0]]))

        numpy.testing.assert_array_almost_equal(characters[0], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        numpy.testing.assert_array_almost_equal(characters[1], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 1, 1, 1, 3, 1, 3, 4, 5],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0]]))

        numpy.testing.assert_array_almost_equal(characters[2], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 4, 1, 5, 1, 3, 1, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_as_tensor_can_handle_multiple_token_indexers_and_empty_fields(self):
        # pylint: disable=protected-access
        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1.empty_field(), self.field1, self.field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict[u"words"].detach().cpu().numpy()
        characters = tensor_dict[u"characters"].detach().cpu().numpy()

        numpy.testing.assert_array_almost_equal(words, numpy.array([[0, 0, 0, 0, 0],
                                                                    [2, 3, 4, 5, 0],
                                                                    [2, 3, 4, 1, 5]]))

        numpy.testing.assert_array_almost_equal(characters[0], numpy.zeros([5, 9]))

        numpy.testing.assert_array_almost_equal(characters[1], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        numpy.testing.assert_array_almost_equal(characters[2], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 1, 1, 1, 3, 1, 3, 4, 5],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0]]))

    def test_printing_doesnt_crash(self):
        list_field = ListField([self.field1, self.field2])
        print(list_field)
