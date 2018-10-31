# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function
from collections import defaultdict

import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSequenceLabelField(AllenNlpTestCase):
    def setUp(self):
        super(TestSequenceLabelField, self).setUp()
        self.text = TextField([Token(t) for t in [u"here", u"are", u"some", u"words", u"."]],
                              {u"words": SingleIdTokenIndexer(u"words")})

    def test_tag_length_mismatch_raises(self):
        with pytest.raises(ConfigurationError):
            wrong_tags = [u"B", u"O", u"O"]
            _ = SequenceLabelField(wrong_tags, self.text)

    def test_count_vocab_items_correctly_indexes_tags(self):
        tags = [u"B", u"I", u"O", u"O", u"O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace=u"labels")

        counter = defaultdict(lambda: defaultdict(int))
        sequence_label_field.count_vocab_items(counter)

        assert counter[u"labels"][u"B"] == 1
        assert counter[u"labels"][u"I"] == 1
        assert counter[u"labels"][u"O"] == 3
        assert set(counter.keys()) == set([u"labels"])

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        b_index = vocab.add_token_to_namespace(u"B", namespace=u'*labels')
        i_index = vocab.add_token_to_namespace(u"I", namespace=u'*labels')
        o_index = vocab.add_token_to_namespace(u"O", namespace=u'*labels')

        tags = [u"B", u"I", u"O", u"O", u"O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace=u"*labels")
        sequence_label_field.index(vocab)

        # pylint: disable=protected-access
        assert sequence_label_field._indexed_labels == [b_index, i_index, o_index, o_index, o_index]
        # pylint: enable=protected-access

    def test_as_tensor_produces_integer_targets(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"B", namespace=u'*labels')
        vocab.add_token_to_namespace(u"I", namespace=u'*labels')
        vocab.add_token_to_namespace(u"O", namespace=u'*labels')

        tags = [u"B", u"I", u"O", u"O", u"O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace=u"*labels")
        sequence_label_field.index(vocab)
        padding_lengths = sequence_label_field.get_padding_lengths()
        tensor = sequence_label_field.as_tensor(padding_lengths).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 1, 2, 2, 2]))

    def test_sequence_label_field_raises_on_incorrect_type(self):

        with pytest.raises(ConfigurationError):
            _ = SequenceLabelField([[], [], [], [], []], self.text)

    def test_class_variables_for_namespace_warnings_work_correctly(self):
        # pylint: disable=protected-access
        tags = [u"B", u"I", u"O", u"O", u"O"]
        assert u"text" not in SequenceLabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.sequence_label_field", level=u"WARNING"):
            _ = SequenceLabelField(tags, self.text, label_namespace=u"text")

        # We've warned once, so we should have set the class variable to False.
        assert u"text" in SequenceLabelField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger=u"allennlp.data.fields.sequence_label_field", level=u"WARNING"):
                _ = SequenceLabelField(tags, self.text, label_namespace=u"text")

        # ... but a new namespace should still log a warning.
        assert u"text2" not in SequenceLabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.sequence_label_field", level=u"WARNING"):
            _ = SequenceLabelField(tags, self.text, label_namespace=u"text2")

    def test_printing_doesnt_crash(self):
        tags = [u"B", u"I", u"O", u"O", u"O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace=u"labels")
        print(sequence_label_field)
