# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import LabelField
from allennlp.data import Vocabulary


class TestLabelField(AllenNlpTestCase):
    def test_as_tensor_returns_integer_tensor(self):
        label = LabelField(5, skip_indexing=True)

        tensor = label.as_tensor(label.get_padding_lengths())
        assert tensor.item() == 5

    def test_label_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"entailment", namespace=u"labels")
        vocab.add_token_to_namespace(u"contradiction", namespace=u"labels")
        vocab.add_token_to_namespace(u"neutral", namespace=u"labels")

        label = LabelField(u"entailment")
        label.index(vocab)
        tensor = label.as_tensor(label.get_padding_lengths())
        assert tensor.item() == 0

    def test_label_field_raises_with_non_integer_labels_and_no_indexing(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField(u"non integer field", skip_indexing=True)

    def test_label_field_raises_with_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField([], skip_indexing=False)

    def test_label_field_empty_field_works(self):
        label = LabelField(u"test")
        empty_label = label.empty_field()
        assert empty_label.label == -1

    def test_class_variables_for_namespace_warnings_work_correctly(self):
        # pylint: disable=protected-access
        assert u"text" not in LabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.label_field", level=u"WARNING"):
            _ = LabelField(u"test", label_namespace=u"text")

        # We've warned once, so we should have set the class variable to False.
        assert u"text" in LabelField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger=u"allennlp.data.fields.label_field", level=u"WARNING"):
                _ = LabelField(u"test2", label_namespace=u"text")

        # ... but a new namespace should still log a warning.
        assert u"text2" not in LabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.label_field", level=u"WARNING"):
            _ = LabelField(u"test", label_namespace=u"text2")

    def test_printing_doesnt_crash(self):
        label = LabelField(u"label", label_namespace=u"namespace")
        print(label)
