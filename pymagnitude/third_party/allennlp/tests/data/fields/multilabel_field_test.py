# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function
import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import MultiLabelField
from allennlp.data.vocabulary import Vocabulary


class TestMultiLabelField(AllenNlpTestCase):
    def test_as_tensor_returns_integer_tensor(self):
        f = MultiLabelField([2, 3], skip_indexing=True, label_namespace=u"test1", num_labels=5)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 0, 1, 1, 0]))

    def test_multilabel_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"rel0", namespace=u"rel_labels")
        vocab.add_token_to_namespace(u"rel1", namespace=u"rel_labels")
        vocab.add_token_to_namespace(u"rel2", namespace=u"rel_labels")

        f = MultiLabelField([u"rel1", u"rel0"], label_namespace=u"rel_labels")
        f.index(vocab)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([1, 1, 0]))

    def test_multilabel_field_raises_with_non_integer_labels_and_no_indexing(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([u"non integer field"], skip_indexing=True)

    def test_multilabel_field_raises_with_no_indexing_and_missing_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([0, 2], skip_indexing=True, num_labels=None)

    def test_multilabel_field_raises_with_no_indexing_and_wrong_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([0, 2, 4], skip_indexing=True, num_labels=3)

    def test_multilabel_field_raises_with_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([1, 2], skip_indexing=False)

    def test_multilabel_field_raises_with_given_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([1, 2], skip_indexing=False, num_labels=4)

    def test_multilabel_field_empty_field_works(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"label1", namespace=u"test_empty_labels")
        vocab.add_token_to_namespace(u"label2", namespace=u"test_empty_labels")

        f = MultiLabelField([], label_namespace=u"test_empty_labels")
        f.index(vocab)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 0]))

    def test_class_variables_for_namespace_warnings_work_correctly(self):
        # pylint: disable=protected-access
        assert u"text" not in MultiLabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.multilabel_field", level=u"WARNING"):
            _ = MultiLabelField([u"test"], label_namespace=u"text")

        # We've warned once, so we should have set the class variable to False.
        assert u"text" in MultiLabelField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger=u"allennlp.data.fields.multilabel_field", level=u"WARNING"):
                _ = MultiLabelField([u"test2"], label_namespace=u"text")

        # ... but a new namespace should still log a warning.
        assert u"text2" not in MultiLabelField._already_warned_namespaces
        with self.assertLogs(logger=u"allennlp.data.fields.multilabel_field", level=u"WARNING"):
            _ = MultiLabelField([u"test"], label_namespace=u"text2")

    def test_printing_doesnt_crash(self):
        field = MultiLabelField([u"label"], label_namespace=u"namespace")
        print(field)
