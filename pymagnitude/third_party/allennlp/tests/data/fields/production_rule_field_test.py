# pylint: disable=no-self-use,invalid-name,protected-access

from __future__ import absolute_import
from collections import defaultdict

from numpy.testing import assert_almost_equal

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.fields import ListField, ProductionRuleField


class TestProductionRuleField(AllenNlpTestCase):
    def setUp(self):
        super(TestProductionRuleField, self).setUp()
        self.vocab = Vocabulary()
        self.s_rule_index = self.vocab.add_token_to_namespace(u"S -> [NP, VP]", namespace=u'rule_labels')
        self.np_index = self.vocab.add_token_to_namespace(u"NP -> test", namespace=u'rule_labels')

    def test_field_counts_vocab_items_correctly(self):
        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)
        assert namespace_token_counts[u"rule_labels"][u"S -> [NP, VP]"] == 1

        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=False)
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)
        assert namespace_token_counts[u"rule_labels"][u"S -> [NP, VP]"] == 0

    def test_index_converts_field_correctly(self):
        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        field.index(self.vocab)
        assert field._rule_id == self.s_rule_index

    def test_padding_lengths_are_computed_correctly(self):
        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        field.index(self.vocab)
        assert field.get_padding_lengths() == {}

    def test_as_tensor_produces_correct_output(self):
        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        field.index(self.vocab)
        tensor_tuple = field.as_tensor(field.get_padding_lengths())
        assert isinstance(tensor_tuple, tuple)
        assert len(tensor_tuple) == 3
        assert tensor_tuple[0] == u'S -> [NP, VP]'
        assert tensor_tuple[1] is True
        assert_almost_equal(tensor_tuple[2].detach().cpu().numpy(), [self.s_rule_index])

        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=False)
        field.index(self.vocab)
        tensor_tuple = field.as_tensor(field.get_padding_lengths())
        assert isinstance(tensor_tuple, tuple)
        assert len(tensor_tuple) == 3
        assert tensor_tuple[0] == u'S -> [NP, VP]'
        assert tensor_tuple[1] is False
        assert tensor_tuple[2] is None

    def test_batch_tensors_does_not_modify_list(self):
        field = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict1 = field.as_tensor(padding_lengths)

        field = ProductionRuleField(u'NP -> test', is_global_rule=True)
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict2 = field.as_tensor(padding_lengths)
        tensor_list = [tensor_dict1, tensor_dict2]
        assert field.batch_tensors(tensor_list) == tensor_list

    def test_doubly_nested_field_works(self):
        field1 = ProductionRuleField(u'S -> [NP, VP]', is_global_rule=True)
        field2 = ProductionRuleField(u'NP -> test', is_global_rule=True)
        field3 = ProductionRuleField(u'VP -> eat', is_global_rule=False)
        list_field = ListField([ListField([field1, field2, field3]),
                                ListField([field1, field2])])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensors = list_field.as_tensor(padding_lengths)
        assert isinstance(tensors, list)
        assert len(tensors) == 2
        assert isinstance(tensors[0], list)
        assert len(tensors[0]) == 3
        assert isinstance(tensors[1], list)
        assert len(tensors[1]) == 3

        tensor_tuple = tensors[0][0]
        assert tensor_tuple[0] == u'S -> [NP, VP]'
        assert tensor_tuple[1] is True
        assert_almost_equal(tensor_tuple[2].detach().cpu().numpy(), [self.s_rule_index])

        tensor_tuple = tensors[0][1]
        assert tensor_tuple[0] == u'NP -> test'
        assert tensor_tuple[1] is True
        assert_almost_equal(tensor_tuple[2].detach().cpu().numpy(), [self.np_index])

        tensor_tuple = tensors[0][2]
        assert tensor_tuple[0] == u'VP -> eat'
        assert tensor_tuple[1] is False
        assert tensor_tuple[2] is None

        tensor_tuple = tensors[1][0]
        assert tensor_tuple[0] == u'S -> [NP, VP]'
        assert tensor_tuple[1] is True
        assert_almost_equal(tensor_tuple[2].detach().cpu().numpy(), [self.s_rule_index])

        tensor_tuple = tensors[1][1]
        assert tensor_tuple[0] == u'NP -> test'
        assert tensor_tuple[1] is True
        assert_almost_equal(tensor_tuple[2].detach().cpu().numpy(), [self.np_index])

        # This item was just padding.
        tensor_tuple = tensors[1][2]
        assert tensor_tuple[0] == u''
        assert tensor_tuple[1] is False
        assert tensor_tuple[2] is None
