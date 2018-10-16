# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import GrammarState

def is_nonterminal(symbol     )        :
    if symbol == u'identity':
        return False
    if u'lambda ' in symbol:
        return False
    if symbol in set([u'x', u'y', u'z']):
        return False
    return True


class TestGrammarState(AllenNlpTestCase):
    def test_is_finished_just_uses_nonterminal_stack(self):
        state = GrammarState([u's'], {}, {}, {}, is_nonterminal)
        assert not state.is_finished()
        state = GrammarState([], {}, {}, {}, is_nonterminal)
        assert state.is_finished()

    def test_get_valid_actions_uses_top_of_stack(self):
        state = GrammarState([u's'], {}, {u's': [1, 2], u't': [3, 4]}, {}, is_nonterminal)
        assert state.get_valid_actions() == [1, 2]
        state = GrammarState([u't'], {}, {u's': [1, 2], u't': [3, 4]}, {}, is_nonterminal)
        assert state.get_valid_actions() == [3, 4]
        state = GrammarState([u'e'], {}, {u's': [1, 2], u't': [3, 4], u'e': []}, {}, is_nonterminal)
        assert state.get_valid_actions() == []

    def test_get_valid_actions_adds_lambda_productions(self):
        state = GrammarState([u's'], {(u's', u'x'): [u's']}, {u's': [1, 2]}, {u's -> x': 5}, is_nonterminal)
        assert state.get_valid_actions() == [1, 2, 5]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        assert state.get_valid_actions() == [1, 2, 5]

    def test_get_valid_actions_adds_lambda_productions_only_for_correct_type(self):
        state = GrammarState([u't'],
                             {(u's', u'x'): [u't']},
                             {u's': [1, 2], u't': [3, 4]},
                             {u's -> x': 5},
                             is_nonterminal)
        assert state.get_valid_actions() == [3, 4]
        # We're doing this assert twice to make sure we haven't accidentally modified the state.
        assert state.get_valid_actions() == [3, 4]

    def test_take_action_gives_correct_next_states_with_non_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        action_indices = object()

        state = GrammarState([u's'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action(u's -> [t, r]')
        expected_next_state = GrammarState([u'r', u't'], {}, valid_actions, action_indices, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = GrammarState([u'r', u't'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action(u't -> identity')
        expected_next_state = GrammarState([u'r'], {}, valid_actions, action_indices, is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

    def test_take_action_crashes_with_mismatched_types(self):
        with pytest.raises(AssertionError):
            state = GrammarState([u's'], {}, {}, {}, is_nonterminal)
            state.take_action(u't -> identity')

    def test_take_action_gives_correct_next_states_with_lambda_productions(self):
        # state.take_action() doesn't read or change these objects, it just passes them through, so
        # we'll use some sentinels to be sure of that.
        valid_actions = object()
        action_indices = object()

        state = GrammarState([u't', u'<s,d>'], {}, valid_actions, action_indices, is_nonterminal)
        next_state = state.take_action(u'<s,d> -> [lambda x, d]')
        expected_next_state = GrammarState([u't', u'd'],
                                           {(u's', u'x'): [u'd']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action(u'd -> [<s,r>, d]')
        expected_next_state = GrammarState([u't', u'd', u'<s,r>'],
                                           {(u's', u'x'): [u'd', u'<s,r>']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action(u'<s,r> -> [lambda y, r]')
        expected_next_state = GrammarState([u't', u'd', u'r'],
                                           {(u's', u'x'): [u'd', u'r'], (u's', u'y'): [u'r']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action(u'r -> identity')
        expected_next_state = GrammarState([u't', u'd'],
                                           {(u's', u'x'): [u'd']},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__

        state = expected_next_state
        next_state = state.take_action(u'd -> x')
        expected_next_state = GrammarState([u't'],
                                           {},
                                           valid_actions,
                                           action_indices,
                                           is_nonterminal)
        assert next_state.__dict__ == expected_next_state.__dict__
