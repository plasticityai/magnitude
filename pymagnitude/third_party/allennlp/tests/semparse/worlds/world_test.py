# pylint: disable=no-self-use,invalid-name,protected-access



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import json

#overrides

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse import ParsingError, World
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import NlvrWorld, WikiTablesWorld
from io import open


class FakeWorldWithoutRecursion(World):
    # pylint: disable=abstract-method
    #overrides
    def all_possible_actions(self):
        # The logical forms this grammar allows are
        # (unary_function argument)
        # (binary_function argument argument)
        actions = [u'@start@ -> t',
                   u't -> [<e,t>, e]',
                   u'<e,t> -> unary_function',
                   u'<e,t> -> [<e,<e,t>>, e]',
                   u'<e,<e,t>> -> binary_function',
                   u'e -> argument']
        return actions


class FakeWorldWithRecursion(FakeWorldWithoutRecursion):
    # pylint: disable=abstract-method
    #overrides
    def all_possible_actions(self):
        # In addition to the forms allowed by ``FakeWorldWithoutRecursion``, this world allows
        # (unary_function (identity .... (argument)))
        # (binary_function (identity .... (argument)) (identity .... (argument)))
        actions = super(FakeWorldWithRecursion, self).all_possible_actions()
        actions.extend([u'e -> [<e,e>, e]',
                        u'<e,e> -> identity'])
        return actions


class TestWorld(AllenNlpTestCase):
    def setUp(self):
        super(TestWorld, self).setUp()
        self.world_without_recursion = FakeWorldWithoutRecursion()
        self.world_with_recursion = FakeWorldWithRecursion()

        test_filename = self.FIXTURES_ROOT / u"data" / u"nlvr" / u"sample_ungrouped_data.jsonl"
        data = [json.loads(line)[u"structured_rep"] for line in open(test_filename).readlines()]
        self.nlvr_world = NlvrWorld(data[0])

        question_tokens = [Token(x) for x in [u'what', u'was', u'the', u'last', u'year', u'2004', u'?']]
        table_file = self.FIXTURES_ROOT / u'data' / u'wikitables' / u'sample_table.tsv'
        table_kg = TableQuestionKnowledgeGraph.read_from_file(table_file, question_tokens)
        self.wikitables_world = WikiTablesWorld(table_kg)

    def test_get_paths_to_root_without_recursion(self):
        argument_paths = self.world_without_recursion.get_paths_to_root(u'e -> argument')
        assert argument_paths == [[u'e -> argument', u't -> [<e,t>, e]', u'@start@ -> t'],
                                  [u'e -> argument', u'<e,t> -> [<e,<e,t>>, e]', u't -> [<e,t>, e]',
                                   u'@start@ -> t']]
        unary_function_paths = self.world_without_recursion.get_paths_to_root(u'<e,t> -> unary_function')
        assert unary_function_paths == [[u'<e,t> -> unary_function', u't -> [<e,t>, e]',
                                         u'@start@ -> t']]
        binary_function_paths =\
                self.world_without_recursion.get_paths_to_root(u'<e,<e,t>> -> binary_function')
        assert binary_function_paths == [[u'<e,<e,t>> -> binary_function',
                                          u'<e,t> -> [<e,<e,t>>, e]', u't -> [<e,t>, e]',
                                          u'@start@ -> t']]

    def test_get_paths_to_root_with_recursion(self):
        argument_paths = self.world_with_recursion.get_paths_to_root(u'e -> argument')
        # Argument now has 4 paths, and the two new paths are with the identity function occurring
        # (only once) within unary and binary functions.
        assert argument_paths == [[u'e -> argument', u't -> [<e,t>, e]', u'@start@ -> t'],
                                  [u'e -> argument', u'<e,t> -> [<e,<e,t>>, e]', u't -> [<e,t>, e]',
                                   u'@start@ -> t'],
                                  [u'e -> argument', u'e -> [<e,e>, e]', u't -> [<e,t>, e]',
                                   u'@start@ -> t'],
                                  [u'e -> argument', u'e -> [<e,e>, e]', u'<e,t> -> [<e,<e,t>>, e]',
                                   u't -> [<e,t>, e]', u'@start@ -> t']]
        identity_paths = self.world_with_recursion.get_paths_to_root(u'<e,e> -> identity')
        # Two identity paths, one through each of unary and binary function productions.
        assert identity_paths == [[u'<e,e> -> identity', u'e -> [<e,e>, e]', u't -> [<e,t>, e]',
                                   u'@start@ -> t'],
                                  [u'<e,e> -> identity', u'e -> [<e,e>, e]',
                                   u'<e,t> -> [<e,<e,t>>, e]', u't -> [<e,t>, e]', u'@start@ -> t']]

    # The tests for get_action_sequence and get_logical_form need a concrete world to be useful;
    # we'll mostly use the NLVR world to test them, as it's a simpler world than the WikiTables
    # world.

    def test_get_action_sequence_removes_currying(self):
        world = self.wikitables_world
        logical_form = (u"(argmax (number 1) (number 1) (fb:row.row.division fb:cell.2) "
                        u"(reverse (lambda x ((reverse fb:row.row.index) (var x))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        assert u'r -> [<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]' in action_sequence

    def test_get_action_sequence_removes_and_retains_var_correctly(self):
        world = self.wikitables_world
        logical_form = (u"((reverse fb:row.row.league) (argmin (number 1) (number 1) "
                        u"(fb:type.object.type fb:type.row) "
                        u"(reverse (lambda x ((reverse fb:row.row.index) (var x))))))")
        parsed_logical_form_without_var = world.parse_logical_form(logical_form)
        action_sequence_without_var = world.get_action_sequence(parsed_logical_form_without_var)
        assert u'<#1,#1> -> var' not in action_sequence_without_var

        parsed_logical_form_with_var = world.parse_logical_form(logical_form,
                                                                remove_var_function=False)
        action_sequence_with_var = world.get_action_sequence(parsed_logical_form_with_var)
        assert u'<#1,#1> -> var' in action_sequence_with_var

    def test_get_logical_form_handles_reverse(self):
        world = self.wikitables_world
        logical_form = u"((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

        logical_form = (u"((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                        u"(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                        u"((reverse fb:row.row.index) (var x)))))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

    def test_get_logical_form_handles_greater_than(self):
        world = self.wikitables_world
        action_sequence = [u'@start@ -> c', u'c -> [<r,c>, r]', u'<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                           u'<<#1,#2>,<#2,#1>> -> reverse', u'<c,r> -> fb:row.row.league',
                           u'r -> [<c,r>, c]', u'<c,r> -> fb:row.row.year', u'c -> [<n,c>, n]',
                           u'<n,c> -> fb:cell.cell.number', u'n -> [<nd,nd>, n]', u'<nd,nd> -> >',
                           u'n -> [<n,n>, n]', u'<n,n> -> number', u'n -> 2004']
        logical_form = world.get_logical_form(action_sequence)
        expected_logical_form = (u'((reverse fb:row.row.league) (fb:row.row.year '
                                 u'(fb:cell.cell.number (> (number 2004)))))')
        assert logical_form == expected_logical_form

    def test_get_logical_form_handles_length_one_terminal_functions(self):
        world = self.wikitables_world
        logical_form = (u"(- ((reverse fb:cell.cell.number) ((reverse fb:row.row.league) "
                        u"(fb:row.row.year fb:cell.usl_a_league))) (number 1))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        action_sequence = world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form

    def test_get_logical_form_with_real_logical_forms(self):
        nlvr_world = self.nlvr_world
        logical_form = (u"(box_count_greater_equals (member_color_count_equals all_boxes 1) 1)")
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        # It makes more sense to compare parsed logical forms instead of actual logical forms.
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)
        logical_form = u"(object_color_all_equals (circle (touch_wall (all_objects))) color_black)"
        parsed_logical_form = nlvr_world.parse_logical_form(logical_form)
        action_sequence = nlvr_world.get_action_sequence(parsed_logical_form)
        reconstructed_logical_form = nlvr_world.get_logical_form(action_sequence)
        parsed_reconstructed_logical_form = nlvr_world.parse_logical_form(reconstructed_logical_form)
        assert parsed_logical_form == parsed_reconstructed_logical_form
        assert nlvr_world.execute(logical_form) == nlvr_world.execute(reconstructed_logical_form)

    def test_get_logical_form_fails_with_incomplete_action_sequence(self):
        nlvr_world = self.nlvr_world
        action_sequence = [u'@start@ -> t', u't -> [<b,t>, b]', u'<b,t> -> box_exists']
        with self.assertRaisesRegex(ParsingError, u'Incomplete action sequence'):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_fails_with_extra_actions(self):
        nlvr_world = self.nlvr_world
        action_sequence = [u'@start@ -> <b,t>', u'<b,t> -> box_exists', u't -> [<b,t>, b]']
        with self.assertRaisesRegex(ParsingError, u'Extra actions'):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_fails_with_action_sequence_in_wrong_order(self):
        nlvr_world = self.nlvr_world
        action_sequence = [u'@start@ -> t', u't -> [<b,t>, b]', u'<b,t> -> box_exists',
                           u'b -> [<c,b>, c]', u'<c,b> -> [<b,<c,b>>, b]',
                           u'b -> all_boxes', u'<b,<c,b>> -> member_color_none_equals',
                           u'c -> color_blue']
        with self.assertRaisesRegex(ParsingError, u'does not match'):
            nlvr_world.get_logical_form(action_sequence)

    def test_get_logical_form_adds_var_correctly(self):
        world = self.wikitables_world
        action_sequence = [u'@start@ -> e', u'e -> [<r,e>, r]', u'<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                           u'<<#1,#2>,<#2,#1>> -> reverse', u'<e,r> -> fb:row.row.league',
                           u'r -> [<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]',
                           u'<d,<d,<#1,<<d,#1>,#1>>>> -> argmin', u'd -> [<e,d>, e]', u'<e,d> -> number',
                           u'e -> 1', u'd -> [<e,d>, e]', u'<e,d> -> number', u'e -> 1',
                           u'r -> [<#1,#1>, r]', u'<#1,#1> -> fb:type.object.type', u'r -> fb:type.row',
                           u'<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                           u"<r,d> -> ['lambda x', d]", u'd -> [<r,d>, r]',
                           u'<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                           u'<d,r> -> fb:row.row.index', u'r -> x']
        logical_form = world.get_logical_form(action_sequence)
        assert u'(var x)' in logical_form
        expected_logical_form = (u"((reverse fb:row.row.league) (argmin (number 1) (number 1) "
                                 u"(fb:type.object.type fb:type.row) "
                                 u"(reverse (lambda x ((reverse fb:row.row.index) (var x))))))")
        parsed_logical_form = world.parse_logical_form(logical_form)
        parsed_expected_logical_form = world.parse_logical_form(expected_logical_form)
        assert parsed_logical_form == parsed_expected_logical_form

    def test_get_logical_form_fails_with_unnecessary_add_var(self):
        world = self.wikitables_world
        action_sequence = [u'@start@ -> e', u'e -> [<r,e>, r]', u'<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                           u'<<#1,#2>,<#2,#1>> -> reverse', u'<e,r> -> fb:row.row.league',
                           u'r -> [<d,<d,<#1,<<d,#1>,#1>>>>, d, d, r, <d,r>]',
                           u'<d,<d,<#1,<<d,#1>,#1>>>> -> argmin', u'd -> [<e,d>, e]', u'<e,d> -> number',
                           u'e -> 1', u'd -> [<e,d>, e]', u'<e,d> -> number', u'e -> 1',
                           u'r -> [<#1,#1>, r]', u'<#1,#1> -> fb:type.object.type', u'r -> fb:type.row',
                           u'<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                           u"<r,d> -> ['lambda x', d]", u'd -> [<r,d>, r]',
                           u'<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                           u'<d,r> -> fb:row.row.index', u'r -> [<#1,#1>, r]', u'<#1,#1> -> var', u'r -> x']
        with self.assertRaisesRegex(ParsingError, u'already had var'):
            world.get_logical_form(action_sequence)

    def test_get_logical_form_with_multiple_negate_filters(self):
        world = self.nlvr_world
        # This is an actual sequence of actions produced by an untrained NlvrSemanticParser
        action_sequence = [u'@start@ -> t', u't -> [<o,<c,t>>, o, c]',
                           u'<o,<c,t>> -> object_color_all_equals', u'o -> [<o,o>, o]',
                           u'<o,o> -> [<<o,o>,<o,o>>, <o,o>]', u'<<o,o>,<o,o>> -> negate_filter',
                           u'<o,o> -> [<<o,o>,<o,o>>, <o,o>]', u'<<o,o>,<o,o>> -> negate_filter',
                           u'<o,o> -> blue', u'o -> [<o,o>, o]', u'<o,o> -> blue', u'o -> [<o,o>, o]',
                           u'<o,o> -> blue', u'o -> all_objects', u'c -> color_blue']
        logical_form = world.get_logical_form(action_sequence)
        # "The color of all blue blue objects that are not not blue is blue".
        expected_logical_form = (u"(object_color_all_equals ((negate_filter (negate_filter blue)) "
                                 u"(blue (blue all_objects))) color_blue)")
        parsed_logical_form = world.parse_logical_form(logical_form)
        parsed_expected_logical_form = world.parse_logical_form(expected_logical_form)
        assert parsed_logical_form == parsed_expected_logical_form
