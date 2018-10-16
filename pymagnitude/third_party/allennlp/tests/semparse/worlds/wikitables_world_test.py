# pylint: disable=no-self-use,invalid-name



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#typing

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import WikiTablesWorld


def check_productions_match(actual_rules           , expected_right_sides           ):
    actual_right_sides = [rule.split(u' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)


class TestWikiTablesWorld(AllenNlpTestCase):
    def setUp(self):
        super(TestWikiTablesWorld, self).setUp()
        question_tokens = [Token(x) for x in [u'what', u'was', u'the', u'last', u'year', u'2000', u'?']]
        self.table_file = self.FIXTURES_ROOT / u'data' / u'wikitables' / u'sample_table.tsv'
        self.table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, question_tokens)
        self.world = WikiTablesWorld(self.table_kg)

    def test_get_valid_actions_returns_correct_set(self):
        # This test is long, but worth it.  These are all of the valid actions in the grammar, and
        # we want to be sure they are what we expect.

        # This test checks that our valid actions for each type match  PNP's, except for the
        # terminal productions for type 'p'.
        valid_actions = self.world.get_valid_actions()
        assert set(valid_actions.keys()) == set([
                u'<#1,#1>',
                u'<#1,<#1,#1>>',
                u'<#1,n>',
                u'<<#1,#2>,<#2,#1>>',
                u'<c,d>',
                u'<c,n>',
                u'<c,p>',
                u'<c,r>',
                u'<d,c>',
                u'<d,d>',
                u'<d,n>',
                u'<d,r>',
                u'<n,<n,<#1,<<#2,#1>,#1>>>>',
                u'<n,<n,<n,d>>>',
                u'<n,<n,n>>',
                u'<n,c>',
                u'<n,d>',
                u'<n,n>',
                u'<n,p>',
                u'<n,r>',
                u'<nd,nd>',
                u'<p,c>',
                u'<p,n>',
                u'<r,c>',
                u'<r,d>',
                u'<r,n>',
                u'<r,p>',
                u'<r,r>',
                u'@start@',
                u'c',
                u'd',
                u'n',
                u'p',
                u'r',])

        check_productions_match(valid_actions[u'<#1,#1>'],
                                [u'!='])

        check_productions_match(valid_actions[u'<#1,<#1,#1>>'],
                                [u'and', u'or'])

        check_productions_match(valid_actions[u'<#1,n>'],
                                [u'count'])

        check_productions_match(valid_actions[u'<<#1,#2>,<#2,#1>>'],
                                [u'reverse'])

        check_productions_match(valid_actions[u'<c,d>'],
                                [u"['lambda x', d]", u'[<<#1,#2>,<#2,#1>>, <d,c>]'])

        check_productions_match(valid_actions[u'<c,n>'],
                                [u"['lambda x', n]", u'[<<#1,#2>,<#2,#1>>, <n,c>]'])

        check_productions_match(valid_actions[u'<c,p>'],
                                [u'[<<#1,#2>,<#2,#1>>, <p,c>]'])

        # Most of these are instance-specific production rules.  These are the columns in the
        # table.  Remember that SEMPRE did things backwards: fb:row.row.division takes a cell ID
        # and returns the row that has that cell in its row.division column.  This is why we have
        # to reverse all of these functions to go from a row to the cell in a particular column.
        check_productions_match(valid_actions[u'<c,r>'],
                                [u'fb:row.row.null',  # This one is global, representing an empty set.
                                 u'fb:row.row.year',
                                 u'fb:row.row.league',
                                 u'fb:row.row.avg_attendance',
                                 u'fb:row.row.division',
                                 u'fb:row.row.regular_season',
                                 u'fb:row.row.playoffs',
                                 u'fb:row.row.open_cup'])

        # These might look backwards, but that's because SEMPRE chose to make them backwards.
        # fb:a.b is a function that takes b and returns a.  So fb:cell.cell.date takes cell.date
        # and returns cell and fb:row.row.index takes row.index and returns row.
        check_productions_match(valid_actions[u'<d,c>'],
                                [u'fb:cell.cell.date',
                                 u'[<<#1,#2>,<#2,#1>>, <c,d>]'])

        check_productions_match(valid_actions[u'<d,d>'],
                                [u"['lambda x', d]", u'[<<#1,#2>,<#2,#1>>, <d,d>]'])

        check_productions_match(valid_actions[u'<d,n>'],
                                [u"['lambda x', n]", u'[<<#1,#2>,<#2,#1>>, <n,d>]'])

        check_productions_match(valid_actions[u'<d,r>'],
                                [u'[<<#1,#2>,<#2,#1>>, <r,d>]'])

        check_productions_match(valid_actions[u'<n,<n,<#1,<<#2,#1>,#1>>>>'],
                                [u'argmax', u'argmin'])

        # "date" is a function that takes three numbers: (date 2018 01 06).
        check_productions_match(valid_actions[u'<n,<n,<n,d>>>'],
                                [u'date'])

        check_productions_match(valid_actions[u'<n,<n,n>>'],
                                [u'-'])

        check_productions_match(valid_actions[u'<n,c>'],
                                [u'fb:cell.cell.num2', u'fb:cell.cell.number',
                                 u'[<<#1,#2>,<#2,#1>>, <c,n>]'])

        check_productions_match(valid_actions[u'<n,d>'],
                                [u"['lambda x', d]", u'[<<#1,#2>,<#2,#1>>, <d,n>]'])

        check_productions_match(valid_actions[u'<n,n>'],
                                [u'avg', u'sum', u'number',
                                 u"['lambda x', n]", u'[<<#1,#2>,<#2,#1>>, <n,n>]'])

        check_productions_match(valid_actions[u'<n,p>'],
                                [u'[<<#1,#2>,<#2,#1>>, <p,n>]'])

        check_productions_match(valid_actions[u'<n,r>'],
                                [u'fb:row.row.index', u'[<<#1,#2>,<#2,#1>>, <r,n>]'])

        check_productions_match(valid_actions[u'<nd,nd>'],
                                [u'<', u'<=', u'>', u'>=', u'min', u'max'])

        # PART_TYPE rules.  A cell part is for when a cell has text that can be split into multiple
        # parts.
        check_productions_match(valid_actions[u'<p,c>'],
                                [u'fb:cell.cell.part'])

        check_productions_match(valid_actions[u'<p,n>'],
                                [u"['lambda x', n]"])

        check_productions_match(valid_actions[u'<r,c>'],
                                [u'[<<#1,#2>,<#2,#1>>, <c,r>]'])

        check_productions_match(valid_actions[u'<r,d>'],
                                [u"['lambda x', d]"])

        check_productions_match(valid_actions[u'<r,n>'],
                                [u"['lambda x', n]", u'[<<#1,#2>,<#2,#1>>, <n,r>]'])

        check_productions_match(valid_actions[u'<r,p>'],
                                [u"['lambda x', p]", u'[<<#1,#2>,<#2,#1>>, <p,r>]'])

        check_productions_match(valid_actions[u'<r,r>'],
                                [u'fb:row.row.next', u'fb:type.object.type', u'[<<#1,#2>,<#2,#1>>, <r,r>]'])

        check_productions_match(valid_actions[u'@start@'],
                                [u'd', u'c', u'p', u'r', u'n'])

        check_productions_match(valid_actions[u'c'],
                                [u'[<#1,#1>, c]',
                                 u'[<#1,<#1,#1>>, c, c]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, c, <n,c>]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, c, <d,c>]',
                                 u'[<d,c>, d]',
                                 u'[<n,c>, n]',
                                 u'[<p,c>, p]',
                                 u'[<r,c>, r]',
                                 u'fb:cell.null',
                                 u'fb:cell.2',
                                 u'fb:cell.2001',
                                 u'fb:cell.2005',
                                 u'fb:cell.4th_round',
                                 u'fb:cell.4th_western',
                                 u'fb:cell.5th',
                                 u'fb:cell.6_028',
                                 u'fb:cell.7_169',
                                 u'fb:cell.did_not_qualify',
                                 u'fb:cell.quarterfinals',
                                 u'fb:cell.usl_a_league',
                                 u'fb:cell.usl_first_division'])

        check_productions_match(valid_actions[u'd'],
                                [u'[<n,<n,<n,d>>>, n, n, n]',
                                 u'[<#1,#1>, d]',
                                 u'[<#1,<#1,#1>>, d, d]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, d, <d,d>]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, d, <n,d>]',
                                 u'[<c,d>, c]',
                                 u'[<nd,nd>, d]'])

        check_productions_match(valid_actions[u'n'],
                                [u'-1',
                                 u'0',
                                 u'1',
                                 u'2000',
                                 u'[<#1,#1>, n]',
                                 u'[<#1,<#1,#1>>, n, n]',
                                 u'[<#1,n>, c]',
                                 u'[<#1,n>, d]',
                                 u'[<#1,n>, n]',
                                 u'[<#1,n>, p]',
                                 u'[<#1,n>, r]',
                                 u'[<c,n>, c]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, n, <d,n>]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, n, <n,n>]',
                                 u'[<n,<n,n>>, n, n]',
                                 u'[<n,n>, n]',
                                 u'[<nd,nd>, n]',
                                 u'[<r,n>, r]'])

        check_productions_match(valid_actions[u'p'],
                                [u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, p, <n,p>]',
                                 u'[<#1,#1>, p]',
                                 u'[<c,p>, c]',
                                 u'[<#1,<#1,#1>>, p, p]',
                                 u'fb:part.4th',
                                 u'fb:part.5th',
                                 u'fb:part.western'])

        check_productions_match(valid_actions[u'r'],
                                [u'fb:type.row',
                                 u'[<#1,#1>, r]',
                                 u'[<#1,<#1,#1>>, r, r]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <d,r>]',
                                 u'[<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]',
                                 u'[<n,r>, n]',
                                 u'[<c,r>, c]',
                                 u'[<r,r>, r]'])

    def test_world_processes_sempre_forms_correctly(self):
        sempre_form = u"((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        # We add columns to the name mapping in sorted order, so "league" and "year" end up as C2
        # and C6.
        assert unicode(expression) == u"R(C6,C2(cell:usl_a_league))"

    def test_world_parses_logical_forms_with_dates(self):
        sempre_form = u"((reverse fb:row.row.league) (fb:row.row.year (fb:cell.cell.date (date 2000 -1 -1))))"
        expression = self.world.parse_logical_form(sempre_form)
        assert unicode(expression) == u"R(C2,C6(D1(D0(num:2000,num:~1,num:~1))))"

    def test_world_parses_logical_forms_with_decimals(self):
        question_tokens = [Token(x) for x in [u'0.2']]
        table_kg = TableQuestionKnowledgeGraph.read_from_file(
                self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_table.tsv", question_tokens)
        world = WikiTablesWorld(table_kg)
        sempre_form = u"(fb:cell.cell.number (number 0.200))"
        expression = world.parse_logical_form(sempre_form)
        assert unicode(expression) == u"I1(I(num:0_200))"

    def test_get_action_sequence_removes_currying_for_all_wikitables_functions(self):
        # minus
        logical_form = u"(- (number 0) (number 1))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert u'n -> [<n,<n,n>>, n, n]' in action_sequence

        # date
        logical_form = u"(count (fb:cell.cell.date (date 2000 -1 -1)))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert u'd -> [<n,<n,<n,d>>>, n, n, n]' in action_sequence

        # argmax
        logical_form = (u"(argmax (number 1) (number 1) (fb:row.row.division fb:cell.2) "
                        u"(reverse (lambda x ((reverse fb:row.row.index) (var x))))")
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert u'r -> [<n,<n,<#1,<<#2,#1>,#1>>>>, n, n, r, <n,r>]' in action_sequence

        # and
        logical_form = u"(and (number 1) (number 1))"
        parsed_logical_form = self.world.parse_logical_form(logical_form)
        action_sequence = self.world.get_action_sequence(parsed_logical_form)
        assert u'n -> [<#1,<#1,#1>>, n, n]' in action_sequence

    def test_parsing_logical_forms_fails_with_unmapped_names(self):
        with pytest.raises(ParsingError):
            _ = self.world.parse_logical_form(u"(number 20)")

    def test_world_has_only_basic_numbers(self):
        valid_actions = self.world.get_valid_actions()
        assert u'n -> -1' in valid_actions[u'n']
        assert u'n -> 0' in valid_actions[u'n']
        assert u'n -> 1' in valid_actions[u'n']
        assert u'n -> 17' not in valid_actions[u'n']
        assert u'n -> 231' not in valid_actions[u'n']
        assert u'n -> 2007' not in valid_actions[u'n']
        assert u'n -> 2107' not in valid_actions[u'n']
        assert u'n -> 1800' not in valid_actions[u'n']

    def test_world_adds_numbers_from_question(self):
        question_tokens = [Token(x) for x in [u'what', u'2007', u'2,107', u'0.2', u'1800s', u'1950s', u'?']]
        table_kg = TableQuestionKnowledgeGraph.read_from_file(
                self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_table.tsv", question_tokens)
        world = WikiTablesWorld(table_kg)
        valid_actions = world.get_valid_actions()
        assert u'n -> 2007' in valid_actions[u'n']
        assert u'n -> 2107' in valid_actions[u'n']

        # It appears that sempre normalizes floating point numbers.
        assert u'n -> 0.200' in valid_actions[u'n']

        # We want to add the end-points to things like "1800s": 1800 and 1900.
        assert u'n -> 1800' in valid_actions[u'n']
        assert u'n -> 1900' in valid_actions[u'n']
        assert u'n -> 1950' in valid_actions[u'n']
        assert u'n -> 1960' in valid_actions[u'n']

    def test_world_returns_correct_actions_with_reverse(self):
        sempre_form = u"((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = [u'@start@ -> c', u'c -> [<r,c>, r]', u'<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                                  u'<<#1,#2>,<#2,#1>> -> reverse', u'<c,r> -> fb:row.row.year',
                                  u'r -> [<c,r>, c]', u'<c,r> -> fb:row.row.league', u'c -> fb:cell.usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_two_reverses(self):
        sempre_form = (u"(max ((reverse fb:cell.cell.date) ((reverse fb:row.row.year) "
                       u"(fb:row.row.league fb:cell.usl_a_league))))")
        expression = self.world.parse_logical_form(sempre_form)
        actions = self.world.get_action_sequence(expression)
        target_action_sequence = [u'@start@ -> d', u'd -> [<nd,nd>, d]', u'<nd,nd> -> max', u'd -> [<c,d>, c]',
                                  u'<c,d> -> [<<#1,#2>,<#2,#1>>, <d,c>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                                  u'<d,c> -> fb:cell.cell.date', u'c -> [<r,c>, r]',
                                  u'<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]', u'<<#1,#2>,<#2,#1>> -> reverse',
                                  u'<c,r> -> fb:row.row.year', u'r -> [<c,r>, c]',
                                  u'<c,r> -> fb:row.row.league', u'c -> fb:cell.usl_a_league']
        assert actions == target_action_sequence

    def test_world_returns_correct_actions_with_lambda_with_var(self):
        sempre_form = (u"((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                       u"(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                       u"((reverse fb:row.row.index) (var x)))))))")
        expression = self.world.parse_logical_form(sempre_form, remove_var_function=False)
        actions_with_var = self.world.get_action_sequence(expression)
        assert u'<#1,#1> -> var' in actions_with_var
        assert u'r -> x' in actions_with_var

    def test_world_returns_correct_actions_with_lambda_without_var(self):
        sempre_form = (u"((reverse fb:cell.cell.date) ((reverse fb:row.row.year) (argmax (number 1) "
                       u"(number 1) (fb:row.row.league fb:cell.usl_a_league) (reverse (lambda x "
                       u"((reverse fb:row.row.index) (var x)))))))")
        expression = self.world.parse_logical_form(sempre_form)
        actions_without_var = self.world.get_action_sequence(expression)
        assert u'<#1,#1> -> var' not in actions_without_var
        assert u'r -> x' in actions_without_var

    @pytest.mark.skip(reason=u"fibonacci recursion currently going on here")
    def test_with_deeply_nested_logical_form(self):
        question_tokens = [Token(x) for x in [u'what', u'was', u'the', u'district', u'?']]
        table_filename = self.FIXTURES_ROOT / u'data' / u'wikitables' / u'table' / u'109.tsv'
        table_kg = TableQuestionKnowledgeGraph.read_from_file(table_filename, question_tokens)
        world = WikiTablesWorld(table_kg)
        logical_form = (u"(count ((reverse fb:cell.cell.number) (or (or (or (or (or (or (or (or "
                        u"(or (or (or (or (or (or (or (or (or (or (or (or (or fb:cell.virginia_1 "
                        u"fb:cell.virginia_10) fb:cell.virginia_11) fb:cell.virginia_12) "
                        u"fb:cell.virginia_13) fb:cell.virginia_14) fb:cell.virginia_15) "
                        u"fb:cell.virginia_16) fb:cell.virginia_17) fb:cell.virginia_18) "
                        u"fb:cell.virginia_19) fb:cell.virginia_2) fb:cell.virginia_20) "
                        u"fb:cell.virginia_21) fb:cell.virginia_22) fb:cell.virginia_3) "
                        u"fb:cell.virginia_4) fb:cell.virginia_5) fb:cell.virginia_6) "
                        u"fb:cell.virginia_7) fb:cell.virginia_8) fb:cell.virginia_9)))")
        print(u"Parsing...")
        world.parse_logical_form(logical_form)

    def _get_world_with_question_tokens(self, tokens             )                   :
        table_kg = TableQuestionKnowledgeGraph.read_from_file(self.table_file, tokens)
        world = WikiTablesWorld(table_kg)
        return world

    def test_get_agenda(self):
        tokens = [Token(x) for x in [u'what', u'was', u'the', u'last', u'year', u'2000', u'?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == set([u'n -> 2000',
                                           u'<c,r> -> fb:row.row.year',
                                           u'<n,<n,<#1,<<#2,#1>,#1>>>> -> argmax'])
        tokens = [Token(x) for x in [u'what', u'was', u'the', u'difference', u'in', u'attendance',
                                     u'between', u'years', u'2001', u'and', u'2005', u'?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains cells here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == set([u'c -> fb:cell.2001',
                                           u'c -> fb:cell.2005',
                                           u'<c,r> -> fb:row.row.year',
                                           u'<n,<n,n>> -> -'])
        tokens = [Token(x) for x in [u'what', u'was', u'the', u'total', u'avg.', u'attendance', u'in',
                                     u'years', u'2001', u'and', u'2005', u'?']]
        world = self._get_world_with_question_tokens(tokens)
        # The agenda contains cells here instead of numbers because 2001 and 2005 actually link to
        # entities in the table whereas 2000 (in the previous case) does not.
        assert set(world.get_agenda()) == set([u'c -> fb:cell.2001',
                                           u'c -> fb:cell.2005',
                                           u'<c,r> -> fb:row.row.year',
                                           u'<c,r> -> fb:row.row.avg_attendance',
                                           u'<n,n> -> sum'])
        tokens = [Token(x) for x in [u'when', u'was', u'the', u'least', u'avg.', u'attendance', u'?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == set([u'<c,r> -> fb:row.row.avg_attendance',
                                           u'<n,<n,<#1,<<#2,#1>,#1>>>> -> argmin'])
        tokens = [Token(x) for x in [u'what', u'is', u'the', u'least', u'avg.', u'attendance', u'?']]
        world = self._get_world_with_question_tokens(tokens)
        assert set(world.get_agenda()) == set([u'<c,r> -> fb:row.row.avg_attendance',
                                           u'<nd,nd> -> min'])
