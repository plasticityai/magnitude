u"""
A ``SqlTableContext`` represents the SQL context in which an utterance appears, with the grammar and
the valid actions.
"""

from __future__ import absolute_import
import re
from collections import defaultdict
#typing

#overrides

from parsimonious.expressions import Sequence, OneOf, Literal
from parsimonious.nodes import Node, NodeVisitor
from parsimonious.grammar import Grammar

# This is the base definition of the SQL grammar written in a simplified sort of
# EBNF notation. The notation here is of the form:
#    nonterminal = right hand side
# Rules that differ only in capitalization of keywords are mapped to the same action by
# the ``SqlVisitor``.  The nonterminal of the first rule is the starting symbol.
# In addition to the grammar here, we add ``col_ref``, ``table_name`` based on the tables
# that ``SqlTableContext`` is initialized with. ``number`` and ``string`` are initialized to
# be empty and later on updated based on the utterances.

SQL_GRAMMAR_STR = ur"""
    statement           = query ws ";" ws
    query               = (ws "(" ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws ")" ws) /
                          (ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws)
                        
    select_results      = col_refs / agg
    agg                 = agg_func ws "(" ws col_ref ws ")"
    agg_func            = "MIN" / "min" / "MAX" / "max" / "COUNT" / "count"
    col_refs            = (col_ref ws "," ws col_refs) / (col_ref)
    table_refs          = (table_name ws "," ws table_refs) / (table_name)
    where_clause        = ("WHERE" ws "(" ws conditions ws ")" ws) / ("WHERE" ws conditions ws)
    
    conditions          = (condition ws conj ws conditions) / 
                          (condition ws conj ws "(" ws conditions ws ")") /
                          ("(" ws conditions ws ")" ws conj ws conditions) /
                          ("(" ws conditions ws ")") /
                          ("not" ws conditions ws ) /
                          ("NOT" ws conditions ws ) /
                          condition
    condition           = in_clause / ternaryexpr / biexpr
    in_clause           = (ws col_ref ws "IN" ws query ws)
    biexpr              = ( col_ref ws binaryop ws value) / (value ws binaryop ws value) / ( col_ref ws "LIKE" ws string)
    binaryop            = "+" / "-" / "*" / "/" / "=" /
                          ">=" / "<=" / ">" / "<"  / "is" / "IS"
    ternaryexpr         = (col_ref ws "not" ws "BETWEEN" ws value ws "AND" ws value ws) /
                          (col_ref ws "NOT" ws "BETWEEN" ws value ws "AND" ws value ws) /
                          (col_ref ws "BETWEEN" ws value ws "AND" ws value ws)
    value               = ("not" ws pos_value) / ("NOT" ws pos_value) /(pos_value)
    pos_value           = ("ALL" ws query) / ("ANY" ws query) / number / boolean / col_ref / string / agg_results / "NULL"
    agg_results         = (ws "("  ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws ")" ws) /
                          (ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws)
    boolean             = "true" / "false"
    ws                  = ~"\s*"i
    conj                = "AND" / "OR" 
    distinct            = ("DISTINCT") / ("")
    number              =  ""
    string              =  ""
"""

KEYWORDS = [u'"SELECT"', u'"FROM"', u'"MIN"', u'"MAX"', u'"COUNT"', u'"WHERE"', u'"NOT"', u'"IN"', u'"LIKE"',
            u'"IS"', u'"BETWEEN"', u'"AND"', u'"ALL"', u'"ANY"', u'"NULL"', u'"OR"', u'"DISTINCT"']

def generate_one_of_string(nonterminal     , literals           )       :
    if literals:
        return  "\n{nonterminal} \t\t = " + u" / ".join([f'"{literal}"' for literal in literals])
    return  '\n{nonterminal} \t\t = ""'

def format_action(nonterminal     , right_hand_side     )       :
    if right_hand_side.upper() in KEYWORDS:
        right_hand_side = right_hand_side.upper()

    if nonterminal == u'string':
        return '{nonterminal} -> ["\'{right_hand_side}\'"]'

    elif nonterminal == u'number':
        return '{nonterminal} -> ["{right_hand_side}"]'

    else:
        right_hand_side = right_hand_side.lstrip(u"(").rstrip(u")")
        child_strings = [token for token in re.split(u" ws |ws | ws", right_hand_side) if token]
        child_strings = [tok.upper() if tok.upper() in KEYWORDS else tok for tok in child_strings]
        return "{nonterminal} -> [{', '.join(child_strings)}]"

class SqlTableContext(object):
    u"""
    A ``SqlTableContext`` represents the SQL context with grammar of SQL and the valid actions
    based on the schema of the tables that it represents.

    Parameters
    ----------
    tables: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables in the dataset, the keys are the names of the tables
        that map to lists of the table's column names.
    """
    def __init__(self, tables                       = None)        :
        self.tables = tables
        self.grammar_str: unicode = self.initialize_grammar_str()
        self.grammar: Grammar = Grammar(self.grammar_str)
        self.valid_actions =  self.initialize_valid_actions()

    def initialize_valid_actions(self)                        :
        u"""
        We initialize the valid actions with the global actions. These include the
        valid actions that result from the grammar and also those that result from
        the tables provided. The keys represent the nonterminals in the grammar
        and the values are lists of the valid actions of that nonterminal.
        """
        valid_actions                      = defaultdict(set)

        for key in self.grammar:
            rhs = self.grammar[key]

            # Sequence represents a series of expressions that match pieces of the text in order.
            # Eg. A -> B C
            if isinstance(rhs, Sequence):
                valid_actions[key].add(format_action(key, u" ".join(rhs._unicode_members()))) # pylint: disable=protected-access

            # OneOf represents a series of expressions, one of which matches the text.
            # Eg. A -> B / C
            elif isinstance(rhs, OneOf):
                for option in rhs._unicode_members(): # pylint: disable=protected-access
                    valid_actions[key].add(format_action(key, option))

            # A string literal, eg. "A"
            elif isinstance(rhs, Literal):
                if rhs.literal != u"":
                    valid_actions[key].add(format_action(key, rhs.literal))
                else:
                    valid_actions[key] = set()

        valid_action_strings = dict((key, sorted(value)) for key, value in list(valid_actions.items()))
        return valid_action_strings

    def initialize_grammar_str(self):
        grammar_str = SQL_GRAMMAR_STR

        if self.tables:
            column_right_hand_sides = [u'"*"']
            for table, columns in list(self.tables.items()):
                column_right_hand_sides.extend([f'("{table}" ws "." ws "{column}")' for column in columns])
            grammar_str += u"\n      col_ref \t\t = " +\
                    u" / ".join(sorted(column_right_hand_sides, reverse=True))

            grammar_str += generate_one_of_string(u'table_name', sorted(list(self.tables.keys()), reverse=True))

        return grammar_str

class SqlVisitor(NodeVisitor):
    u"""
    ``SqlVisitor`` performs a depth-first traversal of the the AST. It takes the parse tree
    and gives us an action sequence that resulted in that parse. Since the visitor has mutable
    state, we define a new ``SqlVisitor`` for each query. To get the action sequence, we create
    a ``SqlVisitor`` and call parse on it, which returns a list of actions. Ex.

        sql_visitor = SqlVisitor(grammar_string)
        action_sequence = sql_visitor.parse(query)

    Parameters
    ----------
    grammar : ``Grammar``
        A Grammar object that we use to parse the text.
    """
    def __init__(self, grammar         )        :
        self.action_sequence =  []
        self.grammar: Grammar = grammar

    #overrides
    def generic_visit(self, node      , visited_children            )             :
        self.add_action(node)
        if node.expr.name == u'statement':
            return self.action_sequence
        return []

    def add_action(self, node      )        :
        u"""
        For each node, we accumulate the rules that generated its children in a list.
        """
        if node.expr.name and node.expr.name != u'ws':
            nonterminal = '{node.expr.name} -> '

            if isinstance(node.expr, Literal):
                right_hand_side = '["{node.text}"]'

            else:
                child_strings = []
                for child in node.__iter__():
                    if child.expr.name == u'ws':
                        continue
                    if child.expr.name != u'':
                        child_strings.append(child.expr.name)
                    else:
                        child_right_side_string = child.expr._as_rhs().lstrip(u"(").rstrip(u")") # pylint: disable=protected-access
                        child_right_side_list = [tok for tok in\
                                                 re.split(u" ws |ws | ws", child_right_side_string) if tok]
                        child_right_side_list = [tok.upper() if tok.upper() in KEYWORDS else tok\
                                                 for tok in child_right_side_list]
                        child_strings.extend(child_right_side_list)
                right_hand_side = u"[" + u", ".join(child_strings) + u"]"

            rule = nonterminal + right_hand_side
            self.action_sequence = [rule] + self.action_sequence
