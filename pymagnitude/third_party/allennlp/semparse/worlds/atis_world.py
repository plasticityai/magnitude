
from __future__ import absolute_import
from copy import deepcopy
#typing
import numpy

from parsimonious.grammar import Grammar

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.sql_table_context import\
        SqlTableContext, SqlVisitor, generate_one_of_string, format_action

from allennlp.data.tokenizers import Token, WordTokenizer
try:
    from itertools import izip
except:
    izip = zip


def get_strings_from_utterance(tokenized_utterance             )                        :
    u"""
    Based on the current utterance, return a dictionary where the keys are the strings in the utterance
    that map to lists of the token indices that they are linked to.
    """
    string_linking_scores                       = defaultdict(list)
    for index, (first_token, second_token) in enumerate(izip(tokenized_utterance, tokenized_utterance[1:])):
        for string in ATIS_TRIGGER_DICT.get(first_token.text.lower(), []):
            string_linking_scores[string].append(index)

        bigram = "{first_token.text} {second_token.text}".lower()
        for string in ATIS_TRIGGER_DICT.get(bigram, []):
            string_linking_scores[string].extend([index, index + 1])

    if tokenized_utterance[-1].text.lower() in ATIS_TRIGGER_DICT:
        for string in ATIS_TRIGGER_DICT[tokenized_utterance[-1].text.lower()]:
            string_linking_scores[string].append(len(tokenized_utterance)-1)

    date = get_date_from_utterance(tokenized_utterance)
    if date:
        for day in DAY_OF_WEEK_INDEX[date.weekday()]:
            string_linking_scores[day] = []

    return string_linking_scores

class AtisWorld(object):
    u"""
    World representation for the Atis SQL domain. This class has a ``SqlTableContext`` which holds the base
    grammars, it then augments this grammar with the entities that are detected from utterances.

    Parameters
    ----------
    utterances: ``List[str]``
        A list of utterances in the interaction, the last element in this list is the
        current utterance that we are interested in.
    """
    sql_table_context = SqlTableContext(TABLES)

    def __init__(self, utterances           , tokenizer=None)        :
        self.utterances =  utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        valid_actions, linking_scores = self.init_all_valid_actions()
        self.valid_actions =  valid_actions

        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores
        self.grammar_str: unicode = self.get_grammar_str()
        self.grammar_with_context: Grammar = Grammar(self.grammar_str)

    def get_valid_actions(self)                        :
        return self.valid_actions

    def init_all_valid_actions(self)                                              :
        u"""
        We initialize the valid actions with the global actions. We then iterate through the
        utterances up to and including the current utterance and add the valid strings.
        """

        valid_actions = deepcopy(self.sql_table_context.valid_actions)
        linking_scores = []
        current_tokenized_utterance = [] if not self.tokenized_utterances\
                else self.tokenized_utterances[-1]

        strings           = set()
        for tokenized_utterance in self.tokenized_utterances:
            string_linking_dict = get_strings_from_utterance(tokenized_utterance)
            strings.update(list(string_linking_dict.keys()))

        # We want to sort things in reverse here to be consistent with the grammar.
        # The parser is greedy which means that if we have a rule that has
        # multiple options for the right hand side, the first one that succeeds is
        # the one that is used. For example, if ``1400`` appears in the query, and
        # both ``1400`` and ``1`` are valid numbers, then we want to try to match
        # ``1400`` first. Otherwise, ``1`` will succeed but nothing will match ``400``.
        # The same applies for strings here.
        strings_list            = sorted(strings, reverse=True)

        # We construct the linking scores for strings from the ``string_linking_dict`` here.
        string_linking_scores = []
        for string in strings_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # string_linking_dict has the strings and linking scores from the last utterance.
            # If the string is not in the last utterance, then the linking scores will be all 0.
            for token_index in string_linking_dict.get(string, []):
                entity_linking[token_index] = 1
            string_linking_scores.append(entity_linking)
        linking_scores.extend(string_linking_scores)

        for string in strings_list:
            action = format_action(u'string', string)
            if action not in valid_actions[u'string']:
                valid_actions[u'string'].append(action)

        numbers = set([u'0', u'1'])
        number_linking_dict                       = {}

        for utterance, tokenized_utterance in izip(self.utterances, self.tokenized_utterances):
            number_linking_dict = get_numbers_from_utterance(utterance, tokenized_utterance)
            numbers.update(list(number_linking_dict.keys()))
        numbers_list            = sorted(numbers, reverse=True)

        # We construct the linking scores for numbers from the ``number_linking_dict`` here.
        number_linking_scores = []
        for number in numbers_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # number_linking_scores has the numbers and linking scores from the last utterance.
            # If the number is not in the last utterance, then the linking scores will be all 0.
            for token_index in number_linking_dict.get(number, []):
                entity_linking[token_index] = 1
            number_linking_scores.append(entity_linking)
        linking_scores.extend(number_linking_scores)

        for number in list(numbers_list):
            action = format_action(u'number', number)
            valid_actions[u'number'].append(action)
        return valid_actions, numpy.array(linking_scores)

    def get_grammar_str(self)       :
        u"""
        Generate a string that can be used to instantiate a ``Grammar`` object. The string is a sequence of
        rules that define the grammar.
        """
        grammar_str_with_context = self.sql_table_context.grammar_str
        numbers = [number.split(u" -> ")[1].lstrip(u'["').rstrip(u'"]') for\
                   number in sorted(self.valid_actions[u'number'], reverse=True)]
        strings = [string .split(u" -> ")[1].lstrip(u'["').rstrip(u'"]') for\
                   string in sorted(self.valid_actions[u'string'], reverse=True)]

        grammar_str_with_context += generate_one_of_string(u"number", numbers)
        grammar_str_with_context += generate_one_of_string(u"string", strings)
        return grammar_str_with_context


    def get_action_sequence(self, query     )             :
        sql_visitor = SqlVisitor(self.grammar_with_context)
        if query:
            action_sequence = sql_visitor.parse(query)
            return action_sequence
        return []

    def all_possible_actions(self)             :
        u"""
        Return a sorted list of strings representing all possible actions
        of the form: nonterminal -> [right_hand_side]
        """
        all_actions = set()
        for _, action_list in list(self.valid_actions.items()):
            for action in action_list:
                all_actions.add(action)
        return sorted(all_actions)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([self.valid_actions == other.valid_actions,
                        numpy.array_equal(self.linking_scores, other.linking_scores),
                        self.utterances == other.utterances,
                        self.grammar_str == other.grammar_str])
        return False
