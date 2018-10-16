
from __future__ import absolute_import
import re
from collections import defaultdict
#typing

#overrides
from unidecode import unidecode

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from io import open
try:
    from itertools import izip
except:
    izip = zip


DEFAULT_NUMBERS = [u'-1', u'0', u'1']
NUMBER_CHARACTERS = set([u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'.', u'-'])
MONTH_NUMBERS = {
        u'january': 1,
        u'jan': 1,
        u'february': 2,
        u'feb': 2,
        u'march': 3,
        u'mar': 3,
        u'april': 4,
        u'apr': 4,
        u'may': 5,
        u'june': 6,
        u'jun': 6,
        u'july': 7,
        u'jul': 7,
        u'august': 8,
        u'aug': 8,
        u'september': 9,
        u'sep': 9,
        u'october': 10,
        u'oct': 10,
        u'november': 11,
        u'nov': 11,
        u'december': 12,
        u'dec': 12,
        }
ORDER_OF_MAGNITUDE_WORDS = {u'hundred': 100, u'thousand': 1000, u'million': 1000000}
NUMBER_WORDS = {
        u'zero': 0,
        u'one': 1,
        u'two': 2,
        u'three': 3,
        u'four': 4,
        u'five': 5,
        u'six': 6,
        u'seven': 7,
        u'eight': 8,
        u'nine': 9,
        u'ten': 10,
        u'first': 1,
        u'second': 2,
        u'third': 3,
        u'fourth': 4,
        u'fifth': 5,
        u'sixth': 6,
        u'seventh': 7,
        u'eighth': 8,
        u'ninth': 9,
        u'tenth': 10,
        }

class TableQuestionKnowledgeGraph(KnowledgeGraph):
    u"""
    A ``TableQuestionKnowledgeGraph`` represents the linkable entities in a table and a question
    about the table.  The linkable entities in a table are the cells and the columns of the table,
    and the linkable entities from the question are the numbers in the question.  We use the
    question to define our space of allowable numbers, because there are infinitely many numbers
    that we could include in our action space, and we really don't want to do that. Additionally, we
    have a method that returns the set of entities in the graph that are relevant to the question,
    and we keep the question for this method. See ``get_linked_agenda_items`` for more information.

    To represent the table as a graph, we make each cell and column a node in the graph, and
    consider a column's neighbors to be all cells in that column (and thus each cell has just one
    neighbor - the column it belongs to).  This is a rather simplistic view of the table. For
    example, we don't store the order of rows.

    We represent numbers as standalone nodes in the graph, without any neighbors.

    Additionally, when we encounter cells that can be split, we create ``fb:part.[something]``
    entities, also without any neighbors.
    """
    def __init__(self,
                 entities          ,
                 neighbors                      ,
                 entity_text                ,
                 question_tokens             )        :
        super(TableQuestionKnowledgeGraph, self).__init__(entities, neighbors, entity_text)
        self.question_tokens = question_tokens
        self._entity_prefixes = defaultdict(list)
        for entity, text in list(self.entity_text.items()):
            parts = text.split()
            if not parts:
                continue
            prefix = parts[0].lower()
            self._entity_prefixes[prefix].append(entity)

    @classmethod
    def read_from_file(cls, filename     , question             )                                 :
        u"""
        We read tables formatted as TSV files here. We assume the first line in the file is a tab
        separated list of column headers, and all subsequent lines are content rows. For example if
        the TSV file is:

        Nation      Olympics    Medals
        USA         1896        8
        China       1932        9

        we read "Nation", "Olympics" and "Medals" as column headers, "USA" and "China" as cells
        under the "Nation" column and so on.
        """
        return cls.read_from_lines(open(filename).readlines(), question)

    @classmethod
    def read_from_lines(cls, lines           , question             )                                 :
        cells = []
        # We assume the first row is column names.
        for row_index, line in enumerate(lines):
            line = line.rstrip(u'\n')
            if row_index == 0:
                columns = line.split(u'\t')
            else:
                cells.append(line.split(u'\t'))
        return cls.read_from_json({u"columns": columns, u"cells": cells, u"question": question})

    @classmethod
    def read_from_json(cls, json_object                )                                 :
        u"""
        We read tables formatted as JSON objects (dicts) here. This is useful when you are reading
        data from a demo. The expected format is::

            {"question": [token1, token2, ...],
             "columns": [column1, column2, ...],
             "cells": [[row1_cell1, row1_cell2, ...],
                       [row2_cell1, row2_cell2, ...],
                       ... ]}
        """
        entity_text                 = {}
        neighbors                              = defaultdict(list)

        # Getting number entities first.  Number entities don't have any neighbors, and their
        # "entity text" is the text from the question that evoked the number.
        question_tokens = json_object[u'question']
        for number, number_text in cls._get_numbers_from_tokens(question_tokens):
            entity_text[number] = number_text
            neighbors[number] = []
        for default_number in DEFAULT_NUMBERS:
            if default_number not in neighbors:
                neighbors[default_number] = []
                entity_text[default_number] = default_number

        # Following Sempre's convention for naming columns.  Sempre gives columns unique names when
        # columns normalize to a collision, so we keep track of these.  We do not give cell text
        # unique names, however, as `fb:cell.x` is actually a function that returns all cells that
        # have text that normalizes to "x".
        column_ids = []
        columns                 = {}
        for column_string in json_object[u'columns']:
            column_string = column_string.replace(u'\\n', u'\n')
            normalized_string = 'fb:row.row.{cls._normalize_string(column_string)}'
            if normalized_string in columns:
                columns[normalized_string] += 1
                normalized_string = '{normalized_string}_{columns[normalized_string]}'
            columns[normalized_string] = 1
            column_ids.append(normalized_string)
            entity_text[normalized_string] = column_string

        # Stores cell text to cell name, making sure that unique text maps to a unique name.
        cell_id_mapping                 = {}
        column_cells                  = [[] for _ in columns]
        for row_index, row_cells in enumerate(json_object[u'cells']):
            assert len(columns) == len(row_cells), (u"Invalid format. Row %d has %d cells, but header has %d"
                                                    u" columns" % (row_index, len(row_cells), len(columns)))
            # Following Sempre's convention for naming cells.
            row_cell_ids = []
            for column_index, cell_string in enumerate(row_cells):
                cell_string = cell_string.replace(u'\\n', u'\n')
                column_cells[column_index].append(cell_string)
                if cell_string in cell_id_mapping:
                    normalized_string = cell_id_mapping[cell_string]
                else:
                    base_normalized_string = 'fb:cell.{cls._normalize_string(cell_string)}'
                    normalized_string = base_normalized_string
                    attempt_number = 1
                    while normalized_string in list(cell_id_mapping.values()):
                        attempt_number += 1
                        normalized_string = "{base_normalized_string}_{attempt_number}"
                    cell_id_mapping[cell_string] = normalized_string
                row_cell_ids.append(normalized_string)
                entity_text[normalized_string] = cell_string
            for column_id, cell_id in izip(column_ids, row_cell_ids):
                neighbors[column_id].append(cell_id)
                neighbors[cell_id].append(column_id)

        for column in column_cells:
            if cls._should_split_column_cells(column):
                for cell_string in column:
                    for part_entity, part_string in cls._get_cell_parts(cell_string):
                        neighbors[part_entity] = []
                        entity_text[part_entity] = part_string
        return cls(set(neighbors.keys()), dict(neighbors), entity_text, question_tokens)

    @staticmethod
    def _normalize_string(string     )       :
        u"""
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,

        return unidecode(string.lower())

    @staticmethod
    def _get_numbers_from_tokens(tokens             )                         :
        u"""
        Finds numbers in the input tokens and returns them as strings.  We do some simple heuristic
        number recognition, finding ordinals and cardinals expressed as text ("one", "first",
        etc.), as well as numerals ("7th", "3rd"), months (mapping "july" to 7), and units
        ("1ghz").

        We also handle year ranges expressed as decade or centuries ("1800s" or "1950s"), adding
        the endpoints of the range as possible numbers to generate.

        We return a list of tuples, where each tuple is the (number_string, token_text) for a
        number found in the input tokens.
        """
        numbers = []
        for i, token in enumerate(tokens):
            number                    = None
            token_text = token.text
            text = token.text.replace(u',', u'').lower()
            if text in NUMBER_WORDS:
                number = NUMBER_WORDS[text]

            magnitude = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ORDER_OF_MAGNITUDE_WORDS:
                    magnitude = ORDER_OF_MAGNITUDE_WORDS[next_token]
                    token_text += u' ' + tokens[i + 1].text

            is_range = False
            if len(text) > 1 and text[-1] == u's' and text[-2] == u'0':
                is_range = True
                text = text[:-1]

            # We strip out any non-digit characters, to capture things like '7th', or '1ghz'.  The
            # way we're doing this could lead to false positives for something like '1e2', but
            # we'll take that risk.  It shouldn't be a big deal.
            text = u''.join(text[i] for i, char in enumerate(text) if char in NUMBER_CHARACTERS)

            try:
                # We'll use a check for float(text) to find numbers, because text.isdigit() doesn't
                # catch things like "-3" or "0.07".
                number = float(text)
            except ValueError:
                pass

            if number is not None:
                number = number * magnitude
                if u'.' in text:
                    number_string = u'%.3f' % number
                else:
                    number_string = u'%d' % number
                numbers.append((number_string, token_text))
                if is_range:
                    # TODO(mattg): both numbers in the range will have the same text, and so the
                    # linking score won't have any way to differentiate them...  We should figure
                    # out a better way to handle this.
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == u'0':
                        num_zeros += 1
                    numbers.append((unicode(int(number + 10 ** num_zeros)), token_text))
        return numbers

    cell_part_regex = re.compile(r',\s|\n|/')
    @classmethod
    def _get_cell_parts(cls, cell_text     )                         :
        u"""
        Splits a cell into parts and returns the parts of the cell.  We return a list of
        ``(entity_name, entity_text)``, where ``entity_name`` is ``fb:part.[something]``, and
        ``entity_text`` is the text of the cell corresponding to that part.  For many cells, there
        is only one "part", and we return a list of length one.

        Note that you shouldn't call this on every cell in the table; SEMPRE decides to make these
        splits only when at least one of the cells in a column looks "splittable".  Only if you're
        splitting the cells in a column should you use this function.
        """
        parts = []
        for part_text in cls.cell_part_regex.split(cell_text):
            part_text = part_text.strip()
            part_entity = 'fb:part.{cls._normalize_string(part_text)}'
            parts.append((part_entity, part_text))
        return parts

    @classmethod
    def _should_split_column_cells(cls, column_cells           )        :
        u"""
        Returns true if there is any cell in this column that can be split.
        """
        return any(cls._should_split_cell(cell_text) for cell_text in column_cells)

    @classmethod
    def _should_split_cell(cls, cell_text     )        :
        u"""
        Checks whether the cell should be split.  We're just doing the same thing that SEMPRE did
        here.
        """
        if u', ' in cell_text or u'\n' in cell_text or u'/' in cell_text:
            return True
        return False

    def get_linked_agenda_items(self)             :
        u"""
        Returns entities that can be linked to spans in the question, that should be in the agenda,
        for training a coverage based semantic parser. This method essentially does a heuristic
        entity linking, to provide weak supervision for a learning to search parser.
        """
        agenda_items            = []
        for entity in self._get_longest_span_matching_entities():
            agenda_items.append(entity)
            # If the entity is a cell, we need to add the column to the agenda as well,
            # because the answer most likely involves getting the row with the cell.
            if u'fb:cell' in entity:
                agenda_items.append(self.neighbors[entity][0])
        return agenda_items

    def _get_longest_span_matching_entities(self):
        question = u" ".join([token.text for token in self.question_tokens])
        matches_starting_at                       = defaultdict(list)
        for index, token in enumerate(self.question_tokens):
            if token.text in self._entity_prefixes:
                for entity in self._entity_prefixes[token.text]:
                    if self.entity_text[entity].lower() in question:
                        matches_starting_at[index].append(entity)
        longest_matches            = []
        for index, matches in list(matches_starting_at.items()):
            longest_matches.append(sorted(matches, key=len)[-1])
        return longest_matches

    #overrides
    def __eq__(self, other):
        if isinstance(self, other.__class__):
            for key in self.__dict__:
                # We need to specially handle question tokens because they are Spacy's ``Token``
                # objects, and equality is not defined for them.
                if key == u"question_tokens":
                    self_tokens = self.__dict__[key]
                    other_tokens = other.__dict__[key]
                    if not all([token1.text == token2.text
                                for token1, token2 in izip(self_tokens, other_tokens)]):
                        return False
                else:
                    if not self.__dict__[key] == other.__dict__[key]:
                        return False
            return True
        return NotImplemented
