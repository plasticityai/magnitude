# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods

from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph


class TestTableQuestionKnowledgeGraph(AllenNlpTestCase):
    def test_read_from_json_handles_simple_cases(self):
        json = {
                u'question': [Token(x) for x in [u'where', u'is', u'mersin', u'?']],
                u'columns': [u'Name in English', u'Location'],
                u'cells': [[u'Paradeniz', u'Mersin'],
                          [u'Lake Gala', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:cell.mersin'])
        assert graph.entities == [u'-1', u'0', u'1', u'fb:cell.edirne', u'fb:cell.lake_gala',
                                  u'fb:cell.mersin', u'fb:cell.paradeniz', u'fb:row.row.location',
                                  u'fb:row.row.name_in_english']
        assert neighbors == set([u'fb:row.row.location'])
        neighbors = set(graph.neighbors[u'fb:row.row.name_in_english'])
        assert neighbors == set([u'fb:cell.paradeniz', u'fb:cell.lake_gala'])
        assert graph.entity_text[u'fb:cell.edirne'] == u'Edirne'
        assert graph.entity_text[u'fb:cell.lake_gala'] == u'Lake Gala'
        assert graph.entity_text[u'fb:cell.mersin'] == u'Mersin'
        assert graph.entity_text[u'fb:cell.paradeniz'] == u'Paradeniz'
        assert graph.entity_text[u'fb:row.row.location'] == u'Location'
        assert graph.entity_text[u'fb:row.row.name_in_english'] == u'Name in English'

        # These are default numbers that should always be in the graph.
        assert graph.neighbors[u'-1'] == []
        assert graph.neighbors[u'0'] == []
        assert graph.neighbors[u'1'] == []
        assert graph.entity_text[u'-1'] == u'-1'
        assert graph.entity_text[u'0'] == u'0'
        assert graph.entity_text[u'1'] == u'1'

    def test_read_from_json_replaces_newlines(self):
        # The csv -> tsv conversion renders '\n' as r'\n' (with a literal slash character), that
        # gets read in a two characters instead of one.  We need to make sure we convert it back to
        # one newline character, so our splitting and other processing works correctly.
        json = {
                u'question': [Token(x) for x in [u'where', u'is', u'mersin', u'?']],
                u'columns': [u'Name\\nin English', u'Location'],
                u'cells': [[u'Paradeniz', u'Mersin'],
                          [u'Lake\\nGala', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.entities == [u'-1', u'0', u'1', u'fb:cell.edirne', u'fb:cell.lake_gala',
                                  u'fb:cell.mersin', u'fb:cell.paradeniz', u'fb:part.gala',
                                  u'fb:part.lake', u'fb:part.paradeniz', u'fb:row.row.location',
                                  u'fb:row.row.name_in_english']
        assert graph.entity_text[u'fb:row.row.name_in_english'] == u'Name\nin English'

    def test_read_from_json_splits_columns_when_necessary(self):
        json = {
                u'question': [Token(x) for x in [u'where', u'is', u'mersin', u'?']],
                u'columns': [u'Name in English', u'Location'],
                u'cells': [[u'Paradeniz', u'Mersin with spaces'],
                          [u'Lake, Gala', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.entities == [u'-1', u'0', u'1', u'fb:cell.edirne', u'fb:cell.lake_gala',
                                  u'fb:cell.mersin_with_spaces', u'fb:cell.paradeniz', u'fb:part.gala',
                                  u'fb:part.lake', u'fb:part.paradeniz', u'fb:row.row.location',
                                  u'fb:row.row.name_in_english']
        assert graph.neighbors[u'fb:part.lake'] == []
        assert graph.neighbors[u'fb:part.gala'] == []
        assert graph.neighbors[u'fb:part.paradeniz'] == []

    def test_read_from_json_handles_numbers_in_question(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                u'question': [Token(x) for x in [u'one', u'4']],
                u'columns': [],
                u'cells': []
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.neighbors[u'1'] == []
        assert graph.neighbors[u'4'] == []
        assert graph.entity_text[u'1'] == u'one'
        assert graph.entity_text[u'4'] == u'4'

    def test_get_cell_parts_returns_cell_text_on_simple_cells(self):
        assert TableQuestionKnowledgeGraph._get_cell_parts(u'Team') == [(u'fb:part.team', u'Team')]
        assert TableQuestionKnowledgeGraph._get_cell_parts(u'2006') == [(u'fb:part.2006', u'2006')]
        assert TableQuestionKnowledgeGraph._get_cell_parts(u'Wolfe Tones') == [(u'fb:part.wolfe_tones',
                                                                               u'Wolfe Tones')]

    def test_get_cell_parts_splits_on_commas(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts(u'United States, Los Angeles')
        assert set(parts) == set([(u'fb:part.united_states', u'United States'),
                              (u'fb:part.los_angeles', u'Los Angeles')])

    def test_get_cell_parts_on_past_failure_cases(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts(u'Checco D\'Angelo\n "Jimmy"')
        assert set(parts) == set([(u'fb:part.checco_d_angelo', u"Checco D\'Angelo"),
                              (u'fb:part._jimmy', u'"Jimmy"')])

    def test_get_cell_parts_handles_multiple_splits(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts(u'this, has / lots\n of , commas')
        assert set(parts) == set([(u'fb:part.this', u'this'),
                              (u'fb:part.has', u'has'),
                              (u'fb:part.lots', u'lots'),
                              (u'fb:part.of', u'of'),
                              (u'fb:part.commas', u'commas')])

    def test_should_split_column_returns_false_when_all_text_is_simple(self):
        assert TableQuestionKnowledgeGraph._should_split_column_cells([u'Team', u'2006', u'Wolfe Tones']) is False

    def test_should_split_column_returns_true_when_one_input_is_splitable(self):
        assert TableQuestionKnowledgeGraph._should_split_column_cells([u'Team, 2006', u'Wolfe Tones']) is True

    def test_read_from_json_handles_diacritics(self):
        json = {
                u'question': [],
                u'columns': [u'Name in English', u'Name in Turkish', u'Location'],
                u'cells': [[u'Lake Van', u'Van Gölü', u'Mersin'],
                          [u'Lake Gala', u'Gala Gölü', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.name_in_turkish'])
        assert neighbors == set([u'fb:cell.van_golu', u'fb:cell.gala_golu'])

        json = {
                u'question': [],
                u'columns': [u'Notes'],
                u'cells': [[u'Ordained as a priest at\nReșița on March, 29th 1936']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.notes'])
        assert neighbors == set([u'fb:cell.ordained_as_a_priest_at_resita_on_march_29th_1936'])

        json = {
                u'question': [],
                u'columns': [u'Player'],
                u'cells': [[u'Mateja Kežman']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.player'])
        assert neighbors == set([u'fb:cell.mateja_kezman'])

        json = {
                u'question': [],
                u'columns': [u'Venue'],
                u'cells': [[u'Arena Națională, Bucharest, Romania']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.venue'])
        assert neighbors == set([u'fb:cell.arena_nationala_bucharest_romania'])

    def test_read_from_json_handles_newlines_in_columns(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                u'question': [],
                u'columns': [u'Peak\\nAUS', u'Peak\\nNZ'],
                u'cells': [[u'1', u'2'],
                          [u'3', u'4']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.peak_aus'])
        assert neighbors == set([u'fb:cell.1', u'fb:cell.3'])
        neighbors = set(graph.neighbors[u'fb:row.row.peak_nz'])
        assert neighbors == set([u'fb:cell.2', u'fb:cell.4'])
        neighbors = set(graph.neighbors[u'fb:cell.1'])
        assert neighbors == set([u'fb:row.row.peak_aus'])

        json = {
                u'question': [],
                u'columns': [u'Title'],
                u'cells': [[u'Dance of the\\nSeven Veils']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.title'])
        assert neighbors == set([u'fb:cell.dance_of_the_seven_veils'])

    def test_read_from_json_handles_diacritics_and_newlines(self):
        json = {
                u'question': [],
                u'columns': [u'Notes'],
                u'cells': [[u'8 districts\nFormed from Orūzgān Province in 2004']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.notes'])
        assert neighbors == set([u'fb:cell.8_districts_formed_from_oruzgan_province_in_2004'])

    def test_read_from_json_handles_crazy_unicode(self):
        json = {
                u'question': [],
                u'columns': [u'Town'],
                u'cells': [[u'Viðareiði'],
                          [u'Funningsfjørður'],
                          [u'Froðba']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.town'])
        assert neighbors == set([
                u'fb:cell.funningsfj_r_ur',
                u'fb:cell.vi_arei_i',
                u'fb:cell.fro_ba',])

        json = {
                u'question': [],
                u'columns': [u'Fate'],
                u'cells': [[u'Sunk at 45°00′N 11°21′W﻿ / ﻿45.000°N 11.350°W'],
                          [u'66°22′32″N 29°20′19″E﻿ / ﻿66.37556°N 29.33861°E']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.fate'])
        assert neighbors == set([u'fb:cell.sunk_at_45_00_n_11_21_w_45_000_n_11_350_w',
                             u'fb:cell.66_22_32_n_29_20_19_e_66_37556_n_29_33861_e'])

        json = {
                u'question': [],
                u'columns': [u'€0.01', u'Σ Points'],
                u'cells': [[u'6,000', u'9.5']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row._0_01'])
        assert neighbors == set([u'fb:cell.6_000'])
        neighbors = set(graph.neighbors[u'fb:row.row._points'])
        assert neighbors == set([u'fb:cell.9_5'])

        json = {
                u'question': [],
                u'columns': [u'Division'],
                u'cells': [[u'1ª Aut. Pref.']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.division'])
        assert neighbors == set([u'fb:cell.1_aut_pref'])

    def test_read_from_json_handles_parentheses_correctly(self):
        json = {
                u'question': [],
                u'columns': [u'Urban settlements'],
                u'cells': [[u'Dzhebariki-Khaya\\n(Джебарики-Хая)'],
                          [u'South Korea (KOR)'],
                          [u'Area (km²)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.urban_settlements'])
        assert neighbors == set([u'fb:cell.dzhebariki_khaya',
                             u'fb:cell.south_korea_kor',
                             u'fb:cell.area_km'])

        json = {
                u'question': [],
                u'columns': [u'Margin\\nof victory'],
                u'cells': [[u'−9 (67-67-68-69=271)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.margin_of_victory'])
        assert neighbors == set([u'fb:cell._9_67_67_68_69_271'])

        json = {
                u'question': [],
                u'columns': [u'Record'],
                u'cells': [[u'4.08 m (13 ft 41⁄2 in)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row.record'])
        assert neighbors == set([u'fb:cell.4_08_m_13_ft_41_2_in'])

    def test_read_from_json_handles_columns_with_duplicate_normalizations(self):
        json = {
                u'question': [],
                u'columns': [u'# of votes', u'% of votes'],
                u'cells': [[u'1', u'2'],
                          [u'3', u'4']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors[u'fb:row.row._of_votes'])
        assert neighbors == set([u'fb:cell.1', u'fb:cell.3'])
        neighbors = set(graph.neighbors[u'fb:row.row._of_votes_2'])
        assert neighbors == set([u'fb:cell.2', u'fb:cell.4'])
        neighbors = set(graph.neighbors[u'fb:cell.1'])
        assert neighbors == set([u'fb:row.row._of_votes'])

    def test_read_from_json_handles_cells_with_duplicate_normalizations(self):
        json = {
                u'question': [],
                u'columns': [u'answer'],
                u'cells': [[u'yes'], [u'yes*'], [u'yes'], [u'yes '], [u'yes*']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)

        # There are three unique text strings that all normalize to "yes", so there are three
        # fb:cell.yes entities.  Hopefully we produce them in the same order as SEMPRE does...
        assert graph.entities == [u'-1', u'0', u'1', u'fb:cell.yes', u'fb:cell.yes_2', u'fb:cell.yes_3',
                                  u'fb:row.row.answer']

    def test_get_numbers_from_tokens_works_for_arabic_numerals(self):
        tokens = [Token(x) for x in [u'7', u'1.0', u'-20']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [(u'7', u'7'), (u'1.000', u'1.0'), (u'-20', u'-20')]

    def test_get_numbers_from_tokens_works_for_ordinal_and_cardinal_numbers(self):
        tokens = [Token(x) for x in [u'one', u'five', u'Seventh']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [(u'1', u'one'), (u'5', u'five'), (u'7', u'Seventh')]

    def test_get_numbers_from_tokens_works_for_months(self):
        tokens = [Token(x) for x in [u'January', u'March', u'october']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [(u'1', u'January'), (u'3', u'March'), (u'10', u'october')]

    def test_get_numbers_from_tokens_works_for_units(self):
        tokens = [Token(x) for x in [u'1ghz', u'3.5mm', u'-2m/s']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [(u'1', u'1ghz'), (u'3.500', u'3.5mm'), (u'-2', u'-2m/s')]

    def test_get_numbers_from_tokens_works_with_magnitude_words(self):
        tokens = [Token(x) for x in [u'one', u'million', u'7', u'thousand']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [(u'1000000', u'one million'), (u'7000', u'7 thousand')]

    def test_get_linked_agenda_items(self):
        json = {
                u'question': [Token(x) for x in [u'where', u'is', u'mersin', u'?']],
                u'columns': [u'Name in English', u'Location'],
                u'cells': [[u'Paradeniz', u'Mersin'],
                          [u'Lake Gala', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.get_linked_agenda_items() == [u'fb:cell.mersin', u'fb:row.row.location']

    def test_get_longest_span_matching_entities(self):
        json = {
                u'question': [Token(x) for x in [u'where', u'is', u'lake', u'big', u'gala', u'?']],
                u'columns': [u'Name in English', u'Location'],
                u'cells': [[u'Paradeniz', u'Lake Big'],
                          [u'Lake Big Gala', u'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph._get_longest_span_matching_entities() == [u'fb:cell.lake_big_gala']
