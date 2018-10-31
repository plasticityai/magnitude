# pylint: disable=invalid-name,no-self-use,protected-access



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.semparse.worlds import WikiTablesWorld
from io import open


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 2
    instance = instances[0]

    assert list(instance.fields.keys()) == set([
            u'question',
            u'table',
            u'world',
            u'actions',
            u'target_action_sequences',
            u'example_lisp_string',
            u'metadata'])

    question_tokens = [u"what", u"was", u"the", u"last", u"year", u"where", u"this", u"team", u"was", u"a",
                       u"part", u"of", u"the", u"usl", u"a", u"-", u"league", u"?"]
    assert [t.text for t in instance.fields[u"question"].tokens] == question_tokens

    entities = instance.fields[u'table'].knowledge_graph.entities
    assert len(entities) == 59
    assert sorted(entities) == [
            # Numbers, which are represented as graph entities, as we link them to the question.
            u'-1',
            u'0',
            u'1',

            # The table cell entity names.
            u'fb:cell.10_727',
            u'fb:cell.11th',
            u'fb:cell.1st',
            u'fb:cell.1st_round',
            u'fb:cell.1st_western',
            u'fb:cell.2',
            u'fb:cell.2001',
            u'fb:cell.2002',
            u'fb:cell.2003',
            u'fb:cell.2004',
            u'fb:cell.2005',
            u'fb:cell.2006',
            u'fb:cell.2007',
            u'fb:cell.2008',
            u'fb:cell.2009',
            u'fb:cell.2010',
            u'fb:cell.2nd',
            u'fb:cell.2nd_pacific',
            u'fb:cell.2nd_round',
            u'fb:cell.3rd_pacific',
            u'fb:cell.3rd_round',
            u'fb:cell.3rd_usl_3rd',
            u'fb:cell.4th_round',
            u'fb:cell.4th_western',
            u'fb:cell.5_575',
            u'fb:cell.5_628',
            u'fb:cell.5_871',
            u'fb:cell.5th',
            u'fb:cell.6_028',
            u'fb:cell.6_260',
            u'fb:cell.6_851',
            u'fb:cell.7_169',
            u'fb:cell.8_567',
            u'fb:cell.9_734',
            u'fb:cell.did_not_qualify',
            u'fb:cell.quarterfinals',
            u'fb:cell.semifinals',
            u'fb:cell.usl_a_league',
            u'fb:cell.usl_first_division',
            u'fb:cell.ussf_d_2_pro_league',

            # Cell parts
            u'fb:part.11th',
            u'fb:part.1st',
            u'fb:part.2nd',
            u'fb:part.3rd',
            u'fb:part.4th',
            u'fb:part.5th',
            u'fb:part.pacific',
            u'fb:part.usl_3rd',
            u'fb:part.western',

            # Column headers
            u'fb:row.row.avg_attendance',
            u'fb:row.row.division',
            u'fb:row.row.league',
            u'fb:row.row.open_cup',
            u'fb:row.row.playoffs',
            u'fb:row.row.regular_season',
            u'fb:row.row.year',
            ]

    # The content of this will be tested indirectly by checking the actions; we'll just make
    # sure we get a WikiTablesWorld object in here.
    assert isinstance(instance.fields[u'world'].as_tensor({}), WikiTablesWorld)

    action_fields = instance.fields[u'actions'].field_list
    actions = [action_field.rule for action_field in action_fields]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).  This is good, because
    # sometimes DPD does silly things that we don't want to reproduce.  But it also means if we
    # break something, we might not notice in the test unless we check this explicitly.
    num_action_sequences = len(instance.fields[u"target_action_sequences"].field_list)
    assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples _by tree size_, which is _not_ the
    # first one in the file, or the shortest logical form by _string length_.  It's also a totally
    # made up logical form, just to demonstrate that we're sorting things correctly.
    action_sequence = instance.fields[u"target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]
    assert actions == [
            u'@start@ -> r',
            u'r -> [<c,r>, c]',
            u'<c,r> -> fb:row.row.league',
            u'c -> fb:cell.3rd_usl_3rd'
            ]


class WikiTablesDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        params = {
                u'lazy': False,
                u'tables_directory': self.FIXTURES_ROOT / u"data" / u"wikitables",
                u'dpd_output_directory': self.FIXTURES_ROOT / u"data" / u"wikitables" / u"dpd_output",
                }
        reader = WikiTablesDatasetReader.from_params(Params(params))
        dataset = reader.read(unicode(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data.examples"))
        assert_dataset_correct(dataset)

    def test_reader_reads_preprocessed_file(self):
        # We're should get the exact same results when reading a pre-processed file as we get when
        # we read the original data.
        reader = WikiTablesDatasetReader()
        dataset = reader.read(unicode(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data_preprocessed.jsonl"))
        assert_dataset_correct(dataset)

    def test_read_respects_max_dpd_tries_when_not_sorting(self):
        tables_directory = self.FIXTURES_ROOT / u"data" / u"wikitables"
        dpd_output_directory = self.FIXTURES_ROOT / u"data" / u"wikitables" / u"dpd_output"
        reader = WikiTablesDatasetReader(lazy=False,
                                         sort_dpd_logical_forms=False,
                                         max_dpd_logical_forms=1,
                                         max_dpd_tries=1,
                                         tables_directory=tables_directory,
                                         dpd_output_directory=dpd_output_directory)
        dataset = reader.read(unicode(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data.examples"))
        instances = list(dataset)
        instance = instances[0]
        actions = [action_field.rule for action_field in instance.fields[u'actions'].field_list]

        # We should have just taken the first logical form from the file, which has the following
        # action sequence.
        action_sequence = instance.fields[u"target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        action_strings = [actions[i] for i in action_indices]
        assert action_strings == [
                u'@start@ -> d',
                u'd -> [<c,d>, c]',
                u'<c,d> -> [<<#1,#2>,<#2,#1>>, <d,c>]',
                u'<<#1,#2>,<#2,#1>> -> reverse',
                u'<d,c> -> fb:cell.cell.date',
                u'c -> [<r,c>, r]',
                u'<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                u'<<#1,#2>,<#2,#1>> -> reverse',
                u'<c,r> -> fb:row.row.year',
                u'r -> [<n,r>, n]',
                u'<n,r> -> fb:row.row.index',
                u'n -> [<nd,nd>, n]',
                u'<nd,nd> -> max',
                u'n -> [<r,n>, r]',
                u'<r,n> -> [<<#1,#2>,<#2,#1>>, <n,r>]',
                u'<<#1,#2>,<#2,#1>> -> reverse',
                u'<n,r> -> fb:row.row.index',
                u'r -> [<c,r>, c]',
                u'<c,r> -> fb:row.row.league',
                u'c -> fb:cell.usl_a_league'
                ]

    def test_parse_example_line(self):
        # pylint: disable=no-self-use,protected-access
        with open(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data.examples") as filename:
            lines = filename.readlines()
        example_info = WikiTablesDatasetReader._parse_example_line(lines[0])
        question = u'what was the last year where this team was a part of the usl a-league?'
        assert example_info == {u'id': u'nt-0',
                                u'question': question,
                                u'table_filename': u'tables/590.csv'}
