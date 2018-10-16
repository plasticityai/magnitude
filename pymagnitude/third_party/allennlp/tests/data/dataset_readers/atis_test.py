# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.data.dataset_readers import AtisDatasetReader
from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.worlds import AtisWorld

class TestAtisReader(AllenNlpTestCase):
    def test_atis_read_from_file(self):
        reader = AtisDatasetReader()
        data_path = AllenNlpTestCase.FIXTURES_ROOT / u"data" / u"atis" / u"sample.json"

        instances = list(reader.read(unicode(data_path)))

        assert len(instances) == 12
        instance = instances[0]

        assert list(instance.fields.keys()) ==\
                set([u'utterance',
                 u'actions',
                 u'world',
                 u'target_action_sequence',
                 u'linking_scores'])

        assert [t.text for t in instance.fields[u"utterance"].tokens] ==\
                [u'show', u'me', u'the', u'one', u'way',
                 u'flights', u'from', u'detroit', u'to',
                 u'westchester', u'county']

        assert isinstance(instance.fields[u'world'].as_tensor({}), AtisWorld)
        # Check that the strings in the actions field has the same actions
        # as the strings in the world.
        valid_strs = set()
        for action in instance.fields[u'actions'].field_list:
            if action.rule.startswith(u'string'):
                valid_strs.add(action.rule)

        world = instance.fields[u'world'].metadata
        assert valid_strs == set(world.valid_actions[u'string'])

        assert world.valid_actions[u'string'] ==\
            [u'string -> ["\'WESTCHESTER COUNTY\'"]',
             u'string -> ["\'NO\'"]',
             u'string -> ["\'HHPN\'"]',
             u'string -> ["\'DTW\'"]',
             u'string -> ["\'DETROIT\'"]',
             u'string -> ["\'DDTT\'"]']

        assert world.valid_actions[u'number'] ==\
                [u'number -> ["1"]',
                 u'number -> ["0"]']

        # We should have generated created linking scores of the shape
        # (num_entities, num_utterance_tokens). We have two types
        # of entities: strings and numbers.
        assert world.linking_scores.shape[0] ==\
                len(world.valid_actions[u'string'] +
                    world.valid_actions[u'number'])
        assert world.linking_scores.shape[1] ==\
                len(instance.fields[u'utterance'].tokens)

        expected_linking_scores = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], # "westchester county" ->
                                                                      # "WESTCHESTER COUNTY"
                                   [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # "one way" -> "NO"
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], # "westchester county" -> "HHPN"
                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # "detroit" -> "DTW"
                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # "detroit" -> "DETROIT"
                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # "detroit" -> "DDTT"
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1 added as default
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] # 0 added as default

        for entity_index, entity in enumerate(world.linking_scores):
            for question_index, _ in enumerate(entity):
                assert world.linking_scores[entity_index][question_index] ==\
                        expected_linking_scores[entity_index][question_index]
