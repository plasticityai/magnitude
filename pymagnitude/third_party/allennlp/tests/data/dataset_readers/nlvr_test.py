# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import NlvrDatasetReader
from allennlp.semparse.worlds import NlvrWorld


class TestNlvrDatasetReader(AllenNlpTestCase):
    def test_reader_reads_ungrouped_data(self):
        test_file = unicode(self.FIXTURES_ROOT / u"data" / u"nlvr" /
                        u"sample_ungrouped_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        assert list(instance.fields.keys()) == set([u'sentence', u'agenda', u'worlds', u'actions', u'labels',
                                          u'identifier'])
        sentence_tokens = instance.fields[u"sentence"].tokens
        expected_tokens = [u'There', u'is', u'a', u'circle', u'closely', u'touching', u'a', u'corner', u'of',
                           u'a', u'box', u'.']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields[u"actions"].field_list]
        assert len(actions) == 115
        agenda = [item.sequence_index for item in instance.fields[u"agenda"].field_list]
        agenda_strings = [actions[rule_id] for rule_id in agenda]
        assert set(agenda_strings) == set([u'<o,o> -> circle',
                                           u'<o,t> -> object_exists',
                                           u'<o,o> -> touch_corner'])
        worlds = [world_field.as_tensor({}) for world_field in instance.fields[u"worlds"].field_list]
        assert isinstance(worlds[0], NlvrWorld)
        label = instance.fields[u"labels"].field_list[0].label
        assert label == u"true"

    def test_agenda_indices_are_correct(self):
        reader = NlvrDatasetReader()
        test_file = unicode(self.FIXTURES_ROOT / u"data" / u"nlvr" /
                        u"sample_ungrouped_data.jsonl")
        dataset = reader.read(test_file)
        instances = list(dataset)
        instance = instances[0]
        sentence_tokens = instance.fields[u"sentence"].tokens
        sentence = u" ".join([t.text for t in sentence_tokens])
        agenda = [item.sequence_index for item in instance.fields[u"agenda"].field_list]
        actions = [action.rule for action in instance.fields[u"actions"].field_list]
        agenda_actions = [actions[i] for i in agenda]
        world = instance.fields[u"worlds"].field_list[0].as_tensor({})
        expected_agenda_actions = world.get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
        assert expected_agenda_actions == agenda_actions

    def test_reader_reads_grouped_data(self):
        test_file = unicode(self.FIXTURES_ROOT / u"data" / u"nlvr" /
                        u"sample_grouped_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]
        assert list(instance.fields.keys()) == set([u'sentence', u'agenda', u'worlds', u'actions', u'labels',
                                          u'identifier'])
        sentence_tokens = instance.fields[u"sentence"].tokens
        expected_tokens = [u'There', u'is', u'a', u'circle', u'closely', u'touching', u'a', u'corner', u'of',
                           u'a', u'box', u'.']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields[u"actions"].field_list]
        assert len(actions) == 115
        agenda = [item.sequence_index for item in instance.fields[u"agenda"].field_list]
        agenda_strings = [actions[rule_id] for rule_id in agenda]
        assert set(agenda_strings) == set([u'<o,o> -> circle',
                                           u'<o,o> -> touch_corner',
                                           u'<o,t> -> object_exists'
                                          ])
        worlds = [world_field.as_tensor({}) for world_field in instance.fields[u"worlds"].field_list]
        assert all([isinstance(world, NlvrWorld) for world in worlds])
        labels = [label.label for label in instance.fields[u"labels"].field_list]
        assert labels == [u"true", u"false", u"true", u"false"]

    def test_reader_reads_processed_data(self):
        # Processed data contains action sequences that yield the correct denotations, obtained from
        # an offline search.
        test_file = unicode(self.FIXTURES_ROOT / u"data" / u"nlvr" /
                        u"sample_processed_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]
        assert list(instance.fields.keys()) == set([u"sentence", u"target_action_sequences",
                                          u"worlds", u"actions", u"labels", u"identifier"])
        all_action_sequence_indices = instance.fields[u"target_action_sequences"].field_list
        assert len(all_action_sequence_indices) == 20
        action_sequence_indices = [item.sequence_index for item in
                                   all_action_sequence_indices[0].field_list]
        actions = [action.rule for action in instance.fields[u"actions"].field_list]
        action_sequence = [actions[rule_id] for rule_id in action_sequence_indices]
        assert action_sequence == [u'@start@ -> t',
                                   u't -> [<o,t>, o]',
                                   u'<o,t> -> object_exists',
                                   u'o -> [<o,o>, o]',
                                   u'<o,o> -> touch_corner',
                                   u'o -> [<o,o>, o]',
                                   u'<o,o> -> circle',
                                   u'o -> all_objects']
