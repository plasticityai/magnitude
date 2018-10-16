# pylint: disable=invalid-name,no-self-use,protected-access


from __future__ import division
from __future__ import absolute_import
from collections import namedtuple
import os
import pytest

from flaky import flaky
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model, WikiTablesMmlSemanticParser
from allennlp.training.metrics.wikitables_accuracy import SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH
try:
    from itertools import izip
except:
    izip = zip


class WikiTablesMmlSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

        # The model tests are run with respect to the module root, so check if abbreviations
        # and grammar already exist there (since we want to clean up module root after test)
        self.module_root_abbreviations_path = self.MODULE_ROOT / u"data" / u"abbreviations.tsv"
        self.module_root_grammar_path = self.MODULE_ROOT / u"data" / u"grow.grammar"
        self.should_remove_root_sempre_abbreviations = not os.path.exists(self.module_root_abbreviations_path)
        self.should_remove_root_sempre_grammar = not os.path.exists(self.module_root_grammar_path)

        super(WikiTablesMmlSemanticParserTest, self).setUp()
        self.set_up_model(unicode(self.FIXTURES_ROOT / u"semantic_parsing" / u"wikitables" / u"experiment.json"),
                          unicode(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data.examples"))

    def tearDown(self):
        super(WikiTablesMmlSemanticParserTest, self).tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)
        if self.should_remove_root_sempre_abbreviations and os.path.exists(self.module_root_abbreviations_path):
            os.remove(self.module_root_abbreviations_path)
        if self.should_remove_root_sempre_grammar and os.path.exists(self.module_root_grammar_path):
            os.remove(self.module_root_grammar_path)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_elmo_mixture_no_features_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / u'semantic_parsing' / u'wikitables' / u'experiment-mixture.json'
        self.ensure_model_can_train_save_and_load(param_file)

    @flaky
    def test_elmo_no_features_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / u'semantic_parsing' / u'wikitables' / u'experiment-elmo-no-features.json'
        self.ensure_model_can_train_save_and_load(param_file, tolerance=1e-2)

    def test_get_neighbor_indices(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = torch.LongTensor([])

        neighbor_indices = self.model._get_neighbor_indices(worlds, num_entities, tensor)

        # Checks for the correct shape meaning dimension 2 has size num_neighbors,
        # padding of -1 is used, and correct neighbor indices.
        assert_almost_equal(neighbor_indices.data.numpy(), [[[-1, -1],
                                                             [3, 4],
                                                             [3, 4],
                                                             [1, 2],
                                                             [1, 2]],
                                                            [[-1, -1],
                                                             [2, -1],
                                                             [1, -1],
                                                             [-1, -1],
                                                             [-1, -1]]])

    def test_get_type_vector(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = torch.LongTensor([])
        type_vector, _ = self.model._get_type_vector(worlds, num_entities, tensor)
        # Verify that both types are present and padding used for non existent entities.
        assert_almost_equal(type_vector.data.numpy(), [[[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 1],
                                                        [0, 0, 0, 1]],
                                                       [[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 1],
                                                        [0, 0, 0, 0],
                                                        [0, 0, 0, 0]]])

    def test_get_linking_probabilities(self):
        worlds, num_entities = self.get_fake_worlds()
        # (batch_size, num_question_tokens, num_entities)
        linking_scores = [[[-2, 1, 0, -3, 2],
                           [4, -1, 5, -3, 4]],
                          [[0, 1, 8, 10, 10],
                           [3, 2, -1, -2, 1]]]
        linking_scores = torch.FloatTensor(linking_scores)
        question_mask = torch.LongTensor([[1, 1], [1, 0]])
        _, entity_type_dict = self.model._get_type_vector(worlds, num_entities, linking_scores)

        # (batch_size, num_question_tokens, num_entities)
        entity_probability = self.model._get_linking_probabilities(worlds, linking_scores, question_mask,
                                                                   entity_type_dict)

        # The following properties in entity_probability are tested for by true_probability:
        # (1) It has all 0.0 probabilities when there is no question token, as seen for the
        #     second word in the second batch.
        # (2) It has 0.0 probabilities when an entity is masked, as seen in the last two entities
        #     for the second batch instance.
        # (3) The probabilities for entities of the same type with the same question token should
        #     sum to at most 1, but not necessarily 1, because some probability mass goes to the
        #     null entity.  We have three entity types here, so each row should sum to at most 3,
        #     and that number will approach 3 as the unnormalized linking scores for each entity
        #     get higher.
        true_probability = [[[0.1192029, 0.5761169, 0.2119416, 0.0058998, 0.8756006],
                             [0.9820138, 0.0024561, 0.9908675, 0.0008947, 0.9811352]],
                            [[0.5, 0.7310586, 0.9996647, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0]]]
        assert_almost_equal(entity_probability.detach().cpu().numpy(), true_probability)

    def get_fake_worlds(self):
        # Generate a toy WikitablesWorld.
        FakeTable = namedtuple(u'FakeTable', [u'entities', u'neighbors'])
        FakeWorld = namedtuple(u'FakeWorld', [u'table_graph'])
        entities = [[u'0', u'fb:cell.2010', u'fb:cell.2011', u'fb:row.row.year', u'fb:row.row.year2'],
                    [u'1', u'fb:cell.2012', u'fb:row.row.year']]
        neighbors = [{u'fb:cell.2010': [u'fb:row.row.year', u'fb:row.row.year2'],
                      u'fb:cell.2011': [u'fb:row.row.year', u'fb:row.row.year2'],
                      u'fb:row.row.year': [u'fb:cell.2010', u'fb:cell.2011'],
                      u'fb:row.row.year2': [u'fb:cell.2010', u'fb:cell.2011'],
                      u'0': [],
                     },
                     {u'fb:cell.2012': [u'fb:row.row.year'],
                      u'fb:row.row.year': [u'fb:cell.2012'],
                      u'1': [],
                     }]

        worlds = [FakeWorld(FakeTable(entity_list, entity2neighbors))
                  for entity_list, entity2neighbors in izip(entities, neighbors)]
        num_entities = max([len(entity_list) for entity_list in entities])
        return worlds, num_entities

    def test_embed_actions_works_with_batched_and_padded_input(self):
        params = Params.from_file(self.param_file)
        model = Model.from_params(vocab=self.vocab, params=params[u'model'])
        action_embedding_weights = model._action_embedder.weight
        rule1 = model.vocab.get_token_from_index(1, u'rule_labels')
        rule1_tensor = torch.LongTensor([1])
        rule2 = model.vocab.get_token_from_index(2, u'rule_labels')
        rule2_tensor = torch.LongTensor([2])
        rule3 = model.vocab.get_token_from_index(3, u'rule_labels')
        rule3_tensor = torch.LongTensor([3])
        actions = [[(rule1, True, rule1_tensor),
                    (rule2, True, rule2_tensor),
                    # This one is padding; the tensors shouldn't matter here.
                    (u'', False, None)],
                   [(rule3, True, rule3_tensor),
                    (u'instance_action', False, None),
                    (rule1, True, rule1_tensor)]]

        embedded_actions, _, _, action_indices = model._embed_actions(actions)
        assert action_indices[(0, 0)] == action_indices[(1, 2)]
        assert action_indices[(1, 1)] == -1
        assert len(set(action_indices.values())) == 4

        # Now we'll go through all three unique actions and make sure the embedding is as we expect.
        action_embedding = embedded_actions[action_indices[(0, 0)]]
        expected_action_embedding = action_embedding_weights[action_indices[(0, 0)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(0, 1)]]
        expected_action_embedding = action_embedding_weights[action_indices[(0, 1)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(1, 0)]]
        expected_action_embedding = action_embedding_weights[action_indices[(1, 0)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

    def test_map_entity_productions(self):
        # (batch_size, num_entities, num_question_tokens) = (3, 4, 5)
        linking_scores = torch.rand(3, 4, 5)
        # Because we only need a small piece of the WikiTablesWorld and TableKnowledgeGraph, we'll
        # just use some namedtuples to fake the part of the API that we need, instead of going to
        # the trouble of constructing the full objects.
        FakeTable = namedtuple(u'FakeTable', [u'entities'])
        FakeWorld = namedtuple(u'FakeWorld', [u'table_graph'])
        entities = [[u'fb:cell.2010', u'fb:cell.2011', u'fb:row.row.year', u'fb:row.row.year2'],
                    [u'fb:cell.2012', u'fb:cell.2013', u'fb:row.row.year'],
                    [u'fb:cell.2010', u'fb:row.row.year']]
        worlds = [FakeWorld(FakeTable(entity_list)) for entity_list in entities]
        # The tensors here for the global actions won't actually be read, so we're not constructing
        # them.
        # it.  Same with the RHS tensors.  NT* here is just saying "some non-terminal".
        actions = [[(u'@START@ -> r', True, None),
                    (u'@START@ -> c', True, None),
                    (u'@START@ -> <c,r>', True, None),
                    (u'c -> fb:cell.2010', False, None),
                    (u'c -> fb:cell.2011', False, None),
                    (u'<c,r> -> fb:row.row.year', False, None),
                    (u'<c,r> -> fb:row.row.year2', False, None)],
                   [(u'@START@ -> c', True, None),
                    (u'c -> fb:cell.2012', False, None),
                    (u'c -> fb:cell.2013', False, None),
                    (u'<c,r> -> fb:row.row.year', False, None)],
                   [(u'@START@ -> c', True, None),
                    (u'c -> fb:cell.2010', False, None),
                    (u'<c,r> -> fb:row.row.year', False, None)]]
        flattened_linking_scores, actions_to_entities =\
                WikiTablesMmlSemanticParser._map_entity_productions(linking_scores, worlds, actions)
        assert_almost_equal(flattened_linking_scores.detach().cpu().numpy(),
                            linking_scores.view(3 * 4, 5).detach().cpu().numpy())
        assert actions_to_entities == {
                (0, 3): 0,
                (0, 4): 1,
                (0, 5): 2,
                (0, 6): 3,
                (1, 1): 4,
                (1, 2): 5,
                (1, 3): 6,
                (2, 1): 8,
                (2, 2): 9,
                }

WikiTablesMmlSemanticParserTest = pytest.mark.java(WikiTablesMmlSemanticParserTest)
