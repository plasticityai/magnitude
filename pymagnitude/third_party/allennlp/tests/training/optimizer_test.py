# pylint: disable=invalid-name


from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator


class TestOptimizer(AllenNlpTestCase):
    def setUp(self):
        super(TestOptimizer, self).setUp()
        self.instances = SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv')
        vocab = Vocabulary.from_instances(self.instances)
        self.model_params = Params({
                u"text_field_embedder": {
                        u"tokens": {
                                u"type": u"embedding",
                                u"embedding_dim": 5
                                }
                        },
                u"encoder": {
                        u"type": u"lstm",
                        u"input_size": 5,
                        u"hidden_size": 7,
                        u"num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(vocab=vocab, params=self.model_params)

    def test_optimizer_basic(self):
        optimizer_params = Params({
                u"type": u"sgd",
                u"lr": 1
        })
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, optimizer_params)
        param_groups = optimizer.param_groups
        assert len(param_groups) == 1
        assert param_groups[0][u'lr'] == 1

    def test_optimizer_parameter_groups(self):
        optimizer_params = Params({
                u"type": u"sgd",
                u"lr": 1,
                u"momentum": 5,
                u"parameter_groups": [
                        # the repeated "bias_" checks a corner case
                        # NOT_A_VARIABLE_NAME displays a warning but does not raise an exception
                        [[u"weight_i", u"bias_", u"bias_", u"NOT_A_VARIABLE_NAME"], {u'lr': 2}],
                        [[u"tag_projection_layer"], {u'lr': 3}],
                ]
        })
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, optimizer_params)
        param_groups = optimizer.param_groups

        assert len(param_groups) == 3
        assert param_groups[0][u'lr'] == 2
        assert param_groups[1][u'lr'] == 3
        # base case uses default lr
        assert param_groups[2][u'lr'] == 1
        for k in range(3):
            assert param_groups[k][u'momentum'] == 5

        # all LSTM parameters except recurrent connections (those with weight_h in name)
        assert len(param_groups[0][u'params']) == 6
        # just the projection weight and bias
        assert len(param_groups[1][u'params']) == 2
        # the embedding + recurrent connections left in the default group
        assert len(param_groups[2][u'params']) == 3


class TestDenseSparseAdam(AllenNlpTestCase):

    def setUp(self):
        super(TestDenseSparseAdam, self).setUp()
        self.instances = SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv')
        self.vocab = Vocabulary.from_instances(self.instances)
        self.model_params = Params({
                u"text_field_embedder": {
                        u"tokens": {
                                u"type": u"embedding",
                                u"embedding_dim": 5,
                                u"sparse": True
                                }
                        },
                u"encoder": {
                        u"type": u"lstm",
                        u"input_size": 5,
                        u"hidden_size": 7,
                        u"num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)

    def test_can_optimise_model_with_dense_and_sparse_params(self):
        optimizer_params = Params({
                u"type": u"dense_sparse_adam"
        })
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, optimizer_params)
        iterator = BasicIterator(2)
        iterator.index_with(self.vocab)
        Trainer(self.model, optimizer, iterator, self.instances).train()
