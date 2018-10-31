# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestBasicTextFieldEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestBasicTextFieldEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace(u"1")
        self.vocab.add_token_to_namespace(u"2")
        self.vocab.add_token_to_namespace(u"3")
        self.vocab.add_token_to_namespace(u"4")
        params = Params({
                u"words1": {
                        u"type": u"embedding",
                        u"embedding_dim": 2
                        },
                u"words2": {
                        u"type": u"embedding",
                        u"embedding_dim": 5
                        },
                u"words3": {
                        u"type": u"embedding",
                        u"embedding_dim": 3
                        }
                })
        self.token_embedder = BasicTextFieldEmbedder.from_params(vocab=self.vocab, params=params)
        self.inputs = {
                u"words1": torch.LongTensor([[0, 2, 3, 5]]),
                u"words2": torch.LongTensor([[1, 4, 3, 2]]),
                u"words3": torch.LongTensor([[1, 5, 1, 2]])
                }

    def test_get_output_dim_aggregates_dimension_from_each_embedding(self):
        assert self.token_embedder.get_output_dim() == 10

    def test_forward_asserts_input_field_match(self):
        self.inputs[u'words4'] = self.inputs[u'words3']
        del self.inputs[u'words3']
        with pytest.raises(ConfigurationError):
            self.token_embedder(self.inputs)
        self.inputs[u'words3'] = self.inputs[u'words4']
        del self.inputs[u'words4']

    def test_forward_concats_resultant_embeddings(self):
        assert self.token_embedder(self.inputs).size() == (1, 4, 10)

    def test_forward_works_on_higher_order_input(self):
        params = Params({
                u"words": {
                        u"type": u"embedding",
                        u"num_embeddings": 20,
                        u"embedding_dim": 2,
                        },
                u"characters": {
                        u"type": u"character_encoding",
                        u"embedding": {
                                u"embedding_dim": 4,
                                u"num_embeddings": 15,
                                },
                        u"encoder": {
                                u"type": u"cnn",
                                u"embedding_dim": 4,
                                u"num_filters": 10,
                                u"ngram_filter_sizes": [3],
                                },
                        }
                })
        token_embedder = BasicTextFieldEmbedder.from_params(vocab=self.vocab, params=params)
        inputs = {
                u'words': (torch.rand(3, 4, 5, 6) * 20).long(),
                u'characters': (torch.rand(3, 4, 5, 6, 7) * 15).long(),
                }
        assert token_embedder(inputs, num_wrapping_dims=2).size() == (3, 4, 5, 6, 12)

    def test_forward_runs_with_non_bijective_mapping(self):
        elmo_fixtures_path = self.FIXTURES_ROOT / u'elmo'
        options_file = unicode(elmo_fixtures_path / u'options.json')
        weight_file = unicode(elmo_fixtures_path / u'lm_weights.hdf5')
        params = Params({
                u"words": {
                        u"type": u"embedding",
                        u"num_embeddings": 20,
                        u"embedding_dim": 2,
                        },
                u"elmo": {
                        u"type": u"elmo_token_embedder",
                        u"options_file": options_file,
                        u"weight_file": weight_file
                        },
                u"embedder_to_indexer_map": {u"words": [u"words"], u"elmo": [u"elmo", u"words"]}
                })
        token_embedder = BasicTextFieldEmbedder.from_params(self.vocab, params)
        inputs = {
                u'words': (torch.rand(3, 6) * 20).long(),
                u'elmo': (torch.rand(3, 6, 50) * 15).long(),
                }
        token_embedder(inputs)

    def test_old_from_params_new_from_params(self):

        old_params = Params({
                u"words1": {
                        u"type": u"embedding",
                        u"embedding_dim": 2
                        },
                u"words2": {
                        u"type": u"embedding",
                        u"embedding_dim": 5
                        },
                u"words3": {
                        u"type": u"embedding",
                        u"embedding_dim": 3
                        }
                })

        with pytest.warns(DeprecationWarning):
            BasicTextFieldEmbedder.from_params(params=old_params, vocab=self.vocab)

        new_params = Params({
                u"token_embedders": {
                        u"words1": {
                                u"type": u"embedding",
                                u"embedding_dim": 2
                                },
                        u"words2": {
                                u"type": u"embedding",
                                u"embedding_dim": 5
                                },
                        u"words3": {
                                u"type": u"embedding",
                                u"embedding_dim": 3
                                }
                        }
                })

        token_embedder = BasicTextFieldEmbedder.from_params(params=new_params, vocab=self.vocab)
        assert token_embedder(self.inputs).size() == (1, 4, 10)
