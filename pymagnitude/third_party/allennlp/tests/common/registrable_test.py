# pylint: disable=no-self-use,invalid-name,too-many-public-methods


from __future__ import with_statement
from __future__ import absolute_import
import pytest
import torch
import torch.nn.init
import torch.optim.lr_scheduler

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn import Initializer
from allennlp.nn.regularizers.regularizer import Regularizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler


class TestRegistrable(AllenNlpTestCase):

    def test_registrable_functionality_works(self):
        # This function tests the basic `Registrable` functionality:
        #
        #   1. The decorator should add things to the list.
        #   2. The decorator should crash when adding a duplicate.
        #   3. If a default is given, it should show up first in the list.
        #
        # What we don't test here is that built-in items are registered correctly.  Those are
        # tested in the other tests below.
        #
        # We'll test this with the Tokenizer class, just to have a concrete class to use, and one
        # that has a default.
        base_class = Tokenizer
        assert u'fake' not in base_class.list_available()

        class Fake(base_class):
            # pylint: disable=abstract-method
            pass

        Fake = base_class.register(u'fake')(Fake)


        assert base_class.by_name(u'fake') == Fake

        default = base_class.default_implementation
        if default is not None:
            assert base_class.list_available()[0] == default
            base_class.default_implementation = u"fake"
            assert base_class.list_available()[0] == u"fake"

            with pytest.raises(ConfigurationError):
                base_class.default_implementation = u"not present"
                base_class.list_available()
            base_class.default_implementation = default

        del Registrable._registry[base_class][u'fake']  # pylint: disable=protected-access

    # TODO(mattg): maybe move all of these into tests for the base class?

    def test_registry_has_builtin_dataset_readers(self):
        assert DatasetReader.by_name(u'snli').__name__ == u'SnliReader'
        assert DatasetReader.by_name(u'sequence_tagging').__name__ == u'SequenceTaggingDatasetReader'
        assert DatasetReader.by_name(u'language_modeling').__name__ == u'LanguageModelingReader'
        assert DatasetReader.by_name(u'squad').__name__ == u'SquadReader'

    def test_registry_has_builtin_iterators(self):
        assert DataIterator.by_name(u'basic').__name__ == u'BasicIterator'
        assert DataIterator.by_name(u'bucket').__name__ == u'BucketIterator'

    def test_registry_has_builtin_tokenizers(self):
        assert Tokenizer.by_name(u'word').__name__ == u'WordTokenizer'
        assert Tokenizer.by_name(u'character').__name__ == u'CharacterTokenizer'

    def test_registry_has_builtin_token_indexers(self):
        assert TokenIndexer.by_name(u'single_id').__name__ == u'SingleIdTokenIndexer'
        assert TokenIndexer.by_name(u'characters').__name__ == u'TokenCharactersIndexer'

    def test_registry_has_builtin_regularizers(self):
        assert Regularizer.by_name(u'l1').__name__ == u'L1Regularizer'
        assert Regularizer.by_name(u'l2').__name__ == u'L2Regularizer'

    def test_registry_has_builtin_initializers(self):
        all_initializers = {
                u"normal": torch.nn.init.normal_,
                u"uniform": torch.nn.init.uniform_,
                u"orthogonal": torch.nn.init.orthogonal_,
                u"constant": torch.nn.init.constant_,
                u"dirac": torch.nn.init.dirac_,
                u"xavier_normal": torch.nn.init.xavier_normal_,
                u"xavier_uniform": torch.nn.init.xavier_uniform_,
                u"kaiming_normal": torch.nn.init.kaiming_normal_,
                u"kaiming_uniform": torch.nn.init.kaiming_uniform_,
                u"sparse": torch.nn.init.sparse_,
                u"eye": torch.nn.init.eye_,
        }
        for key, value in list(all_initializers.items()):
            # pylint: disable=protected-access
            assert Initializer.by_name(key)()._init_function == value

    def test_registry_has_builtin_learning_rate_schedulers(self):
        all_schedulers = {
                u"step": torch.optim.lr_scheduler.StepLR,
                u"multi_step": torch.optim.lr_scheduler.MultiStepLR,
                u"exponential": torch.optim.lr_scheduler.ExponentialLR,
                u"reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
        }
        for key, value in list(all_schedulers.items()):
            assert LearningRateScheduler.by_name(key) == value

    def test_registry_has_builtin_token_embedders(self):
        assert TokenEmbedder.by_name(u"embedding").__name__ == u'Embedding'
        assert TokenEmbedder.by_name(u"character_encoding").__name__ == u'TokenCharactersEncoder'

    def test_registry_has_builtin_text_field_embedders(self):
        assert TextFieldEmbedder.by_name(u"basic").__name__ == u'BasicTextFieldEmbedder'

    def test_registry_has_builtin_seq2seq_encoders(self):
        # pylint: disable=protected-access
        assert Seq2SeqEncoder.by_name(u'gru')._module_class.__name__ == u'GRU'
        assert Seq2SeqEncoder.by_name(u'lstm')._module_class.__name__ == u'LSTM'
        assert Seq2SeqEncoder.by_name(u'rnn')._module_class.__name__ == u'RNN'

    def test_registry_has_builtin_seq2vec_encoders(self):
        assert Seq2VecEncoder.by_name(u'cnn').__name__ == u'CnnEncoder'
        # pylint: disable=protected-access
        assert Seq2VecEncoder.by_name(u'gru')._module_class.__name__ == u'GRU'
        assert Seq2VecEncoder.by_name(u'lstm')._module_class.__name__ == u'LSTM'
        assert Seq2VecEncoder.by_name(u'rnn')._module_class.__name__ == u'RNN'

    def test_registry_has_builtin_similarity_functions(self):
        assert SimilarityFunction.by_name(u"dot_product").__name__ == u'DotProductSimilarity'
        assert SimilarityFunction.by_name(u"bilinear").__name__ == u'BilinearSimilarity'
        assert SimilarityFunction.by_name(u"linear").__name__ == u'LinearSimilarity'
        assert SimilarityFunction.by_name(u"cosine").__name__ == u'CosineSimilarity'
